"""Independent regression tests for frozen inputs and support-union semantics."""
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from saa_baseline import data, intervals, model


ORIGIN = pd.Timestamp("2020-10-30T23:59:59Z")


def at(seconds):
    return ORIGIN + pd.Timedelta(seconds=seconds)


def times(*seconds):
    return pd.DatetimeIndex([at(s) for s in seconds])


def event(*pairs):
    return intervals.SupportEvent(tuple((at(a), at(b)) for a, b in pairs))


def detector_frame(dates=None):
    if dates is None:
        dates = pd.DatetimeIndex(["2020-10-20T00:00:00Z", "2020-10-20T00:00:01Z"])
    frame = pd.DataFrame({c: np.arange(len(dates), dtype=float) + int(c) for c in data.CHANNELS})
    frame["UTC"] = dates
    return frame


class IndependentIntervalReview(unittest.TestCase):
    def test_half_open_boundaries_and_adjacent_intervals(self):
        mask = intervals.rasterize(times(0, 1, 2, 3), [(at(0), at(1)), (at(1), at(2))])
        np.testing.assert_array_equal(mask, [True, True, False, False])
        self.assertEqual(intervals.merge_intervals([(at(1), at(2)), (at(0), at(1))]),
                         [(at(0), at(2))])
        self.assertEqual(intervals.intersection_seconds(event((0, 1)), event((1, 2))), 0)

    def test_gap_at_30_groups_without_inventing_support(self):
        runs = intervals.positive_runs(times(0, 30), [1, 1], cadence=1)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].pieces, ((at(0), at(1)), (at(30), at(31))))
        self.assertEqual(runs[0].seconds, 2)
        self.assertEqual(runs[0].envelope_seconds, 31)

    def test_gap_above_30_splits_at_nanosecond_precision(self):
        t = pd.DatetimeIndex([at(0), at(30), at(60) + pd.Timedelta(nanoseconds=1)])
        runs = intervals.positive_runs(t, [1, 1, 1], cadence=1)
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(runs[0].pieces), 2)
        self.assertEqual(runs[1].start, t[-1])

    def test_micro_jitter_creates_pieces_not_extra_events(self):
        t = pd.DatetimeIndex([at(0), at(1) + pd.Timedelta(nanoseconds=1),
                              at(2) + pd.Timedelta(nanoseconds=2)])
        runs = intervals.positive_runs(t, [1, 1, 1], cadence=1)
        self.assertEqual(len(runs), 1)
        self.assertEqual(len(runs[0].pieces), 3)
        self.assertEqual(runs[0].seconds, 3)
        self.assertAlmostEqual(runs[0].envelope_seconds, 3.000000002, places=12)

    def test_negative_breaks_positive_run(self):
        runs = intervals.positive_runs(times(0, 1, 2), [1, 0, 1], cadence=1)
        self.assertEqual(len(runs), 2)
        self.assertEqual([r.seconds for r in runs], [1, 1])
        self.assertEqual([r.pieces for r in runs], [((at(0), at(1)),), ((at(2), at(3)),)])

    def test_midnight_does_not_break_a_run(self):
        runs = intervals.positive_runs(times(0, 1, 2), [1, 1, 1], cadence=1)
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].pieces, ((at(0), at(3)),))
        self.assertNotEqual(runs[0].start.date(), runs[0].end.date())

    def test_support_is_clipped_by_next_observation_and_analysis_end(self):
        self.assertEqual(intervals.observation_support(times(0, .5, 5), 1, at(5.25)),
                         [(at(0), at(.5)), (at(.5), at(1.5)), (at(5), at(5.25))])

    def test_support_union_intersection_ignores_gaps(self):
        a, b = event((0, 1), (10, 11)), event((1, 10))
        self.assertEqual(intervals.intersection_seconds(a, b), 0)
        self.assertEqual(intervals.event_match([a], [b], .01)["matches"], 0)
        c = event((.5, 10.5))
        self.assertEqual(intervals.intersection_seconds(a, c), 1)
        match = intervals.event_match([a], [c], 1 / 11)
        self.assertEqual(match["matches"], 1)
        self.assertAlmostEqual(match["mean_matched_IoU"], 1 / 11)

    def test_matching_maximizes_cardinality_before_iou(self):
        reference = [event((0, 10)), event((10, 20))]
        predicted = [event((3, 15)), event((0, 4))]
        result = intervals.event_match(reference, predicted, .25)
        self.assertEqual(result["matches"], 2)
        self.assertAlmostEqual(result["mean_matched_IoU"], (.4 + 5 / 17) / 2)
        self.assertEqual(result["fraction_human_events_matched"], 1)
        self.assertEqual(result["fraction_predicted_runs_matched"], 1)

    def test_empty_runs_and_matches_are_defined(self):
        self.assertEqual(intervals.positive_runs(times(), [], 1), [])
        self.assertEqual(intervals.positive_runs(times(0, 1), [0, 0], 1), [])
        for reference, predicted, expected in [([], [], (None, None)),
                                                ([event((0, 1))], [], (0, None)),
                                                ([], [event((0, 1))], (None, 0))]:
            result = intervals.event_match(reference, predicted, .5)
            self.assertEqual(result["matches"], 0)
            self.assertIsNone(result["mean_matched_IoU"])
            self.assertEqual((result["fraction_human_events_matched"],
                              result["fraction_predicted_runs_matched"]), expected)

    def test_invalid_timestamps_and_binary_vectors_fail_closed(self):
        for t in [times(1, 0), times(0, 0), pd.DatetimeIndex([at(0), pd.NaT])]:
            with self.subTest(t=t), self.assertRaises(ValueError):
                intervals.positive_runs(t, [1, 1], 1)
        for labels in [[1], [1, 2], [[1], [0]], [1, np.nan]]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                intervals.positive_runs(times(0, 1), labels, 1)

    def test_invalid_numeric_parameters_and_end_fail_closed(self):
        for cadence in [0, -1, np.nan, np.inf]:
            with self.subTest(cadence=cadence), self.assertRaises(ValueError):
                intervals.observation_support(times(0, 1), cadence)
        for max_gap in [0, -1, np.nan, np.inf]:
            with self.subTest(max_gap=max_gap), self.assertRaises(ValueError):
                intervals.positive_runs(times(0, 1), [1, 1], 1, max_gap=max_gap)
        for end in [at(0), at(1), pd.NaT]:
            with self.subTest(end=end), self.assertRaises(ValueError):
                intervals.observation_support(times(0, 1), 1, end)
        for threshold in [0, -1, np.nan, np.inf, 1.1]:
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                intervals.event_match([], [], threshold)

    def test_random_interval_unions_against_integer_grid_oracle(self):
        rng = np.random.default_rng(20261024)
        for _ in range(100):
            pairs = []
            occupied = []
            for _side in range(2):
                starts = rng.integers(0, 16, size=5)
                lengths = rng.integers(1, 5, size=5)
                side = [(int(s), int(s + n)) for s, n in zip(starts, lengths)]
                pairs.append(event(*side))
                occupied.append(set().union(*(set(range(s, e)) for s, e in side)))
            a, b = pairs
            self.assertEqual(a.seconds, len(occupied[0]))
            self.assertEqual(b.seconds, len(occupied[1]))
            self.assertEqual(intervals.intersection_seconds(a, b), len(occupied[0] & occupied[1]))
            self.assertEqual(intervals.intersection_seconds(b, a), len(occupied[0] & occupied[1]))

    def test_random_matching_against_exhaustive_assignment(self):
        rng = np.random.default_rng(42)
        for _ in range(40):
            sides = []
            for _side in range(2):
                sides.append([event((int(s), int(s + n))) for s, n in
                              zip(rng.integers(0, 10, size=4), rng.integers(1, 8, size=4))])
            reference, predicted = sides
            scores = np.array([[intervals.intersection_seconds(a, b) /
                                (a.seconds + b.seconds - intervals.intersection_seconds(a, b))
                                for b in predicted] for a in reference])
            for threshold in [.1, .3, .5, 1.]:
                def oracle(i=0, used=frozenset()):
                    if i == len(reference):
                        return (0, 0.)
                    best = oracle(i + 1, used)
                    for j in range(len(predicted)):
                        if j not in used and scores[i, j] >= threshold:
                            count, total = oracle(i + 1, used | {j})
                            best = max(best, (count + 1, total + scores[i, j]))
                    return best
                count, total = oracle()
                result = intervals.event_match(reference, predicted, threshold)
                self.assertEqual(result["matches"], count)
                if count:
                    self.assertAlmostEqual(result["mean_matched_IoU"], total / count, places=12)
                else:
                    self.assertIsNone(result["mean_matched_IoU"])


class IndependentDataReview(unittest.TestCase):
    def test_normalization_preserves_bytes_columns_timestamps_order_and_eol(self):
        source_bytes = (b"annotator,date,start,end,label,source\r\n"
                        b'Martin Kakona,2020-10-22,2020-10-22T02:03:04.123456789+00:00,'
                        b'2020-10-22T02:04:00Z,SAA,"source, quoted"\r\n'
                        b'Martin,2020-10-20,2020-10-20T00:00:00Z,2020-10-20T00:00:01Z,SAA,original')
        with tempfile.TemporaryDirectory() as tmp:
            source, normalized, sidecar = [Path(tmp) / name for name in ["input.csv", "copy.csv", "copy.json"]]
            source.write_bytes(source_bytes)
            manifest = data.normalize_annotations(source, normalized, sidecar)
            self.assertEqual(source.read_bytes(), source_bytes)
            expected = source_bytes.replace(b"Martin Kakona,", b"Martin,")
            self.assertEqual(normalized.read_bytes(), expected)
            self.assertEqual(manifest, json.loads(sidecar.read_text()))
            self.assertEqual(manifest["rows"], 2)
            self.assertEqual(manifest["row_provenance"], [
                {"data_row_1_based": 1, "original_annotator": "Martin Kakona"},
                {"data_row_1_based": 2, "original_annotator": "Martin"}])
            a = list(csv.reader(io.StringIO(source_bytes.decode(), newline="")))
            b = list(csv.reader(io.StringIO(expected.decode(), newline="")))
            self.assertEqual(a[0], b[0])
            self.assertEqual([r[1:] for r in a[1:]], [r[1:] for r in b[1:]])

    def test_normalization_rejects_unknown_annotator_without_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, normalized, sidecar = [Path(tmp) / name for name in ["input.csv", "copy.csv", "copy.json"]]
            source.write_text("annotator,date,start,end,label,source\nAnother person,d,s,e,SAA,x\n")
            with self.assertRaises(ValueError):
                data.normalize_annotations(source, normalized, sidecar)
            self.assertFalse(normalized.exists())
            self.assertFalse(sidecar.exists())

    def test_normalization_refuses_source_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, sidecar = Path(tmp) / "input.csv", Path(tmp) / "copy.json"
            original = b"annotator,date,start,end,label,source\n"
            source.write_bytes(original)
            with self.assertRaises(ValueError):
                data.normalize_annotations(source, source, sidecar)
            self.assertEqual(source.read_bytes(), original)

    def test_feature_whitelist_excludes_all_comparison_and_label_columns(self):
        frame = detector_frame()
        expected = np.log1p(frame[list(data.CHANNELS)].to_numpy(float))
        frame["SAA"] = [0, 1]
        frame["SRAG"] = ["not a model input", None]
        frame["marked"] = [True, False]
        frame["label"] = ["SAA", "unmarked"]
        frame["8"], frame["28"] = [np.nan, np.inf], [-1, -2]
        np.testing.assert_array_equal(data.features(frame), expected)
        frame["SAA"], frame["marked"] = [1, 0], [False, True]
        np.testing.assert_array_equal(data.features(frame), expected)
        self.assertEqual(data.features(frame).shape, (2, 19))
        self.assertEqual(data.CHANNELS, tuple(map(str, range(9, 28))))

    def test_duplicate_missing_and_invalid_detector_channels_fail_closed(self):
        frame = detector_frame()
        duplicate = pd.concat([frame, frame[["9"]]], axis=1)
        for invalid in [duplicate, frame.drop(columns="9")]:
            with self.subTest(columns=list(invalid)), self.assertRaises(ValueError):
                data.features(invalid)
        for value in [np.nan, np.inf, -1]:
            bad = frame.copy()
            bad.loc[0, "9"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                data.features(bad)

    def test_partition_boundary_membership_and_event_guard(self):
        frame = detector_frame(pd.DatetimeIndex(["2020-10-30T23:59:59Z", "2020-10-31T00:00:00Z",
                                                 "2020-11-03T23:59:59Z", "2020-11-04T00:00:00Z",
                                                 "2020-11-07T23:59:59Z"]))
        masks = data.partition_masks(frame, [])
        np.testing.assert_array_equal(masks["train"], [1, 0, 0, 0, 0])
        np.testing.assert_array_equal(masks["validation_not_retuned"], [0, 1, 1, 0, 0])
        np.testing.assert_array_equal(masks["retrospective_holdout"], [0, 0, 0, 1, 1])
        for cut in [data.CUT_TRAIN, data.CUT_TEST]:
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                data.partition_masks(frame, [(cut - pd.Timedelta(seconds=1), cut + pd.Timedelta(seconds=1))])
            data.partition_masks(frame, [(cut - pd.Timedelta(seconds=1), cut)])
            data.partition_masks(frame, [(cut, cut + pd.Timedelta(seconds=1))])


class IndependentModelReview(unittest.TestCase):
    def test_frozen_configuration_and_training_inputs(self):
        expected = {"n_estimators": 100, "max_depth": 10, "min_samples_leaf": 30,
                    "class_weight": "balanced", "random_state": 42, "n_jobs": 2}
        self.assertEqual(model.RF_PARAMETERS, expected)
        self.assertEqual(model.THRESHOLD, .75)
        frame = detector_frame()
        frame["SAA"], frame["SRAG"], frame["label"] = [1, 0], [1, 0], [1, 0]
        with patch.object(model, "RandomForestClassifier") as rf:
            estimator = model.train(frame, [0, 1])
            rf.assert_called_once_with(**expected)
            self.assertIs(estimator, rf.return_value)
            x, y = estimator.fit.call_args.args
            np.testing.assert_array_equal(x, data.features(frame))
            np.testing.assert_array_equal(y, [0, 1])
            self.assertEqual(x.shape[1], 19)

    def test_training_rejects_exact_cutoff_later_and_missing_timestamps(self):
        for forbidden in [data.CUT_TRAIN, data.CUT_TEST, pd.Timestamp("2020-11-07", tz="UTC"), pd.NaT]:
            frame = detector_frame(pd.DatetimeIndex([pd.Timestamp("2020-10-20", tz="UTC"), forbidden]))
            with self.subTest(forbidden=forbidden), patch.object(model, "RandomForestClassifier") as rf:
                with self.assertRaises(ValueError):
                    model.train(frame, [0, 1])
                rf.assert_not_called()

    def test_training_rejects_label_matrices_and_invalid_classes(self):
        for labels in [[[0, 1], [1, 0]], [[0], [1]], [0], [0, 2], [0, np.nan], [0, 0]]:
            with self.subTest(labels=labels), patch.object(model, "RandomForestClassifier") as rf:
                with self.assertRaises(ValueError):
                    model.train(detector_frame(), labels)
                rf.assert_not_called()

    def test_threshold_is_inclusive_and_class_order_is_respected(self):
        class Estimator:
            classes_ = np.array([1, 0])

            def predict_proba(self, x):
                self.x = x
                return np.array([[.75, .25], [.749999, .250001]])

        estimator = Estimator()
        probabilities, predicted = model.predict(estimator, detector_frame())
        np.testing.assert_array_equal(probabilities, [.75, .749999])
        np.testing.assert_array_equal(predicted, [True, False])
        self.assertEqual(estimator.x.shape, (2, 19))

    def test_predict_cli_ignores_optional_invalid_comparison_mask(self):
        from saa_baseline import __main__ as cli

        class Estimator:
            classes_ = np.array([0, 1])

            def predict_proba(self, x):
                return np.array([[.2, .8], [.9, .1]])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw, artifact, output = root / "raw.csv", root / "model.joblib", root / "predictions.csv"
            artifact.write_bytes(b"mock artifact, not loaded")
            frame = detector_frame(pd.DatetimeIndex(["2020-11-04T00:00:00Z", "2020-11-04T00:00:01Z"]))
            frame["SAA"] = ["not used", np.nan]
            frame["label"] = ["not used either", "also ignored"]
            frame.to_csv(raw, index=False)
            argv = ["saa_baseline", "predict", "--model", str(artifact), "--raw", str(raw),
                    "--output", str(output)]
            with patch("sys.argv", argv), patch.object(cli, "load_model", return_value=(Estimator(), {})), \
                    patch("sys.stdout", new_callable=io.StringIO):
                cli.main()
            result = pd.read_csv(output)
            np.testing.assert_array_equal(result.model_prediction, [1, 0])
            self.assertEqual(list(result), ["UTC", "model_probability", "model_prediction"])


if __name__ == "__main__":
    unittest.main()
