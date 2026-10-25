"""Descriptive, retrospective analysis. This module cannot certify independence."""
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.metrics import confusion_matrix, f1_score
from .data import (CHANNELS, ANALYSIS_START, ANALYSIS_END, CUT_TEST, analysis_data,
                   load_annotations, partition_masks, sha256)
from .intervals import (SupportEvent, event_match, intersection_seconds,
                        legacy_envelope_runs, merge_intervals, positive_runs, rasterize)
from .model import RF_PARAMETERS, THRESHOLD, predict, save_model, train


def agreement(reference, prediction):
    a, b = np.asarray(reference), np.asarray(prediction)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or not np.isin(a, [0, 1]).all() or not np.isin(b, [0, 1]).all():
        raise ValueError("Aligned binary vectors are required.")
    if not len(a):
        return {"n": 0, "both_marked": 0, "A_only": 0, "B_only": 0,
                "neither_marked": 0, "positive_overlap_F1": None, "positive_overlap_Jaccard": None}
    tn, fp, fn, tp = confusion_matrix(a, b, labels=[0, 1]).ravel()
    return {"n": len(a), "both_marked": int(tp), "A_only": int(fn),
            "B_only": int(fp), "neither_marked": int(tn),
            "positive_overlap_F1": float(f1_score(a, b, zero_division=0)),
            "positive_overlap_Jaccard": float(tp / (tp + fp + fn)) if tp + fp + fn else None}


def write_event_tables(events, folder, stem):
    folder = Path(folder)
    summary, pieces = [], []
    for i, event in enumerate(events):
        summary.append({"run_id": i, "start": event.start, "end": event.end,
                        "supported_seconds": event.seconds,
                        "envelope_seconds": event.envelope_seconds,
                        "unsupported_inside_envelope_seconds": event.envelope_seconds - event.seconds,
                        "support_pieces": len(event.pieces), "human_reviewed": False})
        for j, (s, e) in enumerate(event.pieces):
            pieces.append({"run_id": i, "piece_id": j, "start": s, "end": e})
    pd.DataFrame(summary, columns=["run_id", "start", "end", "supported_seconds", "envelope_seconds",
                                  "unsupported_inside_envelope_seconds", "support_pieces", "human_reviewed"]).to_csv(folder / f"{stem}_runs.csv", index=False)
    pd.DataFrame(pieces, columns=["run_id", "piece_id", "start", "end"]).to_csv(folder / f"{stem}_support_pieces.csv", index=False)


def evaluate_predictions(predictions, annotations, cadence, end=None):
    p = predictions.copy()
    if not {"UTC", "model_prediction"} <= set(p):
        raise ValueError("Predictions require UTC and model_prediction.")
    p["UTC"] = pd.to_datetime(p.UTC, utc=True, format="ISO8601")
    if not p.model_prediction.isin([0, 1]).all():
        raise ValueError("model_prediction must be binary.")
    human = merge_intervals(zip(annotations.start, annotations.end))
    support = positive_runs(p.UTC, p.model_prediction, cadence, analysis_end=end)
    y = rasterize(p.UTC, human)
    # Evaluate only complete marks in the explicitly supplied observation window.
    # If a human interval crosses the window, reject rather than silently clip it.
    if len(p):
        window_start = p.UTC.min()
        window_end = end if end is not None else p.UTC.max() + pd.Timedelta(seconds=cadence)
        if any(s < window_start < e or s < window_end < e for s, e in human):
            raise ValueError("A reference interval crosses the prediction window; specify an eligible window.")
        human = [(s, e) for s, e in human if s >= window_start and e <= window_end]
    else:
        human = []
    reference = [SupportEvent((h,)) for h in human]
    scores = [event_match(reference, support, t) for t in (0.1, 0.3, 0.5)]
    return agreement(y, p.model_prediction), scores, support


def create_figure(frame, human, srag, pred_events, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    start = pd.Timestamp("2020-11-03T23:00:00Z")
    end = pd.Timestamp("2020-11-04T03:00:00Z")
    g = frame[(frame.UTC >= start) & (frame.UTC < end)]
    fig, (ax, masks) = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True,
                                  gridspec_kw={"height_ratios": [3.4, 1.6]}, constrained_layout=True)
    for _, group in g.groupby("segment_id"):
        ax.plot(group.UTC, group[list(CHANNELS)].sum(axis=1), lw=0.8, color="#243b63")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_ylabel("Sum of detector counts\n(channels 9–27; no inferred unit)")
    ax.set_title("Continuous reference overlays across midnight", loc="left", fontweight="bold")
    groups = [("Martin marks", [SupportEvent((h,)) for h in human], "#24805b", 2),
              ("Supplied SRAG", srag, "#d58b25", 1),
              ("Fixed RF ≥ 0.75", pred_events, "#8155a2", 0)]
    for name, events, color, ypos in groups:
        for event in events:
            for s, e in event.pieces:
                if e > start and s < end:
                    s, e = max(s, start), min(e, end)
                    masks.broken_barh([(mdates.date2num(s), (e - s).total_seconds() / 86400)],
                                      (ypos - 0.24, 0.48), facecolors=color)
    masks.set_yticks([0, 1, 2], [g[0] for g in groups[::-1]])
    masks.set_ylim(-0.6, 2.6)
    masks.set_xlabel("UTC · 3–4 November 2020 · midnight is not an event boundary")
    masks.set_xlim(start, end)
    masks.xaxis.set_major_formatter(mdates.DateFormatter("%d Nov\n%H:%M", tz="UTC"))
    masks.axvline(pd.Timestamp("2020-11-04T00:00Z"), color="#666", ls=":", lw=0.8)
    ax.grid(alpha=0.15)
    masks.set_title("Interval masks, not adjudicated physical passages; RF shown only after the retrospective split", fontsize=9, loc="left")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def reproduce(raw_path, marks_path, output, legacy_dir=None):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    frame, marks, human, cadence, hashes = analysis_data(raw_path, marks_path)
    masks = partition_masks(frame, human)
    estimator = train(frame.loc[masks["train"]], frame.marked.to_numpy()[masks["train"]].astype(int))
    save_model(estimator, out / "frozen_rf.joblib", {"source_hashes": hashes,
               "cadence_seconds": cadence, "train_end_exclusive": "2020-10-31T00:00:00Z",
               "independent_final_test_used": False})
    held = frame.loc[masks["retrospective_holdout"]].copy()
    probability, prediction = predict(estimator, held)
    p = held[["UTC", "segment_id", "SAA", "marked"]].copy()
    p["model_probability"] = probability
    p["model_prediction"] = prediction
    p.to_csv(out / "model_predictions.csv", index=False)
    split_table = pd.DataFrame([{"partition": name, "n": int(mask.sum()),
                                "start": frame.loc[mask, "UTC"].min(),
                                "last_observation": frame.loc[mask, "UTC"].max()}
                               for name, mask in masks.items()])
    split_table.to_csv(out / "split_manifest.csv", index=False)
    sample_scores, support_scores, pred_events = evaluate_predictions(p, marks, cadence, ANALYSIS_END)
    retrospective = pd.DataFrame([
        {"comparison": "frozen RF vs Martin marked/unmarked", **sample_scores},
        {"comparison": "frozen RF vs SRAG (secondary overlap)", **agreement(p.SAA, prediction)},
        {"comparison": "always unmarked vs Martin", **agreement(p.marked, np.zeros(len(p), dtype=int))}])
    retrospective.to_csv(out / "retrospective_model_agreement.csv", index=False)
    pd.DataFrame(support_scores).to_csv(out / "supported_event_overlap.csv", index=False)
    write_event_tables(pred_events, out, "model")
    legacy_pred = legacy_envelope_runs(p.UTC, prediction, cadence)
    reference = [SupportEvent((h,)) for h in human if h[0] >= CUT_TEST]
    legacy_scores = [event_match(reference, legacy_pred, t) for t in (0.1, 0.3, 0.5)]
    pd.DataFrame(legacy_scores).to_csv(out / "legacy_envelope_event_overlap.csv", index=False)
    srag = positive_runs(frame.UTC, frame.SAA, cadence, analysis_end=ANALYSIS_END)
    legacy_srag = legacy_envelope_runs(frame.UTC, frame.SAA, cadence)
    write_event_tables(srag, out, "supplied_SRAG")
    contain = []
    for i, h in enumerate(human):
        event = SupportEvent((h,))
        supported = sum(intersection_seconds(event, r) for r in srag)
        envelope = sum(intersection_seconds(event, r) for r in legacy_srag)
        contain.append({"human_event": i, "start": h[0], "end": h[1],
                        "human_duration_seconds": event.seconds,
                        "SRAG_supported_overlap_fraction": supported / event.seconds,
                        "legacy_SRAG_envelope_overlap_fraction": envelope / event.seconds})
    containment = pd.DataFrame(contain)
    containment.to_csv(out / "human_SRAG_containment_sensitivity.csv", index=False)
    timing = []
    for shift in (-60, 0, 60):
        delta = pd.Timedelta(seconds=shift)
        y = rasterize(frame.UTC, [(s + delta, e + delta) for s, e in human])
        timing.append({"human_shift_seconds": shift, **agreement(frame.SAA, y),
                       "human_marked_outside_SRAG": int((y & ~frame.SAA.astype(bool)).sum())})
    pd.DataFrame(timing).to_csv(out / "timing_sensitivity.csv", index=False)
    sensitivity = []
    for name, fixed, old in [("RF retrospective positive runs", pred_events, legacy_pred),
                              ("Supplied SRAG positive runs, whole window", srag, legacy_srag)]:
        sensitivity.append({"series": name, "positive_runs": len(fixed),
                            "legacy_positive_runs": len(old),
                            "support_pieces": sum(len(e.pieces) for e in fixed),
                            "supported_seconds": sum(e.seconds for e in fixed),
                            "legacy_envelope_seconds": sum(e.seconds for e in old),
                            "unsupported_seconds_previously_bridged": sum(e.seconds for e in old) - sum(e.seconds for e in fixed)})
    pd.DataFrame(sensitivity).to_csv(out / "support_sensitivity.csv", index=False)
    parity = {"legacy_prediction_comparison_run": False}
    if legacy_dir is not None:
        old = pd.read_csv(Path(legacy_dir) / "model_predictions.csv")
        old_times = pd.to_datetime(old.UTC, utc=True, format="ISO8601")
        if not np.array_equal(old_times.to_numpy(), p.UTC.to_numpy()):
            raise AssertionError("Legacy prediction timestamps differ.")
        max_abs = float(np.max(np.abs(old.model_probability.to_numpy() - probability)))
        disagreements = int(np.sum(old.model_prediction.to_numpy() != prediction))
        if disagreements or max_abs > 1e-14:
            raise AssertionError("Frozen RF did not reproduce legacy predictions.")
        parity = {"legacy_prediction_comparison_run": True, "rows": len(p),
                  "prediction_disagreements": disagreements,
                  "maximum_absolute_probability_difference": max_abs,
                  "legacy_predictions_sha256": sha256(Path(legacy_dir) / "model_predictions.csv")}
    create_figure(frame, human, srag, pred_events, out / "continuous_window.png")
    manifest = {
        "prepared_utc": "2026-10-24", "source_hashes": hashes,
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "analysis_window": [str(ANALYSIS_START), str(ANALYSIS_END)],
        "analysis_rows": len(frame), "annotation_rows": len(marks), "human_intervals": len(human),
        "canonical_annotator": "Martin", "annotator_count": 1,
        "median_cadence_seconds": cadence, "max_acquisition_gap_seconds": 30.,
        "model": "RandomForestClassifier", "parameters": RF_PARAMETERS, "threshold": THRESHOLD,
        "unmarked_is_verified_negative": False, "SRAG_is_ground_truth": False,
        "independent_final_test_used": False, "independent_test_available": False,
        "human_annotation_completed": False, "publication_authorized": False,
        "license_selected": False, "whole_source_file_is_developer_accessible": True,
        "support_policy": "Capped sample-support union within positive runs; no calendar cuts or gap filling",
        "support_pieces_are_physical_events": False,
        "retrospective_sample_scores": sample_scores,
        "support_sensitivity": sensitivity, "supported_event_scores": support_scores,
        "legacy_prediction_parity": parity,
        "human_events_fully_contained_legacy": int((containment.legacy_SRAG_envelope_overlap_fraction >= 1 - 1e-9).sum()),
        "human_events_fully_contained_supported": int((containment.SRAG_supported_overlap_fraction >= 1 - 1e-9).sum()),
        "note": "Private retrospective reproduction only. A volunteer-managed undisclosed eligible segment is not supplied. No new human labels."}
    (out / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
