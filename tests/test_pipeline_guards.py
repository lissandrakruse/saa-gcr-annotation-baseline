"""Synthetic software fixtures only; no fabricated human/scientific observations."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from saa_baseline.analysis import agreement, evaluate_predictions
from saa_baseline.data import CHANNELS, CUT_TRAIN, CUT_TEST, load_raw, partition_masks
from saa_baseline.intervals import SupportEvent, event_match, observation_support, positive_runs
from saa_baseline.notebook_runner import execute_notebook


class PipelineGuards(unittest.TestCase):
    def test_empty_sample_agreement_explicit_nulls(self):
        score = agreement([], [])
        self.assertEqual(score['n'], 0)
        self.assertIsNone(score['positive_overlap_F1'])

    def test_binary_length_guard(self):
        for x, y in [([0], []), ([2], [0]), ([[0]], [[0]])]:
            with self.assertRaises(ValueError):
                agreement(x, y)

    def test_crossing_split_rejected(self):
        frame = pd.DataFrame({'UTC': pd.to_datetime(['2020-10-30T00:00Z', '2020-11-01T00:00Z', '2020-11-05T00:00Z'])})
        with self.assertRaises(ValueError):
            partition_masks(frame, [(CUT_TRAIN-pd.Timedelta(seconds=1),CUT_TRAIN+pd.Timedelta(seconds=1))])

    def test_half_open_partitions(self):
        frame = pd.DataFrame({'UTC': [CUT_TRAIN-pd.Timedelta(seconds=1),CUT_TRAIN,CUT_TEST]})
        parts = partition_masks(frame, [])
        np.testing.assert_array_equal(parts['train'],[1,0,0])
        np.testing.assert_array_equal(parts['validation_not_retuned'],[0,1,0])
        np.testing.assert_array_equal(parts['retrospective_holdout'],[0,0,1])

    def test_crossing_evaluation_window_rejected(self):
        p = pd.DataFrame({'UTC': pd.to_datetime(['2020-11-04T00:00:00Z','2020-11-04T00:00:10Z']), 'model_prediction':[1,1]})
        m = pd.DataFrame({'start':pd.to_datetime(['2020-11-03T23:59:59Z']), 'end':pd.to_datetime(['2020-11-04T00:00:05Z'])})
        with self.assertRaises(ValueError):
            evaluate_predictions(p,m,10,pd.Timestamp('2020-11-04T00:00:20Z'))

    def test_duplicate_csv_headers_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'bad.csv'
            p.write_text('UTC,9,9\n2020-01-01T00:00:00Z,1,2\n')
            with self.assertRaises(ValueError):
                load_raw(p)

    def test_exact_gap_threshold_and_cap(self):
        t=pd.to_datetime(['2020-01-01T00:00:00Z','2020-01-01T00:00:30Z'])
        runs=positive_runs(t,[1,1],10,max_gap=30)
        self.assertEqual(len(runs),1)
        self.assertEqual(len(runs[0].pieces),2)
        self.assertEqual(runs[0].seconds,20)
        self.assertEqual(runs[0].envelope_seconds,40)

    def test_last_support_clipped_to_analysis_end(self):
        t=pd.to_datetime(['2020-01-01T00:00:00Z'])
        end=pd.Timestamp('2020-01-01T00:00:05Z')
        self.assertEqual(observation_support(t,10,end),[(t[0],end)])

    def test_nan_binary_predictions_rejected(self):
        with self.assertRaises(ValueError):
            positive_runs(pd.to_datetime(['2020-01-01T00:00:00Z']),[np.nan],10)

    def test_notebook_runner_executes_normal_python(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            source=p/'input.ipynb'; dest=p/'output.ipynb'
            source.write_text(json.dumps({'nbformat':4,'nbformat_minor':5,'metadata':{},'cells':[
                {'cell_type':'code','id':'test','metadata':{},'source':['x=2\n','print(x+3)\n'],'outputs':[],'execution_count':None}]}))
            self.assertEqual(execute_notebook(source,dest,p),1)
            result=json.loads(dest.read_text())
            self.assertEqual(result['cells'][0]['outputs'][0]['text'],'5\n')


if __name__=='__main__':
    unittest.main()
