"""Clinical evaluation invariants: denominators, illegal scores and paired rows."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from fphelper.data import LOW, HIGH
from metrics import evaluate, legal_mask, numeric_mask, paired, binary_metrics
from run import run_legacy, check_environment


class EvaluationTests(unittest.TestCase):
    def test_incomplete_and_invalid_outputs_are_different(self):
        value = np.tile(LOW.astype(float), (3, 1))
        value[1, 0] = np.nan
        value[2, 3] = 0
        np.testing.assert_array_equal(numeric_mask(value), [True, False, True])
        np.testing.assert_array_equal(legal_mask(value), [True, False, False])

    def test_out_of_range_prediction_is_not_dropped_from_confusion_or_mae(self):
        truth = np.tile(LOW, (2, 1))
        pred = truth.copy()
        pred[0, 3] = 0
        result = evaluate(truth, pred)
        item = result['subscores']['vol_forehead_wrinkle']
        self.assertEqual(item['mae'], .5)
        self.assertEqual(item['illegal_predictions'], 1)
        self.assertEqual(np.asarray(item['confusion_matrix']).sum(), 2)
        self.assertIn(0, item['confusion_levels'])
        self.assertEqual(result['total_mae'], 2)
        self.assertEqual(result['invalid_item_rows'], 1)

    def test_raw_negative_total_is_retained(self):
        truth = np.tile(HIGH, (3, 1))
        pred = truth.copy()
        pred[:, 3:8] = 1
        result = evaluate(truth, pred)
        self.assertEqual(result['invalid_raw_total_rows'], 3)
        self.assertEqual(result['total_mae'], 80)
        self.assertEqual(result['bias'], -80)

    def test_three_dimensions_have_equal_weight(self):
        truth = np.tile(LOW, (4, 1))
        pred = truth.copy()
        pred[:, 3:8] = HIGH[3:8]
        result = evaluate(truth, pred)
        self.assertAlmostEqual(result['equal_dimension_normalized_mae'], 1 / 3)
        self.assertEqual(result['dimension_normalized_mae']['voluntary'], 1)

    def test_paired_comparison_uses_identical_complete_rows(self):
        truth = np.tile(LOW, (4, 1))
        first, second = truth.astype(float), truth.astype(float)
        first[0, 0] = np.nan
        second[1, 0] = np.nan
        first[2, 3] += 1
        second[3, 3] += 2
        result = paired(truth, first, second)
        self.assertEqual(result['count'], 2)
        self.assertEqual(result['first_mae'], 2)
        self.assertEqual(result['second_mae'], 4)
        self.assertEqual(result['mae_difference'], -2)

    def test_zero_dominance_does_not_imply_synkinesis_detection(self):
        result = binary_metrics([0] * 99 + [1], [0] * 100)
        self.assertEqual(result['recall'], 0)
        self.assertEqual(result['false_positive_rate'], 0)
        self.assertEqual(result['fn'], 1)
        self.assertIsNone(result['precision'])

    def test_identical_varying_scores_have_perfect_absolute_agreement(self):
        truth = np.array([LOW, HIGH, LOW + (HIGH - LOW) // 2])
        result = evaluate(truth, truth.copy())
        self.assertEqual(result['total_mae'], 0)
        self.assertAlmostEqual(result['icc_absolute_single'], 1)
        self.assertEqual(result['within_5'], 1)

    def test_parent_cleans_temporary_images_when_worker_times_out(self):
        import tempfile
        import json
        touched = []
        def timeout(command, **kwargs):
            job = json.loads(Path(command[-1]).read_text())
            image = Path(job['temp_root']) / 'temporary-patient-frame.jpg'
            image.write_bytes(b'synthetic')
            touched.append(image)
            raise subprocess.TimeoutExpired(command, 1)
        with tempfile.TemporaryDirectory() as directory:
            with patch('run.subprocess.run', side_effect=timeout):
                result = run_legacy({'method': 'synthetic'}, Path(directory), 1)
            self.assertEqual(result['error_type'], 'TimeoutExpired')
            self.assertFalse(touched[0].exists())

    def test_cache_reuse_rejects_changed_or_unversioned_runtime(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            check_environment(root, {'python': 'synthetic-version-A'})
            check_environment(root, {'python': 'synthetic-version-A'})
            with self.assertRaises(RuntimeError):
                check_environment(root, {'python': 'synthetic-version-B'})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'legacy-cache').mkdir()
            (root / 'legacy-cache/result.json').write_text('{}')
            with self.assertRaises(RuntimeError):
                check_environment(root, {'python': 'synthetic-version-A'})


if __name__ == '__main__':
    unittest.main()
