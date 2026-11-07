import json
import unittest
import numpy as np
from scipy.optimize import check_grad
from fphelper.data import decode_label, totals, LOW, HIGH
from fphelper.ordinal import objective, OrderedLogit
from fphelper.training import prepared_split, inner_splits
from fphelper.metrics import absolute_agreement_icc


class TrainingTests(unittest.TestCase):
    def test_score_bounds_and_missing_labels(self):
        label = {"rest symmetry": {"eye": 0, "cheek": 0, "mouth": 0},
                 "voluntary symmetry": {k: 5 for k in ("forehead wrinkle", "eye closure", "smile", "snarl", "lip pucker")},
                 "synkinesis": {k: 0 for k in ("forehead wrinkle", "eye closure", "smile", "snarl", "lip pucker")}}
        self.assertEqual(totals(decode_label(json.dumps(label))), 100)
        label["voluntary symmetry"]["smile"] = 0
        with self.assertRaises(ValueError):
            decode_label(json.dumps(label))
        label["voluntary symmetry"]["smile"] = True
        with self.assertRaises(ValueError):
            decode_label(json.dumps(label))
        del label["voluntary symmetry"]["smile"]
        with self.assertRaises(KeyError):
            decode_label(json.dumps(label))

    def test_ordinal_gradient_matches_numerical_derivative(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(12, 4))
        z = (rng.integers(1, 6, size=12)[:, None] > np.arange(1, 5)).astype(float)
        parameters = rng.normal(size=8)
        error = check_grad(lambda p: objective(p, x, z, 0.1)[0],
                           lambda p: objective(p, x, z, 0.1)[1], parameters)
        self.assertLess(error, 1e-6)

    def test_ordinal_learns_ordered_signal_and_legal_probabilities(self):
        x = np.linspace(-2, 2, 120)[:, None]
        y = np.digitize(x[:, 0], [-1.2, -0.4, 0.4, 1.2]) + 1
        model = OrderedLogit(1, 5, 0.01).fit(x, y)
        prediction = model.predict(x)
        self.assertLess(np.mean(np.abs(prediction - y)), 0.3)
        self.assertTrue(np.all(np.diff(prediction) >= 0))
        probability = model.predict_proba(x)
        self.assertTrue(np.all(probability >= 0))
        np.testing.assert_allclose(probability.sum(axis=1), 1)
        constant = OrderedLogit(0, 3).fit(x, np.zeros(len(x), dtype=int))
        np.testing.assert_array_equal(constant.predict(x), 0)

    def test_projector_never_fits_validation_rows(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(25, 6, 24))
        altered = x.copy()
        altered[20:] += 1000
        train, test = np.arange(20), np.arange(20, 25)
        first, first_appearance, *_ = prepared_split(x, x, train, test, 7)
        second, second_appearance, *_ = prepared_split(altered, altered, train, test, 7)
        for a, b in zip(first.scalers + first_appearance.scalers, second.scalers + second_appearance.scalers):
            np.testing.assert_array_equal(a.mean_, b.mean_)
        for a, b in zip(first.pcas + first_appearance.pcas, second.pcas + second_appearance.pcas):
            np.testing.assert_array_equal(a.components_, b.components_)

    def test_duplicate_video_group_isolation(self):
        labels = np.tile(LOW, (30, 1))
        groups = np.repeat(np.arange(15), 2)
        for train, test in inner_splits(labels, groups, 7):
            self.assertFalse(set(groups[train]) & set(groups[test]))
        self.assertAlmostEqual(absolute_agreement_icc([1, 2, 3], [1, 2, 3]), 1)
        self.assertLess(absolute_agreement_icc([1, 2, 3], [11, 12, 13]), 0.1)


if __name__ == "__main__":
    unittest.main()
