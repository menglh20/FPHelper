"""Version folders select independent feature families while preserving imports."""
import importlib
from pathlib import Path
import unittest

import numpy as np

from fphelper.data import LOW
from fphelper.versions import VERSION_FOLDERS, get_model


class AlgorithmVersionTests(unittest.TestCase):
    def test_registered_versions_exist_under_algorithm(self):
        root = Path(__file__).resolve().parents[2] / 'algorithm'
        for family, folder in VERSION_FOLDERS.items():
            module = get_model(family)
            self.assertEqual(module.FAMILY, family)
            self.assertEqual(module.VERSION, folder)
            self.assertEqual(Path(module.__file__).resolve(), root / folder / 'model.py')

    def test_feature_selection_and_fusion(self):
        geometry = [np.full((3, 2), i) for i in range(6)]
        appearance = [np.full((3, 4), i + 20) for i in range(6)]
        self.assertIs(get_model('SB1').inputs_for(geometry, appearance), geometry)
        self.assertIs(get_model('SB2').inputs_for(geometry, appearance), appearance)
        result = get_model('SB3').inputs_for(geometry, appearance)
        for g, a, result in zip(geometry, appearance, result):
            np.testing.assert_array_equal(result[:, :2], g)
            np.testing.assert_array_equal(result[:, 2:], a)

    def test_baseline_uses_training_mode_with_original_tie_break(self):
        labels = np.tile(LOW, (4, 1))
        labels[:2, 3] = 2
        model = get_model('SB0')
        fitted = model.fit(labels)
        prediction = model.predict(fitted, 2)
        self.assertEqual(prediction.shape, (2, 13))
        self.assertTrue((prediction[:, 3] == 1).all())

    def test_saved_model_class_and_ordinal_module_paths_stay_available(self):
        training = importlib.import_module('fphelper.training')
        ordinal = importlib.import_module('fphelper.ordinal')
        self.assertEqual(training.SunnybrookModel.__module__, 'fphelper.training')
        self.assertEqual(ordinal.OrderedLogit.__module__, 'fphelper.ordinal')
        self.assertIn('/algorithm/shared/fphelper/', str(Path(training.__file__).resolve()))


if __name__ == '__main__':
    unittest.main()
