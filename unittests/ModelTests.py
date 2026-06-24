import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model import model_train, model_load, model_predict

class ModelTest(unittest.TestCase):

    def test_01_train(self):
        """
        Test that model training runs without error.
        """
        result = model_train(data_dir=os.path.join('data', 'cs-test'), test=True)
        self.assertIsInstance(result, dict)
        self.assertIn('model_version', result)
        self.assertIn('runtime', result)
        self.assertIn('metrics', result)

    def test_02_load(self):
        """
        Test that model can be loaded after training.
        """
        model = model_load(test=True)
        self.assertIsNotNone(model)

    def test_03_predict(self):
        """
        Test that model returns a valid prediction.
        """
        result = model_predict(country='United Kingdom', year=2018, month=1, day=5, test=True)
        self.assertIn('y_pred', result)
        self.assertIsInstance(result['y_pred'], list)
        self.assertTrue(len(result['y_pred']) > 0)

    def test_04_predict_all(self):
        """
        Test prediction for multiple countries.
        """
        countries = ['United Kingdom', 'Germany', 'France']
        for country in countries:
            result = model_predict(country=country, year=2018, month=6, day=1, test=True)
            self.assertIn('y_pred', result)

if __name__ == '__main__':
    unittest.main()
