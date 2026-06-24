import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from logger import update_predict_log, update_train_log, load_log

class LoggerTest(unittest.TestCase):

    def test_01_train_log(self):
        """
        Test that training log is updated without error.
        """
        update_train_log(mode='test', runtime=1.0, model_version='0.1', metrics={'rmse': 100.0})
        logs = load_log('train_log.csv')
        self.assertIsInstance(logs, list)
        self.assertTrue(len(logs) > 0)

    def test_02_predict_log(self):
        """
        Test that predict log is updated without error.
        """
        update_predict_log(country='United Kingdom', y_pred=[1200.0], target_date='2018-01-01', model_version='0.1', runtime=0.01)
        logs = load_log('predict_log.csv')
        self.assertIsInstance(logs, list)
        self.assertTrue(len(logs) > 0)

    def test_03_log_contents(self):
        """
        Test that log entries contain expected keys.
        """
        logs = load_log('predict_log.csv')
        if logs:
            self.assertIn('country', logs[0])
            self.assertIn('model_version', logs[0])

if __name__ == '__main__':
    unittest.main()
