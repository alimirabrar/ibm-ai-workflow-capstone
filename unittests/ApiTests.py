import os
import sys
import json
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

class ApiTest(unittest.TestCase):

    def setUp(self):
        app.app.testing = True
        self.client = app.app.test_client()

    def test_01_index(self):
        """
        Test that the index endpoint returns alive status.
        """
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'alive')

    def test_02_train(self):
        """
        Test that train endpoint works with valid input.
        """
        payload = {'mode': 'test'}
        response = self.client.post('/train', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')

    def test_03_predict(self):
        """
        Test that predict endpoint returns valid prediction.
        """
        payload = {
            'mode': 'test',
            'query': {'country': 'United Kingdom', 'year': 2018, 'month': 1, 'day': 5}
        }
        response = self.client.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'ok')
        self.assertIn('y_pred', data)

    def test_04_predict_all(self):
        """
        Test predict for multiple countries.
        """
        countries = ['United Kingdom', 'Germany', 'France']
        for country in countries:
            payload = {
                'mode': 'test',
                'query': {'country': country, 'year': 2018, 'month': 6, 'day': 1}
            }
            response = self.client.post('/predict', data=json.dumps(payload), content_type='application/json')
            self.assertEqual(response.status_code, 200)

    def test_05_predict_missing_params(self):
        """
        Test that predict returns error for missing params.
        """
        payload = {'mode': 'test', 'query': {'country': 'United Kingdom'}}
        response = self.client.post('/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
