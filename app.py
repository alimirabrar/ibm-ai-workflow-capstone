import os
import json
import re
from flask import Flask, request, jsonify
from model import model_train, model_predict, model_load
from logger import update_predict_log, update_train_log

app = Flask(__name__)

@app.route('/')
def index():
    return jsonify({'status': 'alive', 'message': 'AAVAIL Revenue Prediction API'})

@app.route('/train', methods=['POST'])
def train():
    if not request.json:
        return jsonify({'status': 'error', 'message': 'Missing JSON body'}), 400
    mode = request.json.get('mode', 'test')
    if mode not in ['test', 'prod']:
        return jsonify({'status': 'error', 'message': 'mode must be test or prod'}), 400
    result = model_train(data_dir=os.path.join('data', 'cs-' + mode), test=True if mode=='test' else False)
    update_train_log(mode=mode, runtime=result.get('runtime', 0), model_version=result.get('model_version', '0.1'), metrics=result.get('metrics', {}))
    return jsonify({'status': 'ok', 'mode': mode, 'result': result})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.json:
        return jsonify({'status': 'error', 'message': 'Missing JSON body'}), 400
    query = request.json.get('query', {})
    if not query:
        return jsonify({'status': 'error', 'message': 'Missing query'}), 400
    country = query.get('country', None)
    year = query.get('year', None)
    month = query.get('month', None)
    day = query.get('day', None)
    if not all([country, year, month, day]):
        return jsonify({'status': 'error', 'message': 'query must contain country, year, month, day'}), 400
    mode = request.json.get('mode', 'test')
    result = model_predict(country=country, year=int(year), month=int(month), day=int(day), test=True if mode=='test' else False)
    update_predict_log(country=country, y_pred=result['y_pred'], target_date='{}-{}-{}'.format(year,month,day), model_version=result.get('model_version','0.1'), runtime=result.get('runtime',0))
    return jsonify({'status': 'ok', 'y_pred': result['y_pred'], 'model_version': result.get('model_version','0.1')})

@app.route('/logs/<filename>', methods=['GET'])
def logs(filename):
    log_dir = os.path.join('logs')
    log_path = os.path.join(log_dir, filename)
    if not os.path.isfile(log_path):
        return jsonify({'status': 'error', 'message': 'file not found'}), 404
    with open(log_path, 'r') as f:
        content = f.read()
    return content, 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
