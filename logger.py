import os
import csv
import time
from datetime import date

LOG_DIR = 'logs'

def _ensure_log_dir():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def update_predict_log(country, y_pred, target_date, model_version, runtime):
    """
    Log predictions to a CSV file.
    """
    _ensure_log_dir()
    log_file = os.path.join(LOG_DIR, 'predict_log.csv')
    header = ['timestamp', 'country', 'y_pred', 'target_date', 'model_version', 'runtime']
    row = [str(date.today()), country, str(y_pred), target_date, model_version, str(runtime)]
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def update_train_log(mode, runtime, model_version, metrics):
    """
    Log training runs to a CSV file.
    """
    _ensure_log_dir()
    log_file = os.path.join(LOG_DIR, 'train_log.csv')
    header = ['timestamp', 'mode', 'runtime', 'model_version', 'metrics']
    row = [str(date.today()), mode, str(runtime), model_version, str(metrics)]
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

def load_log(log_file):
    """
    Load a log file and return as list of dicts.
    """
    log_path = os.path.join(LOG_DIR, log_file)
    if not os.path.isfile(log_path):
        return []
    rows = []
    with open(log_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows

if __name__ == '__main__':
    update_train_log(mode='test', runtime=1.5, model_version='0.1', metrics={'rmse': 100.0})
    update_predict_log(country='United Kingdom', y_pred=[1200.0], target_date='2018-01-01', model_version='0.1', runtime=0.01)
    print('Predict log:', load_log('predict_log.csv'))
    print('Train log:', load_log('train_log.csv'))
