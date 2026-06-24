import os
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

MODEL_VERSION = '0.1'
MODEL_DIR = 'models'

def _load_data(data_dir, test=True):
    """
    Load JSON data files and return a combined DataFrame.
    """
    if not os.path.isdir(data_dir):
        raise Exception('data directory not found: {}'.format(data_dir))
    files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
    dfs = []
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        with open(fpath, 'r') as f:
            data = pd.read_json(f)
        dfs.append(data)
    df = pd.concat(dfs, ignore_index=True)
    return df

def _engineer_features(df):
    """
    Create feature matrix from the dataframe.
    Returns X (features) and y (target)
    """
    df = df.copy()
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['year'] = df['invoice_date'].dt.year
    df['month'] = df['invoice_date'].dt.month
    df['day'] = df['invoice_date'].dt.day
    df['day_of_week'] = df['invoice_date'].dt.dayofweek
    df['price'] = df['price'].astype(float)
    df['quantity'] = df['quantity'].astype(float)
    df['revenue'] = df['price'] * df['quantity']
    feature_cols = ['year', 'month', 'day', 'day_of_week', 'quantity', 'price']
    target_col = 'revenue'
    X = df[feature_cols].values
    y = df[target_col].values
    return X, y

def model_train(data_dir=os.path.join('data', 'cs-test'), test=True):
    """
    Train the model and save to disk.
    """
    start = time.time()
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    df = _load_data(data_dir, test=test)
    X, y = _engineer_features(df)
    n_samples = int(0.8 * len(X))
    X_train, X_test = X[:n_samples], X[n_samples:]
    y_train, y_test = y[:n_samples], y[n_samples:]
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    runtime = round(time.time() - start, 2)
    model_name = 'model_{}.pkl'.format(MODEL_VERSION)
    model_path = os.path.join(MODEL_DIR, model_name)
    with open(model_path, 'wb') as f:
        pickle.dump(pipe, f)
    return {'model_version': MODEL_VERSION, 'runtime': runtime, 'metrics': {'rmse': round(rmse, 2)}, 'n_samples': len(X)}

def model_load(test=True):
    """
    Load a saved model.
    """
    model_path = os.path.join(MODEL_DIR, 'model_{}.pkl'.format(MODEL_VERSION))
    if not os.path.exists(model_path):
        raise Exception('model not found: {}'.format(model_path))
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def model_predict(country, year, month, day, test=True):
    """
    Make a prediction for a given country and date.
    """
    start = time.time()
    model = model_load(test=test)
    # Use dummy day_of_week and recent averages for prediction
    import datetime
    target = datetime.date(year, month, day)
    day_of_week = target.weekday()
    # Use typical averages for quantity and price
    X_pred = np.array([[year, month, day, day_of_week, 50, 25.0]])
    y_pred = model.predict(X_pred)
    runtime = round(time.time() - start, 4)
    return {'y_pred': [round(float(y_pred[0]), 2)], 'model_version': MODEL_VERSION, 'runtime': runtime}

if __name__ == '__main__':
    result = model_train()
    print('Training result:', result)
    pred = model_predict('United Kingdom', 2018, 1, 1, test=True)
    print('Prediction:', pred)
