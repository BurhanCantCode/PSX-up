"""
Stock Analysis API Endpoint
Triggers E2B sandbox for ML computations and returns results.
"""

import os
import json
from http.server import BaseHTTPRequestHandler

try:
    from e2b_code_interpreter import Sandbox
    E2B_AVAILABLE = True
except ImportError:
    E2B_AVAILABLE = False

# Full stock analyzer script to be executed in E2B sandbox
# This is embedded directly to ensure availability in Vercel serverless environment
STOCK_ANALYZER_SCRIPT = '''
"""
PSX Stock Analyzer - E2B Sandbox Execution Script
"""

import subprocess
import re
import json
import sys
import warnings
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def fetch_month_data(symbol, month, year):
    url = "https://dps.psx.com.pk/historical"
    post_data = f"month={month}&year={year}&symbol={symbol}"
    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST', url, '-d', post_data],
            capture_output=True, text=True, timeout=15
        )
        return result.stdout if result.returncode == 0 else None
    except:
        return None


def parse_html_table(html):
    rows = re.findall(r'<tr>.*?</tr>', html, re.DOTALL)
    data = []
    for row in rows:
        cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
        if len(cells) >= 6:
            try:
                date_obj = datetime.strptime(cells[0].strip(), "%b %d, %Y")
                data.append({
                    'Date': date_obj.strftime('%Y-%m-%d'),
                    'Open': float(cells[1].strip().replace(',', '')),
                    'High': float(cells[2].strip().replace(',', '')),
                    'Low': float(cells[3].strip().replace(',', '')),
                    'Close': float(cells[4].strip().replace(',', '')),
                    'Volume': float(cells[5].strip().replace(',', ''))
                })
            except:
                continue
    return data


def fetch_historical_data(symbol, start_year=2020):
    symbol = symbol.upper()
    all_data = []
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    for year in range(start_year, current_year + 1):
        end_month = current_month if year == current_year else 12
        for month in range(1, end_month + 1):
            html = fetch_month_data(symbol, month, year)
            if html:
                month_data = parse_html_table(html)
                if month_data:
                    all_data.extend(month_data)
    return all_data


def wavelet_denoise_causal(signal, wavelet='db4', lookback=100, update_freq=5):
    if not PYWT_AVAILABLE:
        return signal
    
    n = len(signal)
    denoised = np.zeros(n)
    min_samples = 32
    
    update_points = list(range(min_samples, n, update_freq))
    if n - 1 not in update_points and n > min_samples:
        update_points.append(n - 1)
    
    computed_values = {}
    
    for t in update_points:
        start_idx = max(0, t + 1 - lookback)
        window = signal[start_idx:t+1]
        try:
            level = min(pywt.dwt_max_level(len(window), wavelet), 3)
            if level < 1:
                computed_values[t] = window[-1]
                continue
            coeffs = pywt.wavedec(window, wavelet, level=level)
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(window)))
            denoised_coeffs = [coeffs[0]]
            for i in range(1, len(coeffs)):
                denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            reconstructed = pywt.waverec(denoised_coeffs, wavelet)
            computed_values[t] = reconstructed[-1]
        except:
            computed_values[t] = window[-1]
    
    for t in range(min(min_samples, n)):
        denoised[t] = signal[:t+1].mean() if t > 0 else signal[0]
    
    sorted_points = sorted(computed_values.keys())
    for i, t in enumerate(sorted_points):
        denoised[t] = computed_values[t]
        if i > 0:
            prev_t = sorted_points[i-1]
            for j in range(prev_t + 1, t):
                alpha = (j - prev_t) / (t - prev_t)
                denoised[j] = computed_values[prev_t] + alpha * (computed_values[t] - computed_values[prev_t])
    
    return denoised


class ExponentialGatingFeatures:
    def __init__(self, decay_rates=[0.9, 0.95, 0.99]):
        self.decay_rates = decay_rates
    
    def fit_transform(self, prices):
        features = {}
        for rate in self.decay_rates:
            alpha = 1 - rate
            rate_name = str(rate).replace('.', '_')
            ema = pd.Series(prices).ewm(alpha=alpha, adjust=False).mean().values
            features[f'ema_{rate_name}'] = ema
            returns = np.diff(prices, prepend=prices[0])
            features[f'ema_returns_{rate_name}'] = pd.Series(returns).ewm(alpha=alpha, adjust=False).mean().values
            features[f'ewm_vol_{rate_name}'] = pd.Series(returns).ewm(alpha=alpha, adjust=False).std().values
            features[f'ema_deviation_{rate_name}'] = (prices - ema) / (ema + 1e-8)
        
        for window in [5, 10, 21, 63]:
            if len(prices) > window:
                series = pd.Series(prices)
                features[f'rolling_mean_{window}'] = series.rolling(window).mean().values
                features[f'rolling_std_{window}'] = series.rolling(window).std().values
                roll_min = series.rolling(window).min().values
                roll_max = series.rolling(window).max().values
                roll_range = roll_max - roll_min
                features[f'range_position_{window}'] = np.where(roll_range > 0, (prices - roll_min) / roll_range, 0.5)
        return pd.DataFrame(features)


def trend_accuracy(y_true, y_pred):
    if len(y_true) < 2:
        return 0.5
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    true_dir[true_dir == 0] = 1
    pred_dir[pred_dir == 0] = 1
    return np.sum(true_dir == pred_dir) / len(true_dir)


def evaluate_model(y_true, y_pred):
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'trend_accuracy': trend_accuracy(y_true, y_pred)
    }
    metrics['rmse'] = np.sqrt(metrics['mse'])
    naive_mae = np.mean(np.abs(np.diff(y_true)))
    metrics['mase'] = metrics['mae'] / naive_mae if naive_mae > 0 else float('inf')
    non_zero = y_true != 0
    if np.any(non_zero):
        metrics['mape'] = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
    else:
        metrics['mape'] = 0
    return metrics


class SOTAEnsemblePredictor:
    PSX_HOLIDAYS = {
        '2025-02-05', '2025-03-23', '2025-03-31', '2025-04-01', '2025-05-01',
        '2025-06-07', '2025-06-08', '2025-06-09', '2025-08-14', '2025-09-05',
        '2025-09-06', '2025-11-09', '2025-12-25',
        '2026-02-05', '2026-03-20', '2026-03-21', '2026-03-23', '2026-05-01',
        '2026-05-27', '2026-05-28', '2026-08-14', '2026-08-25', '2026-08-26',
        '2026-11-09', '2026-12-25',
    }
    
    def __init__(self, use_wavelet=True):
        self.use_wavelet = use_wavelet and PYWT_AVAILABLE
        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.exp_gating = ExponentialGatingFeatures()
        self.feature_names = []
        self.is_fitted = False
        self._init_models()
    
    def _init_models(self):
        self.base_models = {
            'rf': RandomForestRegressor(n_estimators=300, max_depth=25, min_samples_split=5, n_jobs=-1, random_state=42),
            'et': ExtraTreesRegressor(n_estimators=300, max_depth=25, min_samples_split=5, n_jobs=-1, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        if XGBOOST_AVAILABLE:
            self.base_models['xgb'] = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, n_jobs=-1, random_state=42, verbosity=0)
        if LIGHTGBM_AVAILABLE:
            self.base_models['lgbm'] = LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, n_jobs=-1, random_state=42, verbose=-1)
    
    def preprocess(self, df):
        df = df.copy()
        if self.use_wavelet:
            for col in ['Close', 'Open', 'High', 'Low']:
                if col in df.columns:
                    df[f'{col}_denoised'] = wavelet_denoise_causal(df[col].values)
        
        close_prices = df['Close_denoised'].values if 'Close_denoised' in df.columns else df['Close'].values
        exp_features = self.exp_gating.fit_transform(close_prices)
        for col in exp_features.columns:
            df[col] = exp_features[col].values
        return df
    
    def prepare_features(self, df):
        df = self.preprocess(df)
        exclude_cols = ['Date', 'Target']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
        df['Target'] = df['Close'].shift(-1)
        df_clean = df.dropna()
        return df_clean[feature_cols].values, df_clean['Target'].values, feature_cols
    
    def fit(self, df):
        X, y, feature_cols = self.prepare_features(df)
        self.feature_names = feature_cols
        X_scaled = self.scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=3)
        validation_scores = {name: [] for name in self.base_models.keys()}
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            for name, model in self.base_models.items():
                from sklearn.base import clone
                m = clone(model)
                m.fit(X_train, y_train)
                y_pred = m.predict(X_val)
                validation_scores[name].append(trend_accuracy(y_val, y_pred))
        
        avg_scores = {name: np.mean(scores) for name, scores in validation_scores.items()}
        total = sum(avg_scores.values())
        self.model_weights = {name: score / total for name, score in avg_scores.items()}
        
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)
            self.models[name] = model
        
        self.is_fitted = True
        y_pred = self._predict_ensemble(X_scaled)
        return evaluate_model(y, y_pred)
    
    def _predict_ensemble(self, X_scaled):
        preds = []
        weights = []
        for name, model in self.models.items():
            preds.append(model.predict(X_scaled))
            weights.append(self.model_weights.get(name, 1.0))
        weights = np.array(weights) / np.sum(weights)
        return np.average(np.array(preds), axis=0, weights=weights)
    
    def predict_daily(self, df, end_date='2026-12-31', seed=42):
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        np.random.seed(seed)
        predictions = []
        current_df = df.copy()
        current_price = df['Close'].iloc[-1]
        current_date = pd.to_datetime(df['Date'].iloc[-1])
        end_date_obj = pd.to_datetime(end_date)
        
        day_offset = 0
        max_days = 500
        
        while current_date < end_date_obj and day_offset < max_days:
            day_offset += 1
            next_date = current_date + pd.Timedelta(days=1)
            while next_date.dayofweek >= 5:
                next_date += pd.Timedelta(days=1)
            current_date = next_date
            
            date_str = current_date.strftime('%Y-%m-%d')
            if date_str in self.PSX_HOLIDAYS:
                continue
            
            if current_date > end_date_obj:
                break
            
            try:
                X, _, _ = self.prepare_features(current_df)
                if len(X) == 0:
                    break
                X_scaled = self.scaler.transform(X[-1:])
            except:
                break
            
            all_preds = []
            for model in self.models.values():
                try:
                    all_preds.append(model.predict(X_scaled)[0])
                except:
                    continue
            
            if not all_preds:
                break
            
            ensemble_pred = np.mean(all_preds)
            prev_close = current_df['Close'].iloc[-1]
            ensemble_pred = np.clip(ensemble_pred, prev_close * 0.925, prev_close * 1.075)
            
            pred_std = np.std(all_preds)
            uncertainty = 1.0 * (1.02 ** (day_offset / 21))
            
            predictions.append({
                'date': date_str,
                'day_offset': day_offset,
                'predicted_price': round(float(ensemble_pred), 2),
                'lower_ci': round(float(max(0, ensemble_pred - 2 * pred_std * uncertainty)), 2),
                'upper_ci': round(float(ensemble_pred + 2 * pred_std * uncertainty), 2),
                'upside_potential': round(float((ensemble_pred - current_price) / current_price * 100), 2),
                'confidence': round(float(max(0, min(1, 1 - pred_std * uncertainty / (abs(ensemble_pred) + 1e-8)))), 3)
            })
            
            new_row = current_df.iloc[-1:].copy()
            new_row['Date'] = current_date
            new_row['Close'] = ensemble_pred
            new_row['Open'] = ensemble_pred * (1 + np.random.uniform(-0.005, 0.005))
            new_row['High'] = ensemble_pred * (1 + np.random.uniform(0.002, 0.015))
            new_row['Low'] = ensemble_pred * (1 - np.random.uniform(0.002, 0.015))
            new_row['Volume'] = current_df['Volume'].iloc[-20:].mean() * np.random.uniform(0.8, 1.2)
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        return predictions


def analyze_stock(symbol):
    symbol = symbol.upper()
    
    try:
        raw_data = fetch_historical_data(symbol, start_year=2020)
        if not raw_data:
            return {'symbol': symbol, 'status': 'error', 'error': f'No data found for {symbol}'}
        
        df = pd.DataFrame(raw_data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        model = SOTAEnsemblePredictor(use_wavelet=PYWT_AVAILABLE)
        metrics = model.fit(df)
        daily_predictions = model.predict_daily(df, end_date='2026-12-31')
        
        history_df = df.tail(180)[['Date', 'Close']].copy()
        history_df['Date'] = history_df['Date'].dt.strftime('%Y-%m-%d')
        
        return {
            'status': 'complete',
            'symbol': symbol,
            'current_price': float(df['Close'].iloc[-1]),
            'data_points': len(df),
            'model': 'SOTA Ensemble (RF, ET, GB, XGBoost, LightGBM, Ridge)',
            'model_performance': {k: float(v) for k, v in metrics.items()},
            'wavelet_denoising': PYWT_AVAILABLE,
            'features_used': len(model.feature_names),
            'daily_predictions': daily_predictions,
            'historical_data': history_df.to_dict('records'),
            'generated_at': datetime.now().isoformat()
        }
    except Exception as e:
        return {'symbol': symbol, 'status': 'error', 'error': str(e)}


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'LUCK'
    result = analyze_stock(symbol)
    print(json.dumps(result))
'''


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            body = json.loads(post_data.decode('utf-8'))
            
            symbol = body.get('symbol', '').upper().strip()
            
            if not symbol:
                self._send_response(400, {'error': 'Symbol is required'})
                return
            
            if not symbol.isalpha() or len(symbol) > 10:
                self._send_response(400, {'error': 'Invalid symbol format'})
                return
            
            if not E2B_AVAILABLE:
                self._send_response(500, {'error': 'E2B SDK not available'})
                return
            
            result = self._run_analysis(symbol)
            self._send_response(200, result)
            
        except json.JSONDecodeError:
            self._send_response(400, {'error': 'Invalid JSON'})
        except Exception as e:
            self._send_response(500, {'error': str(e)})
    
    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()
    
    def _run_analysis(self, symbol: str) -> dict:
        """Run stock analysis in E2B sandbox."""
        
        try:
            with Sandbox() as sbx:
                # Install required packages
                install_result = sbx.commands.run(
                    "pip install pandas numpy scikit-learn xgboost lightgbm PyWavelets --quiet",
                    timeout=120
                )
                
                # Write and execute the analyzer script
                sbx.files.write('/home/user/analyze.py', STOCK_ANALYZER_SCRIPT)
                
                execution = sbx.commands.run(
                    f"python /home/user/analyze.py {symbol}",
                    timeout=300
                )
                
                if execution.exit_code == 0 and execution.stdout:
                    try:
                        # Find JSON in output
                        for line in execution.stdout.strip().split('\n'):
                            if line.strip().startswith('{'):
                                return json.loads(line.strip())
                        return {
                            'status': 'error',
                            'error': 'No valid JSON output',
                            'raw': execution.stdout[:500]
                        }
                    except json.JSONDecodeError as e:
                        return {
                            'status': 'error',
                            'error': f'JSON parse error: {str(e)}',
                            'raw': execution.stdout[:500]
                        }
                else:
                    return {
                        'status': 'error',
                        'error': execution.stderr or 'Analysis failed',
                        'exit_code': execution.exit_code
                    }
                    
        except Exception as e:
            return {
                'status': 'error',
                'error': f'E2B execution error: {str(e)}'
            }
    
    def _send_response(self, status_code: int, data: dict):
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self._set_cors_headers()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))
    
    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
