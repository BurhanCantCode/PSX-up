#!/usr/bin/env python3
"""
🎯 WILLIAMS %R CLASSIFIER (Matching Awan et al. 2021)

This module implements the exact methodology from:
"Prediction of KSE-100 using SVM with Williams %R" - 85% accuracy

Key differences from regression model:
1. CLASSIFICATION (up/down) not regression (exact price)
2. Uses Williams %R as PRIMARY feature
3. Weekly prediction horizon (less noise than daily)
4. SVM with RBF kernel as primary model
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, Tuple


class WilliamsRClassifier:
    """
    SVM Classifier using Williams %R as primary feature.
    
    Based on Awan et al. (2021) methodology for PSX prediction.
    """
    
    def __init__(self, prediction_horizon: int = 5):
        """
        Args:
            prediction_horizon: Days ahead to predict (5 = weekly)
        """
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.model = SVC(
            kernel='rbf',
            C=100,
            gamma='scale',
            class_weight='balanced',  # Handle imbalanced classes
            probability=True,
            random_state=42
        )
        self.is_fitted = False
        
        # Feature columns (Williams %R focused)
        self.feature_cols = [
            'williams_r',
            'williams_overbought',
            'williams_oversold',
            'disparity_5',
            'disparity_10',
            'rsi_14',
            'macd',
            'momentum_5',
        ]
    
    def prepare_classification_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and binary target for classification.
        
        Target: 1 if price goes UP in next N days, 0 if DOWN
        """
        df = df.copy()
        
        # Calculate Williams %R if not present
        if 'williams_r' not in df.columns:
            high = df['High']
            low = df['Low']
            close = df['Close']
            period = 14
            highest_high = high.rolling(period).max()
            lowest_low = low.rolling(period).min()
            df['williams_r'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-8)
            df['williams_overbought'] = (df['williams_r'] > -20).astype(int)
            df['williams_oversold'] = (df['williams_r'] < -80).astype(int)
        
        # Calculate other features if missing
        close = df['Close']
        if 'disparity_5' not in df.columns:
            ma5 = close.rolling(5).mean()
            df['disparity_5'] = 100 * (close - ma5) / (ma5 + 1e-8)
        if 'disparity_10' not in df.columns:
            ma10 = close.rolling(10).mean()
            df['disparity_10'] = 100 * (close - ma10) / (ma10 + 1e-8)
        if 'rsi_14' not in df.columns:
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            df['rsi_14'] = 100 - (100 / (1 + rs))
        if 'macd' not in df.columns:
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            df['macd'] = ema12 - ema26
        if 'momentum_5' not in df.columns:
            df['momentum_5'] = close / close.shift(5) - 1
        
        # Create binary target: 1 if price UP in N days, 0 if DOWN
        df['future_return'] = close.shift(-self.prediction_horizon) / close - 1
        df['target'] = (df['future_return'] > 0).astype(int)
        
        # Drop NaN
        df = df.dropna(subset=self.feature_cols + ['target'])
        
        X = df[self.feature_cols].values
        y = df['target'].values
        
        return X, y
    
    def fit(self, df: pd.DataFrame, verbose: bool = True) -> Dict:
        """
        Train the classifier with walk-forward validation.
        """
        X, y = self.prepare_classification_data(df)
        
        if verbose:
            print(f"📊 Training Williams %R Classifier")
            print(f"   Samples: {len(X)}")
            print(f"   Features: {len(self.feature_cols)}")
            print(f"   Prediction horizon: {self.prediction_horizon} days")
            print(f"   Class balance: {np.mean(y):.2%} UP, {1-np.mean(y):.2%} DOWN")
        
        # Walk-forward cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Scale
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train
            self.model.fit(X_train_scaled, y_train)
            
            # Validate
            y_pred = self.model.predict(X_val_scaled)
            acc = accuracy_score(y_val, y_pred)
            cv_scores.append(acc)
        
        # Final training on all data
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        mean_acc = np.mean(cv_scores)
        
        if verbose:
            print(f"\n   📈 Cross-Validation Results:")
            print(f"      Mean Accuracy: {mean_acc:.2%}")
            print(f"      Fold Scores: {[f'{s:.2%}' for s in cv_scores]}")
        
        return {
            'mean_accuracy': mean_acc,
            'cv_scores': cv_scores,
            'samples': len(X),
        }
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict direction (UP=1, DOWN=0).
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X, _ = self.prepare_classification_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict probability of UP direction.
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet")
        
        X, _ = self.prepare_classification_data(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]


def run_classifier_backtest(symbol: str, verbose: bool = True) -> Dict:
    """
    Run backtest of Williams %R Classifier matching Awan et al. methodology.
    
    Train on pre-2024 data, test on 2024.
    """
    import json
    
    data_file = Path(__file__).parent.parent / "data" / f"{symbol}_historical_with_indicators.json"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Split at 2024
    cutoff = pd.to_datetime('2024-01-01')
    train_df = df[df['Date'] < cutoff].copy()
    test_df = df[df['Date'] >= cutoff].copy()
    
    if verbose:
        print("=" * 70)
        print(f"🎯 WILLIAMS %R CLASSIFIER BACKTEST: {symbol}")
        print("=" * 70)
        print(f"\n   Training: {len(train_df)} records ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
        print(f"   Testing: {len(test_df)} records ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    # Train classifier
    clf = WilliamsRClassifier(prediction_horizon=5)  # Weekly predictions
    train_metrics = clf.fit(train_df, verbose=verbose)
    
    # Get test data with proper features
    X_test, y_test = clf.prepare_classification_data(test_df)
    
    # Predict
    X_test_scaled = clf.scaler.transform(X_test)
    y_pred = clf.model.predict(X_test_scaled)
    
    # Metrics
    test_accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    if verbose:
        print(f"\n   📊 TEST RESULTS (2024):")
        print(f"      🎯 ACCURACY: {test_accuracy:.2%}")
        print(f"\n      Confusion Matrix:")
        print(f"         Predicted DOWN | Predicted UP")
        print(f"         Actual DOWN: {cm[0][0]:4d} | {cm[0][1]:4d}")
        print(f"         Actual UP:   {cm[1][0]:4d} | {cm[1][1]:4d}")
        print(f"\n      Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['DOWN', 'UP']))
        
        # Compare with research benchmark
        print(f"\n   📋 vs Research Benchmark:")
        if test_accuracy >= 0.85:
            print(f"      ✅ MATCHES Awan et al. (2021) benchmark of 85%!")
        elif test_accuracy >= 0.70:
            print(f"      👍 GOOD accuracy (70-85%), approaching research benchmark")
        elif test_accuracy >= 0.55:
            print(f"      ⚠️ MODERATE accuracy, better than random but below research")
        else:
            print(f"      ❌ Below random chance - check feature quality")
    
    return {
        'symbol': symbol,
        'train_accuracy': train_metrics['mean_accuracy'],
        'test_accuracy': test_accuracy,
        'prediction_horizon': clf.prediction_horizon,
        'samples_train': len(train_df),
        'samples_test': len(test_df),
        'confusion_matrix': cm.tolist(),
    }


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings('ignore')
    
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else 'LUCK'
    
    try:
        results = run_classifier_backtest(symbol)
        
        print("\n" + "=" * 70)
        print("🎯 CLASSIFIER BACKTEST COMPLETE")
        print("=" * 70)
        print(f"\n   Symbol: {results['symbol']}")
        print(f"   Train Accuracy (CV): {results['train_accuracy']:.2%}")
        print(f"   Test Accuracy (2024): {results['test_accuracy']:.2%}")
        print(f"   Prediction Horizon: {results['prediction_horizon']} days (weekly)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   First run the analyzer on {symbol} to generate historical data.")
