#!/usr/bin/env python3
"""
🧪 BACKTESTING SCRIPT: Research Model Validation

This script trains on data up to 2023 and tests predictions against actual 2024 prices.
Reports: Accuracy, Precision, Recall, RMSE, MAE, Direction Accuracy, Sharpe Ratio

Usage:
    python backend/backtest_research_model.py LUCK
    python backend/backtest_research_model.py PSO
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Tuple

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.research_model import PSXResearchModel, get_realistic_benchmarks
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix
)


def load_historical_data(symbol: str) -> pd.DataFrame:
    """Load historical data for a symbol."""
    data_file = Path(__file__).parent.parent / "data" / f"{symbol}_historical_with_indicators.json"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    return df


def split_train_test(df: pd.DataFrame, cutoff_date: str = '2024-01-01') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into training (before cutoff) and test (after cutoff)."""
    cutoff = pd.to_datetime(cutoff_date)
    
    train = df[df['Date'] < cutoff].copy()
    test = df[df['Date'] >= cutoff].copy()
    
    return train, test


def calculate_direction_metrics(actual: np.ndarray, predicted: np.ndarray, prev_prices: np.ndarray) -> Dict:
    """
    Calculate direction prediction metrics.
    
    Direction = 1 if price went UP from previous day, 0 if DOWN
    """
    # Calculate actual directions (1 = up, 0 = down)
    actual_direction = (actual > prev_prices).astype(int)
    predicted_direction = (predicted > prev_prices).astype(int)
    
    accuracy = accuracy_score(actual_direction, predicted_direction)
    precision = precision_score(actual_direction, predicted_direction, zero_division=0)
    recall = recall_score(actual_direction, predicted_direction, zero_division=0)
    f1 = f1_score(actual_direction, predicted_direction, zero_division=0)
    
    cm = confusion_matrix(actual_direction, predicted_direction)
    
    return {
        'direction_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'true_up': int((actual_direction == 1).sum()),
        'true_down': int((actual_direction == 0).sum()),
        'predicted_up': int((predicted_direction == 1).sum()),
        'predicted_down': int((predicted_direction == 0).sum())
    }


def calculate_price_metrics(actual: np.ndarray, predicted: np.ndarray) -> Dict:
    """Calculate price prediction metrics."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def calculate_trading_metrics(actual: np.ndarray, predicted: np.ndarray, 
                               prev_prices: np.ndarray, initial_capital: float = 100000,
                               transaction_cost: float = 0.01) -> Dict:
    """
    Calculate trading performance metrics.
    
    Simple strategy: Buy if predicted UP, Sell if predicted DOWN
    """
    capital = initial_capital
    position = 0  # 0 = no position, 1 = long
    trades = 0
    wins = 0
    
    returns = []
    
    for i in range(len(actual)):
        predicted_direction = 1 if predicted[i] > prev_prices[i] else 0
        actual_direction = 1 if actual[i] > prev_prices[i] else 0
        
        if predicted_direction == 1 and position == 0:
            # Buy
            position = 1
            entry_price = prev_prices[i]
            trades += 1
        elif predicted_direction == 0 and position == 1:
            # Sell
            position = 0
            exit_price = actual[i]
            trade_return = (exit_price - entry_price) / entry_price
            
            # Apply transaction cost
            trade_return -= transaction_cost
            
            capital *= (1 + trade_return)
            returns.append(trade_return)
            
            if trade_return > 0:
                wins += 1
    
    # Close final position
    if position == 1:
        exit_price = actual[-1]
        trade_return = (exit_price - entry_price) / entry_price - transaction_cost
        capital *= (1 + trade_return)
        returns.append(trade_return)
        if trade_return > 0:
            wins += 1
    
    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate = wins / len(returns) if returns else 0
    
    # Sharpe Ratio (annualized, assuming daily returns)
    if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (avg_return * 252) / (std_return * np.sqrt(252)) if std_return > 0 else 0
    else:
        sharpe = 0
    
    return {
        'total_return': total_return,
        'final_capital': capital,
        'num_trades': trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe
    }


def run_backtest(symbol: str, verbose: bool = True) -> Dict:
    """
    Run full backtest for a symbol.
    
    Train on data up to 2023, test on 2024.
    """
    print("=" * 70)
    print(f"🧪 BACKTESTING RESEARCH MODEL: {symbol}")
    print("=" * 70)
    
    # Load data
    print(f"\n📂 Loading historical data for {symbol}...")
    df = load_historical_data(symbol)
    print(f"   Total records: {len(df)}")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    
    # Split train/test
    print(f"\n✂️ Splitting data at 2024-01-01...")
    train_df, test_df = split_train_test(df, cutoff_date='2024-01-01')
    print(f"   Training: {len(train_df)} records ({train_df['Date'].min().date()} to {train_df['Date'].max().date()})")
    print(f"   Testing: {len(test_df)} records ({test_df['Date'].min().date()} to {test_df['Date'].max().date()})")
    
    if len(test_df) < 10:
        print("❌ Not enough test data (need at least 10 records)")
        return {}
    
    # Train model on training data only
    print(f"\n🔬 Training research model on pre-2024 data...")
    model = PSXResearchModel(use_wavelet=True, symbol=symbol)
    train_metrics = model.fit(train_df, verbose=False)
    
    print(f"   Training ensemble accuracy: {train_metrics['ensemble_accuracy']:.2%}")
    
    # Preprocess ALL data ONCE (much faster than per-day)
    print(f"\n🔮 Preprocessing all data for testing...")
    full_df_processed = model.preprocess(df)
    
    # Make sure all feature columns exist
    for col in model.feature_cols:
        if col not in full_df_processed.columns:
            full_df_processed[col] = 0
    
    # Get test portion (after cutoff)
    cutoff_idx = len(train_df)
    test_processed = full_df_processed.iloc[cutoff_idx:].copy()
    train_processed = full_df_processed.iloc[:cutoff_idx].copy()
    
    print(f"   Preprocessed test data: {len(test_processed)} records")
    
    # Generate predictions for test period
    print(f"\n🔮 Generating predictions for test period...")
    
    test_dates = test_df['Date'].values
    test_prices = test_df['Close'].values
    predictions = []
    
    # Predict each test day using features from PREVIOUS day
    for i in range(len(test_processed)):
        if i == 0:
            # First test day: use last training day's features
            X = train_processed[model.feature_cols].iloc[-1:].fillna(0).values
        else:
            # Use previous test day's features
            X = test_processed[model.feature_cols].iloc[i-1:i].fillna(0).values
        
        # Predict
        pred_price = model.ensemble.predict(X)[0]
        predictions.append(pred_price)
        
        if (i + 1) % 100 == 0:
            print(f"   Processed {i + 1}/{len(test_processed)} days...")
    
    predictions = np.array(predictions)
    actual_prices = test_prices
    prev_prices = np.concatenate([[train_df['Close'].iloc[-1]], test_prices[:-1]])
    
    # Calculate metrics
    print(f"\n📊 CALCULATING METRICS...")
    
    # Price metrics
    price_metrics = calculate_price_metrics(actual_prices, predictions)
    print(f"\n   📈 Price Prediction Metrics:")
    print(f"      MAE: PKR {price_metrics['mae']:.2f}")
    print(f"      RMSE: PKR {price_metrics['rmse']:.2f}")
    print(f"      MAPE: {price_metrics['mape']:.2f}%")
    print(f"      R²: {price_metrics['r2']:.4f}")
    
    # Direction metrics
    direction_metrics = calculate_direction_metrics(actual_prices, predictions, prev_prices)
    print(f"\n   🎯 Direction Prediction Metrics:")
    print(f"      Direction Accuracy: {direction_metrics['direction_accuracy']:.2%}")
    print(f"      Precision: {direction_metrics['precision']:.2%}")
    print(f"      Recall: {direction_metrics['recall']:.2%}")
    print(f"      F1 Score: {direction_metrics['f1_score']:.2%}")
    print(f"      Confusion Matrix:")
    print(f"         Predicted DOWN | Predicted UP")
    cm = direction_metrics['confusion_matrix']
    print(f"         Actual DOWN: {cm[0][0]:4d} | {cm[0][1]:4d}")
    print(f"         Actual UP:   {cm[1][0]:4d} | {cm[1][1]:4d}")
    
    # Trading metrics
    trading_metrics = calculate_trading_metrics(actual_prices, predictions, prev_prices)
    print(f"\n   💰 Trading Metrics (with 1% transaction cost):")
    print(f"      Total Return: {trading_metrics['total_return']:.2f}%")
    print(f"      Final Capital: PKR {trading_metrics['final_capital']:,.2f}")
    print(f"      Number of Trades: {trading_metrics['num_trades']}")
    print(f"      Win Rate: {trading_metrics['win_rate']:.2%}")
    print(f"      Sharpe Ratio: {trading_metrics['sharpe_ratio']:.2f}")
    
    # Compare to benchmarks
    benchmarks = get_realistic_benchmarks()
    print(f"\n   📋 Comparison to Research Benchmarks:")
    
    acc = direction_metrics['direction_accuracy']
    if acc > benchmarks['direction_accuracy']['likely_overfit']:
        print(f"      ⚠️ Accuracy {acc:.2%} > 75% - LIKELY OVERFIT!")
    elif acc > benchmarks['direction_accuracy']['realistic_good']:
        print(f"      ✅ Accuracy {acc:.2%} in GOOD range (55-75%)")
    elif acc > benchmarks['direction_accuracy']['realistic_average']:
        print(f"      👍 Accuracy {acc:.2%} in AVERAGE range (50-55%)")
    else:
        print(f"      ❌ Accuracy {acc:.2%} below random chance")
    
    sharpe = trading_metrics['sharpe_ratio']
    if sharpe > benchmarks['sharpe_ratio']['likely_overfit']:
        print(f"      ⚠️ Sharpe {sharpe:.2f} > 2.0 - LIKELY OVERFIT!")
    elif sharpe > 0.5:
        print(f"      ✅ Sharpe {sharpe:.2f} in realistic range (0.5-2.0)")
    else:
        print(f"      ❌ Sharpe {sharpe:.2f} below target")
    
    # Buy and hold comparison
    buy_hold_return = (test_prices[-1] - test_prices[0]) / test_prices[0] * 100
    print(f"\n   📊 Buy & Hold Return for 2024: {buy_hold_return:.2f}%")
    print(f"   📊 Strategy Return: {trading_metrics['total_return']:.2f}%")
    if trading_metrics['total_return'] > buy_hold_return:
        print(f"      ✅ Strategy BEAT buy & hold by {trading_metrics['total_return'] - buy_hold_return:.2f}%")
    else:
        print(f"      ❌ Strategy UNDERPERFORMED by {buy_hold_return - trading_metrics['total_return']:.2f}%")
    
    print("\n" + "=" * 70)
    print("🧪 BACKTEST COMPLETE")
    print("=" * 70)
    
    return {
        'symbol': symbol,
        'train_size': len(train_df),
        'test_size': len(test_df),
        'train_metrics': train_metrics,
        'price_metrics': price_metrics,
        'direction_metrics': direction_metrics,
        'trading_metrics': trading_metrics,
        'buy_hold_return': buy_hold_return
    }


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    
    # Get symbol from command line or default to LUCK
    symbol = sys.argv[1].upper() if len(sys.argv) > 1 else 'LUCK'
    
    try:
        results = run_backtest(symbol)
        
        # Save results
        results_file = Path(__file__).parent.parent / "data" / f"{symbol}_backtest_results.json"
        
        # Convert numpy types to Python types for JSON
        def convert_to_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_json(v) for k, v in obj.items()}
            return obj
        
        with open(results_file, 'w') as f:
            json.dump(convert_to_json(results), f, indent=2)
        
        print(f"\n📁 Results saved to: {results_file}")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print(f"   First run the analyzer on {symbol} to generate historical data.")
