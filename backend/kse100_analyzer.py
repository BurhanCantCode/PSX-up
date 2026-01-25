#!/usr/bin/env python3
"""
KSE100 Index Analyzer
Fetches historical KSE100 data and generates predictions using the same models as stocks.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import json

from backend.external_features import fetch_kse100, fetch_kse100_month
from backend.research_model import PSXResearchModel
from backend.validated_indicators import calculate_validated_indicators


def fetch_kse100_historical_data(start_year: int = 2020, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetch KSE100 historical data and convert to standard format.
    Uses caching and incremental updates to ensure latest data is always used for training.
    
    Args:
        start_year: Year to start fetching from
        force_refresh: If True, force a full refresh ignoring cache
    
    Returns DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    BASE_DIR = Path(__file__).parent.parent
    cache_file = BASE_DIR / "data" / "KSE100_historical_with_indicators.json"
    cache_file.parent.mkdir(exist_ok=True)
    
    # Check for cached data
    cached_data = None
    needs_refresh = force_refresh
    
    if not force_refresh and cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
            
            if cached_data and len(cached_data) > 0:
                # Check if cached data is fresh (within 1 day of today)
                cached_dates = [pd.to_datetime(d.get('Date', '')).date() for d in cached_data if 'Date' in d]
                if cached_dates:
                    last_cached_date = max(cached_dates)
                    today = datetime.now().date()
                    days_old = (today - last_cached_date).days
                    
                    if days_old >= 1:
                        needs_refresh = True
                        print(f"\n🔄 Cached KSE100 data is {days_old} day(s) old. Checking for latest data...")
                    else:
                        print(f"\n✅ Using cached KSE100 data (up to {last_cached_date}, {days_old} day(s) old)")
        except Exception as e:
            print(f"⚠️ Error reading cache: {e}. Fetching fresh data...")
            needs_refresh = True
    
    if needs_refresh:
        print(f"\n📊 Fetching KSE100 historical data from {start_year}...")
        
        # If we have cached data, do incremental update
        if cached_data and len(cached_data) > 0:
            # Find last date in cached data
            cached_dates = [pd.to_datetime(d.get('Date', '')).date() for d in cached_data if 'Date' in d]
            if cached_dates:
                last_date = max(cached_dates)
                print(f"📥 Found cached data up to {last_date}. Fetching new data from {last_date} onwards...")
                
                # Fetch only new data (from last date onwards to today)
                # We'll fetch from the month containing the last date (to catch any new days in that month)
                # and all subsequent months up to today
                last_dt = datetime.combine(last_date, datetime.min.time())
                start_fetch_year = last_dt.year
                start_fetch_month = last_dt.month  # Start from same month to catch new days
                
                # Fetch new data
                current_date = datetime.now()
                new_data_list = []
                
                for year in range(start_fetch_year, current_date.year + 1):
                    start_m = start_fetch_month if year == start_fetch_year else 1
                    end_m = current_date.month if year == current_date.year else 12
                    
                    for month in range(start_m, end_m + 1):
                        month_data = fetch_kse100_month(month, year)
                        new_data_list.extend(month_data)
                
                if new_data_list:
                    # Convert new data to standard format
                    new_df = pd.DataFrame(new_data_list)
                    new_df['date'] = pd.to_datetime(new_df['date'])
                    
                    # Convert to standard format
                    new_standard = []
                    for _, row in new_df.iterrows():
                        new_standard.append({
                            'Date': row['date'].strftime('%Y-%m-%d'),
                            'Open': float(row['kse100_open']),
                            'High': float(row['kse100_high']),
                            'Low': float(row['kse100_low']),
                            'Close': float(row['kse100_close']),
                            'Volume': float(row['kse100_volume'])
                        })
                    
                    # Merge with cached data (remove duplicates by date)
                    seen_dates = {d.get('Date') for d in cached_data}
                    original_count = len(cached_data)
                    new_dates_added = []
                    for record in new_standard:
                        if record['Date'] not in seen_dates:
                            cached_data.append(record)
                            seen_dates.add(record['Date'])
                            new_dates_added.append(record['Date'])

                    # Sort by date
                    cached_data.sort(key=lambda x: x.get('Date', ''))
                    actual_new = len(cached_data) - original_count
                    if actual_new > 0:
                        print(f"✅ Added {actual_new} new trading days: {new_dates_added}")
                        print(f"   Total records: {len(cached_data)}")
                    else:
                        print(f"✅ Cache is up-to-date. No new trading days available from PSX.")
                else:
                    print(f"✅ No new data found. Using existing {len(cached_data)} records")
            else:
                # Cached data exists but no valid dates, do full fetch
                cached_data = None
        
        # If no cached data or full refresh needed, fetch everything
        if not cached_data:
            # Fetch KSE100 data
            kse100_df = fetch_kse100(start_year=start_year)
            
            if kse100_df.empty:
                raise ValueError("Failed to fetch KSE100 data")
            
            # Convert to standard format (same as stock data)
            cached_data = []
            for _, row in kse100_df.iterrows():
                cached_data.append({
                    'Date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    'Open': float(row['kse100_open']),
                    'High': float(row['kse100_high']),
                    'Low': float(row['kse100_low']),
                    'Close': float(row['kse100_close']),
                    'Volume': float(row['kse100_volume'])
                })
        
        # Save updated cache
        with open(cache_file, 'w') as f:
            json.dump(cached_data, f, indent=2)
        print(f"💾 Saved {len(cached_data)} records to cache")
    
    # Convert cached data to DataFrame
    df = pd.DataFrame(cached_data)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicates (keep last)
    df = df.drop_duplicates(subset=['Date'], keep='last').reset_index(drop=True)
    
    # Calculate basic indicators
    df = calculate_validated_indicators(df)
    
    print(f"✅ Loaded {len(df)} KSE100 records ({df['Date'].min().date()} to {df['Date'].max().date()})")
    
    return df


def analyze_kse100(horizon: int = 365) -> Dict:
    """
    Analyze KSE100 index and generate predictions.
    
    Args:
        horizon: Number of days to predict ahead (default 365 for full year)
    
    Returns:
        Dictionary with analysis results in same format as stock analyzer
    """
    print("\n" + "=" * 60)
    print("📈 KSE100 INDEX ANALYSIS")
    print("=" * 60)
    
    # 1. Fetch historical data
    df = fetch_kse100_historical_data(start_year=2020)
    
    if len(df) < 100:
        raise ValueError(f"Insufficient KSE100 data: {len(df)} records (need at least 100)")
    
    current_price = float(df['Close'].iloc[-1])
    current_date = df['Date'].iloc[-1]
    
    print(f"\n📊 Current KSE100: {current_price:,.2f} (as of {current_date.date()})")
    
    # 2. Train model
    print("\n🧠 Training prediction model...")
    # For KSE100, we don't need external KSE100 features (it IS the index)
    # But we still want USD/PKR, oil, etc.
    model = PSXResearchModel(use_wavelet=True, symbol='KSE100', use_returns_model=True)

    # Note: fit() and predict_daily() will call preprocess() internally
    # So we pass the raw df, not preprocessed data
    metrics = model.fit(df, verbose=True)
    
    print(f"   Model R2: {metrics.get('ensemble_accuracy', 0):.2%}")
    print(f"   Trend Accuracy: {metrics.get('trend_accuracy', 0):.2%}")
    
    # 3. Generate predictions
    print(f"\n🔮 Generating {horizon}-day predictions...")

    if horizon == 'full' or horizon >= 365:
        # Generate daily predictions through end of 2026
        # Use force_full_year=True to bypass 60-day cap for visualization
        end_date = '2026-12-31'
        predictions = model.predict_daily(df, days=365, force_full_year=True)
    else:
        # Generate predictions for specified horizon
        predictions = model.predict_daily(df, days=horizon)
    
    if not predictions:
        raise ValueError("Failed to generate predictions")
    
    # 4. Format results
    last_pred = predictions[-1] if predictions else {}
    
    # Format historical data for charting
    historical_data = []
    for _, row in df.tail(180).iterrows():  # Last 180 days for chart
        historical_data.append({
            'Date': row['Date'].strftime('%Y-%m-%d') if hasattr(row['Date'], 'strftime') else str(row['Date']),
            'Close': float(row['Close'])
        })
    
    # Format daily predictions
    daily_predictions = []
    for pred in predictions:
        daily_predictions.append({
            'date': pred.get('date', ''),
            'predicted_price': pred.get('predicted_price', 0),
            'upside_potential': pred.get('upside_potential', 0),
            'confidence': pred.get('confidence', 0.5)
        })
    
    result = {
        'symbol': 'KSE100',
        'current_price': current_price,
        'model': 'Research Model (Wavelet + External Features)',
        'model_performance': {
            'r2': metrics.get('ensemble_accuracy', 0),
            'trend_accuracy': metrics.get('trend_accuracy', 0),
            'mase': metrics.get('mase', 0),
            'wavelet_denoising': True
        },
        'daily_predictions': daily_predictions,
        'historical_data': historical_data,
        'generated_at': datetime.now().isoformat()
    }
    
    # Save to cache
    cache_file = Path(__file__).parent.parent / "data" / "KSE100_research_predictions_2026.json"
    cache_file.parent.mkdir(exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n✅ KSE100 analysis complete!")
    print(f"   Current: {current_price:,.2f}")
    if last_pred:
        print(f"   Predicted: {last_pred.get('predicted_price', 0):,.2f}")
        print(f"   Upside: {last_pred.get('upside_potential', 0):+.2f}%")
    
    return result


if __name__ == "__main__":
    # Test the analyzer
    result = analyze_kse100(horizon=365)
    print("\n✅ Test complete!")

