#!/usr/bin/env python3
"""
🔍 SIMPLE PSX STOCK SCREENER
Runs model on top stocks one at a time for 2026 predictions.
"""

import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

# Top PSX stocks to analyze
TOP_STOCKS = [
    'OGDC', 'PPL', 'PSO',       # Oil & Gas
    'HBL', 'UBL', 'MCB', 'MEBL', # Banks  
    'LUCK', 'DGKC', 'MLCF',     # Cement
    'FFC', 'EFERT', 'FFBL',     # Fertilizer
    'HUBC', 'ENGRO', 'SYS',     # Others
]

def fetch_stock_data(symbol: str) -> pd.DataFrame:
    """Fetch stock EOD data from PSX."""
    import subprocess
    import json
    from datetime import datetime
    
    url = f"https://dps.psx.com.pk/timeseries/eod/{symbol}"
    result = subprocess.run(
        ['curl', '-s', url],
        capture_output=True, text=True, timeout=30
    )
    
    if result.returncode != 0:
        return None
    
    try:
        response = json.loads(result.stdout)
        data = response.get('data', [])
        if not data:
            return None
        
        # Format: [timestamp, close, volume, open]
        records = []
        for row in data:
            ts, close, volume, open_price = row[0], row[1], row[2], row[3]
            dt = datetime.fromtimestamp(ts)
            # Estimate high/low from open/close
            high = max(open_price, close) * 1.005
            low = min(open_price, close) * 0.995
            records.append({
                'Date': dt,
                'Open': float(open_price),
                'High': float(high),
                'Low': float(low),
                'Close': float(close),
                'Volume': float(volume)
            })
        
        df = pd.DataFrame(records)
        return df.sort_values('Date').reset_index(drop=True)
    except:
        return None


def analyze_single_stock(symbol: str) -> dict:
    """Run full analysis on one stock."""
    try:
        from research_model import PSXResearchModel
    except ImportError:
        from backend.research_model import PSXResearchModel
    
    print(f"\n{'='*50}")
    print(f"📊 {symbol}")
    print(f"{'='*50}")
    
    try:
        # Fetch data
        df = fetch_stock_data(symbol)
        if df is None or len(df) < 100:
            return {'symbol': symbol, 'error': 'Insufficient data'}
        
        current_price = df['Close'].iloc[-1]
        print(f"   Current: PKR {current_price:.2f}")
        
        # Create model and train
        model = PSXResearchModel(symbol=symbol, use_wavelet=True)
        model.fit(df, verbose=False)
        
        # Predict 60 days (covers Feb 2026)
        predictions = model.predict_daily(df, days=60)
        
        if not predictions:
            return {'symbol': symbol, 'error': 'Prediction failed'}
        
        # Get Feb prediction (around day 45)
        feb_idx = min(44, len(predictions)-1)
        feb_pred = predictions[feb_idx]
        # Get end prediction
        year_pred = predictions[-1]
        
        feb_upside = feb_pred.get('upside_potential', 0)
        year_upside = year_pred.get('upside_potential', 0)
        
        print(f"   Feb 2026: PKR {feb_pred.get('predicted_price', 0):.2f} ({feb_upside:+.1f}%)")
        print(f"   60-Day:   PKR {year_pred.get('predicted_price', 0):.2f} ({year_upside:+.1f}%)")
        
        # Get news impact
        news_bias = 0
        try:
            from backend.brecorder_scraper import get_brecorder_news_for_symbol
            news = get_brecorder_news_for_symbol(symbol)
            news_bias = news.get('impact', {}).get('news_bias', 0)
            print(f"   News: {news_bias:+.2f}")
        except:
            pass
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'feb_price': feb_pred['predicted_price'],
            'feb_upside': feb_upside,
            'year_price': year_pred['predicted_price'],
            'year_upside': year_upside,
            'news_bias': news_bias,
        }
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)[:50]}")
        return {'symbol': symbol, 'error': str(e)[:100]}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--stock', '-s', help='Analyze single stock')
    parser.add_argument('--all', '-a', action='store_true', help='Analyze all top stocks')
    args = parser.parse_args()
    
    if args.stock:
        result = analyze_single_stock(args.stock.upper())
        print(f"\nResult: {result}")
    elif args.all:
        results = []
        for symbol in TOP_STOCKS:
            result = analyze_single_stock(symbol)
            if 'error' not in result:
                results.append(result)
        
        # Sort and display rankings
        if results:
            print("\n" + "="*60)
            print("🏆 BEST FOR FEB 2026 EXIT:")
            for r in sorted(results, key=lambda x: x['feb_upside'], reverse=True)[:5]:
                print(f"   {r['symbol']}: {r['feb_upside']:+.1f}%")
            
            print("\n🏆 BEST FOR 2026 OVERALL:")
            for r in sorted(results, key=lambda x: x['year_upside'], reverse=True)[:5]:
                print(f"   {r['symbol']}: {r['year_upside']:+.1f}%")
    else:
        # Default: run first 3 stocks as test
        print("Running test on 3 stocks...")
        for symbol in TOP_STOCKS[:3]:
            result = analyze_single_stock(symbol)
            print(f"Result: {result.get('symbol')}: Feb={result.get('feb_upside', 'ERR')}, Year={result.get('year_upside', 'ERR')}")
