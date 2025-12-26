#!/usr/bin/env python3
"""
PSX Stock Screener - Quick Market Scan
Quickly screens multiple stocks to identify top performers without full SOTA training.
Uses lightweight metrics: momentum, volatility, recent returns, volume.
"""

import asyncio
import json
import requests
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

# Major PSX stocks - KSE-100 constituents
PSX_MAJOR_STOCKS = [
    # Banks
    'HBL', 'UBL', 'MCB', 'NBP', 'ABL', 'BAFL', 'BAHL', 'MEBL', 'BOP',
    # Oil & Gas
    'OGDC', 'PPL', 'POL', 'PSO', 'SNGP', 'SSGC', 'MARI', 'ATRL',
    # Cement
    'LUCK', 'DGKC', 'MLCF', 'KOHC', 'FCCL', 'CHCC', 'PIOC',
    # Fertilizer
    'ENGRO', 'FFC', 'EFERT', 'FATIMA',
    # Power
    'HUBC', 'KEL', 'KAPCO', 'NCPL',
    # Telecom
    'PTC', 'PTML',
    # Pharma
    'SEARL', 'GLAXO', 'HINOON',
    # Autos
    'INDU', 'PSMC', 'HCAR', 'MTL',
    # Others
    'SYS', 'TRG', 'LOTCHEM', 'ISL', 'COLG', 'NESTLE', 'UNITY'
]

def fetch_month_data_curl(symbol: str, month: int, year: int) -> str:
    """Fetch data using curl POST (same as stock analyzer)"""
    import subprocess
    url = "https://dps.psx.com.pk/historical"
    post_data = f"month={month}&year={year}&symbol={symbol}"
    
    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST', url, '-d', post_data],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout
        return None
    except:
        return None


def parse_html_table_regex(html: str) -> List[Dict]:
    """Parse HTML table using regex (same as stock analyzer)"""
    import re
    rows = re.findall(r'<tr>.*?</tr>', html, re.DOTALL)
    data = []
    
    for row in rows:
        cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
        
        if len(cells) >= 6:
            try:
                date_str = cells[0].strip()
                date_obj = datetime.strptime(date_str, "%b %d, %Y")
                
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


def fetch_recent_data(symbol: str, months: int = 3) -> List[Dict]:
    """Fetch recent months of data for quick screening"""
    import time
    data = []
    current_date = datetime.now()
    
    for i in range(months):
        target_date = current_date - timedelta(days=30 * i)
        month = target_date.month
        year = target_date.year
        
        html = fetch_month_data_curl(symbol, month, year)
        if html:
            month_data = parse_html_table_regex(html)
            if month_data:
                data.extend(month_data)
        
        time.sleep(0.1)  # Small delay between requests
    
    return data


def calculate_screening_metrics(symbol: str, data: List[Dict]) -> Dict:
    """Calculate quick screening metrics for a stock"""
    if not data or len(data) < 10:
        return None
    
    try:
        df = pd.DataFrame(data)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        current_price = df['Close'].iloc[-1]
        
        # Returns
        if len(df) >= 5:
            return_1w = (df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100
        else:
            return_1w = 0
            
        if len(df) >= 21:
            return_1m = (df['Close'].iloc[-1] / df['Close'].iloc[-21] - 1) * 100
        else:
            return_1m = return_1w
            
        return_total = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
        
        # Momentum (price above moving averages)
        sma_20 = df['Close'].rolling(20).mean().iloc[-1] if len(df) >= 20 else current_price
        sma_50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else current_price
        
        momentum_score = 0
        if current_price > sma_20:
            momentum_score += 1
        if current_price > sma_50:
            momentum_score += 1
        if sma_20 > sma_50:
            momentum_score += 1
        
        # Volatility (lower is more stable)
        returns = df['Close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 5 else 0
        
        # Volume trend
        avg_volume = df['Volume'].mean()
        recent_volume = df['Volume'].iloc[-5:].mean() if len(df) >= 5 else avg_volume
        volume_trend = (recent_volume / avg_volume - 1) * 100 if avg_volume > 0 else 0
        
        # RSI (14-period)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1] if len(df) >= 15 else 50
        
        # Trend strength (percentage of up days)
        up_days = (df['Close'].diff() > 0).sum()
        trend_strength = up_days / len(df) * 100
        
        # Calculate composite score (higher is better)
        # Weights: momentum (30%), return (25%), RSI sweet spot (20%), low volatility (15%), volume (10%)
        rsi_score = 100 - abs(rsi - 55)  # Optimal RSI around 55 (bullish but not overbought)
        vol_score = max(0, 100 - volatility)  # Lower volatility = higher score
        
        composite_score = (
            momentum_score / 3 * 30 +          # 0-30 points
            min(max(return_1m, -20), 20) + 20 +  # 0-40 points, normalized
            rsi_score / 100 * 20 +             # 0-20 points
            vol_score / 100 * 15 +             # 0-15 points
            min(max(volume_trend, -10), 10) + 10  # 0-20 points
        )
        
        return {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'return_1w': round(return_1w, 2),
            'return_1m': round(return_1m, 2),
            'return_total': round(return_total, 2),
            'momentum_score': momentum_score,
            'volatility': round(volatility, 2),
            'rsi': round(rsi, 1),
            'volume_trend': round(volume_trend, 2),
            'trend_strength': round(trend_strength, 1),
            'composite_score': round(composite_score, 1),
            'data_points': len(df)
        }
    except Exception as e:
        return None


def screen_stock(symbol: str) -> Dict:
    """Screen a single stock"""
    print(f"  Scanning {symbol}...", end=' ')
    data = fetch_recent_data(symbol, months=3)
    if data:
        metrics = calculate_screening_metrics(symbol, data)
        if metrics:
            print(f"✓ ({metrics['data_points']} records)")
            return metrics
    print("✗ (no data)")
    return None


def run_screener(stocks: List[str] = None, max_workers: int = 5) -> List[Dict]:
    """Run the stock screener on multiple symbols"""
    import time
    
    if stocks is None:
        stocks = PSX_MAJOR_STOCKS
    
    print("="*70)
    print("🔍 PSX STOCK SCREENER - Quick Market Scan")
    print("="*70)
    print(f"Scanning {len(stocks)} stocks (this takes ~1-2 minutes)...")
    print()
    
    results = []
    
    # Sequential processing to avoid API blocking
    for i, symbol in enumerate(stocks, 1):
        result = screen_stock(symbol)
        if result:
            results.append(result)
        
        # Progress indicator
        if i % 10 == 0:
            print(f"  ... {i}/{len(stocks)} done")
        
        time.sleep(0.2)  # Delay between stocks
    
    # Sort by composite score
    results.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return results


def print_results(results: List[Dict], top_n: int = 15):
    """Print screening results"""
    print()
    print("="*70)
    print(f"📊 TOP {min(top_n, len(results))} STOCKS BY COMPOSITE SCORE")
    print("="*70)
    print()
    print(f"{'Rank':<5} {'Symbol':<8} {'Price':<10} {'1W %':<8} {'1M %':<8} {'RSI':<6} {'Vol%':<8} {'Score':<8}")
    print("-"*70)
    
    for i, stock in enumerate(results[:top_n], 1):
        color_1m = '+' if stock['return_1m'] >= 0 else ''
        color_1w = '+' if stock['return_1w'] >= 0 else ''
        
        print(f"{i:<5} {stock['symbol']:<8} {stock['current_price']:<10.2f} "
              f"{color_1w}{stock['return_1w']:<7.1f} {color_1m}{stock['return_1m']:<7.1f} "
              f"{stock['rsi']:<6.0f} {stock['volatility']:<7.1f} {stock['composite_score']:<8.1f}")
    
    print()
    print("Legend:")
    print("  • 1W % / 1M % = Returns over 1 week / 1 month")
    print("  • RSI = Relative Strength Index (30-70 normal, >70 overbought, <30 oversold)")
    print("  • Vol% = Annualized volatility (lower = more stable)")
    print("  • Score = Composite screening score (higher = better)")
    print()
    print("💡 Recommendation: Run full SOTA analysis on top 3-5 stocks")
    print()


def save_results(results: List[Dict], filename: str = None):
    """Save results to JSON file"""
    if filename is None:
        filename = f"screening_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    output_path = Path(__file__).parent.parent / "data" / filename
    with open(output_path, 'w') as f:
        json.dump({
            'generated_at': datetime.now().isoformat(),
            'stocks_scanned': len(results),
            'results': results
        }, f, indent=2)
    
    print(f"📁 Results saved to: {output_path}")
    return output_path


# API endpoint for web interface
async def screen_stocks_api(symbols: List[str] = None) -> Dict:
    """API endpoint for stock screening"""
    if symbols is None:
        symbols = PSX_MAJOR_STOCKS[:20]  # Limit to 20 for API
    
    results = run_screener(symbols, max_workers=3)
    
    return {
        'success': True,
        'stocks_scanned': len(symbols),
        'results_found': len(results),
        'top_stocks': results[:10],
        'all_results': results
    }


if __name__ == "__main__":
    print()
    print("Starting PSX Stock Screener...")
    print()
    
    # Run on major stocks
    results = run_screener()
    
    if results:
        print_results(results)
        save_results(results)
        
        # Show recommendation
        print("🏆 TOP PICKS FOR DETAILED ANALYSIS:")
        print("-"*40)
        for i, stock in enumerate(results[:5], 1):
            signal = "🟢 BULLISH" if stock['return_1m'] > 0 and stock['momentum_score'] >= 2 else "🟡 NEUTRAL" if stock['return_1m'] > -5 else "🔴 BEARISH"
            print(f"  {i}. {stock['symbol']}: {signal} (Score: {stock['composite_score']:.1f})")
        print()
    else:
        print("❌ No results. Check your internet connection.")
