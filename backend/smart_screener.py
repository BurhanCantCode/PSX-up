#!/usr/bin/env python3
"""
🧠 SMART STOCK SCREENER - Uses ML Model Predictions
Screens stocks based on actual model predictions, not just technical indicators.
This is what the user actually wants - stocks with high predicted upside!
"""

import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd

# Major PSX stocks
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
    'SEARL', 'GLAXO', 'HINOON', 'HALEON',
    # Autos
    'INDU', 'PSMC', 'HCAR', 'MTL',
    # IT/Tech
    'SYS', 'TRG',
    # Others
    'LOTCHEM', 'ISL', 'COLG', 'NESTLE', 'UNITY', 'AGIL', 'PAKOXY', 'KTML', 'IPAK', 'FABL', 'DCR', 'BFAGRO'
]


def load_stock_analysis(symbol: str) -> Optional[Dict]:
    """Load complete analysis for a symbol if it exists"""
    analysis_file = Path(__file__).parent.parent / "data" / f"{symbol}_complete_analysis.json"
    
    if not analysis_file.exists():
        return None
    
    try:
        with open(analysis_file, 'r') as f:
            data = json.load(f)
        return data
    except:
        return None


def calculate_smart_score(analysis: Dict) -> float:
    """
    Calculate smart score based on ML model predictions.
    Higher score = better opportunity.
    """
    score = 0.0
    
    # 1. Prediction Upside (0-40 points)
    pred_reasoning = analysis.get('prediction_reasoning', {})
    predicted_upside = pred_reasoning.get('prediction_upside', 0)
    direction = pred_reasoning.get('direction', 'NEUTRAL')
    
    if direction == 'BULLISH':
        if predicted_upside > 20:
            score += 40
        elif predicted_upside > 10:
            score += 30
        elif predicted_upside > 5:
            score += 20
        elif predicted_upside > 0:
            score += 10
    elif direction == 'BEARISH':
        score -= abs(predicted_upside) * 2  # Penalize bearish
    
    # 2. Forecast Summary (0-30 points)
    forecast = analysis.get('forecast_summary', {})
    total_change = forecast.get('total_expected_change_pct', 0)
    
    if total_change > 15:
        score += 30
    elif total_change > 10:
        score += 20
    elif total_change > 5:
        score += 15
    elif total_change > 0:
        score += 10
    else:
        score -= abs(total_change)  # Penalize negative
    
    # 3. Signal Quality (0-20 points)
    bullish_count = pred_reasoning.get('bullish_count', 0)
    bearish_count = pred_reasoning.get('bearish_count', 0)
    
    signal_ratio = bullish_count / (bearish_count + 1)  # Avoid division by zero
    if signal_ratio >= 3:
        score += 20
    elif signal_ratio >= 2:
        score += 15
    elif signal_ratio >= 1:
        score += 10
    elif bearish_count > bullish_count * 2:
        score -= 10  # Heavy bearish signals
    
    # 4. Sentiment (0-10 points)
    sentiment = analysis.get('sentiment', {})
    sent_signal = sentiment.get('signal', 'NEUTRAL')
    sent_confidence = sentiment.get('confidence', 0)
    
    if sent_signal == 'BUY':
        score += 10 * sent_confidence
    elif sent_signal == 'HOLD':
        score += 5 * sent_confidence
    elif sent_signal == 'SELL':
        score -= 10
    
    # 5. Data Freshness (0-10 points, penalty for old data)
    generated_at = analysis.get('generated_at', '')
    try:
        gen_date = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
        days_old = (datetime.now() - gen_date.replace(tzinfo=None)).days
        
        if days_old <= 1:
            score += 10
        elif days_old <= 3:
            score += 8
        elif days_old <= 7:
            score += 5
        elif days_old <= 14:
            score += 2
        else:
            score -= 5  # Penalize old data
    except:
        score -= 5  # No date = old data
    
    # 6. Monthly Forecast Quality (0-10 points)
    monthly_forecast = analysis.get('monthly_forecast', [])
    if monthly_forecast:
        # Check if next month is bullish
        next_month = monthly_forecast[0] if monthly_forecast else {}
        next_trend = next_month.get('trend', 'NEUTRAL')
        next_change = next_month.get('predicted_change_pct', 0)
        
        if next_trend == 'BULLISH' and next_change > 5:
            score += 10
        elif next_trend == 'BULLISH' and next_change > 0:
            score += 7
        elif next_trend == 'NEUTRAL' and next_change > 0:
            score += 5
        elif next_trend == 'BEARISH':
            score -= abs(next_change)  # Penalize bearish next month
    
    return round(score, 1)


def screen_stock(symbol: str) -> Optional[Dict]:
    """Screen a single stock using ML model predictions"""
    analysis = load_stock_analysis(symbol)
    
    if not analysis:
        return None
    
    # Calculate smart score
    score = calculate_smart_score(analysis)
    
    # Get key metrics
    pred_reasoning = analysis.get('prediction_reasoning', {})
    forecast = analysis.get('forecast_summary', {})
    sentiment = analysis.get('sentiment', {})
    current_price = analysis.get('current_price', 0)
    
    # Get monthly forecast
    monthly_forecast = analysis.get('monthly_forecast', [])
    next_month = monthly_forecast[0] if monthly_forecast else {}
    
    # Determine signal
    predicted_upside = pred_reasoning.get('prediction_upside', 0)
    direction = pred_reasoning.get('direction', 'NEUTRAL')
    
    if score >= 80 and predicted_upside > 10:
        signal = 'STRONG BUY'
    elif score >= 60 and predicted_upside > 5:
        signal = 'BUY'
    elif score >= 40 or (predicted_upside > 0 and direction == 'BULLISH'):
        signal = 'HOLD'
    elif score < 20 or (predicted_upside < -5 and direction == 'BEARISH'):
        signal = 'SELL'
    else:
        signal = 'HOLD'
    
    return {
        'symbol': symbol,
        'current_price': current_price,
        'predicted_upside': round(predicted_upside, 1),
        'direction': direction,
        'signal': signal,
        'smart_score': score,
        'total_forecast': round(forecast.get('total_expected_change_pct', 0), 1),
        'sentiment': sentiment.get('signal', 'NEUTRAL'),
        'sentiment_confidence': round(sentiment.get('confidence', 0) * 100, 0),
        'bullish_signals': pred_reasoning.get('bullish_count', 0),
        'bearish_signals': pred_reasoning.get('bearish_count', 0),
        'next_month_trend': next_month.get('trend', 'NEUTRAL'),
        'next_month_change': round(next_month.get('predicted_change_pct', 0), 1),
        'data_age_days': _calculate_age(analysis.get('generated_at', ''))
    }


def _calculate_age(generated_at: str) -> int:
    """Calculate days since analysis was generated"""
    try:
        gen_date = datetime.fromisoformat(generated_at.replace('Z', '+00:00'))
        days_old = (datetime.now() - gen_date.replace(tzinfo=None)).days
        return days_old
    except:
        return 999


def run_smart_screener(stocks: List[str] = None, min_score: float = 40, limit: int = 20) -> List[Dict]:
    """
    Run smart screener using ML model predictions.
    
    Args:
        stocks: List of symbols to screen (default: all major PSX stocks)
        min_score: Minimum smart score to include (default: 40)
        limit: Maximum number of results (default: 20)
    
    Returns:
        List of screened stocks sorted by smart score
    """
    if stocks is None:
        stocks = PSX_MAJOR_STOCKS
    
    print("=" * 80)
    print("🧠 SMART STOCK SCREENER - Using ML Model Predictions")
    print("=" * 80)
    print(f"Scanning {len(stocks)} stocks for opportunities...")
    print()
    
    results = []
    for symbol in stocks:
        result = screen_stock(symbol)
        if result and result['smart_score'] >= min_score:
            results.append(result)
    
    # Sort by smart score (highest first)
    results.sort(key=lambda x: x['smart_score'], reverse=True)
    
    # Limit results
    if limit:
        results = results[:limit]
    
    return results


def print_screener_results(results: List[Dict]):
    """Print screener results in a nice format"""
    if not results:
        print("\n❌ No stocks found matching criteria")
        print("   Try lowering min_score or running analysis on more stocks")
        return
    
    print(f"\n✅ Found {len(results)} opportunities:\n")
    print("=" * 80)
    
    for i, stock in enumerate(results, 1):
        emoji = "🟢" if stock['direction'] == 'BULLISH' else "🔴" if stock['direction'] == 'BEARISH' else "🟡"
        signal_emoji = "🔥" if stock['signal'] == 'STRONG BUY' else "✅" if stock['signal'] == 'BUY' else "⏸️" if stock['signal'] == 'HOLD' else "❌"
        fresh = "✅" if stock['data_age_days'] <= 7 else "⚠️" if stock['data_age_days'] <= 14 else "❌"
        
        print(f"{i}. {stock['symbol']} {emoji} {signal_emoji} {fresh} (Score: {stock['smart_score']:.1f})")
        print(f"   Price: Rs {stock['current_price']:.2f}")
        print(f"   Predicted Upside: {stock['predicted_upside']:+.1f}% ({stock['direction']})")
        print(f"   Signal: {stock['signal']}")
        print(f"   12-Month Forecast: {stock['total_forecast']:+.1f}%")
        print(f"   Sentiment: {stock['sentiment']} ({stock['sentiment_confidence']:.0f}% confidence)")
        print(f"   Signals: {stock['bullish_signals']} bullish, {stock['bearish_signals']} bearish")
        print(f"   Next Month: {stock['next_month_trend']} ({stock['next_month_change']:+.1f}%)")
        print(f"   Data Age: {stock['data_age_days']} days")
        print()


if __name__ == "__main__":
    # Run smart screener
    results = run_smart_screener(min_score=40, limit=20)
    print_screener_results(results)
    
    print("=" * 80)
    print("💡 To analyze a stock: python3 -m backend.main --symbol SYMBOL")
    print("=" * 80)
