"""
📅 MONTHLY FORECAST MODULE

Generates detailed monthly forecasts with:
- Monthly aggregation of daily predictions
- PSX seasonal event mapping (Ramadan, fiscal year, etc.)
- News event correlation for each month
- Plain-English reasoning for predicted direction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict


# ============================================================================
# PSX SEASONAL CALENDAR
# ============================================================================

PSX_CALENDAR = {
    1: {  # January
        'name': 'January',
        'fiscal': 'Q2 end (Dec FY companies)',
        'events': ['Budget planning season', 'Q2 results announcements'],
        'typical_pattern': 'mixed',
        'notes': 'Post-holiday recovery, results season begins'
    },
    2: {  # February
        'name': 'February',
        'fiscal': 'Q3 start',
        'events': ['Annual result filings due'],
        'typical_pattern': 'neutral',
        'notes': 'Quiet month, awaiting Q2 results'
    },
    3: {  # March
        'name': 'March',
        'fiscal': 'Q3',
        'events': ['Ramadan may start', 'Pre-Ramadan positioning'],
        'typical_pattern': 'volatile',
        'notes': 'Ramadan effect begins - trading volumes often decline'
    },
    4: {  # April
        'name': 'April',
        'fiscal': 'Q3',
        'events': ['Ramadan/EID', 'Volume typically low'],
        'typical_pattern': 'bearish_to_neutral',
        'notes': 'Ramadan fasting reduces trading activity'
    },
    5: {  # May
        'name': 'May',
        'fiscal': 'Q3 end',
        'events': ['Post-EID recovery', 'EID ul-Fitr holiday'],
        'typical_pattern': 'bullish',
        'notes': 'EID bonus buying, market typically rebounds'
    },
    6: {  # June
        'name': 'June',
        'fiscal': 'Year end (Jun FY companies)',
        'events': ['Final results season', 'Dividend announcements', 'Budget announcement'],
        'typical_pattern': 'volatile',
        'notes': 'Major results season, budget speculation'
    },
    7: {  # July
        'name': 'July',
        'fiscal': 'Q1 start (new FY)',
        'events': ['New fiscal year', 'Annual reports', 'AGM season'],
        'typical_pattern': 'bullish',
        'notes': 'Fresh fiscal year optimism, dividend payouts'
    },
    8: {  # August
        'name': 'August',
        'fiscal': 'Q1',
        'events': ['Independence Day', 'Q1 preview expectations'],
        'typical_pattern': 'neutral',
        'notes': 'Lower summer volumes, Muharram period'
    },
    9: {  # September
        'name': 'September',
        'fiscal': 'Q1 end',
        'events': ['Q1 results', 'SBP monetary policy'],
        'typical_pattern': 'mixed',
        'notes': 'Q1 results drive sector moves'
    },
    10: {  # October
        'name': 'October',
        'fiscal': 'Q2 start',
        'events': ['EID ul-Adha possible', 'Winter construction demand'],
        'typical_pattern': 'neutral_to_bullish',
        'notes': 'Cement sector picks up, infrastructure spending'
    },
    11: {  # November
        'name': 'November',
        'fiscal': 'Q2',
        'events': ['Winter demand', 'Earnings previews'],
        'typical_pattern': 'neutral',
        'notes': 'Quiet pre-results period'
    },
    12: {  # December
        'name': 'December',
        'fiscal': 'Q2 end',
        'events': ['Year-end positioning', 'Tax selling', 'Q2 results expected'],
        'typical_pattern': 'volatile',
        'notes': 'Year-end profit taking or positioning'
    }
}

# News impact duration mapping
NEWS_IMPACT_DURATION = {
    'dividend': 30,
    'merger': 90,
    'acquisition': 60,
    'profit': 21,
    'loss': 21,
    'expansion': 45,
    'investment': 60,
    'regulatory': 30,
    'lawsuit': 60,
    'privatization': 90,
    'result': 14,
    'earnings': 14,
}


# ============================================================================
# MONTHLY FORECAST GENERATOR
# ============================================================================

def generate_monthly_forecast(
    daily_predictions: List[Dict],
    sentiment_result: Optional[Dict],
    df: pd.DataFrame,
    symbol: str
) -> List[Dict]:
    """
    Aggregate daily predictions into monthly summaries with detailed reasoning.
    
    Args:
        daily_predictions: List of daily prediction dicts from model
        sentiment_result: News sentiment analysis from Groq
        df: Historical price DataFrame
        symbol: Stock symbol
    
    Returns:
        List of monthly forecast dicts with reasoning
    """
    if not daily_predictions:
        return []
    
    # Group predictions by month
    monthly_groups = defaultdict(list)
    for pred in daily_predictions:
        date_str = pred.get('date', '')
        if date_str:
            try:
                month_key = date_str[:7]  # '2025-01'
                monthly_groups[month_key].append(pred)
            except:
                continue
    
    # Get current price and technical indicators
    current_price = float(df['Close'].iloc[-1])
    latest_rsi = float(df['rsi_14'].iloc[-1]) if 'rsi_14' in df.columns else 50
    price_vs_ema50 = _calculate_ema_position(df)
    
    # Extract news events from sentiment
    news_events = _extract_news_events(sentiment_result) if sentiment_result else []
    key_events = sentiment_result.get('key_events', []) if sentiment_result else []
    catalysts = sentiment_result.get('catalysts', []) if sentiment_result else []
    risks = sentiment_result.get('risks', []) if sentiment_result else []
    
    monthly_forecasts = []
    
    for month_key in sorted(monthly_groups.keys()):
        month_preds = monthly_groups[month_key]
        
        # Parse month
        try:
            year = int(month_key[:4])
            month_num = int(month_key[5:7])
            month_name = f"{PSX_CALENDAR[month_num]['name']} {year}"
        except:
            month_name = month_key
            month_num = 1
        
        # Calculate monthly stats
        prices = [p['predicted_price'] for p in month_preds]
        start_price = prices[0]
        end_price = prices[-1]
        change_pct = ((end_price - start_price) / start_price) * 100 if start_price > 0 else 0
        
        # Determine trend
        if change_pct > 3:
            trend = 'BULLISH'
            trend_emoji = '📈'
        elif change_pct < -3:
            trend = 'BEARISH'
            trend_emoji = '📉'
        else:
            trend = 'NEUTRAL'
            trend_emoji = '➡️'
        
        # Average confidence
        avg_confidence = np.mean([p.get('confidence', 0.5) for p in month_preds])
        
        # Get seasonal context
        seasonal_info = PSX_CALENDAR.get(month_num, {})
        seasonal_factors = [
            seasonal_info.get('fiscal', ''),
            *seasonal_info.get('events', [])
        ]
        seasonal_factors = [f for f in seasonal_factors if f]  # Remove empty
        
        # Match news to this month
        matched_news = _match_news_to_month(news_events, month_key, month_num)
        
        # Generate primary driver reasoning
        primary_driver = _generate_primary_driver(
            trend=trend,
            change_pct=change_pct,
            matched_news=matched_news,
            seasonal_info=seasonal_info,
            key_events=key_events,
            catalysts=catalysts,
            risks=risks,
            month_num=month_num,
            symbol=symbol
        )
        
        # Build technical signals
        technical_signals = _build_technical_signals(
            latest_rsi, price_vs_ema50, trend, month_key
        )
        
        # Key dates in this month
        key_dates = _get_key_dates(month_preds, month_num)
        
        monthly_forecasts.append({
            'month': month_key,
            'month_name': month_name,
            'trend': trend,
            'trend_emoji': trend_emoji,
            'predicted_change_pct': round(change_pct, 2),
            'price_at_start': round(start_price, 2),
            'price_at_end': round(end_price, 2),
            'trading_days': len(month_preds),
            'reasoning': {
                'primary_driver': primary_driver,
                'news_events': matched_news[:3],  # Top 3
                'seasonal_factors': seasonal_factors,
                'technical_signals': technical_signals,
            },
            'confidence': round(avg_confidence, 3),
            'key_dates': key_dates,
            'seasonal_pattern': seasonal_info.get('typical_pattern', 'unknown'),
        })
    
    return monthly_forecasts


def _calculate_ema_position(df: pd.DataFrame) -> str:
    """Calculate price position relative to EMA 50."""
    if 'ema_50' not in df.columns and 'EMA_50' not in df.columns:
        return 'unknown'
    
    ema_col = 'ema_50' if 'ema_50' in df.columns else 'EMA_50'
    close = df['Close'].iloc[-1]
    ema50 = df[ema_col].iloc[-1]
    
    if pd.isna(ema50) or ema50 == 0:
        return 'unknown'
    
    pct = ((close - ema50) / ema50) * 100
    
    if pct > 5:
        return f'Trading {pct:.1f}% above 50-day EMA (strong uptrend)'
    elif pct > 0:
        return f'Trading {pct:.1f}% above 50-day EMA'
    elif pct > -5:
        return f'Trading {abs(pct):.1f}% below 50-day EMA'
    else:
        return f'Trading {abs(pct):.1f}% below 50-day EMA (weak trend)'


def _extract_news_events(sentiment_result: Dict) -> List[Dict]:
    """Extract news headlines from sentiment result."""
    news_items = sentiment_result.get('news_items', [])
    
    events = []
    for item in news_items:
        events.append({
            'headline': item.get('title', item.get('headline', '')),
            'date': item.get('date', ''),
            'source': item.get('source', item.get('source_name', 'Unknown')),
            'impact': _classify_news_impact(item.get('title', ''))
        })
    
    return events


def _classify_news_impact(headline: str) -> str:
    """Classify headline as positive/negative/neutral."""
    headline_lower = headline.lower()
    
    positive_keywords = ['profit', 'dividend', 'growth', 'surge', 'rise', 'expansion', 'invest', 'positive']
    negative_keywords = ['loss', 'decline', 'fall', 'drop', 'lawsuit', 'penalty', 'debt', 'negative']
    
    for kw in positive_keywords:
        if kw in headline_lower:
            return 'positive'
    
    for kw in negative_keywords:
        if kw in headline_lower:
            return 'negative'
    
    return 'neutral'


def _match_news_to_month(news_events: List[Dict], month_key: str, month_num: int) -> List[str]:
    """
    Match news events to a specific month based on date or impact duration.
    For future months, inherit recent news with ongoing impact.
    """
    matched = []
    
    for event in news_events:
        headline = event.get('headline', '')
        if not headline:
            continue
        
        event_date = event.get('date', '')
        
        # If event has a date in this month, include it
        if event_date and event_date.startswith(month_key):
            matched.append(headline)
            continue
        
        # For future months, check if news has ongoing impact
        headline_lower = headline.lower()
        for keyword, duration in NEWS_IMPACT_DURATION.items():
            if keyword in headline_lower and duration >= 30:
                # This is a news item with long-lasting impact
                matched.append(f"{headline} (ongoing impact)")
                break
    
    # Remove duplicates while preserving order
    seen = set()
    unique_matched = []
    for m in matched:
        if m.lower() not in seen:
            seen.add(m.lower())
            unique_matched.append(m)
    
    return unique_matched[:5]


def _generate_primary_driver(
    trend: str,
    change_pct: float,
    matched_news: List[str],
    seasonal_info: Dict,
    key_events: List[str],
    catalysts: List[str],
    risks: List[str],
    month_num: int,
    symbol: str
) -> str:
    """Generate human-readable primary driver explanation."""
    
    reasons = []
    
    # 1. News-driven reasoning (highest priority)
    if matched_news:
        if trend == 'BULLISH':
            reasons.append(f"Positive news momentum: {matched_news[0][:60]}...")
        elif trend == 'BEARISH':
            reasons.append(f"Negative news pressure: {matched_news[0][:60]}...")
    
    # 2. Catalyst-driven (from LLM analysis)
    if catalysts and trend == 'BULLISH':
        reasons.append(f"Key catalyst: {catalysts[0]}")
    elif risks and trend == 'BEARISH':
        reasons.append(f"Key risk: {risks[0]}")
    
    # 3. Seasonal factors
    seasonal_events = seasonal_info.get('events', [])
    if seasonal_events:
        if 'Ramadan' in ' '.join(seasonal_events):
            reasons.append("Ramadan period typically sees lower volumes")
        elif 'EID' in ' '.join(seasonal_events):
            reasons.append("Post-EID period often sees market recovery")
        elif 'Dividend' in ' '.join(seasonal_events) or month_num in [6, 7]:
            reasons.append("Dividend announcement season may drive interest")
        elif 'results' in ' '.join(seasonal_events).lower():
            reasons.append(f"{seasonal_info.get('fiscal', '')} - results announcements expected")
    
    # 4. Default technical reasoning
    if not reasons:
        if trend == 'BULLISH':
            reasons.append(f"Technical momentum projecting +{change_pct:.1f}% gain")
        elif trend == 'BEARISH':
            reasons.append(f"Mean reversion expected, projecting {change_pct:.1f}% adjustment")
        else:
            reasons.append("Consolidation phase expected with limited directional move")
    
    return "; ".join(reasons[:2])  # Max 2 reasons


def _build_technical_signals(rsi: float, ema_position: str, trend: str, month_key: str) -> List[str]:
    """Build list of technical signal descriptions."""
    signals = []
    
    # RSI interpretation
    if rsi > 70:
        signals.append(f"RSI at {rsi:.0f} (overbought - caution)")
    elif rsi < 30:
        signals.append(f"RSI at {rsi:.0f} (oversold - potential bounce)")
    elif rsi > 50:
        signals.append(f"RSI at {rsi:.0f} (bullish bias)")
    else:
        signals.append(f"RSI at {rsi:.0f} (bearish bias)")
    
    # EMA position
    if ema_position != 'unknown':
        signals.append(ema_position)
    
    return signals


def _get_key_dates(month_preds: List[Dict], month_num: int) -> List[Dict]:
    """Identify key dates in the month (highest/lowest predictions)."""
    if len(month_preds) < 3:
        return []
    
    # Find highest and lowest prediction days
    sorted_by_price = sorted(month_preds, key=lambda x: x.get('predicted_price', 0))
    
    key_dates = []
    
    # Lowest price day
    low_pred = sorted_by_price[0]
    key_dates.append({
        'date': low_pred['date'],
        'event': 'Predicted low',
        'price': round(low_pred['predicted_price'], 2)
    })
    
    # Highest price day
    high_pred = sorted_by_price[-1]
    key_dates.append({
        'date': high_pred['date'],
        'event': 'Predicted high',
        'price': round(high_pred['predicted_price'], 2)
    })
    
    return key_dates


# ============================================================================
# SUMMARY GENERATOR
# ============================================================================

def generate_forecast_summary(monthly_forecasts: List[Dict]) -> Dict:
    """
    Generate an overall summary of the monthly forecasts.
    """
    if not monthly_forecasts:
        return {'error': 'No forecasts available'}
    
    bullish_months = [m for m in monthly_forecasts if m['trend'] == 'BULLISH']
    bearish_months = [m for m in monthly_forecasts if m['trend'] == 'BEARISH']
    neutral_months = [m for m in monthly_forecasts if m['trend'] == 'NEUTRAL']
    
    # Overall direction
    if len(bullish_months) > len(bearish_months) * 1.5:
        overall = 'BULLISH'
        emoji = '📈'
    elif len(bearish_months) > len(bullish_months) * 1.5:
        overall = 'BEARISH'
        emoji = '📉'
    else:
        overall = 'MIXED'
        emoji = '↔️'
    
    # Total predicted change
    total_change = sum(m['predicted_change_pct'] for m in monthly_forecasts)
    
    # Best and worst months
    sorted_months = sorted(monthly_forecasts, key=lambda x: x['predicted_change_pct'])
    worst_month = sorted_months[0] if sorted_months else None
    best_month = sorted_months[-1] if sorted_months else None
    
    return {
        'overall_direction': overall,
        'overall_emoji': emoji,
        'bullish_months': len(bullish_months),
        'bearish_months': len(bearish_months),
        'neutral_months': len(neutral_months),
        'total_expected_change_pct': round(total_change, 2),
        'best_month': {
            'month_name': best_month['month_name'],
            'change_pct': best_month['predicted_change_pct'],
            'reason': best_month['reasoning']['primary_driver']
        } if best_month else None,
        'worst_month': {
            'month_name': worst_month['month_name'],
            'change_pct': worst_month['predicted_change_pct'],
            'reason': worst_month['reasoning']['primary_driver']
        } if worst_month else None,
        'avg_confidence': round(np.mean([m['confidence'] for m in monthly_forecasts]), 3)
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Simple test with mock data
    print("="*60)
    print("Testing Monthly Forecast Generator")
    print("="*60)
    
    # Mock daily predictions
    mock_predictions = []
    base_price = 800.0
    for i in range(60):  # 2 months
        date = datetime(2025, 1, 1) + timedelta(days=i)
        if date.weekday() < 5:  # Skip weekends
            # Simulate price movement
            if date.month == 1:
                price = base_price * (1 + i * 0.002)  # Bullish Jan
            else:
                price = base_price * (1 - (i - 30) * 0.001)  # Slight bearish Feb
            
            mock_predictions.append({
                'date': date.strftime('%Y-%m-%d'),
                'predicted_price': price,
                'confidence': 0.7 + np.random.uniform(-0.1, 0.1)
            })
    
    # Mock sentiment
    mock_sentiment = {
        'news_items': [
            {'title': 'Lucky Cement announces 15% dividend', 'date': '2025-01-05', 'source': 'BR'},
            {'title': 'Cement exports rise 12%', 'date': '2025-01-10', 'source': 'Dawn'},
        ],
        'key_events': ['Dividend announcement'],
        'catalysts': ['Strong construction demand'],
        'risks': ['Rising coal prices'],
    }
    
    # Mock DataFrame
    mock_df = pd.DataFrame({
        'Close': [800, 805, 810],
        'rsi_14': [55, 58, 60],
        'ema_50': [780, 782, 785],
    })
    
    # Generate forecast
    forecasts = generate_monthly_forecast(
        mock_predictions, 
        mock_sentiment, 
        mock_df, 
        'LUCK'
    )
    
    print(f"\nGenerated {len(forecasts)} monthly forecasts:")
    for f in forecasts:
        print(f"\n{f['trend_emoji']} {f['month_name']}: {f['trend']}")
        print(f"   Change: {f['predicted_change_pct']:+.2f}%")
        print(f"   Reason: {f['reasoning']['primary_driver']}")
        if f['reasoning']['news_events']:
            print(f"   News: {f['reasoning']['news_events'][0][:50]}...")
    
    # Summary
    summary = generate_forecast_summary(forecasts)
    print(f"\n{summary['overall_emoji']} OVERALL: {summary['overall_direction']}")
    print(f"   Bullish months: {summary['bullish_months']}")
    print(f"   Bearish months: {summary['bearish_months']}")
