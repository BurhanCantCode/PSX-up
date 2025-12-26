#!/usr/bin/env python3
"""
🔥 HOT STOCKS FINDER
Extracts trending stocks from news headlines WITHOUT using Claude (free).
Used for homepage "trending now" section.
"""

import subprocess
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import Counter


# All PSX stock symbols and their variations
PSX_STOCKS = {
    # Banks
    'HBL': ['HBL', 'Habib Bank', 'Habib Bank Limited'],
    'UBL': ['UBL', 'United Bank', 'UBL Bank'],
    'MCB': ['MCB', 'MCB Bank', 'Muslim Commercial'],
    'NBP': ['NBP', 'National Bank'],
    'ABL': ['ABL', 'Allied Bank'],
    'BAFL': ['BAFL', 'Bank Alfalah', 'Alfalah'],
    'BAHL': ['BAHL', 'Bank Al Habib', 'Al Habib'],
    'MEBL': ['MEBL', 'Meezan Bank', 'Meezan'],
    'BOP': ['BOP', 'Bank of Punjab'],
    
    # Oil & Gas
    'OGDC': ['OGDC', 'Oil and Gas Development', 'OGDCL'],
    'PPL': ['PPL', 'Pakistan Petroleum'],
    'POL': ['POL', 'Pakistan Oilfields'],
    'PSO': ['PSO', 'Pakistan State Oil'],
    'SNGP': ['SNGP', 'Sui Northern'],
    'SSGC': ['SSGC', 'Sui Southern'],
    'MARI': ['MARI', 'Mari Petroleum', 'Mari Gas'],
    'ATRL': ['ATRL', 'Attock Refinery'],
    
    # Cement
    'LUCK': ['LUCK', 'Lucky Cement', 'Lucky'],
    'DGKC': ['DGKC', 'DG Khan Cement', 'DG Cement'],
    'MLCF': ['MLCF', 'Maple Leaf Cement'],
    'KOHC': ['KOHC', 'Kohat Cement'],
    'FCCL': ['FCCL', 'Fauji Cement'],
    'CHCC': ['CHCC', 'Cherat Cement'],
    'PIOC': ['PIOC', 'Pioneer Cement'],
    
    # Fertilizer
    'ENGRO': ['ENGRO', 'Engro', 'Engro Corporation'],
    'FFC': ['FFC', 'Fauji Fertilizer'],
    'EFERT': ['EFERT', 'Engro Fertilizers'],
    'FATIMA': ['FATIMA', 'Fatima Fertilizer', 'Fatima Group'],
    
    # Power
    'HUBC': ['HUBC', 'Hub Power', 'HUBCO'],
    'KEL': ['KEL', 'K-Electric', 'Karachi Electric'],
    'KAPCO': ['KAPCO', 'Kot Addu Power'],
    'NCPL': ['NCPL', 'Nishat Chunian'],
    
    # Technology
    'SYS': ['SYS', 'Systems Limited', 'Systems Ltd'],
    'TRG': ['TRG', 'TRG Pakistan'],
    'AVN': ['AVN', 'Avanceon'],
    'NETSOL': ['NETSOL', 'NetSol'],
    
    # Pharma
    'SEARL': ['SEARL', 'Searle'],
    'GLAXO': ['GLAXO', 'GSK', 'GlaxoSmithKline'],
    'HINOON': ['HINOON', 'Highnoon'],
    
    # Auto
    'INDU': ['INDU', 'Indus Motor', 'Toyota Indus'],
    'PSMC': ['PSMC', 'Pak Suzuki'],
    'HCAR': ['HCAR', 'Honda Atlas', 'Honda Cars'],
    'MTL': ['MTL', 'Millat Tractors'],
    
    # Others
    'NESTLE': ['NESTLE', 'Nestle Pakistan'],
    'COLG': ['COLG', 'Colgate'],
    'ISL': ['ISL', 'Ismail Industries'],
    'UNITY': ['UNITY', 'Unity Foods'],
    'LOTCHEM': ['LOTCHEM', 'Lotte Chemical'],
    'PTC': ['PTC', 'Pakistan Tobacco'],
}

# Bullish keywords (positive news)
BULLISH_KEYWORDS = [
    'profit', 'dividend', 'growth', 'record', 'surge', 'rally', 'gain',
    'expansion', 'investment', 'acquire', 'merger', 'strong', 'beat',
    'outperform', 'upgrade', 'bullish', 'buy', 'upside', 'breakthrough'
]

# Bearish keywords (negative news)
BEARISH_KEYWORDS = [
    'loss', 'decline', 'drop', 'fall', 'crash', 'plunge', 'weak',
    'investigation', 'fraud', 'penalty', 'downgrade', 'sell', 'bearish'
]


def fetch_url(url: str, timeout: int = 10) -> str:
    """Fetch URL content using curl (no Selenium needed)."""
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', str(timeout),
             '-A', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
             url],
            capture_output=True, text=True, timeout=timeout + 5
        )
        if result.returncode == 0:
            return result.stdout
    except:
        pass
    return ""


def extract_stocks_from_text(text: str) -> List[Tuple[str, str, bool]]:
    """
    Extract stock mentions from text.
    Returns: [(symbol, context, is_bullish), ...]
    """
    mentions = []
    text_upper = text.upper()
    text_lower = text.lower()
    
    for symbol, aliases in PSX_STOCKS.items():
        for alias in aliases:
            if alias.upper() in text_upper:
                # Find context around mention
                idx = text_upper.find(alias.upper())
                context = text[max(0, idx-50):min(len(text), idx+100)]
                
                # Determine sentiment
                context_lower = context.lower()
                bullish = any(kw in context_lower for kw in BULLISH_KEYWORDS)
                bearish = any(kw in context_lower for kw in BEARISH_KEYWORDS)
                
                is_bullish = bullish and not bearish
                
                mentions.append((symbol, context.strip(), is_bullish))
                break  # Found this stock, don't check other aliases
    
    return mentions


def scrape_psx_announcements() -> List[Dict]:
    """Scrape latest PSX announcements - improved extraction."""
    news = []
    
    url = "https://dps.psx.com.pk/announcements"
    html = fetch_url(url, timeout=15)
    
    if html:
        # Try multiple patterns
        # Pattern 1: Table rows
        rows = re.findall(r'<tr[^>]*>(.*?)</tr>', html, re.DOTALL | re.IGNORECASE)
        
        for row in rows[:50]:
            row_text = re.sub(r'<[^>]+>', ' ', row)  # Strip HTML tags
            row_text = ' '.join(row_text.split())  # Normalize whitespace
            
            if len(row_text) > 30:
                # Check for stock symbols in the text
                for symbol in PSX_STOCKS.keys():
                    if symbol in row_text.upper():
                        is_bullish = any(kw in row_text.lower() for kw in BULLISH_KEYWORDS)
                        news.append({
                            'symbol': symbol,
                            'headline': row_text[:200],
                            'date': datetime.now().strftime('%Y-%m-%d'),
                            'source': 'PSX',
                            'is_bullish': is_bullish
                        })
        
        # Pattern 2: Any text containing stock symbols
        paragraphs = re.findall(r'>([^<]{50,300})<', html)
        for para in paragraphs:
            para_clean = para.strip()
            for symbol in PSX_STOCKS.keys():
                if symbol in para_clean.upper():
                    is_bullish = any(kw in para_clean.lower() for kw in BULLISH_KEYWORDS)
                    news.append({
                        'symbol': symbol,
                        'headline': para_clean[:200],
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'PSX',
                        'is_bullish': is_bullish
                    })
    
    # Deduplicate
    seen = set()
    unique = []
    for item in news:
        key = (item['symbol'], item['headline'][:50])
        if key not in seen:
            seen.add(key)
            unique.append(item)
    
    return unique[:30]


def scrape_top_stocks_activity() -> List[Dict]:
    """Get stocks with recent activity by checking multiple company pages."""
    activity = []
    
    # Check top stocks for recent announcements
    top_symbols = ['LUCK', 'HBL', 'OGDC', 'ENGRO', 'SYS', 'UBL', 'FFC', 'HUBC', 'MCB', 'PPL']
    
    for symbol in top_symbols:
        url = f"https://dps.psx.com.pk/company/{symbol}"
        html = fetch_url(url, timeout=8)
        
        if html:
            # Look for announcements section
            announcements = re.findall(r'<td[^>]*>([^<]{20,200})</td>', html)
            
            for ann in announcements[:3]:
                ann_clean = ann.strip()
                if len(ann_clean) > 25 and not ann_clean.startswith('20'):  # Skip dates
                    is_bullish = any(kw in ann_clean.lower() for kw in BULLISH_KEYWORDS)
                    activity.append({
                        'symbol': symbol,
                        'headline': ann_clean[:200],
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'PSX Company',
                        'is_bullish': is_bullish
                    })
    
    return activity


def scrape_business_news() -> List[Dict]:
    """Scrape business news headlines."""
    news = []
    
    # Business Recorder - try multiple URLs
    urls = [
        "https://www.brecorder.com/",
        "https://www.brecorder.com/business",
    ]
    
    for url in urls:
        html = fetch_url(url, timeout=10)
        
        if html:
            # Extract all text that might be headlines
            matches = re.findall(r'>([A-Z][^<]{30,200})<', html)
            
            for text in matches[:30]:
                text_clean = text.strip()
                # Must contain a stock-related keyword
                if any(kw in text_clean.lower() for kw in ['psx', 'stock', 'share', 'cement', 'bank', 'oil', 'profit', 'dividend']):
                    for symbol in PSX_STOCKS.keys():
                        if symbol in text_clean.upper() or any(alias.upper() in text_clean.upper() for alias in PSX_STOCKS[symbol]):
                            is_bullish = any(kw in text_clean.lower() for kw in BULLISH_KEYWORDS)
                            news.append({
                                'symbol': symbol,
                                'headline': text_clean,
                                'date': datetime.now().strftime('%Y-%m-%d'),
                                'source': 'Business Recorder',
                                'is_bullish': is_bullish
                            })
    
    return news


def get_trending_stocks() -> Dict:
    """
    Main function: Get trending stocks from news WITHOUT using Claude.
    This is FREE to call and used for homepage.
    """
    print("🔥 Finding trending stocks from news...")
    
    all_news = []
    
    # Scrape multiple sources
    print("  📰 Checking PSX announcements...")
    psx_news = scrape_psx_announcements()
    all_news.extend(psx_news)
    print(f"     Found {len(psx_news)} stock mentions")
    
    print("  📰 Checking Business Recorder...")
    br_news = scrape_business_news()
    all_news.extend(br_news)
    print(f"     Found {len(br_news)} stock mentions")
    
    print("  📰 Checking top company pages...")
    activity = scrape_top_stocks_activity()
    all_news.extend(activity)
    print(f"     Found {len(activity)} company updates")
    
    # Count stock mentions
    mention_counts = Counter(item['symbol'] for item in all_news)
    bullish_counts = Counter(item['symbol'] for item in all_news if item.get('is_bullish'))
    
    # Build trending list
    trending = []
    for symbol, count in mention_counts.most_common(15):
        bullish = bullish_counts.get(symbol, 0)
        headlines = [item['headline'] for item in all_news if item['symbol'] == symbol][:3]
        
        # Determine trend
        bullish_ratio = bullish / count if count > 0 else 0
        if bullish_ratio > 0.6:
            trend = '🟢 BULLISH'
        elif bullish_ratio < 0.3:
            trend = '🔴 BEARISH'
        else:
            trend = '🟡 MIXED'
        
        trending.append({
            'symbol': symbol,
            'mentions': count,
            'bullish_mentions': bullish,
            'trend': trend,
            'headlines': headlines,
            'reason': f'{count} mentions in recent news'
        })
    
    return {
        'trending_stocks': trending,
        'total_news_scraped': len(all_news),
        'sources': ['PSX Announcements', 'Business Recorder'],
        'generated_at': datetime.now().isoformat(),
        'note': 'Based on news mentions (no AI, free to call)'
    }


def get_hot_stocks_for_homepage() -> List[Dict]:
    """
    Simplified function for homepage: Get top 5 hot stocks.
    """
    result = get_trending_stocks()
    
    top_stocks = []
    for stock in result['trending_stocks'][:5]:
        top_stocks.append({
            'symbol': stock['symbol'],
            'trend': stock['trend'],
            'mentions': stock['mentions'],
            'headline': stock['headlines'][0] if stock['headlines'] else 'In the news today'
        })
    
    return top_stocks


# Cache for homepage (to avoid scraping on every request)
_trending_cache = None
_cache_time = None

def get_cached_trending_stocks(max_age_minutes: int = 15) -> Dict:
    """Get trending stocks with caching to reduce scraping."""
    global _trending_cache, _cache_time
    
    now = datetime.now()
    
    if _trending_cache and _cache_time:
        age = (now - _cache_time).total_seconds() / 60
        if age < max_age_minutes:
            return _trending_cache
    
    _trending_cache = get_trending_stocks()
    _cache_time = now
    
    return _trending_cache


if __name__ == "__main__":
    print("="*60)
    print("🔥 HOT STOCKS FINDER (No AI, Free)")
    print("="*60)
    print()
    
    result = get_trending_stocks()
    
    print()
    print("="*60)
    print("📊 TRENDING STOCKS:")
    print("="*60)
    print()
    
    for i, stock in enumerate(result['trending_stocks'][:10], 1):
        print(f"{i}. {stock['symbol']} - {stock['trend']}")
        print(f"   {stock['mentions']} mentions ({stock['bullish_mentions']} bullish)")
        if stock['headlines']:
            print(f"   📰 {stock['headlines'][0][:60]}...")
        print()
    
    print(f"Total news items scraped: {result['total_news_scraped']}")
    print(f"Sources: {', '.join(result['sources'])}")
