"""
🔍 ENHANCED NEWS FETCHER
Multi-source news aggregation with company alias expansion, parent company detection,
and sector-wide news capture. Designed to catch major market-moving events like
UAE investments, IMF loans, policy changes that affect stock groups.
"""

import re
import subprocess
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Set, Optional
import math

# ============================================================================
# COMPANY ALIASES & PARENT COMPANIES
# Maps stock symbols to all related search terms
# ============================================================================

COMPANY_ALIASES = {
    # Fauji Group - CRITICAL (UAE $1B news missed without this)
    'FFC': {
        'names': ['Fauji Fertilizer', 'FFC', 'Fauji Fertilizer Company'],
        'parent': 'Fauji Foundation',
        'sector': 'fertilizer',
        'sector_peers': ['FFBL', 'EFERT', 'FATIMA'],
    },
    'FFBL': {
        'names': ['Fauji Fertilizer Bin Qasim', 'FFBL', 'Fauji Bin Qasim'],
        'parent': 'Fauji Foundation',
        'sector': 'fertilizer',
        'sector_peers': ['FFC', 'EFERT', 'FATIMA'],
    },
    'FCCL': {
        'names': ['Fauji Cement', 'FCCL', 'Fauji Cement Company'],
        'parent': 'Fauji Foundation',
        'sector': 'cement',
        'sector_peers': ['LUCK', 'DGKC', 'MLCF', 'PIOC'],
    },
    
    # Oil & Gas
    'PSO': {
        'names': ['Pakistan State Oil', 'PSO', 'State Oil'],
        'parent': None,
        'sector': 'oil_marketing',
        'sector_peers': ['SHEL', 'APL', 'HASCOL'],
    },
    'PPL': {
        'names': ['Pakistan Petroleum', 'PPL', 'Pakistan Petroleum Limited'],
        'parent': None,
        'sector': 'exploration_production',
        'sector_peers': ['OGDC', 'POL', 'MARI'],
    },
    'OGDC': {
        'names': ['OGDC', 'Oil and Gas Development Company', 'Oil Gas Development'],
        'parent': None,
        'sector': 'exploration_production',
        'sector_peers': ['PPL', 'POL', 'MARI'],
    },
    'POL': {
        'names': ['Pakistan Oilfields', 'POL', 'Pakistan Oilfields Limited'],
        'parent': 'Attock Group',
        'sector': 'exploration_production',
        'sector_peers': ['OGDC', 'PPL', 'MARI'],
    },
    
    # Cement
    'LUCK': {
        'names': ['Lucky Cement', 'LUCK', 'Lucky Cement Limited'],
        'parent': 'Lucky Group',
        'sector': 'cement',
        'sector_peers': ['DGKC', 'MLCF', 'PIOC', 'FCCL', 'CHCC', 'KOHC'],
    },
    'DGKC': {
        'names': ['DG Khan Cement', 'DGKC', 'DG Cement'],
        'parent': 'Nishat Group',
        'sector': 'cement',
        'sector_peers': ['LUCK', 'MLCF', 'PIOC', 'FCCL'],
    },
    
    # Banks
    'HBL': {
        'names': ['Habib Bank', 'HBL', 'Habib Bank Limited'],
        'parent': 'Aga Khan Fund',
        'sector': 'banks',
        'sector_peers': ['UBL', 'MCB', 'NBP', 'ABL', 'BAFL', 'MEBL'],
    },
    'UBL': {
        'names': ['United Bank', 'UBL', 'United Bank Limited'],
        'parent': 'Bestway Group',
        'sector': 'banks',
        'sector_peers': ['HBL', 'MCB', 'NBP', 'ABL', 'BAFL'],
    },
    'MEBL': {
        'names': ['Meezan Bank', 'MEBL', 'Meezan'],
        'parent': None,
        'sector': 'islamic_banks',
        'sector_peers': ['HBL', 'UBL', 'MCB', 'BAHL'],
    },
    
    # Tech
    'SYS': {
        'names': ['Systems Limited', 'SYS', 'Systems Ltd'],
        'parent': None,
        'sector': 'technology',
        'sector_peers': ['TRG', 'NETSOL'],
    },
    'TRG': {
        'names': ['TRG Pakistan', 'TRG', 'The Resource Group'],
        'parent': None,
        'sector': 'technology',
        'sector_peers': ['SYS', 'NETSOL'],
    },
    
    # Fertilizer
    'EFERT': {
        'names': ['Engro Fertilizers', 'EFERT', 'Engro Fert'],
        'parent': 'Engro Corporation',
        'sector': 'fertilizer',
        'sector_peers': ['FFC', 'FFBL', 'FATIMA'],
    },
    'FATIMA': {
        'names': ['Fatima Fertilizer', 'FATIMA', 'Fatima Group'],
        'parent': None,
        'sector': 'fertilizer',
        'sector_peers': ['FFC', 'FFBL', 'EFERT'],
    },
    
    # Engro Group
    'ENGRO': {
        'names': ['Engro Corporation', 'ENGRO', 'Engro Corp'],
        'parent': None,
        'sector': 'conglomerate',
        'sector_peers': ['EFERT', 'EPCL', 'EFOOD'],
    },
    
    # Power
    'HUBC': {
        'names': ['Hub Power', 'HUBCO', 'Hub Power Company'],
        'parent': None,
        'sector': 'power',
        'sector_peers': ['KAPCO', 'NCPL', 'KEL', 'NPL'],
    },
    'KEL': {
        'names': ['K-Electric', 'KEL', 'Karachi Electric'],
        'parent': None,
        'sector': 'power',
        'sector_peers': ['HUBC', 'KAPCO', 'NCPL'],
    },
}

# Sector keywords for broad searches
SECTOR_KEYWORDS = {
    'fertilizer': ['fertilizer Pakistan', 'urea prices', 'DAP prices', 'fertilizer subsidy'],
    'cement': ['cement sector Pakistan', 'cement exports', 'construction Pakistan', 'CPEC projects'],
    'oil_marketing': ['OMC Pakistan', 'fuel prices Pakistan', 'petroleum levy', 'oil imports'],
    'exploration_production': ['oil gas Pakistan', 'petroleum exploration', 'OGRA', 'oil discovery'],
    'banks': ['banking sector Pakistan', 'SBP policy rate', 'KIBOR', 'monetary policy', 'ADR ratio'],
    'islamic_banks': ['islamic banking Pakistan', 'sukuk', 'sharia compliant'],
    'technology': ['IT exports Pakistan', 'software exports', 'tech Pakistan'],
    'power': ['power sector Pakistan', 'circular debt', 'electricity tariff', 'NEPRA'],
    'conglomerate': [],
}

# Macro news categories that affect all stocks
MACRO_CATEGORIES = [
    'IMF Pakistan',
    'SBP policy rate',
    'KIBOR rate',
    'USD PKR exchange',
    'Pakistan forex reserves',
    'UAE Pakistan investment',
    'Saudi Arabia Pakistan investment',
    'China CPEC Pakistan',
    'Pakistan budget',
    'KSE-100 index',
    'PSX market today',
]


# ============================================================================
# MULTI-SOURCE NEWS SCRAPING
# ============================================================================

NEWS_SOURCES = {
    'business_recorder': {
        'search_url': 'https://www.brecorder.com/?s={}',
        'fallback_url': 'https://www.brecorder.com/search?query={}',
        'priority': 1,
    },
    'dawn': {
        'search_url': 'https://www.dawn.com/search?q={}',
        'fallback_url': 'https://www.dawn.com/news/business',
        'priority': 1,
    },
    'pakistan_today': {
        'search_url': 'https://www.pakistantoday.com.pk/?s={}',
        'fallback_url': 'https://www.pakistantoday.com.pk/category/business/',
        'priority': 1,
    },
    'express_tribune': {
        'search_url': 'https://tribune.com.pk/?s={}',
        'fallback_url': 'https://tribune.com.pk/business/psx',
        'priority': 2,
    },
    'geo_news': {
        'search_url': 'https://www.geo.tv/search/{}',
        'fallback_url': 'https://www.geo.tv/category/business',
        'priority': 2,
    },
    'minute_mirror': {
        'search_url': 'https://minutemirror.com.pk/?s={}',
        'fallback_url': 'https://minutemirror.com.pk/category/business/',
        'priority': 3,
    },
}


def fetch_news_curl(url: str, timeout: int = 10) -> str:
    """Fetch URL content using curl"""
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', str(timeout), 
             '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
             url],
            capture_output=True, text=True, timeout=timeout + 5
        )
        return result.stdout if result.returncode == 0 else ""
    except:
        return ""


def extract_articles_from_html(html: str, source: str) -> List[Dict]:
    """Extract article titles and links from HTML"""
    articles = []
    
    # Common patterns for article headlines
    patterns = [
        r'<a[^>]+href="([^"]+)"[^>]*>([^<]{20,200})</a>',  # Standard links
        r'<h[123][^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]{20,200})</a>',  # Headlines
        r'class="[^"]*title[^"]*"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>([^<]{20,200})</a>',
    ]
    
    seen = set()
    for pattern in patterns:
        matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)
        for url, title in matches:
            title = re.sub(r'\s+', ' ', title).strip()
            title_lower = title.lower()
            
            # Filter out navigation/junk
            if len(title) < 25:
                continue
            if any(x in title_lower for x in ['menu', 'category', 'privacy', 'contact', 'about us']):
                continue
            if title_lower in seen:
                continue
            
            seen.add(title_lower)
            articles.append({
                'title': title[:300],
                'url': url if url.startswith('http') else '',
                'source': source,
                'date': datetime.now().strftime('%Y-%m-%d'),
            })
            
            if len(articles) >= 10:
                break
        
        if len(articles) >= 10:
            break
    
    return articles


def get_search_queries(symbol: str) -> List[str]:
    """Get all search queries for a symbol (expanded with aliases + PSX context)"""
    queries = []
    
    if symbol in COMPANY_ALIASES:
        info = COMPANY_ALIASES[symbol]
        
        # Add company names WITH PSX context for better relevance
        for name in info['names'][:2]:
            queries.append(f"{name} PSX")  # e.g., "Lucky Cement PSX"
        
        # Also add full company name without PSX (for general business news)
        if len(info['names']) > 0:
            queries.append(info['names'][0])  # e.g., "Lucky Cement" 
        
        # Add parent company if exists (important for group news like Fauji)
        if info.get('parent'):
            queries.append(info['parent'])  # e.g., "Fauji Foundation"
    else:
        # Unknown symbol - use symbol + PSX context
        queries.append(f"{symbol} PSX")
        queries.append(f"{symbol} Pakistan stock")
    
    return list(dict.fromkeys(queries))  # Dedupe while preserving order


def fetch_multi_source_news(symbol: str, max_per_source: int = 5) -> List[Dict]:
    """
    Fetch news from multiple sources with expanded queries.
    Only keeps articles that actually mention the search terms.
    """
    all_news = []
    seen_titles = set()
    queries = get_search_queries(symbol)
    
    print(f"\n📰 ENHANCED NEWS FETCH: {symbol}")
    print(f"   Queries: {queries[:4]}...")
    
    # Fetch from each source
    for source_name, source_config in NEWS_SOURCES.items():
        try:
            source_news = []
            
            # Try each query
            for query in queries[:3]:  # Limit to top 3 queries
                search_url = source_config['search_url'].format(query.replace(' ', '+'))
                html = fetch_news_curl(search_url)
                
                if html:
                    articles = extract_articles_from_html(html, source_name.replace('_', ' ').title())
                    
                    for article in articles:
                        title_lower = article['title'].lower()
                        if title_lower not in seen_titles:
                            # STRICT RELEVANCE CHECK: Only keep if title mentions ANY search term
                            is_relevant = any(
                                q.lower() in title_lower 
                                for q in queries
                            )
                            
                            # ONLY add if relevant
                            if is_relevant:
                                seen_titles.add(title_lower)
                                article['is_direct'] = True
                                source_news.append(article)
                
                if len(source_news) >= max_per_source:
                    break
            
            if source_news:
                print(f"   ✅ {source_name}: {len(source_news)} relevant articles")
                all_news.extend(source_news[:max_per_source])
            
        except Exception as e:
            print(f"   ⚠️ {source_name}: Error - {str(e)[:30]}")
    
    # Also fetch macro news that could affect the stock (these are labeled differently)
    macro_news = fetch_macro_news()
    for item in macro_news:
        item['is_direct'] = False
        item['is_macro'] = True
    all_news.extend(macro_news)
    
    # Sort by relevance (direct mentions first) and date
    all_news.sort(key=lambda x: (not x.get('is_direct', False), x.get('date', '')), reverse=True)
    
    print(f"   📊 Total: {len(all_news)} relevant articles")
    return all_news


def fetch_macro_news() -> List[Dict]:
    """Fetch macro economic news that affects all PSX stocks"""
    macro_articles = []
    seen = set()
    
    # Quick search for key macro topics
    for topic in MACRO_CATEGORIES[:3]:  # Limit to avoid slow down
        try:
            search_url = f"https://www.brecorder.com/?s={topic.replace(' ', '+')}"
            html = fetch_news_curl(search_url, timeout=5)
            
            if html:
                articles = extract_articles_from_html(html, 'Business Recorder')
                for article in articles[:2]:
                    if article['title'].lower() not in seen:
                        seen.add(article['title'].lower())
                        article['is_macro'] = True
                        article['is_direct'] = False
                        macro_articles.append(article)
        except:
            continue
    
    return macro_articles[:5]  # Max 5 macro articles


# ============================================================================
# SENTIMENT SCORING WITH TIME DECAY
# ============================================================================

SOURCE_CREDIBILITY = {
    'business recorder': 1.0,
    'dawn': 1.0,
    'pakistan today': 0.9,
    'express tribune': 0.9,
    'psx': 1.0,
    'geo news': 0.8,
    'minute mirror': 0.7,
}


def calculate_news_bias(
    news_items: List[Dict],
    sentiment_scores: List[float]  # From LLM analysis
) -> Dict:
    """
    Calculate time-weighted news bias.
    
    Returns:
        {
            'bias': float (-1 to +1),
            'confidence': float (0 to 1),
            'signals': list of signal descriptions
        }
    """
    if not news_items or not sentiment_scores:
        return {'bias': 0, 'confidence': 0, 'signals': []}
    
    total_weight = 0
    weighted_sentiment = 0
    signals = []
    
    for i, (item, sentiment) in enumerate(zip(news_items, sentiment_scores)):
        # Time decay (7-day half-life)
        try:
            pub_date = datetime.strptime(item.get('date', ''), '%Y-%m-%d')
            days_old = (datetime.now() - pub_date).days
        except:
            days_old = 0
        
        decay = math.exp(-days_old / 7)
        
        # Source credibility
        source = item.get('source', '').lower()
        cred = SOURCE_CREDIBILITY.get(source, 0.5)
        
        # Direct vs sector relevance
        relevance = 1.0 if item.get('is_direct') else 0.5
        
        # Macro importance
        if item.get('is_macro'):
            relevance = 0.7  # Medium importance
        
        weight = decay * cred * relevance
        weighted_sentiment += sentiment * weight
        total_weight += weight
        
        # Track significant signals
        if abs(sentiment) > 0.3:
            direction = "🟢 Bullish" if sentiment > 0 else "🔴 Bearish"
            signals.append({
                'title': item['title'][:80],
                'source': item.get('source', 'Unknown'),
                'direction': direction,
                'strength': abs(sentiment)
            })
    
    bias = weighted_sentiment / total_weight if total_weight > 0 else 0
    confidence = min(total_weight / 5, 1.0)  # Max confidence at 5 weighted items
    
    return {
        'bias': round(bias, 3),
        'confidence': round(confidence, 3),
        'signals': signals[:5]  # Top 5 signals
    }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def get_enhanced_news_for_symbol(symbol: str) -> Dict:
    """
    Main function: Get comprehensive news for a symbol.
    
    Returns:
        {
            'news_items': list of articles,
            'queries_used': list of search terms,
            'sources_checked': list of sources,
            'parent_company': str or None,
            'sector': str,
            'sector_peers': list
        }
    """
    symbol = symbol.upper()
    
    news_items = fetch_multi_source_news(symbol)
    queries = get_search_queries(symbol)
    
    company_info = COMPANY_ALIASES.get(symbol, {})
    
    return {
        'news_items': news_items,
        'queries_used': queries,
        'sources_checked': list(NEWS_SOURCES.keys()),
        'parent_company': company_info.get('parent'),
        'sector': company_info.get('sector', 'unknown'),
        'sector_peers': company_info.get('sector_peers', []),
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    # Test with FFC (should catch Fauji Foundation news)
    print("\n" + "=" * 60)
    print("Testing Enhanced News Fetcher")
    print("=" * 60)
    
    result = get_enhanced_news_for_symbol('FFC')
    
    print(f"\nQueries used: {result['queries_used']}")
    print(f"Parent company: {result['parent_company']}")
    print(f"Sector: {result['sector']}")
    print(f"\nNews found ({len(result['news_items'])} articles):")
    
    for item in result['news_items'][:5]:
        direct = "✓" if item.get('is_direct') else " "
        print(f"  [{direct}] {item['source']}: {item['title'][:60]}...")
