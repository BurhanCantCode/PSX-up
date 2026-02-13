#!/usr/bin/env python3
"""
🔮 BUSINESS RECORDER DEEP ARTICLE SCRAPER
Fetches full article content from BR Research for rich sentiment analysis.
This enables the Fortune Teller to see actual financial metrics, not just headlines.
"""

import re
import json
import subprocess
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path

# Try Selenium for JS-rendered content
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False


# ============================================================================
# STOCK TO COMPANY MAPPING (for search)
# ============================================================================

STOCK_SEARCH_TERMS = {
    'BWHL': ['Baluchistan Wheel', 'BWHL', 'Balochistan Wheel'],
    'PSO': ['Pakistan State Oil', 'PSO'],
    'LUCK': ['Lucky Cement', 'LUCK'],
    'HBL': ['Habib Bank', 'HBL'],
    'UBL': ['United Bank', 'UBL'],
    'MCB': ['MCB Bank', 'Muslim Commercial Bank'],
    'ENGRO': ['Engro Corporation', 'Engro'],
    'SYS': ['Systems Limited', 'SYS'],
    'FFC': ['Fauji Fertilizer', 'FFC'],
    'PPL': ['Pakistan Petroleum', 'PPL'],
    'OGDC': ['OGDC', 'Oil and Gas Development'],
    'MARI': ['Mari Petroleum', 'MARI'],
    'HUBC': ['Hub Power', 'HUBCO'],
    'MEBL': ['Meezan Bank', 'MEBL'],
    'BAFL': ['Bank Alfalah', 'BAFL'],
    'POL': ['Pakistan Oilfields', 'POL'],
    'MTL': ['Millat Tractors', 'MTL'],
}


# ============================================================================
# BUSINESS RECORDER ARTICLE FETCHER
# ============================================================================

def fetch_br_research_articles(symbol: str, max_articles: int = 3) -> List[Dict]:
    """
    Search Business Recorder for research articles about a stock.
    Returns a list of articles with full content.
    """
    search_terms = STOCK_SEARCH_TERMS.get(symbol.upper(), [symbol])
    articles = []
    
    for term in search_terms[:2]:  # Try up to 2 search terms
        search_url = f"https://www.brecorder.com/?s={term.replace(' ', '+')}"
        
        try:
            # Use curl to fetch search results
            result = subprocess.run(
                ['curl', '-s', '-L', '--max-time', '15',
                 '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
                 search_url],
                capture_output=True, text=True, timeout=20
            )
            
            if result.returncode == 0 and result.stdout:
                html = result.stdout
                
                # Find article links - look for BR Research articles specifically
                # Pattern: look for links with article IDs
                article_urls = re.findall(
                    r'href="(https://www\.brecorder\.com/news/\d+)"',
                    html
                )
                
                # Also check for research articles
                research_urls = re.findall(
                    r'href="(https://www\.brecorder\.com/news/\d+/[^"]*)"',
                    html
                )
                
                all_urls = list(set(article_urls + research_urls))
                
                # Fetch full content for each article
                for url in all_urls[:max_articles]:
                    if url not in [a.get('url') for a in articles]:
                        article = fetch_article_content(url)
                        if article and is_relevant_article(article, symbol, search_terms):
                            articles.append(article)
                            
                            if len(articles) >= max_articles:
                                break
                
                if articles:
                    break  # Found articles, don't need to try more search terms
                    
        except Exception as e:
            print(f"  ⚠️ Error searching BR: {e}")
            continue
    
    return articles


def fetch_article_content(url: str) -> Optional[Dict]:
    """
    Fetch the full content of a Business Recorder article.
    Extracts title, date, author, and full text.
    """
    try:
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '20',
             '-H', 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
             url],
            capture_output=True, text=True, timeout=25
        )
        
        if result.returncode != 0 or not result.stdout:
            return None
        
        html = result.stdout
        
        # Extract title
        title_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
        title = title_match.group(1).strip() if title_match else "Unknown Title"
        
        # Alternative title extraction
        if title == "Unknown Title":
            title_match = re.search(r'<title>([^<]+)</title>', html)
            if title_match:
                title = title_match.group(1).split('|')[0].strip()
        
        # Extract date
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', html)
        if not date_match:
            date_match = re.search(r'(\w+ \d{1,2},? \d{4})', html)
        date = date_match.group(1) if date_match else datetime.now().strftime('%Y-%m-%d')
        
        # Extract article body - look for article content divs
        content = ""
        
        # Try different content extraction patterns
        patterns = [
            r'<div[^>]*class="[^"]*story-content[^"]*"[^>]*>(.*?)</div>',
            r'<div[^>]*class="[^"]*entry-content[^"]*"[^>]*>(.*?)</div>',
            r'<article[^>]*>(.*?)</article>',
            r'<div[^>]*class="[^"]*post-content[^"]*"[^>]*>(.*?)</div>',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
            if match:
                raw_content = match.group(1)
                # Strip HTML tags
                content = re.sub(r'<[^>]+>', ' ', raw_content)
                content = re.sub(r'\s+', ' ', content).strip()
                if len(content) > 200:
                    break
        
        # Fallback: extract all paragraph text
        if len(content) < 200:
            paragraphs = re.findall(r'<p[^>]*>([^<]{50,})</p>', html)
            content = ' '.join(paragraphs)
            content = re.sub(r'<[^>]+>', ' ', content)
            content = re.sub(r'\s+', ' ', content).strip()
        
        if len(content) < 100:
            return None
        
        # Extract financial metrics mentioned in the article
        metrics = extract_financial_metrics(content)
        
        return {
            'url': url,
            'title': title,
            'date': date,
            'content': content[:5000],  # Limit to 5000 chars
            'content_length': len(content),
            'source': 'Business Recorder Research',
            'financial_metrics': metrics,
            'fetched_at': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"  ⚠️ Error fetching article {url}: {e}")
        return None


def is_relevant_article(article: Dict, symbol: str, search_terms: List[str]) -> bool:
    """
    Check if an article is actually relevant to the stock.
    """
    title_lower = article.get('title', '').lower()
    content_lower = article.get('content', '').lower()
    
    # Check for symbol or company name in title or content
    for term in [symbol.lower()] + [t.lower() for t in search_terms]:
        if term in title_lower or term in content_lower:
            return True
    
    return False


def extract_financial_metrics(text: str) -> Dict:
    """
    Extract key financial metrics from article text.
    Returns structured data about revenue, profit, margins, etc.
    """
    metrics = {}
    
    # Revenue/Sales patterns
    revenue_patterns = [
        r'(?:net sales|revenue|sales).*?(?:Rs\.?|PKR)\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|mn|bn)',
        r'(?:net sales|revenue|sales).*?(\d+(?:\.\d+)?)\s*(?:percent|%)\s*(?:increase|growth|grew)',
        r'(\d+(?:\.\d+)?)\s*(?:percent|%).*?(?:revenue|sales)\s*(?:growth|increase)',
    ]
    
    for pattern in revenue_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['revenue_mentioned'] = match.group(0)[:100]
            break
    
    # Profit patterns
    profit_patterns = [
        r'(?:net profit|profit after tax|PAT|earnings).*?(?:Rs\.?|PKR)\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|mn|bn)',
        r'(?:net profit|profit).*?(\d+(?:\.\d+)?)\s*(?:percent|%)\s*(?:increase|decrease|grew|declined)',
        r'(?:EPS|earnings per share).*?(?:Rs\.?|PKR)\s*([\d,]+(?:\.\d+)?)',
    ]
    
    for pattern in profit_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['profit_mentioned'] = match.group(0)[:100]
            break
    
    # Margin patterns
    margin_patterns = [
        r'(?:gross profit|GP|operating|net profit)\s*margin.*?(\d+(?:\.\d+)?)\s*(?:percent|%)',
        r'(\d+(?:\.\d+)?)\s*(?:percent|%).*?(?:margin)',
    ]
    
    for pattern in margin_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['margin_mentioned'] = match.group(0)[:100]
            break
    
    # Dividend patterns
    dividend_patterns = [
        r'(?:dividend|cash dividend|payout).*?(?:Rs\.?|PKR)\s*([\d,]+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*(?:percent|%).*?(?:dividend|payout)',
    ]
    
    for pattern in dividend_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metrics['dividend_mentioned'] = match.group(0)[:100]
            break
    
    # Growth indicators
    growth_words = ['growth', 'increase', 'expansion', 'recovery', 'improve', 'rise', 'surge', 'higher']
    decline_words = ['decline', 'decrease', 'drop', 'fall', 'lower', 'reduced', 'loss', 'down']
    
    text_lower = text.lower()
    growth_count = sum(1 for w in growth_words if w in text_lower)
    decline_count = sum(1 for w in decline_words if w in text_lower)
    
    metrics['growth_keywords'] = growth_count
    metrics['decline_keywords'] = decline_count
    metrics['sentiment_bias'] = 'positive' if growth_count > decline_count else 'negative' if decline_count > growth_count else 'neutral'
    
    return metrics


# ============================================================================
# FUNDAMENTAL DATA FETCHER (from PSX Terminal)
# ============================================================================

def fetch_live_fundamentals(symbol: str) -> Dict:
    """
    Fetch live fundamental data from PSX Terminal.
    Returns P/E ratio, dividend yield, market cap, etc.
    """
    fundamentals = {'symbol': symbol, 'source': 'PSX Terminal'}
    
    try:
        # Fetch from symbol endpoint
        symbol_url = f"https://psxterminal.com/symbol/{symbol}/__data.json?market=REG&x-sveltekit-invalidated=01"
        
        result = subprocess.run(
            ['curl', '-s', '-H', 'User-Agent: Mozilla/5.0', symbol_url],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            data_array = data.get('nodes', [{}])[1].get('data', [])
            
            # Parse symbol data
            for item in data_array:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if key in ['peRatio', 'PE', 'pe']:
                            pe = _resolve_reference(value, data_array)
                            if isinstance(pe, (int, float)) and 0 < pe < 500:
                                fundamentals['pe_ratio'] = float(pe)
                        
                        elif key in ['marketCap', 'market_cap']:
                            mc = _resolve_reference(value, data_array)
                            if mc:
                                fundamentals['market_cap'] = mc
                        
                        elif key in ['dividendYield', 'dividend_yield', 'totalYield']:
                            dy = _resolve_reference(value, data_array)
                            if isinstance(dy, (int, float)):
                                fundamentals['dividend_yield'] = float(dy)
                        
                        elif key in ['latestPrice', 'price', 'close']:
                            price = _resolve_reference(value, data_array)
                            if isinstance(price, (int, float)) and price > 0:
                                fundamentals['price'] = float(price)
        
        # Also fetch from yields endpoint for dividend data
        yields_url = "https://psxterminal.com/yields/__data.json?x-sveltekit-invalidated=01"
        result = subprocess.run(
            ['curl', '-s', '-H', 'User-Agent: Mozilla/5.0', yields_url],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0 and result.stdout:
            data = json.loads(result.stdout)
            data_array = data.get('nodes', [{}])[1].get('data', [])
            
            # Find the symbol in yields data
            for i, item in enumerate(data_array):
                if isinstance(item, dict) and 'latestPrice' in item:
                    if i + 1 < len(data_array) and data_array[i + 1] == symbol:
                        # Found our symbol
                        if 'totalYield' in item and 'dividend_yield' not in fundamentals:
                            dy = _resolve_reference(item['totalYield'], data_array)
                            if isinstance(dy, (int, float)):
                                fundamentals['dividend_yield'] = float(dy)
        
        # Fetch from market overview for P/E
        if 'pe_ratio' not in fundamentals:
            market_url = "https://psxterminal.com/market/__data.json?x-sveltekit-invalidated=01"
            result = subprocess.run(
                ['curl', '-s', '-H', 'User-Agent: Mozilla/5.0', market_url],
                capture_output=True, text=True, timeout=15
            )
            
            if result.returncode == 0 and result.stdout:
                data = json.loads(result.stdout)
                data_array = data.get('nodes', [{}])[1].get('data', [])
                
                # Find symbol in mapping
                for item in data_array:
                    if isinstance(item, dict) and symbol in item:
                        idx = item[symbol]
                        if isinstance(idx, int) and idx < len(data_array):
                            stock_data = data_array[idx]
                            if isinstance(stock_data, dict) and 'peRatio' in stock_data:
                                pe = _resolve_reference(stock_data['peRatio'], data_array)
                                if isinstance(pe, (int, float)) and 0 < pe < 500:
                                    fundamentals['pe_ratio'] = float(pe)
        
        fundamentals['fetched_at'] = datetime.now().isoformat()
        
    except Exception as e:
        print(f"  ⚠️ Error fetching fundamentals for {symbol}: {e}")
    
    return fundamentals


def _resolve_reference(ref, data_array: List, depth: int = 0) -> any:
    """Resolve SvelteKit data reference."""
    if depth > 5:
        return None
    
    if isinstance(ref, int) and 0 <= ref < len(data_array):
        resolved = data_array[ref]
        if isinstance(resolved, dict):
            if 'numeric' in resolved:
                return _resolve_reference(resolved['numeric'], data_array, depth + 1)
            elif 'raw' in resolved:
                try:
                    return float(str(resolved['raw']).replace(',', ''))
                except:
                    return resolved['raw']
            return resolved
        return resolved
    return ref


def calculate_quality_score(fundamentals: Dict) -> float:
    """
    Calculate a quality score for a stock based on fundamentals.
    Higher score = higher quality stock that shouldn't be over-sold.
    
    Score from 0.0 to 1.0
    """
    score = 0.5  # Start neutral
    
    # P/E ratio scoring (lower is better for value)
    pe = fundamentals.get('pe_ratio')
    if pe:
        if pe < 8:
            score += 0.15  # Deep value
        elif pe < 12:
            score += 0.10  # Value
        elif pe < 18:
            score += 0.05  # Fair
        elif pe > 30:
            score -= 0.10  # Expensive
    
    # Dividend yield scoring (higher is better for stability)
    div_yield = fundamentals.get('dividend_yield')
    if div_yield:
        if div_yield > 8:
            score += 0.15  # High yield
        elif div_yield > 5:
            score += 0.10  # Good yield
        elif div_yield > 2:
            score += 0.05  # Modest yield
    
    # Cap score between 0 and 1
    return max(0.0, min(1.0, score))


# ============================================================================
# MAIN API FUNCTION
# ============================================================================

def get_enriched_stock_data(symbol: str) -> Dict:
    """
    Get enriched data for a stock including:
    - Full article content from BR Research
    - Live fundamentals from PSX Terminal
    - Quality score calculation
    
    This is what gets passed to the AI for rich analysis.
    """
    symbol = symbol.upper()
    
    print(f"\n📚 FETCHING ENRICHED DATA: {symbol}")
    print("=" * 50)
    
    # 1. Fetch BR Research articles
    print("📰 Searching Business Recorder Research...")
    articles = fetch_br_research_articles(symbol, max_articles=2)
    print(f"   Found {len(articles)} relevant articles")
    
    # 2. Fetch live fundamentals
    print("📊 Fetching live fundamentals from PSX Terminal...")
    fundamentals = fetch_live_fundamentals(symbol)
    has_fundamentals = 'pe_ratio' in fundamentals or 'dividend_yield' in fundamentals
    print(f"   P/E: {fundamentals.get('pe_ratio', 'N/A')}, Div Yield: {fundamentals.get('dividend_yield', 'N/A')}%")
    
    # 3. Calculate quality score
    quality_score = calculate_quality_score(fundamentals)
    print(f"   Quality Score: {quality_score:.2f}")
    
    # 4. Build enriched data
    enriched = {
        'symbol': symbol,
        'articles': articles,
        'article_count': len(articles),
        'fundamentals': fundamentals,
        'quality_score': quality_score,
        'has_rich_data': len(articles) > 0 or has_fundamentals,
        'enriched_at': datetime.now().isoformat()
    }
    
    # 5. Create summary text for AI
    summary_parts = []
    
    if fundamentals.get('pe_ratio'):
        summary_parts.append(f"P/E Ratio: {fundamentals['pe_ratio']:.1f}")
    if fundamentals.get('dividend_yield'):
        summary_parts.append(f"Dividend Yield: {fundamentals['dividend_yield']:.1f}%")
    if fundamentals.get('price'):
        summary_parts.append(f"Current Price: PKR {fundamentals['price']:.2f}")
    
    for article in articles[:2]:
        metrics = article.get('financial_metrics', {})
        if metrics.get('revenue_mentioned'):
            summary_parts.append(f"Revenue: {metrics['revenue_mentioned']}")
        if metrics.get('profit_mentioned'):
            summary_parts.append(f"Profit: {metrics['profit_mentioned']}")
        if metrics.get('margin_mentioned'):
            summary_parts.append(f"Margin: {metrics['margin_mentioned']}")
    
    enriched['fundamental_summary'] = ' | '.join(summary_parts) if summary_parts else 'No fundamental data available'
    
    print(f"✅ Enriched data ready")
    print(f"   Summary: {enriched['fundamental_summary'][:100]}...")
    
    return enriched


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🔮 BUSINESS RECORDER DEEP ARTICLE SCRAPER - TEST")
    print("=" * 70)
    
    # Test with BWHL
    enriched = get_enriched_stock_data('BWHL')
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"Symbol: {enriched['symbol']}")
    print(f"Articles Found: {enriched['article_count']}")
    print(f"Quality Score: {enriched['quality_score']:.2f}")
    print(f"Has Rich Data: {enriched['has_rich_data']}")
    print(f"Fundamental Summary: {enriched['fundamental_summary']}")
    
    if enriched['articles']:
        print("\nArticles:")
        for i, article in enumerate(enriched['articles'], 1):
            print(f"\n{i}. {article['title']}")
            print(f"   URL: {article['url']}")
            print(f"   Date: {article['date']}")
            print(f"   Content Length: {article['content_length']} chars")
            if article.get('financial_metrics'):
                print(f"   Metrics: {article['financial_metrics']}")
