#!/usr/bin/env python3
"""
🔮 PRODUCTION-GRADE AI SENTIMENT ANALYZER
Uses Premium News Fetcher (10+ sources) + Groq LLM for intelligent analysis.
Designed to be a "fortune teller" for PSX stocks.
"""

import os
import json
import subprocess
import re
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv

# Import premium news fetcher
try:
    from backend.premium_news_fetcher import fetch_premium_news, get_news_for_sentiment_analysis
    PREMIUM_FETCHER_AVAILABLE = True
except ImportError:
    try:
        from premium_news_fetcher import fetch_premium_news, get_news_for_sentiment_analysis
        PREMIUM_FETCHER_AVAILABLE = True
    except ImportError:
        PREMIUM_FETCHER_AVAILABLE = False
        print("⚠️  Premium news fetcher not available, using fallback")

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Selenium imports
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from webdriver_manager.chrome import ChromeDriverManager
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium not installed")

# Groq imports
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not installed")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cache settings
CACHE_DIR = Path(__file__).parent.parent / "data" / "news_cache"
CACHE_DURATION_HOURS = 4  # Cache news for 4 hours

# Stock symbol to company name mapping
STOCK_COMPANIES = {
    'LUCK': ('Lucky Cement', 'Lucky Cement Limited', 'LUCK'),
    'HBL': ('Habib Bank', 'Habib Bank Limited', 'HBL'),
    'UBL': ('United Bank', 'United Bank Limited', 'UBL'),
    'MCB': ('MCB Bank', 'MCB Bank Limited', 'Muslim Commercial Bank'),
    'OGDC': ('OGDC', 'Oil and Gas Development Company', 'Oil Gas Development'),
    'PPL': ('Pakistan Petroleum', 'PPL', 'Pakistan Petroleum Limited'),
    'PSO': ('Pakistan State Oil', 'PSO', 'State Oil'),
    'ENGRO': ('Engro', 'Engro Corporation', 'Engro Corp'),
    'FFC': ('Fauji Fertilizer', 'FFC', 'Fauji Fertilizer Company'),
    'FATIMA': ('Fatima Fertilizer', 'Fatima', 'Fatima Group'),
    'HUBC': ('Hub Power', 'HUBCO', 'Hub Power Company'),
    'SYS': ('Systems Limited', 'SYS', 'Systems Ltd'),
    'TRG': ('TRG Pakistan', 'TRG', 'The Resource Group'),
    'NESTLE': ('Nestle Pakistan', 'Nestle', 'NESTLE'),
    'MARI': ('Mari Petroleum', 'MARI', 'Mari Gas'),
    'ISL': ('Ismail Industries', 'ISL', 'Ismail'),
    'KAPCO': ('Kot Addu Power', 'KAPCO', 'Kot Addu'),
    'NCPL': ('Nishat Chunian Power', 'NCPL', 'Nishat Power'),
    'MEBL': ('Meezan Bank', 'MEBL', 'Meezan'),
    'SEARL': ('Searle Pakistan', 'SEARL', 'Searle'),
    'PIOC': ('Pioneer Cement', 'PIOC', 'Pioneer'),
    'DGKC': ('DG Khan Cement', 'DGKC', 'DG Cement'),
    'MLCF': ('Maple Leaf Cement', 'MLCF', 'Maple Leaf'),
    'KOHC': ('Kohat Cement', 'KOHC', 'Kohat'),
    'KEL': ('K-Electric', 'KEL', 'Karachi Electric'),
    'NBP': ('National Bank', 'NBP', 'National Bank Pakistan'),
    'ABL': ('Allied Bank', 'ABL', 'Allied Bank Limited'),
    'BAFL': ('Bank Alfalah', 'BAFL', 'Alfalah'),
    'BAHL': ('Bank Al Habib', 'BAHL', 'Al Habib'),
    'POL': ('Pakistan Oilfields', 'POL', 'Pakistan Oilfields Limited'),
    'ATRL': ('Attock Refinery', 'ATRL', 'Attock'),
    'EFERT': ('Engro Fertilizers', 'EFERT', 'Engro Fert'),
    'CHCC': ('Cherat Cement', 'CHCC', 'Cherat'),
    'FCCL': ('Fauji Cement', 'FCCL', 'Fauji Cement Company'),
    'PTC': ('Pakistan Tobacco', 'PTC', 'Pak Tobacco'),
    'GLAXO': ('GlaxoSmithKline', 'GSK Pakistan', 'Glaxo'),
    'INDU': ('Indus Motor', 'INDU', 'Toyota Indus'),
    'HCAR': ('Honda Atlas Cars', 'HCAR', 'Honda Cars'),
    'MTL': ('Millat Tractors', 'MTL', 'Millat'),
}

# News source configurations
NEWS_SOURCES = {
    'business_recorder': {
        'search_url': 'https://www.brecorder.com/?s={}',
        'selectors': {
            'articles': '.story-title a, .entry-title a, article h2 a, .td-module-title a',
            'dates': '.entry-date, .td-post-date, time, .post-date',
        }
    },
    'dawn': {
        'search_url': 'https://www.dawn.com/search?q={}',
        'selectors': {
            'articles': 'h2 a, .story__title a, article h3 a',
            'dates': '.timestamp, time, .story__time',
        }
    },
    'tribune': {
        'search_url': 'https://tribune.com.pk/?s={}',
        'selectors': {
            'articles': '.entry-title a, h2.title a, article h2 a',
            'dates': '.entry-date, time, .post-date',
        }
    },
    'arynews': {
        'search_url': 'https://arynews.tv/?s={}',
        'selectors': {
            'articles': 'h3 a, .entry-title a, article h2 a',
            'dates': '.entry-date, time',
        }
    }
}


# ============================================================================
# SELENIUM DRIVER MANAGEMENT
# ============================================================================

_driver_lock = threading.Lock()
_driver = None

def get_selenium_driver():
    """Get or create a headless Chrome driver (singleton)"""
    global _driver
    
    if not SELENIUM_AVAILABLE:
        return None
    
    with _driver_lock:
        if _driver is None:
            try:
                chrome_options = Options()
                chrome_options.add_argument('--headless=new')
                chrome_options.add_argument('--no-sandbox')
                chrome_options.add_argument('--disable-dev-shm-usage')
                chrome_options.add_argument('--disable-gpu')
                chrome_options.add_argument('--window-size=1920,1080')
                chrome_options.add_argument('--disable-extensions')
                chrome_options.add_argument('--disable-infobars')
                chrome_options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
                
                service = Service(ChromeDriverManager().install())
                _driver = webdriver.Chrome(service=service, options=chrome_options)
                _driver.set_page_load_timeout(20)
                print("✅ Chrome WebDriver initialized")
            except Exception as e:
                print(f"❌ Failed to initialize WebDriver: {e}")
                return None
        
        return _driver


def close_driver():
    """Close the Selenium driver"""
    global _driver
    with _driver_lock:
        if _driver:
            _driver.quit()
            _driver = None


# ============================================================================
# CACHING SYSTEM
# ============================================================================

def get_cache_path(symbol: str) -> Path:
    """Get cache file path for a symbol"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{symbol.upper()}_news.json"


def load_cached_news(symbol: str) -> Optional[Dict]:
    """Load cached news if not expired"""
    cache_path = get_cache_path(symbol)
    
    if cache_path.exists():
        try:
            with open(cache_path, 'r') as f:
                cached = json.load(f)
            
            cached_time = datetime.fromisoformat(cached.get('cached_at', '2000-01-01'))
            if datetime.now() - cached_time < timedelta(hours=CACHE_DURATION_HOURS):
                return cached
        except:
            pass
    
    return None


def save_news_to_cache(symbol: str, news_data: Dict):
    """Save news to cache"""
    cache_path = get_cache_path(symbol)
    news_data['cached_at'] = datetime.now().isoformat()
    
    try:
        with open(cache_path, 'w') as f:
            json.dump(news_data, f, indent=2)
    except:
        pass


# ============================================================================
# NEWS SCRAPING WITH SELENIUM
# ============================================================================

def scrape_news_selenium(driver, source_name: str, search_url: str, selectors: Dict, search_term: str) -> List[Dict]:
    """Scrape news from a source using Selenium"""
    news_items = []
    
    try:
        url = search_url.format(search_term.replace(' ', '+'))
        driver.get(url)
        
        # Wait for content to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Try to find articles
        articles = []
        for selector in selectors['articles'].split(', '):
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                articles.extend(elements)
            except:
                continue
        
        # Extract article info
        seen_titles = set()
        for article in articles[:15]:
            try:
                title = article.text.strip()
                href = article.get_attribute('href') or ''
                
                if title and len(title) > 15 and title.lower() not in seen_titles:
                    seen_titles.add(title.lower())
                    
                    # Try to find associated date
                    date_str = datetime.now().strftime('%Y-%m-%d')
                    
                    news_items.append({
                        'title': title[:300],
                        'url': href,
                        'date': date_str,
                        'source': source_name.replace('_', ' ').title()
                    })
            except:
                continue
        
    except Exception as e:
        print(f"  ⚠️ Error scraping {source_name}: {str(e)[:50]}")
    
    return news_items


def scrape_psx_announcements(symbol: str) -> List[Dict]:
    """Scrape PSX company announcements page"""
    news_items = []
    
    try:
        import subprocess
        url = f"https://dps.psx.com.pk/company/{symbol.upper()}"
        
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '15', url],
            capture_output=True, text=True, timeout=20
        )
        
        if result.returncode == 0 and result.stdout:
            html = result.stdout
            
            # Find announcement table rows
            rows = re.findall(r'<tr[^>]*>.*?</tr>', html, re.DOTALL | re.IGNORECASE)
            
            for row in rows:
                # Extract date and title
                cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
                if len(cells) >= 2:
                    date_text = cells[0].strip()
                    title = ' '.join(cells[1:3]).strip()
                    
                    if len(title) > 10:
                        news_items.append({
                            'title': title[:300],
                            'date': date_text,
                            'source': 'PSX',
                            'url': url
                        })
    except:
        pass
    
    return news_items[:10]


def fetch_all_news(symbol: str, company_names: Tuple[str, ...]) -> List[Dict]:
    """Fetch news from all sources using Selenium"""
    all_news = []
    
    # First check PSX announcements (no Selenium needed)
    print(f"  📰 Checking PSX announcements...")
    psx_news = scrape_psx_announcements(symbol)
    all_news.extend(psx_news)
    print(f"     Found {len(psx_news)} PSX items")
    
    # Get Selenium driver
    driver = get_selenium_driver()
    
    if driver:
        # Search terms: symbol + company names
        search_terms = [symbol] + list(company_names[:2])
        
        for source_name, config in NEWS_SOURCES.items():
            print(f"  🌐 Searching {source_name.replace('_', ' ').title()}...")
            
            for search_term in search_terms:
                try:
                    items = scrape_news_selenium(
                        driver,
                        source_name,
                        config['search_url'],
                        config['selectors'],
                        search_term
                    )
                    
                    # Filter to only include relevant news
                    relevant_items = []
                    for item in items:
                        title_lower = item['title'].lower()
                        if any(term.lower() in title_lower for term in [symbol] + list(company_names)):
                            relevant_items.append(item)
                    
                    all_news.extend(relevant_items)
                    
                    if relevant_items:
                        print(f"     Found {len(relevant_items)} relevant items for '{search_term}'")
                        break  # Found news, no need to try other search terms
                except Exception as e:
                    print(f"     ⚠️ Error: {str(e)[:40]}")
                    continue
    else:
        print("  ⚠️ Selenium not available, using curl fallback...")
        # Fallback to curl
        for name in company_names[:2]:
            all_news.extend(fetch_news_curl(name))
    
    # Deduplicate
    seen = set()
    unique_news = []
    for item in all_news:
        key = item['title'].lower()[:50]
        if key not in seen:
            seen.add(key)
            unique_news.append(item)
    
    return unique_news[:20]


def fetch_news_curl(search_term: str) -> List[Dict]:
    """Fallback: fetch news using curl"""
    news = []
    
    try:
        url = f"https://www.brecorder.com/?s={search_term.replace(' ', '+')}"
        result = subprocess.run(
            ['curl', '-s', '-L', '--max-time', '10', url],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode == 0:
            # Extract links with titles
            matches = re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]{20,150})</a>', result.stdout)
            for href, title in matches[:5]:
                if search_term.lower() in title.lower():
                    news.append({
                        'title': title.strip(),
                        'url': href,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'source': 'Business Recorder'
                    })
    except:
        pass
    
    return news


# ============================================================================
# GROQ INTEGRATION
# ============================================================================

def get_groq_client():
    """Get Groq client"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")
    return Groq(api_key=api_key)


def analyze_with_ai(symbol: str, company_name: str, news_items: List[Dict]) -> Dict:
    """Use Groq (Llama 3.3 70B) for intelligent sentiment analysis with anti-hallucination guardrails"""
    
    if not GROQ_AVAILABLE:
        return fallback_analysis(news_items)
    
    # Get current date for context
    current_date = datetime.now().strftime('%Y-%m-%d')
    
    # Format news with explicit dates
    if news_items:
        news_text = "\n".join([
            f"• [{item.get('date', 'unknown date')}] [{item.get('source_name', item.get('source', 'Unknown'))}] {item['title']}"
            for item in news_items[:15]
        ])
    else:
        news_text = "No recent news found."
    
    prompt = f"""You are a CONSERVATIVE Pakistani stock market analyst. Today's date is {current_date}.

CRITICAL RULES - READ CAREFULLY:
1. ONLY cite facts that appear DIRECTLY in the news headlines below
2. NEVER fabricate acquisitions, mergers, or deals that aren't explicitly mentioned
3. NEVER give specific percentage predictions like "will rise 5%" - say "unclear" instead
4. If a deal/acquisition is mentioned, check the date - OLD news (>3 months) may be STALE or WITHDRAWN
5. Catalysts must be ONLY from the actual news - do NOT invent potential catalysts
6. When uncertain, say "unclear" or "insufficient data"
7. Be SKEPTICAL - a headline about an old AGM is not a buy signal

Analyze the following news about {symbol} ({company_name}):

{news_text}

Based STRICTLY on the news above, provide analysis. 

RESPOND IN JSON FORMAT ONLY:
{{
    "sentiment_score": <float from -1.0 to +1.0, use 0 if unclear>,
    "signal": "<BUY|HOLD|SELL>",
    "confidence": <float 0.0-1.0, lower if news is sparse or old>,
    "verified_events": ["ONLY list events that appear verbatim in headlines above"],
    "price_impact": {{
        "estimate": "<use 'unclear' unless there is very specific financial data>",
        "timeframe": "<unclear if not specified in news>",
        "reasoning": "<brief reasoning based ONLY on provided headlines>"
    }},
    "risks": ["only risks mentioned or implied in the headlines"],
    "catalysts": ["ONLY catalysts explicitly mentioned in headlines - do NOT invent any"],
    "data_quality": "<good|limited|poor> - based on how much actionable news we have",
    "summary": "<2-3 factual sentences ONLY referencing the actual headlines, acknowledge uncertainty>"
}}

ANTI-HALLUCINATION CHECKLIST before responding:
- Did I only cite facts from the headlines above? 
- Did I avoid making up specific % predictions?
- Did I avoid inventing acquisitions/deals not in the headlines?
- Am I being appropriately uncertain given sparse data?

Return ONLY valid JSON."""

    try:
        client = get_groq_client()
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.05  # Lower temperature = more conservative
        )
        
        response_text = completion.choices[0].message.content.strip()
        result = json.loads(response_text)
        
        result['model'] = 'llama-3.3-70b-versatile'
        result['analyzed_at'] = datetime.now().isoformat()
        
        # Rename verified_events to key_events for compatibility
        if 'verified_events' in result:
            result['key_events'] = result.pop('verified_events')
        
        # Map to simpler signal for UI
        signal_map = {
            'STRONG_BUY': 'BULLISH',
            'BUY': 'BULLISH',
            'HOLD': 'NEUTRAL',
            'SELL': 'BEARISH',
            'STRONG_SELL': 'BEARISH'
        }
        result['signal_simple'] = signal_map.get(result.get('signal', 'HOLD'), 'NEUTRAL')
        
        return result
        
    except Exception as e:
        print(f"❌ Groq error: {e}")
        return fallback_analysis(news_items)


def fallback_analysis(news_items: List[Dict]) -> Dict:
    """Fallback when Claude is unavailable"""
    text = ' '.join([item.get('title', '') for item in news_items]).lower()
    
    bullish = sum(1 for w in ['profit', 'growth', 'dividend', 'acquire', 'expansion', 'record', 'surge'] if w in text)
    bearish = sum(1 for w in ['loss', 'decline', 'drop', 'fraud', 'investigation', 'shutdown'] if w in text)
    
    if bullish > bearish:
        score, signal = min(0.5, bullish * 0.15), 'BULLISH'
    elif bearish > bullish:
        score, signal = max(-0.5, -bearish * 0.15), 'BEARISH'
    else:
        score, signal = 0, 'NEUTRAL'
    
    return {
        'sentiment_score': score,
        'signal': signal,
        'signal_simple': signal,
        'confidence': 0.3,
        'key_events': [],
        'price_impact': {'estimate': 'unclear', 'timeframe': 'unclear', 'reasoning': 'Keyword analysis only'},
        'summary': 'Analysis based on keyword matching (AI unavailable)',
        'model': 'fallback',
        'analyzed_at': datetime.now().isoformat()
    }


# ============================================================================
# MAIN API FUNCTIONS
# ============================================================================

def get_stock_sentiment(symbol: str, use_cache: bool = True) -> Dict:
    """
    🔮 Main function: Get comprehensive AI-powered sentiment for a stock.
    This is the "fortune teller" function.
    Now uses PREMIUM NEWS FETCHER with 10+ sources!
    """
    symbol = symbol.upper()
    
    # Get company names
    company_names = STOCK_COMPANIES.get(symbol, (symbol,))
    company_name = company_names[0] if company_names else symbol
    
    print(f"\n🔮 SENTIMENT ANALYSIS: {symbol} ({company_name})")
    print("=" * 50)
    
    # Check cache
    if use_cache:
        cached = load_cached_news(symbol)
        if cached:
            print("📦 Using cached analysis (less than 4 hours old)")
            return cached
    
    # Fetch news using PREMIUM fetcher (10+ sources!)
    if PREMIUM_FETCHER_AVAILABLE:
        print("🚀 Using PREMIUM NEWS FETCHER (10+ sources)...")
        premium_result = fetch_premium_news(symbol, use_cache=False, verbose=True)
        news_items = premium_result.get('news_items', [])
        sources_searched = premium_result.get('sources_searched', [])
        print(f"📊 Total news items found: {len(news_items)}")
    else:
        # Fallback to basic fetcher
        print("📰 Fetching news from multiple sources...")
        news_items = fetch_all_news(symbol, company_names)
        sources_searched = list(NEWS_SOURCES.keys()) + ['PSX']
        print(f"📊 Total news items found: {len(news_items)}")
    
    # Display found news
    if news_items:
        print("\n📋 Headlines found:")
        for item in news_items[:5]:
            source = item.get('source_name', item.get('source', 'Unknown'))
            print(f"   • [{source}] {item['title'][:70]}...")
    
    # Analyze with AI (Groq)
    print("\n🤖 Analyzing with Groq (Llama 3.3)...")
    analysis = analyze_with_ai(symbol, company_name, news_items)
    
    # Build complete result
    result = {
        'symbol': symbol,
        'company': company_name,
        'news_count': len(news_items),
        'news_items': news_items[:15],  # More news items now!
        'sources_searched': sources_searched,
        **analysis
    }
    
    # Generate emoji
    signal = result.get('signal_simple', result.get('signal', 'NEUTRAL'))
    if signal in ['BULLISH', 'STRONG_BUY', 'BUY']:
        result['signal_emoji'] = '🟢'
    elif signal in ['BEARISH', 'STRONG_SELL', 'SELL']:
        result['signal_emoji'] = '🔴'
    else:
        result['signal_emoji'] = '🟡'
    
    # Cache result
    save_news_to_cache(symbol, result)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"🎯 RESULT: {result.get('signal_emoji', '🟡')} {result.get('signal', 'NEUTRAL')}")
    print(f"📈 Sentiment Score: {result.get('sentiment_score', 0):.2f}")
    print(f"🎯 Confidence: {result.get('confidence', 0):.0%}")
    if result.get('price_impact'):
        pi = result['price_impact']
        print(f"💰 Expected Impact: {pi.get('estimate', 'unclear')} over {pi.get('timeframe', 'unclear')}")
    print(f"📝 Summary: {result.get('summary', 'N/A')}")
    
    return result


# Backward compatibility aliases
def analyze_news_for_stock(symbol: str, news_items: List[Dict] = None) -> Dict:
    return get_stock_sentiment(symbol)


async def get_market_sentiment() -> Dict:
    """Get overall market sentiment"""
    stocks = ['OGDC', 'HBL', 'LUCK']
    total, count = 0, 0
    
    for symbol in stocks:
        try:
            result = get_stock_sentiment(symbol, use_cache=True)
            total += result.get('sentiment_score', 0)
            count += 1
        except:
            continue
    
    avg = total / count if count > 0 else 0
    
    return {
        'market_sentiment': round(avg, 3),
        'signal': 'BULLISH' if avg > 0.15 else 'BEARISH' if avg < -0.15 else 'NEUTRAL',
        'emoji': '🟢' if avg > 0.15 else '🔴' if avg < -0.15 else '🟡',
        'stocks_analyzed': count,
        'analyzed_at': datetime.now().isoformat()
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🔮 PSX FORTUNE TELLER - AI-Powered Sentiment Analysis")
    print("="*70)
    print()
    print(f"✅ Groq: {'Available' if GROQ_AVAILABLE else 'Not Available'}")
    print(f"✅ API Key: {'Loaded' if os.getenv('GROQ_API_KEY') else 'Missing'}")
    print()
    
    # Test with a stock
    test_symbols = ['LUCK', 'SYS']
    
    for symbol in test_symbols:
        result = get_stock_sentiment(symbol, use_cache=False)
        print()
    
    # Cleanup
    close_driver()
    print("\n✅ Done!")
