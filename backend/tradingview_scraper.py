"""
🔍 TRADINGVIEW TECHNICAL INDICATORS SCRAPER
Scrapes real technical indicators from TradingView to fix incorrect local calculations.
Uses Selenium infrastructure from brecorder_scraper.py.

Key Features:
- Scrapes RSI, MACD, Stochastic, ADX, Moving Averages
- Gets Buy/Sell/Neutral recommendation counts
- Rate limiting: 5 second minimum between requests
- Caching: 15 min during market hours, 60 min off-market
- Fallback to local calculation if scrape fails
"""

import re
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import threading

# Cache directory
CACHE_DIR = Path(__file__).parent.parent / "data" / "tradingview_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Selenium imports (reuse from brecorder_scraper)
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False

# Rate limiting
_last_request_time = {}
_request_lock = threading.Lock()
MIN_REQUEST_INTERVAL = 5  # seconds

# Driver management (reuse pattern from brecorder_scraper)
_driver_lock = threading.Lock()
_driver = None


def get_driver():
    """Get or create Selenium driver"""
    global _driver
    if not SELENIUM_AVAILABLE:
        return None
    
    with _driver_lock:
        if _driver is None:
            try:
                options = Options()
                options.add_argument('--headless=new')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                options.add_argument('--disable-blink-features=AutomationControlled')
                options.add_experimental_option('excludeSwitches', ['enable-automation'])
                options.add_argument('--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36')
                
                _driver = webdriver.Chrome(options=options)
                _driver.set_page_load_timeout(30)
            except Exception as e:
                print(f"❌ Failed to init TradingView driver: {e}")
                return None
        return _driver


def close_driver():
    """Close Selenium driver"""
    global _driver
    with _driver_lock:
        if _driver:
            _driver.quit()
            _driver = None


def is_market_hours() -> bool:
    """Check if it's during PSX market hours (9:30 AM - 3:30 PM PKT, Mon-Fri)"""
    now = datetime.now()
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    # Check if during market hours (simplified - adjust for PKT if needed)
    hour = now.hour
    minute = now.minute
    start_time = 9 * 60 + 30  # 9:30 AM in minutes
    end_time = 15 * 60 + 30   # 3:30 PM in minutes
    current_time = hour * 60 + minute
    
    return start_time <= current_time <= end_time


def rate_limit_check(symbol: str):
    """Enforce rate limiting between requests"""
    with _request_lock:
        now = time.time()
        last_time = _last_request_time.get(symbol, 0)
        elapsed = now - last_time
        
        if elapsed < MIN_REQUEST_INTERVAL:
            sleep_time = MIN_REQUEST_INTERVAL - elapsed
            print(f"   ⏱️ Rate limiting: sleeping {sleep_time:.1f}s for {symbol}")
            time.sleep(sleep_time)
        
        _last_request_time[symbol] = time.time()


def scrape_tradingview_technicals(symbol: str) -> Optional[Dict]:
    """
    Scrape technical indicators from TradingView.
    URL: https://www.tradingview.com/symbols/PSX-{SYMBOL}/technicals/
    
    Returns:
        {
            'rsi_14': float,
            'macd': float,
            'macd_signal': float,
            'stochastic_k': float,
            'stochastic_d': float,
            'adx': float,
            'sma_20': float,
            'sma_50': float,
            'sma_200': float,
            'ema_20': float,
            'ema_50': float,
            'recommendation_buy': int,
            'recommendation_sell': int,
            'recommendation_neutral': int,
            'overall_recommendation': str (BUY/SELL/NEUTRAL)
        }
    """
    symbol = symbol.upper()
    
    # Check cache first
    cache_max_age = 15 if is_market_hours() else 60  # minutes
    cached = load_from_cache(symbol, cache_max_age)
    if cached:
        print(f"   ✅ TradingView cache hit for {symbol}")
        return cached
    
    # Rate limiting
    rate_limit_check(symbol)
    
    driver = get_driver()
    if not driver:
        print(f"   ⚠️ No Selenium driver for TradingView")
        return None
    
    url = f"https://www.tradingview.com/symbols/PSX-{symbol}/technicals/"
    
    try:
        print(f"   🌐 Scraping TradingView: {symbol}")
        driver.get(url)
        
        # Wait for page load - try multiple selectors
        try:
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "[class*='speedometer'], [class*='gauge'], .tv-widget-technicals"))
            )
        except:
            # Alternative: just wait for any content
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        
        # Give it extra time to load all indicators
        time.sleep(3)
        
        # Parse technical indicators
        technicals = {}
        
        # Extract recommendation gauge (Buy/Sell/Neutral counts)
        try:
            # Try multiple selectors for the gauge
            gauge = None
            for selector in ["[class*='speedometer']", "[class*='gauge']", "[class*='technical-summary']"]:
                try:
                    gauge = driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except:
                    continue
            
            if not gauge:
                # Try to find text containing "Buy" or "Sell"
                gauge = driver.find_element(By.XPATH, "//*[contains(text(), 'Buy') or contains(text(), 'Sell')]")
            
            recommendation_text = gauge.text.strip() if gauge else ""
            
            # Parse counts from text like "Buy (15) Neutral (10) Sell (3)"
            buy_match = re.search(r'Buy.*?(\d+)', recommendation_text, re.IGNORECASE)
            sell_match = re.search(r'Sell.*?(\d+)', recommendation_text, re.IGNORECASE)
            neutral_match = re.search(r'Neutral.*?(\d+)', recommendation_text, re.IGNORECASE)
            
            technicals['recommendation_buy'] = int(buy_match.group(1)) if buy_match else 0
            technicals['recommendation_sell'] = int(sell_match.group(1)) if sell_match else 0
            technicals['recommendation_neutral'] = int(neutral_match.group(1)) if neutral_match else 0
            
            # Determine overall recommendation
            buy_count = technicals['recommendation_buy']
            sell_count = technicals['recommendation_sell']
            
            if buy_count > sell_count * 1.5:
                technicals['overall_recommendation'] = 'BUY'
            elif sell_count > buy_count * 1.5:
                technicals['overall_recommendation'] = 'SELL'
            else:
                technicals['overall_recommendation'] = 'NEUTRAL'
                
        except Exception as e:
            print(f"   ⚠️ Could not parse recommendations: {e}")
            technicals['recommendation_buy'] = 0
            technicals['recommendation_sell'] = 0
            technicals['recommendation_neutral'] = 0
            technicals['overall_recommendation'] = 'NEUTRAL'
        
        # Extract individual indicators from the table
        # TradingView shows indicators in a table with name and value columns
        try:
            # Find all indicator rows - try multiple selectors
            rows = []
            for selector in ["tr[class*='row']", "tr", "table tr"]:
                try:
                    rows = driver.find_elements(By.CSS_SELECTOR, selector)
                    if rows:
                        break
                except:
                    continue
            
            # Map TradingView indicator names to our column names
            # Use partial matching for flexibility
            indicator_map = {
                'Relative Strength Index': 'rsi_14',
                'MACD Level': 'macd_level',
                'Stochastic %K': 'stochastic_k',
                'Average Directional Index': 'adx',
                'Awesome Oscillator': 'awesome_oscillator',
                'Momentum': 'momentum_10',
                'Williams Percent Range': 'williams_r',
                'Bull Bear Power': 'bull_bear_power',
                'Ultimate Oscillator': 'ultimate_oscillator',
                'Commodity Channel Index': 'cci',
                'Stochastic RSI': 'stochastic_rsi',
                # Moving Averages
                'Exponential Moving Average (10)': 'ema_10',
                'Simple Moving Average (10)': 'sma_10',
                'Exponential Moving Average (20)': 'ema_20',
                'Simple Moving Average (20)': 'sma_20',
                'Exponential Moving Average (30)': 'ema_30',
                'Simple Moving Average (30)': 'sma_30',
                'Exponential Moving Average (50)': 'ema_50',
                'Simple Moving Average (50)': 'sma_50',
                'Exponential Moving Average (100)': 'ema_100',
                'Simple Moving Average (100)': 'sma_100',
                'Volume Weighted Moving Average': 'vwma',
                'Hull Moving Average': 'hma',
            }
            
            for row in rows:
                try:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        name = cells[0].text.strip()
                        value_text = cells[1].text.strip()
                        
                        # Try to extract numeric value (handle negative numbers and decimals)
                        # Remove any currency symbols, commas, etc.
                        value_text_clean = value_text.replace(',', '').replace('$', '')
                        value_match = re.search(r'([-−+]?\d+\.?\d*)', value_text_clean)
                        
                        if value_match:
                            value_str = value_match.group(1).replace('−', '-')  # Replace minus sign
                            try:
                                value = float(value_str)
                                
                                # Map to our key names using partial matching
                                for key, target in indicator_map.items():
                                    if key in name:  # Exact substring match
                                        technicals[target] = value
                                        break
                            except ValueError:
                                continue
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"   ⚠️ Could not parse indicator table: {e}")
        
        # If we got at least RSI or some indicators, consider it a success
        if technicals and len(technicals) > 3:  # Need at least a few indicators
            if 'rsi_14' in technicals:
                print(f"   ✅ TradingView scraped successfully for {symbol} - RSI: {technicals['rsi_14']:.2f}, Total indicators: {len(technicals)}")
            else:
                print(f"   ✅ TradingView scraped {len(technicals)} indicators for {symbol} (no RSI)")
            save_to_cache(symbol, technicals)
            return technicals
        else:
            print(f"   ⚠️ TradingView scrape failed/incomplete for {symbol} (got {len(technicals)} indicators)")
            return None
            
    except Exception as e:
        print(f"   ❌ TradingView scrape failed for {symbol}: {str(e)[:100]}")
        return None


def save_to_cache(symbol: str, data: Dict):
    """Save data to cache"""
    cache_file = CACHE_DIR / f"{symbol.upper()}_technicals.json"
    try:
        cache_data = {
            'symbol': symbol,
            'cached_at': datetime.now().isoformat(),
            'data': data
        }
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        print(f"   ⚠️ Cache save failed: {e}")


def load_from_cache(symbol: str, max_age_minutes: int = 15) -> Optional[Dict]:
    """Load data from cache if not too old"""
    cache_file = CACHE_DIR / f"{symbol.upper()}_technicals.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cached_at = datetime.fromisoformat(cache_data.get('cached_at', '2000-01-01'))
            age = datetime.now() - cached_at
            
            if age < timedelta(minutes=max_age_minutes):
                return cache_data.get('data')
        except Exception as e:
            print(f"   ⚠️ Cache load failed: {e}")
    
    return None


def get_tradingview_indicators(symbol: str, fallback_local: Optional[Dict] = None) -> Dict:
    """
    Main function: Get TradingView technical indicators with fallback.
    
    Args:
        symbol: Stock symbol (e.g., 'UBL')
        fallback_local: Local calculated indicators to use if scrape fails
    
    Returns:
        Dict with technical indicators and metadata
    """
    symbol = symbol.upper()
    
    # Try to scrape
    scraped = scrape_tradingview_technicals(symbol)
    
    if scraped:
        return {
            'symbol': symbol,
            'source': 'tradingview',
            'indicators': scraped,
            'fetched_at': datetime.now().isoformat()
        }
    
    # Fallback to local
    if fallback_local:
        print(f"   ℹ️ Using local indicators for {symbol} (TradingView unavailable)")
        return {
            'symbol': symbol,
            'source': 'local_fallback',
            'indicators': fallback_local,
            'fetched_at': datetime.now().isoformat()
        }
    
    print(f"   ⚠️ No indicators available for {symbol}")
    return {
        'symbol': symbol,
        'source': 'none',
        'indicators': {},
        'fetched_at': datetime.now().isoformat()
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing TradingView Scraper")
    print("=" * 60)
    
    # Test UBL (should show RSI ~77, not 99)
    result = get_tradingview_indicators('UBL')
    
    print(f"\nSource: {result['source']}")
    print(f"Indicators:")
    for key, value in result['indicators'].items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test caching (should hit cache within 15 min)
    print("\n" + "=" * 60)
    print("Testing cache (should be instant)...")
    result2 = get_tradingview_indicators('UBL')
    print(f"Source: {result2['source']}")
    
    close_driver()
    print("\n✅ Done!")
