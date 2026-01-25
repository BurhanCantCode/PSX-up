#!/usr/bin/env python3
"""
📊 EXTERNAL FEATURES MODULE
Research-backed external data for PSX prediction model.

Per peer-reviewed research (2020-2025):
- USD/PKR is the MOST CRITICAL external predictor for PSX
- KSE-100 beta explains most stock movement
- Oil prices affect energy sector (OGDC, PPL, PSO)
- Gold correlates with PKR weakness

Data Sources:
- USD/PKR: Yahoo Finance (PKR=X) - 5+ years available
- KSE-100: PSX DPS API (dps.psx.com.pk)
- Oil/Gold: Yahoo Finance (CL=F, GC=F)
- KIBOR: Hardcoded (SBP updates manually)
"""

import numpy as np
import pandas as pd
import subprocess
import re
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# TradingView scraper
try:
    from backend.tradingview_scraper import get_tradingview_indicators
    TRADINGVIEW_AVAILABLE = True
except ImportError:
    TRADINGVIEW_AVAILABLE = False

# Try yfinance import
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("⚠️ yfinance not installed. Run: pip install yfinance")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Cache settings
CACHE_DIR = Path(__file__).parent.parent / "data" / "external_cache"

# Current KIBOR/SBP Policy Rate (update periodically from SBP website)
# Source: https://www.sbp.org.pk/ecodata/kibor_index.asp
KIBOR_RATE = 0.13  # 13% as of Dec 2024 (policy rate cut from 15%)


# ============================================================================
# USD/PKR EXCHANGE RATE
# ============================================================================

def fetch_usd_pkr(start_date: str = None, end_date: str = None, 
                  period: str = "5y") -> pd.DataFrame:
    """
    Fetch USD/PKR exchange rate from Yahoo Finance.
    
    Research: USD/PKR is the #1 external predictor for PSX.
    Shows ~0.1 correlation with KSE-100, but high predictive power
    for emerging market stocks.
    
    Args:
        start_date: Start date (YYYY-MM-DD) or None for period-based
        end_date: End date (YYYY-MM-DD) or None for period-based  
        period: Period string if dates not specified (1mo, 3mo, 1y, 5y)
    
    Returns:
        DataFrame with columns: date, usdpkr_close, usdpkr_change, 
        usdpkr_volatility, usdpkr_trend
    """
    if not YFINANCE_AVAILABLE:
        print("⚠️ yfinance not available, returning empty DataFrame")
        return pd.DataFrame()
    
    try:
        if start_date and end_date:
            data = yf.download('PKR=X', start=start_date, end=end_date, progress=False)
        else:
            data = yf.download('PKR=X', period=period, progress=False)
        
        if data.empty:
            print("⚠️ No USD/PKR data returned")
            return pd.DataFrame()
        
        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        df = pd.DataFrame({
            'date': data.index,
            'usdpkr_close': data['Close'].values,
            'usdpkr_change': data['Close'].pct_change().values,
            'usdpkr_volatility': data['Close'].pct_change().rolling(20).std().values,
            # PKR weakening trend (bad for stocks)
            'usdpkr_trend': (data['Close'] / data['Close'].shift(20) - 1).values,
            # Is PKR strengthening? (good signal)
            'usdpkr_strengthening': (data['Close'] < data['Close'].shift(5)).astype(int).values
        })
        
        df = df.reset_index(drop=True)
        print(f"✅ Fetched {len(df)} USD/PKR data points")
        return df
        
    except Exception as e:
        print(f"❌ Error fetching USD/PKR: {e}")
        return pd.DataFrame()


# ============================================================================
# KSE-100 INDEX (From PSX DPS API)
# ============================================================================

def fetch_kse100_month(month: int, year: int) -> List[Dict]:
    """Fetch KSE-100 data for a specific month from PSX."""
    url = "https://dps.psx.com.pk/historical"
    post_data = f"month={month}&year={year}&symbol=KSE100"
    
    try:
        result = subprocess.run(
            ['curl', '-s', '-X', 'POST', url, '-d', post_data],
            capture_output=True, text=True, timeout=15
        )
        
        if result.returncode != 0:
            return []
        
        # Parse HTML table
        rows = re.findall(r'<tr>.*?</tr>', result.stdout, re.DOTALL)
        data = []
        
        for row in rows:
            cells = re.findall(r'<td[^>]*>([^<]+)</td>', row)
            if len(cells) >= 6:
                try:
                    date_str = cells[0].strip()
                    date_obj = datetime.strptime(date_str, "%b %d, %Y")
                    
                    data.append({
                        'date': date_obj.strftime('%Y-%m-%d'),
                        'kse100_open': float(cells[1].replace(',', '')),
                        'kse100_high': float(cells[2].replace(',', '')),
                        'kse100_low': float(cells[3].replace(',', '')),
                        'kse100_close': float(cells[4].replace(',', '')),
                        'kse100_volume': float(cells[5].replace(',', ''))
                    })
                except:
                    continue
        
        return data
    except Exception as e:
        print(f"⚠️ Error fetching KSE100 {month}/{year}: {e}")
        return []


def fetch_kse100(start_year: int = 2020, end_date: str = None) -> pd.DataFrame:
    """
    Fetch KSE-100 index data from PSX DPS API.
    
    Research: Market beta (vs KSE-100) explains most stock movement.
    This is the #1 signal for individual stock prediction.
    
    Args:
        start_year: Year to start fetching from
        end_date: End date (defaults to today)
    
    Returns:
        DataFrame with KSE-100 OHLCV data
    """
    current_date = datetime.now()
    end_year = current_date.year
    end_month = current_date.month
    
    if end_date:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        end_year = end_dt.year
        end_month = end_dt.month
    
    all_data = []
    
    for year in range(start_year, end_year + 1):
        start_m = 1 if year > start_year else 1
        end_m = end_month if year == end_year else 12
        
        for month in range(start_m, end_m + 1):
            month_data = fetch_kse100_month(month, year)
            all_data.extend(month_data)
    
    if not all_data:
        print("⚠️ No KSE-100 data fetched")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Add derived features
    df['kse100_return'] = df['kse100_close'].pct_change()
    df['kse100_volatility'] = df['kse100_return'].rolling(20).std()
    df['kse100_trend'] = df['kse100_close'] / df['kse100_close'].shift(20) - 1
    df['kse100_above_sma50'] = (df['kse100_close'] > df['kse100_close'].rolling(50).mean()).astype(int)
    df['kse100_above_sma200'] = (df['kse100_close'] > df['kse100_close'].rolling(200).mean()).astype(int)
    
    print(f"✅ Fetched {len(df)} KSE-100 data points ({df['date'].min().date()} to {df['date'].max().date()})")
    return df


# ============================================================================
# OIL & COMMODITIES
# ============================================================================

def fetch_commodities(start_date: str = None, end_date: str = None,
                      period: str = "5y") -> pd.DataFrame:
    """
    Fetch oil and gold prices from Yahoo Finance.
    
    Research: Oil prices affect energy sector (OGDC, PPL, PSO).
    Gold often correlates with PKR weakness (flight to safety).
    
    Returns:
        DataFrame with oil and gold prices
    """
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame()
    
    try:
        # Fetch both commodities
        if start_date and end_date:
            oil = yf.download('CL=F', start=start_date, end=end_date, progress=False)
            gold = yf.download('GC=F', start=start_date, end=end_date, progress=False)
        else:
            oil = yf.download('CL=F', period=period, progress=False)
            gold = yf.download('GC=F', period=period, progress=False)
        
        # Handle multi-level columns
        for df in [oil, gold]:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
        
        # Merge on date
        result = pd.DataFrame({'date': oil.index})
        
        if not oil.empty:
            result['oil_close'] = oil['Close'].values
            result['oil_change'] = oil['Close'].pct_change().values
            result['oil_trend'] = (oil['Close'] / oil['Close'].shift(20) - 1).values
        
        if not gold.empty:
            # Align gold to oil dates
            gold_aligned = gold.reindex(oil.index, method='ffill')
            result['gold_close'] = gold_aligned['Close'].values
            result['gold_change'] = gold_aligned['Close'].pct_change().values
            result['gold_trend'] = (gold_aligned['Close'] / gold_aligned['Close'].shift(20) - 1).values
        
        result = result.reset_index(drop=True)
        print(f"✅ Fetched {len(result)} commodity data points")
        return result
        
    except Exception as e:
        print(f"❌ Error fetching commodities: {e}")
        return pd.DataFrame()


# ============================================================================
# STOCK BETA CALCULATION
# ============================================================================

def calculate_stock_beta(stock_returns: np.ndarray, 
                         market_returns: np.ndarray, 
                         window: int = 63) -> np.ndarray:
    """
    Calculate rolling beta of stock vs KSE-100.
    
    Research: Market beta explains most stock movement.
    Beta > 1: More volatile than market
    Beta < 1: Less volatile than market
    
    Args:
        stock_returns: Array of stock daily returns
        market_returns: Array of KSE-100 daily returns
        window: Rolling window (63 days = ~3 months)
    
    Returns:
        Array of rolling beta values
    """
    betas = np.full(len(stock_returns), np.nan)
    
    for t in range(window, len(stock_returns)):
        stock_window = stock_returns[t-window:t]
        market_window = market_returns[t-window:t]
        
        # Remove NaN
        mask = ~(np.isnan(stock_window) | np.isnan(market_window))
        if mask.sum() < window // 2:
            continue
        
        # Beta = Cov(stock, market) / Var(market)
        cov = np.cov(stock_window[mask], market_window[mask])[0, 1]
        var = np.var(market_window[mask])
        betas[t] = cov / var if var > 1e-8 else 1.0
    
    return betas


def calculate_correlation(series1: np.ndarray, series2: np.ndarray, 
                          window: int = 63) -> np.ndarray:
    """
    Calculate rolling correlation between two series.
    Useful for USD/PKR vs stock correlation.
    """
    corrs = np.full(len(series1), np.nan)
    
    for t in range(window, len(series1)):
        s1 = series1[t-window:t]
        s2 = series2[t-window:t]
        
        mask = ~(np.isnan(s1) | np.isnan(s2))
        if mask.sum() < window // 2:
            continue
        
        corrs[t] = np.corrcoef(s1[mask], s2[mask])[0, 1]
    
    return corrs


# ============================================================================
# KIBOR / INTEREST RATE
# ============================================================================

def get_kibor_rate() -> float:
    """
    Get current KIBOR/SBP policy rate.
    
    Note: SBP only provides PDFs, not an API.
    This is manually updated from: https://www.sbp.org.pk/ecodata/kibor_index.asp
    
    Returns:
        Current KIBOR rate as decimal (e.g., 0.13 for 13%)
    """
    return KIBOR_RATE


def get_kibor_features(length: int) -> pd.DataFrame:
    """
    Generate KIBOR-based features.
    Since rate changes infrequently, we provide:
    - Current rate (static)
    - High rate indicator (affects bank stocks)
    """
    rate = get_kibor_rate()
    
    return pd.DataFrame({
        'kibor_rate': [rate] * length,
        'kibor_high': [1 if rate > 0.10 else 0] * length,  # High if > 10%
        'kibor_very_high': [1 if rate > 0.15 else 0] * length  # Very high if > 15%
    })


# ============================================================================
# MERGE EXTERNAL FEATURES WITH STOCK DATA
# ============================================================================

def merge_external_features(stock_df: pd.DataFrame, 
                            symbol: str = None,
                            cache: bool = True) -> pd.DataFrame:
    """
    Merge all external features with stock DataFrame.
    
    This is the main API for the SOTA model.
    
    Args:
        stock_df: DataFrame with stock data (must have 'Date' column)
        symbol: Stock symbol (for sector-specific features)
        cache: Whether to cache external data
    
    Returns:
        DataFrame with both stock and external features
    """
    if 'Date' not in stock_df.columns:
        print("⚠️ stock_df must have 'Date' column")
        return stock_df
    
    df = stock_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get date range
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')
    start_year = df['Date'].min().year
    
    print(f"\n📊 MERGING EXTERNAL FEATURES")
    print(f"   Date range: {start_date} to {end_date}")
    print("=" * 50)
    
    # 1. USD/PKR
    print("\n1. Fetching USD/PKR...")
    usdpkr = fetch_usd_pkr(start_date=start_date, end_date=end_date)
    if not usdpkr.empty:
        usdpkr['date'] = pd.to_datetime(usdpkr['date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            usdpkr.sort_values('date'),
            left_on='Date',
            right_on='date',
            direction='backward'
        )
        df = df.drop(columns=['date'], errors='ignore')
        print(f"   ✅ Added {len([c for c in df.columns if 'usdpkr' in c])} USD/PKR features")
    
    # 2. KSE-100 features
    if symbol and symbol.upper() != 'KSE100':
        # For regular stocks: fetch KSE100 as external feature
        print("\n2. Fetching KSE-100...")
        kse100 = fetch_kse100(start_year=start_year, end_date=end_date)
        if not kse100.empty:
            df = pd.merge_asof(
                df.sort_values('Date'),
                kse100.sort_values('date'),
                left_on='Date',
                right_on='date',
                direction='backward'
            )
            df = df.drop(columns=['date'], errors='ignore')
            
            # Calculate beta vs KSE-100
            if 'Close' in df.columns and 'kse100_return' in df.columns:
                stock_returns = df['Close'].pct_change().fillna(0).values
                df['stock_beta'] = calculate_stock_beta(stock_returns, df['kse100_return'].fillna(0).values)
                
                # Correlation with KSE-100
                df['kse100_correlation'] = calculate_correlation(
                    df['Close'].pct_change().fillna(0).values,
                    df['kse100_return'].fillna(0).values
                )
            
            print(f"   ✅ Added {len([c for c in df.columns if 'kse100' in c.lower() or 'beta' in c.lower()])} KSE-100 features")
    else:
        # For KSE100 itself: use its own data as features (real features, not zeros!)
        print("\n2. Adding KSE-100 features from own data...")
        
        # Use KSE100's own OHLCV data as features
        df['kse100_open'] = df['Open'].values
        df['kse100_high'] = df['High'].values
        df['kse100_low'] = df['Low'].values
        df['kse100_close'] = df['Close'].values
        df['kse100_volume'] = df['Volume'].values
        
        # Calculate returns and momentum from KSE100's own data
        df['kse100_return'] = df['Close'].pct_change().fillna(0)
        df['kse100_volatility'] = df['kse100_return'].rolling(20).std().fillna(0)
        df['kse100_trend'] = (df['Close'] / df['Close'].shift(20) - 1).fillna(0)
        
        # Moving average signals
        sma50 = df['Close'].rolling(50).mean()
        sma200 = df['Close'].rolling(200).mean()
        df['kse100_above_sma50'] = (df['Close'] > sma50).astype(int).fillna(0)
        df['kse100_above_sma200'] = (df['Close'] > sma200).astype(int).fillna(0)
        
        # Beta and correlation with itself (perfect correlation)
        df['stock_beta'] = 1.0  # KSE100 has beta of 1.0 with itself
        df['kse100_correlation'] = 1.0  # Perfect correlation with itself
        
        # Additional momentum features from KSE100's own data
        df['kse100_momentum_5'] = (df['Close'] / df['Close'].shift(5) - 1).fillna(0)
        df['kse100_momentum_10'] = (df['Close'] / df['Close'].shift(10) - 1).fillna(0)
        df['kse100_momentum_20'] = (df['Close'] / df['Close'].shift(20) - 1).fillna(0)
        
        print(f"   ✅ Added {len([c for c in df.columns if 'kse100' in c.lower() or 'beta' in c.lower()])} real KSE-100 features from own data")
    
    # 3. Commodities (Oil, Gold)
    print("\n3. Fetching Commodities...")
    commodities = fetch_commodities(start_date=start_date, end_date=end_date)
    if not commodities.empty:
        commodities['date'] = pd.to_datetime(commodities['date'])
        df = pd.merge_asof(
            df.sort_values('Date'),
            commodities.sort_values('date'),
            left_on='Date',
            right_on='date',
            direction='backward'
        )
        df = df.drop(columns=['date'], errors='ignore')
        
        # Sector-specific: Energy stocks correlate with oil
        energy_symbols = ['OGDC', 'PPL', 'PSO', 'POL', 'MARI', 'ATRL']
        if symbol and symbol.upper() in energy_symbols:
            if 'oil_change' in df.columns and 'Close' in df.columns:
                df['oil_correlation'] = calculate_correlation(
                    df['Close'].pct_change().values,
                    df['oil_change'].values
                )
            print(f"   ✅ Added oil correlation for energy stock {symbol}")
        
        print(f"   ✅ Added {len([c for c in df.columns if 'oil' in c.lower() or 'gold' in c.lower()])} commodity features")
    
    # 4. KIBOR
    print("\n4. Adding KIBOR features...")
    kibor_df = get_kibor_features(len(df))
    for col in kibor_df.columns:
        df[col] = kibor_df[col].values
    print(f"   ✅ Added {len(kibor_df.columns)} KIBOR features")
    
    # 5. TradingView Technical Indicators (ALWAYS fetch fresh - no old cache)
    if TRADINGVIEW_AVAILABLE and symbol:
        print(f"\n5. Fetching TradingView indicators for {symbol}...")
        
        # Delete old cache to force fresh fetch
        from pathlib import Path
        cache_file = Path(__file__).parent.parent / "data" / "tradingview_cache" / f"{symbol.upper()}_technicals.json"
        if cache_file.exists():
            try:
                cache_file.unlink()
                print(f"   🔄 Cleared old TradingView cache for fresh fetch")
            except:
                pass
        
        # Fetch from TradingView (fresh data)
        tv_result = get_tradingview_indicators(symbol, fallback_local=None)
        tv_indicators = tv_result.get('indicators', {})
        
        if tv_indicators and tv_result['source'] == 'tradingview':
            # Add TradingView values as ADDITIONAL features
            # These are CURRENT real-time values, so broadcast to all rows (model uses lag features)
            tv_count = 0
            
            # KEY OSCILLATORS (Most Important)
            if 'rsi_14' in tv_indicators:
                df['tv_rsi_14'] = tv_indicators['rsi_14']
                print(f"   📊 TradingView RSI: {tv_indicators['rsi_14']:.2f}")
                tv_count += 1
            
            if 'macd_level' in tv_indicators:
                df['tv_macd'] = tv_indicators['macd_level']
                tv_count += 1
            
            if 'stochastic_k' in tv_indicators:
                df['tv_stochastic_k'] = tv_indicators['stochastic_k']
                tv_count += 1
            
            if 'adx' in tv_indicators:
                df['tv_adx'] = tv_indicators['adx']
                tv_count += 1
            
            if 'momentum_10' in tv_indicators:
                df['tv_momentum'] = tv_indicators['momentum_10']
                tv_count += 1
            
            if 'williams_r' in tv_indicators:
                df['tv_williams_r'] = tv_indicators['williams_r']
                tv_count += 1
            
            if 'cci' in tv_indicators:
                df['tv_cci'] = tv_indicators['cci']
                tv_count += 1
            
            if 'awesome_oscillator' in tv_indicators:
                df['tv_awesome'] = tv_indicators['awesome_oscillator']
                tv_count += 1
            
            if 'bull_bear_power' in tv_indicators:
                df['tv_bull_bear'] = tv_indicators['bull_bear_power']
                tv_count += 1
            
            # MOVING AVERAGES (Important for trend)
            if 'ema_10' in tv_indicators:
                df['tv_ema_10'] = tv_indicators['ema_10']
                tv_count += 1
            
            if 'sma_10' in tv_indicators:
                df['tv_sma_10'] = tv_indicators['sma_10']
                tv_count += 1
            
            if 'ema_20' in tv_indicators:
                df['tv_ema_20'] = tv_indicators['ema_20']
                tv_count += 1
            
            if 'sma_20' in tv_indicators:
                df['tv_sma_20'] = tv_indicators['sma_20']
                tv_count += 1
            
            if 'ema_50' in tv_indicators:
                df['tv_ema_50'] = tv_indicators['ema_50']
                tv_count += 1
            
            if 'sma_50' in tv_indicators:
                df['tv_sma_50'] = tv_indicators['sma_50']
                tv_count += 1
            
            if 'ema_100' in tv_indicators:
                df['tv_ema_100'] = tv_indicators['ema_100']
                tv_count += 1
            
            if 'sma_100' in tv_indicators:
                df['tv_sma_100'] = tv_indicators['sma_100']
                tv_count += 1
            
            # COMPOSITE FEATURES
            # Price vs EMAs (important trend signals)
            current_price = df['Close'].iloc[-1]
            
            if 'ema_20' in tv_indicators and tv_indicators['ema_20'] > 0:
                df['tv_price_vs_ema20'] = (current_price / tv_indicators['ema_20'] - 1) * 100
                tv_count += 1
            
            if 'ema_50' in tv_indicators and tv_indicators['ema_50'] > 0:
                df['tv_price_vs_ema50'] = (current_price / tv_indicators['ema_50'] - 1) * 100
                tv_count += 1
            
            # Recommendation counts
            if 'recommendation_buy' in tv_indicators:
                df['tv_rec_buy'] = tv_indicators['recommendation_buy']
                df['tv_rec_sell'] = tv_indicators['recommendation_sell']
                df['tv_rec_neutral'] = tv_indicators['recommendation_neutral']
                
                # Calculate recommendation score (-1 to +1)
                total = (tv_indicators['recommendation_buy'] + 
                        tv_indicators['recommendation_sell'] + 
                        tv_indicators['recommendation_neutral'])
                if total > 0:
                    df['tv_recommendation_score'] = (
                        (tv_indicators['recommendation_buy'] - tv_indicators['recommendation_sell']) / total
                    )
                tv_count += 4
            
            print(f"   ✅ Added {tv_count} TradingView indicators/features (source: TradingView)")
        elif tv_indicators and tv_result['source'] == 'local_fallback':
            print(f"   ℹ️ Using local indicators (TradingView unavailable)")
        else:
            print(f"   ⚠️ TradingView unavailable, using local indicators")
    
    # 6. USD/PKR correlation with stock
    if 'usdpkr_change' in df.columns and 'Close' in df.columns:
        df['usdpkr_correlation'] = calculate_correlation(
            df['Close'].pct_change().values,
            df['usdpkr_change'].values
        )
    
    # Sort back to original order
    df = df.sort_values('Date').reset_index(drop=True)
    
    print("\n" + "=" * 50)
    print(f"✅ EXTERNAL FEATURES COMPLETE")
    print(f"   Total features added: {len([c for c in df.columns if c not in stock_df.columns])}")
    print(f"   Final DataFrame: {len(df)} rows x {len(df.columns)} columns")
    
    return df


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("📊 EXTERNAL FEATURES MODULE - TEST")
    print("=" * 70)
    
    # Test USD/PKR
    print("\n1. Testing USD/PKR fetch...")
    usdpkr = fetch_usd_pkr(period="1mo")
    if not usdpkr.empty:
        print(f"   Latest: {usdpkr['usdpkr_close'].iloc[-1]:.2f} PKR/USD")
    
    # Test KSE-100
    print("\n2. Testing KSE-100 fetch...")
    kse100 = fetch_kse100(start_year=2024)
    if not kse100.empty:
        print(f"   Latest: {kse100['kse100_close'].iloc[-1]:,.2f}")
    
    # Test Commodities
    print("\n3. Testing commodities fetch...")
    commodities = fetch_commodities(period="1mo")
    if not commodities.empty:
        print(f"   Oil: ${commodities['oil_close'].iloc[-1]:.2f}")
        print(f"   Gold: ${commodities['gold_close'].iloc[-1]:.2f}")
    
    # Test merge with dummy stock data
    print("\n4. Testing merge with stock data...")
    dummy_stock = pd.DataFrame({
        'Date': pd.date_range('2024-01-01', periods=100, freq='D'),
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(100000, 1000000, 100)
    })
    
    merged = merge_external_features(dummy_stock, symbol='OGDC')
    print(f"   New columns: {[c for c in merged.columns if c not in dummy_stock.columns]}")
    
    print("\n✅ All tests complete!")
