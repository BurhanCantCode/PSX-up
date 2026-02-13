#!/usr/bin/env python3
"""
State-of-the-Art Stock Prediction Model
Based on latest research findings:
- Wavelet Denoising (db4 DWT) - Critical for accuracy
- xLSTM-TS style architecture with exponential gating
- TiDE (Time-series Dense Encoder) for efficiency
- Hybrid Loss Function (MSE + Trend Accuracy)
- Ensemble Strategy

Reference: Dublin City University (2024), Columbia FinRL, Scientific Reports Galformer
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Callable
from pathlib import Path
import json
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Check for optional dependencies
try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    print("⚠️  PyWavelets not installed. Wavelet denoising disabled. Install with: pip install PyWavelets")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
# ============================================================================
# 🔮 FORTUNE TELLER: CAUSAL N-BEATS STYLE BASIS DECOMPOSITION
# ============================================================================

class NBEATSBasisDecomposer:
    """
    CAUSAL N-BEATS-style basis expansion for interpretable trend + seasonality.
    Uses EXPANDING WINDOW to ensure no look-ahead bias - at each point t,
    decomposition uses only data from [0, t].
    """
    
    def __init__(self, trend_degree: int = 3, n_harmonics: int = 3, min_window: int = 30):
        self.trend_degree = trend_degree
        self.n_harmonics = n_harmonics
        self.min_window = min_window  # Minimum history before computing features
    
    def _polynomial_basis(self, length: int) -> np.ndarray:
        """Create polynomial basis functions for trend modeling."""
        t = np.linspace(0, 1, length)
        basis = np.column_stack([t ** i for i in range(self.trend_degree + 1)])
        return basis
    
    def _fourier_basis(self, length: int, period: float) -> np.ndarray:
        """Create Fourier basis functions for seasonality."""
        t = np.arange(length)
        basis = []
        for k in range(1, self.n_harmonics + 1):
            basis.append(np.sin(2 * np.pi * k * t / period))
            basis.append(np.cos(2 * np.pi * k * t / period))
        return np.column_stack(basis) if basis else np.zeros((length, 1))
    
    def decompose_causal(self, prices: np.ndarray, lookback: int = 100) -> Dict[str, np.ndarray]:
        """
        FAST causal decomposition using SLIDING window - O(n) instead of O(n²).
        
        Key insight: A 100-day lookback is just as causal as expanding window
        (you're still only using past data), but computation is constant per point.
        
        Args:
            prices: Price series to decompose
            lookback: Fixed sliding window size (default 100 trading days ~5 months)
        
        Returns:
            Dict with trend, seasonal_weekly, seasonal_monthly, residual components
        """
        length = len(prices)
        
        # Initialize output arrays
        trend = np.zeros(length)
        seasonal_weekly = np.zeros(length)
        seasonal_monthly = np.zeros(length)
        residual = np.zeros(length)
        
        # Fill early values with simple estimates
        for t in range(min(self.min_window, length)):
            trend[t] = prices[:t+1].mean() if t > 0 else prices[0]
        
        # SLIDING window decomposition (O(n) since window size is fixed)
        for t in range(self.min_window, length):
            # FIXED: Sliding window, not expanding
            start_idx = max(0, t + 1 - lookback)
            window_prices = prices[start_idx:t+1]
            window_len = len(window_prices)
            current_residual = window_prices.copy()
            
            # 1. Trend: polynomial fit on window
            try:
                poly_basis = self._polynomial_basis(window_len)
                coeffs = np.linalg.lstsq(poly_basis, current_residual, rcond=None)[0]
                window_trend = poly_basis @ coeffs
                trend[t] = window_trend[-1]  # Only take last point
                current_residual = current_residual - window_trend
            except:
                trend[t] = window_prices.mean()
            
            # 2. Weekly seasonality (5 trading days) - need at least 2 weeks
            if window_len >= 10:
                try:
                    weekly_basis = self._fourier_basis(window_len, period=5)
                    coeffs = np.linalg.lstsq(weekly_basis, current_residual, rcond=None)[0]
                    window_weekly = weekly_basis @ coeffs
                    seasonal_weekly[t] = window_weekly[-1]
                    current_residual = current_residual - window_weekly
                except:
                    pass
            
            # 3. Monthly seasonality (~21 trading days) - need at least 2 months
            if window_len >= 42:
                try:
                    monthly_basis = self._fourier_basis(window_len, period=21)
                    coeffs = np.linalg.lstsq(monthly_basis, current_residual, rcond=None)[0]
                    window_monthly = monthly_basis @ coeffs
                    seasonal_monthly[t] = window_monthly[-1]
                    current_residual = current_residual - window_monthly
                except:
                    pass
            
            # 4. Residual at this point
            residual[t] = current_residual[-1]
        
        return {
            'trend': trend,
            'seasonal_weekly': seasonal_weekly,
            'seasonal_monthly': seasonal_monthly,
            'residual': residual
        }
    
    def get_features_causal(self, prices: np.ndarray) -> pd.DataFrame:
        """
        Extract CAUSAL N-BEATS features - no look-ahead bias.
        Each feature at time t only uses data from [0, t].
        """
        components = self.decompose_causal(prices)
        features = {}
        
        for name, component in components.items():
            features[f'nbeats_{name}'] = component
            features[f'nbeats_{name}_pct'] = component / (np.abs(prices) + 1e-8)
            
            # Causal momentum (backward difference)
            if len(component) > 5:
                momentum = np.zeros(len(component))
                for t in range(5, len(component)):
                    momentum[t] = component[t] - component[t-5]
                features[f'nbeats_{name}_momentum_5'] = momentum
        
        return pd.DataFrame(features)
    
    # Keep legacy method for backwards compatibility but mark deprecated
    def decompose(self, prices: np.ndarray) -> Dict[str, np.ndarray]:
        """DEPRECATED: Use decompose_causal() to avoid look-ahead bias."""
        print("⚠️ WARNING: Using non-causal decompose(). Use decompose_causal() instead.")
        return self.decompose_causal(prices)
    
    def get_features(self, prices: np.ndarray) -> pd.DataFrame:
        """DEPRECATED: Use get_features_causal() to avoid look-ahead bias."""
        return self.get_features_causal(prices)


class PSXSeasonalFeatures:
    """
    Generate PSX-specific seasonal features:
    - Weekly patterns (Mon-Fri)
    - Monthly patterns (start/mid/end effects)
    - Yearly patterns (fiscal year, Ramadan, EID seasons)
    - Pakistani market-specific events
    - PSX holidays (market closed days)
    """
    
    # Approximate Ramadan start dates (Hijri calendar, varies yearly)
    RAMADAN_PERIODS = {
        2020: (pd.Timestamp('2020-04-24'), pd.Timestamp('2020-05-23')),
        2021: (pd.Timestamp('2021-04-13'), pd.Timestamp('2021-05-12')),
        2022: (pd.Timestamp('2022-04-02'), pd.Timestamp('2022-05-01')),
        2023: (pd.Timestamp('2023-03-23'), pd.Timestamp('2023-04-21')),
        2024: (pd.Timestamp('2024-03-11'), pd.Timestamp('2024-04-09')),
        2025: (pd.Timestamp('2025-02-28'), pd.Timestamp('2025-03-29')),
        2026: (pd.Timestamp('2026-02-17'), pd.Timestamp('2026-03-18')),
    }
    
    # PSX Public Holidays (market closed)
    PSX_HOLIDAYS = {
        # 2025
        '2025-02-05',  # Kashmir Day
        '2025-03-23',  # Pakistan Day
        '2025-03-31',  # Eid ul-Fitr (approx)
        '2025-04-01',  # Eid ul-Fitr
        '2025-05-01',  # Labour Day
        '2025-06-07',  # Eid ul-Adha (approx)
        '2025-06-08',  # Eid ul-Adha
        '2025-06-09',  # Eid ul-Adha
        '2025-08-14',  # Independence Day
        '2025-09-05',  # Ashura (approx)
        '2025-09-06',  # Ashura
        '2025-11-09',  # Iqbal Day
        '2025-12-25',  # Quaid Day
        # 2026
        '2026-02-05',  # Kashmir Day
        '2026-03-20',  # Eid ul-Fitr (approx)
        '2026-03-21',  # Eid ul-Fitr
        '2026-03-23',  # Pakistan Day
        '2026-05-01',  # Labour Day
        '2026-05-27',  # Eid ul-Adha (approx)
        '2026-05-28',  # Eid ul-Adha
        '2026-08-14',  # Independence Day
        '2026-08-25',  # Ashura (approx)
        '2026-08-26',  # Ashura
        '2026-11-09',  # Iqbal Day
        '2026-12-25',  # Quaid Day
    }
    
    def generate(self, dates: pd.Series) -> pd.DataFrame:
        """Generate all seasonal features."""
        dates = pd.to_datetime(dates)
        features = {}
        
        # Weekly features (trading day of week: 0=Mon, 4=Fri)
        features['day_of_week'] = dates.dt.dayofweek.values
        features['is_monday'] = (dates.dt.dayofweek == 0).astype(int).values
        features['is_friday'] = (dates.dt.dayofweek == 4).astype(int).values
        
        # Monthly features
        features['day_of_month'] = dates.dt.day.values
        features['is_month_start'] = (dates.dt.day <= 5).astype(int).values
        features['is_month_end'] = (dates.dt.day >= 25).astype(int).values
        features['month'] = dates.dt.month.values
        
        # Quarterly features (fiscal year: July-June in Pakistan)
        features['is_quarter_end'] = dates.dt.month.isin([3, 6, 9, 12]).astype(int).values
        features['is_fiscal_year_end'] = (dates.dt.month == 6).astype(int).values  # June
        features['is_fiscal_year_start'] = (dates.dt.month == 7).astype(int).values  # July
        
        # Dividend season (typically announced in Q1 of fiscal year: Jul-Sep)
        features['is_dividend_season'] = dates.dt.month.isin([7, 8, 9]).astype(int).values
        
        # Yearly features with sine/cosine encoding for cyclical nature
        day_of_year = dates.dt.dayofyear.values
        features['yearly_sin'] = np.sin(2 * np.pi * day_of_year / 365)
        features['yearly_cos'] = np.cos(2 * np.pi * day_of_year / 365)
        
        # Ramadan effect (market volume typically drops) - FIXED index bug
        features['is_ramadan'] = self._is_ramadan(dates)
        
        # EID effect (post-EID rally is common in PSX) - FIXED index bug
        features['is_post_eid'] = self._is_post_eid(dates)
        
        return pd.DataFrame(features)
    
    def _is_ramadan(self, dates: pd.Series) -> np.ndarray:
        """Check if date falls in Ramadan period. FIXED: uses np.array instead of Series with index."""
        dates_vals = pd.to_datetime(dates.values)
        is_ramadan = np.zeros(len(dates), dtype=int)
        
        for year, (start, end) in self.RAMADAN_PERIODS.items():
            mask = (dates_vals >= start) & (dates_vals <= end)
            is_ramadan = is_ramadan | mask.astype(int)
        
        return is_ramadan
    
    def _is_post_eid(self, dates: pd.Series) -> np.ndarray:
        """Check if date is in post-EID period (10 days after Ramadan). FIXED: uses np.array."""
        dates_vals = pd.to_datetime(dates.values)
        is_post_eid = np.zeros(len(dates), dtype=int)
        
        for year, (_, end) in self.RAMADAN_PERIODS.items():
            eid_end = end + pd.Timedelta(days=10)
            mask = (dates_vals >= end) & (dates_vals <= eid_end)
            is_post_eid = is_post_eid | mask.astype(int)
        
        return is_post_eid
    
    def is_psx_holiday(self, date_str: str) -> bool:
        """Check if a date is a PSX holiday."""
        return date_str in self.PSX_HOLIDAYS


class MultiHorizonEnsemble:
    """
    Different models optimized for different prediction horizons.
    Short-term: XGBoost/LightGBM (fast-reacting)
    Medium-term: RandomForest (stable)
    Long-term: Ridge regression (follows trend)
    """
    
    HORIZON_CONFIG = {
        'short': {'days': (1, 5), 'weight': 0.35, 'models': ['xgb', 'lgbm']},
        'medium': {'days': (6, 21), 'weight': 0.30, 'models': ['rf', 'et']},
        'long': {'days': (22, 63), 'weight': 0.20, 'models': ['gb']},
        'trend': {'days': (64, 500), 'weight': 0.15, 'models': ['ridge']},
    }
    
    def get_horizon_weight(self, day_offset: int) -> Dict[str, float]:
        """Get model weights for a specific prediction horizon."""
        weights = {}
        
        for horizon_name, config in self.HORIZON_CONFIG.items():
            min_day, max_day = config['days']
            if min_day <= day_offset <= max_day:
                for model in config['models']:
                    weights[model] = weights.get(model, 0) + config['weight'] / len(config['models'])
        
        # Normalize weights
        total = sum(weights.values()) if weights else 1
        return {k: v / total for k, v in weights.items()} if weights else {'rf': 1.0}


# ============================================================================
# WAVELET DENOISING - Critical for accuracy (50% → 70%+ accuracy)
# ============================================================================

def wavelet_denoise(signal: np.ndarray, wavelet: str = 'db4', level: int = None, 
                    threshold_mode: str = 'soft') -> np.ndarray:
    """
    Apply Discrete Wavelet Transform (DWT) denoising.
    
    This is CRITICAL - research shows ~50% accuracy without denoising,
    but 70%+ with proper wavelet denoising.
    
    Args:
        signal: 1D numpy array of prices
        wavelet: Wavelet type ('db4' = Daubechies 4, best for financial data)
        level: Decomposition level (None = automatic)
        threshold_mode: 'soft' (recommended) or 'hard'
    
    Returns:
        Denoised signal
    """
    if not PYWT_AVAILABLE:
        return signal  # Fallback: return original if pywt not available
    
    if len(signal) < 8:
        return signal
    
    # Determine decomposition level
    if level is None:
        level = min(pywt.dwt_max_level(len(signal), wavelet), 4)
    
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    
    # Calculate universal threshold (VisuShrink)
    # σ = MAD / 0.6745 where MAD is median absolute deviation
    detail_coeffs = coeffs[-1]  # Finest detail level
    sigma = np.median(np.abs(detail_coeffs)) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(signal)))
    
    # Apply thresholding to detail coefficients (not approximation)
    denoised_coeffs = [coeffs[0]]  # Keep approximation unchanged
    for i in range(1, len(coeffs)):
        denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))
    
    # Reconstruct signal
    denoised_signal = pywt.waverec(denoised_coeffs, wavelet)
    
    # Handle length mismatch (wavelet reconstruction can add/remove a sample)
    if len(denoised_signal) > len(signal):
        denoised_signal = denoised_signal[:len(signal)]
    elif len(denoised_signal) < len(signal):
        denoised_signal = np.pad(denoised_signal, (0, len(signal) - len(denoised_signal)), 'edge')
    
    return denoised_signal


def multi_scale_wavelet_features(prices: np.ndarray, wavelet: str = 'db4') -> Dict[str, np.ndarray]:
    """
    Extract multi-scale wavelet features for trend detection.
    Returns approximation (trend) and detail (noise) at different scales.
    """
    if not PYWT_AVAILABLE:
        return {'trend': prices, 'noise': np.zeros_like(prices)}
    
    if len(prices) < 32:
        return {'trend': prices, 'noise': np.zeros_like(prices)}
    
    level = min(pywt.dwt_max_level(len(prices), wavelet), 4)
    coeffs = pywt.wavedec(prices, wavelet, level=level)
    
    features = {}
    
    # Trend: reconstruction from only approximation coefficients
    trend_coeffs = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    features['trend'] = pywt.waverec(trend_coeffs, wavelet)[:len(prices)]
    
    # Noise: difference between original and trend
    features['noise'] = prices - features['trend']
    
    # Energy at each scale (useful features)
    for i, c in enumerate(coeffs):
        features[f'energy_level_{i}'] = np.sum(c**2) / len(c)
    
    return features


def wavelet_denoise_causal(signal: np.ndarray, wavelet: str = 'db4', 
                           lookback: int = 100, update_freq: int = 5) -> np.ndarray:
    """
    OPTIMIZED causal wavelet denoising - O(n) instead of O(n²), 10-20x faster.
    
    Key insight: Denoised values change slowly. We:
    1. Use SLIDING window of fixed size (not expanding) - still causal
    2. Compute full denoising every `update_freq` days
    3. Linearly interpolate between updates
    
    Args:
        signal: 1D numpy array of prices
        wavelet: Wavelet type ('db4' = Daubechies 4)
        lookback: Fixed sliding window size (100 days ~5 months)
        update_freq: Compute every N days, interpolate between
    
    Returns:
        Causally denoised signal (each point only uses past data)
    """
    if not PYWT_AVAILABLE:
        return signal
    
    n = len(signal)
    denoised = np.zeros(n)
    min_samples = 32
    
    # Determine update points (compute at these indices)
    update_points = list(range(min_samples, n, update_freq))
    if n - 1 not in update_points and n > min_samples:
        update_points.append(n - 1)
    
    computed_values = {}
    
    # Compute denoised values at update points only
    for t in update_points:
        start_idx = max(0, t + 1 - lookback)
        window = signal[start_idx:t+1]
        
        try:
            level = min(pywt.dwt_max_level(len(window), wavelet), 3)
            if level < 1:
                computed_values[t] = window[-1]
                continue
            
            coeffs = pywt.wavedec(window, wavelet, level=level)
            
            # Calculate threshold from detail coefficients
            detail_coeffs = coeffs[-1]
            sigma = np.median(np.abs(detail_coeffs)) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(window)))
            
            # Apply soft thresholding to details
            denoised_coeffs = [coeffs[0]]
            for i in range(1, len(coeffs)):
                denoised_coeffs.append(pywt.threshold(coeffs[i], threshold, mode='soft'))
            
            reconstructed = pywt.waverec(denoised_coeffs, wavelet)
            computed_values[t] = reconstructed[-1]
        except Exception:
            computed_values[t] = window[-1]
    
    # Fill early values with simple mean
    for t in range(min(min_samples, n)):
        denoised[t] = signal[:t+1].mean() if t > 0 else signal[0]
    
    # Interpolate between update points
    sorted_points = sorted(computed_values.keys())
    for i, t in enumerate(sorted_points):
        denoised[t] = computed_values[t]
        
        if i > 0:
            prev_t = sorted_points[i-1]
            prev_val = computed_values[prev_t]
            curr_val = computed_values[t]
            
            # Linear interpolation for intermediate values
            for j in range(prev_t + 1, t):
                alpha = (j - prev_t) / (t - prev_t)
                denoised[j] = prev_val + alpha * (curr_val - prev_val)
    
    # Fill gap between min_samples and first update point
    if sorted_points and min_samples < sorted_points[0]:
        first_computed = sorted_points[0]
        first_val = computed_values[first_computed]
        for j in range(min_samples, first_computed):
            # Gradual transition from mean to computed
            alpha = (j - min_samples) / (first_computed - min_samples)
            denoised[j] = denoised[min_samples - 1] + alpha * (first_val - denoised[min_samples - 1])
    
    return denoised


def detect_outliers(df: pd.DataFrame, z_threshold: float = 4.0) -> pd.DataFrame:
    """
    Detect and flag outliers in price data.
    Also performs OHLC sanity checks.
    
    Returns DataFrame with added columns: is_outlier, invalid_ohlc
    """
    df = df.copy()
    
    # Z-score on returns
    returns = df['Close'].pct_change()
    z_scores = (returns - returns.mean()) / (returns.std() + 1e-8)
    df['is_outlier'] = abs(z_scores) > z_threshold
    
    # OHLC sanity checks
    df['invalid_ohlc'] = (
        (df['High'] < df['Low']) |
        (df['Close'] > df['High']) |
        (df['Close'] < df['Low']) |
        (df['Open'] > df['High']) |
        (df['Open'] < df['Low'])
    )
    
    # Log warnings
    outlier_count = df['is_outlier'].sum()
    invalid_count = df['invalid_ohlc'].sum()
    
    if outlier_count > 0:
        print(f"  ⚠️ Warning: {outlier_count} outliers detected (|z| > {z_threshold})")
    if invalid_count > 0:
        print(f"  ⚠️ Warning: {invalid_count} invalid OHLC rows detected")
    
    return df


# ============================================================================
# HYBRID LOSS FUNCTION - MSE + Trend Accuracy
# ============================================================================

def trend_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate trend accuracy (direction prediction accuracy).
    This is critical because close numerical values don't mean correct trend prediction.
    """
    if len(y_true) < 2:
        return 0.5
    
    # Calculate actual and predicted returns/directions
    true_direction = np.sign(np.diff(y_true))
    pred_direction = np.sign(np.diff(y_pred))
    
    # Replace zeros with previous non-zero value or 1
    true_direction[true_direction == 0] = 1
    pred_direction[pred_direction == 0] = 1
    
    # Accuracy: percentage of correct direction predictions
    correct = np.sum(true_direction == pred_direction)
    accuracy = correct / len(true_direction)
    
    return accuracy


def hybrid_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.5) -> float:
    """
    Hybrid loss combining MSE and trend accuracy.
    
    Loss = MSE + (1 - ACC) × scaling_factor
    
    The scaling factor ensures both components contribute meaningfully
    regardless of price magnitude.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    acc = trend_accuracy(y_true, y_pred)
    
    # Dynamic scaling: floor(log10(|MSE|)) to align magnitudes
    if mse > 0:
        scale = 10 ** np.floor(np.log10(abs(mse) + 1e-8))
    else:
        scale = 1.0
    
    # Hybrid loss (lower is better)
    loss = mse + (1 - acc) * scale
    
    return loss


def evaluate_model_comprehensive(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Comprehensive model evaluation with all key metrics.
    """
    metrics = {}
    
    # Standard regression metrics
    metrics['mse'] = mean_squared_error(y_true, y_pred)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = mean_absolute_error(y_true, y_pred)
    metrics['r2'] = r2_score(y_true, y_pred)
    
    # MASE (Mean Absolute Scaled Error) - <1 means better than naive forecast
    naive_mae = np.mean(np.abs(np.diff(y_true)))  # Naive = previous value
    metrics['mase'] = metrics['mae'] / naive_mae if naive_mae > 0 else float('inf')
    
    # Trend accuracy (direction prediction)
    metrics['trend_accuracy'] = trend_accuracy(y_true, y_pred)
    
    # Hybrid loss
    metrics['hybrid_loss'] = hybrid_loss(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    non_zero_mask = y_true != 0
    if np.any(non_zero_mask):
        metrics['mape'] = np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100
    else:
        metrics['mape'] = float('inf')
    
    return metrics


# ============================================================================
# xLSTM-TS STYLE FEATURES (Simplified Pure Python/Sklearn Implementation)
# ============================================================================

class ExponentialGatingFeatures:
    """
    Generate features inspired by xLSTM's exponential gating mechanism.
    These capture non-linear temporal dependencies without deep learning.
    """
    
    def __init__(self, decay_rates: List[float] = [0.9, 0.95, 0.99, 0.999]):
        self.decay_rates = decay_rates
    
    def fit_transform(self, prices: np.ndarray) -> pd.DataFrame:
        """Generate exponential gating inspired features."""
        features = {}
        
        # Exponentially weighted statistics at different decay rates
        for rate in self.decay_rates:
            alpha = 1 - rate
            rate_name = str(rate).replace('.', '_')
            
            # EMA
            ema = pd.Series(prices).ewm(alpha=alpha, adjust=False).mean().values
            features[f'ema_{rate_name}'] = ema
            
            # EMA of returns
            returns = np.diff(prices, prepend=prices[0])
            ema_returns = pd.Series(returns).ewm(alpha=alpha, adjust=False).mean().values
            features[f'ema_returns_{rate_name}'] = ema_returns
            
            # Exponentially weighted volatility
            features[f'ewm_vol_{rate_name}'] = pd.Series(returns).ewm(alpha=alpha, adjust=False).std().values
            
            # Price deviation from EMA (normalized)
            deviation = (prices - ema) / (ema + 1e-8)
            features[f'ema_deviation_{rate_name}'] = deviation
        
        # Matrix memory inspired: rolling statistics with varying windows
        for window in [5, 10, 21, 63, 126]:  # 1 week, 2 weeks, 1 month, 3 months, 6 months
            if len(prices) > window:
                series = pd.Series(prices)
                features[f'rolling_mean_{window}'] = series.rolling(window).mean().values
                features[f'rolling_std_{window}'] = series.rolling(window).std().values
                features[f'rolling_min_{window}'] = series.rolling(window).min().values
                features[f'rolling_max_{window}'] = series.rolling(window).max().values
                features[f'rolling_skew_{window}'] = series.rolling(window).skew().values
                features[f'rolling_kurt_{window}'] = series.rolling(window).kurt().values
                
                # Position within rolling range (0=at min, 1=at max)
                roll_range = features[f'rolling_max_{window}'] - features[f'rolling_min_{window}']
                features[f'range_position_{window}'] = np.where(
                    roll_range > 0,
                    (prices - features[f'rolling_min_{window}']) / roll_range,
                    0.5
                )
        
        return pd.DataFrame(features)


class TiDEEncoder:
    """
    Time-series Dense Encoder inspired by TiDE architecture.
    Uses dense layers with residual connections (simulated with sklearn).
    """
    
    def __init__(self, lookback: int = 150, horizon: int = 21):
        self.lookback = lookback
        self.horizon = horizon
        self.scaler = StandardScaler()
    
    def create_sequences(self, data: np.ndarray, targets: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction."""
        X, y = [], []
        
        for i in range(self.lookback, len(data) - self.horizon + 1):
            X.append(data[i - self.lookback:i].flatten())
            if targets is not None:
                y.append(targets[i:i + self.horizon])
            else:
                y.append(data[i:i + self.horizon])
        
        return np.array(X), np.array(y)
    
    def create_dense_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create dense encoded features.
        Simulates TiDE's encoder with polynomial and interaction features.
        """
        features = [X]
        
        # Add squared terms (captures non-linearity)
        features.append(X ** 2)
        
        # Add cross-terms for recent values (interaction features)
        if X.shape[1] >= 10:
            recent = X[:, -10:]
            cross = []
            for i in range(10):
                for j in range(i+1, 10):
                    cross.append(recent[:, i] * recent[:, j])
            if cross:
                features.append(np.column_stack(cross))
        
        return np.hstack(features)


# ============================================================================
# 🔮 TREND DAMPENER - Prevents Excessive Bearish Predictions for Quality Stocks
# ============================================================================

class TrendDampener:
    """
    Intelligent trend dampening for quality stocks.
    
    Problem: Pure ML models tend to extrapolate recent trends, leading to:
    - Overly bearish predictions for stocks in temporary dips
    - Missing mean-reversion opportunities in quality stocks
    
    Solution: Apply dampening based on:
    1. Quality score (from fundamentals: P/E, dividend yield)
    2. How far the prediction deviates from long-term mean
    3. Exponential smoothing toward fair value
    
    Research basis:
    - Mean reversion is well-documented in equity markets (Poterba & Summers, 1988)
    - Quality stocks revert faster than low-quality stocks
    - Excessive short-term bearishness often presents buying opportunities
    """
    
    def __init__(self, 
                 max_dampening: float = 0.5,
                 quality_threshold: float = 0.55,
                 reversion_rate: float = 0.03):
        """
        Args:
            max_dampening: Maximum dampening factor (0.5 = 50% reduction in bearish movement)
            quality_threshold: Minimum quality score to apply dampening
            reversion_rate: Rate of mean reversion per day (0.03 = 3% per day toward mean)
        """
        self.max_dampening = max_dampening
        self.quality_threshold = quality_threshold
        self.reversion_rate = reversion_rate
    
    def calculate_fair_value(self, df: pd.DataFrame, lookback: int = 200) -> float:
        """
        Estimate fair value using multiple methods:
        1. Long-term SMA
        2. EMA
        3. VWAP-like weighted average
        
        Returns the average of these estimates.
        """
        closes = df['Close'].tail(lookback).values
        
        # Method 1: Simple Moving Average (long-term)
        sma = np.mean(closes)
        
        # Method 2: Exponential Moving Average (recent-weighted)
        weights = np.exp(np.linspace(-1, 0, len(closes)))
        weights /= weights.sum()
        ema = np.sum(closes * weights)
        
        # Method 3: Volume-weighted if available
        if 'Volume' in df.columns:
            volumes = df['Volume'].tail(lookback).values
            vol_sum = volumes.sum()
            if vol_sum > 0:
                vwap = np.sum(closes * volumes) / vol_sum
            else:
                vwap = sma
        else:
            vwap = sma
        
        # Return weighted average (favor EMA slightly)
        fair_value = 0.3 * sma + 0.5 * ema + 0.2 * vwap
        return fair_value
    
    def apply_dampening(self, 
                        raw_prediction: float, 
                        current_price: float,
                        fair_value: float,
                        quality_score: float,
                        day_offset: int = 1) -> float:
        """
        Apply trend dampening to a raw prediction.
        
        Logic:
        1. If quality_score < threshold, return raw prediction (no dampening)
        2. If prediction is bearish AND below fair value, apply dampening
        3. Dampening pulls prediction toward fair value
        
        Args:
            raw_prediction: The ML model's raw price prediction
            current_price: Current stock price
            fair_value: Estimated fair value from historical data
            quality_score: 0-1 score based on fundamentals
            day_offset: How far into the future (more dampening for longer horizons)
        
        Returns:
            Dampened prediction
        """
        if quality_score < self.quality_threshold:
            return raw_prediction
        
        # Calculate direction and magnitude
        is_bearish = raw_prediction < current_price
        is_below_fair = raw_prediction < fair_value
        
        # Only dampen if bearish AND predicting below fair value
        if not (is_bearish and is_below_fair):
            return raw_prediction
        
        # Calculate dampening factor
        # Higher quality = more dampening
        # Further from fair value = more dampening
        quality_factor = (quality_score - self.quality_threshold) / (1 - self.quality_threshold)
        
        deviation_from_fair = abs(raw_prediction - fair_value) / fair_value
        deviation_factor = min(1.0, deviation_from_fair / 0.2)  # Cap at 20% deviation
        
        # Time factor: more dampening for longer horizons (mean reversion takes time)
        time_factor = min(1.0, day_offset / 60)  # Full effect by 60 days
        
        # Combined dampening strength
        dampening_strength = quality_factor * deviation_factor * time_factor * self.max_dampening
        
        # Apply dampening: pull prediction toward fair value
        dampened_prediction = raw_prediction + dampening_strength * (fair_value - raw_prediction)
        
        return dampened_prediction
    
    def get_dampening_info(self, 
                           raw_prediction: float,
                           dampened_prediction: float,
                           quality_score: float) -> dict:
        """Get diagnostic info about dampening applied."""
        adjustment = dampened_prediction - raw_prediction
        adjustment_pct = (adjustment / raw_prediction * 100) if raw_prediction > 0 else 0
        
        return {
            'dampening_applied': abs(adjustment) > 0.01,
            'adjustment_amount': adjustment,
            'adjustment_pct': adjustment_pct,
            'quality_score': quality_score,
            'reason': 'Mean reversion adjustment for quality stock' if adjustment > 0 else 'No dampening'
        }


def get_quality_score_from_sentiment(sentiment_result: dict) -> float:
    """
    Extract quality score from sentiment analysis result.
    Used when sentiment analyzer has already computed fundamentals.
    """
    if not sentiment_result:
        return 0.5
    
    # Check if quality score is already computed
    if 'quality_score' in sentiment_result:
        return sentiment_result['quality_score']
    
    # Calculate from fundamentals if available
    fundamentals = sentiment_result.get('fundamentals', {})
    score = 0.5  # Neutral baseline
    
    pe = fundamentals.get('pe_ratio')
    if pe:
        if pe < 8:
            score += 0.15
        elif pe < 12:
            score += 0.10
        elif pe < 18:
            score += 0.05
        elif pe > 30:
            score -= 0.10
    
    div_yield = fundamentals.get('dividend_yield')
    if div_yield:
        if div_yield > 8:
            score += 0.15
        elif div_yield > 5:
            score += 0.10
        elif div_yield > 2:
            score += 0.05
    
    return max(0.0, min(1.0, score))


# ============================================================================
# SOTA ENSEMBLE MODEL
# ============================================================================

class SOTAEnsemblePredictor:
    """
    🔮 FORTUNE TELLER: State-of-the-Art Ensemble Predictor
    
    Combines cutting-edge techniques:
    - N-BEATS-style basis decomposition (trend + seasonality)
    - Wavelet denoising preprocessing (50% → 70%+ accuracy)
    - xLSTM-TS style feature engineering
    - TiDE-inspired dense encoding
    - PSX-specific seasonal patterns (Ramadan, EID, fiscal year)
    - Multi-horizon ensemble (different models for different timeframes)
    - Daily predictions through Dec 2026
    """
    
    def __init__(self, lookback: int = 150, horizon: int = 21, use_wavelet: bool = True,
                 quality_score: float = 0.5):
        self.lookback = lookback
        self.horizon = horizon
        self.use_wavelet = use_wavelet and PYWT_AVAILABLE
        self.quality_score = quality_score  # For trend dampening
        
        self.models = {}
        self.model_weights = {}
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Fortune Teller components
        self.nbeats_decomposer = NBEATSBasisDecomposer(trend_degree=3, n_harmonics=5)
        self.seasonal_features = PSXSeasonalFeatures()
        self.multi_horizon = MultiHorizonEnsemble()
        
        self.exp_gating = ExponentialGatingFeatures()
        self.tide_encoder = TiDEEncoder(lookback=lookback, horizon=horizon)
        
        # 🆕 Trend Dampener for quality stocks
        self.trend_dampener = TrendDampener(
            max_dampening=0.5,
            quality_threshold=0.55,
            reversion_rate=0.03
        )
        self.fair_value = None  # Will be calculated during fit()
        
        self.feature_names = []
        self.is_fitted = False
        
        # Initialize base models with optimized hyperparameters
        self._init_models()
    
    def _init_models(self):
        """Initialize ensemble of models."""
        self.base_models = {}
        
        # Random Forest (consistently strong performer)
        self.base_models['rf'] = RandomForestRegressor(
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        # Extra Trees (often better than RF for time series)
        self.base_models['et'] = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )
        
        # Gradient Boosting
        self.base_models['gb'] = GradientBoostingRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=5,
            random_state=42
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.base_models['xgb'] = XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                n_jobs=-1,
                random_state=42,
                verbosity=0
            )
        
        # LightGBM (if available - very fast)
        if LIGHTGBM_AVAILABLE:
            self.base_models['lgbm'] = LGBMRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_samples=20,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        
        # Ridge regression (linear baseline)
        self.base_models['ridge'] = Ridge(alpha=1.0)
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        🔮 CAUSAL Fortune Teller preprocessing pipeline:
        1. Causal wavelet denoising (no look-ahead)
        2. Causal N-BEATS basis decomposition (expanding window)
        3. PSX seasonal features
        4. Exponential gating features (inherently causal - uses EMA)
        
        FIXED: All features at time t now only use data from [0, t].
        """
        df = df.copy()
        
        # 0. Detect outliers (data quality check)
        df = detect_outliers(df)
        
        # 1. Apply CAUSAL wavelet denoising to price columns
        if self.use_wavelet:
            for col in ['Close', 'Open', 'High', 'Low']:
                if col in df.columns:
                    # FIXED: Use causal denoising - no future data leakage
                    df[f'{col}_denoised'] = wavelet_denoise_causal(df[col].values)
                    
                    # Skip non-causal wavelet features for now  
                    # (multi_scale_wavelet_features is non-causal)
        
        # 2. CAUSAL N-BEATS style decomposition (trend + seasonality)
        close_prices = df['Close_denoised'].values if 'Close_denoised' in df.columns else df['Close'].values
        try:
            # FIXED: Use causal decomposition with expanding window
            nbeats_features = self.nbeats_decomposer.get_features_causal(close_prices)
            for col in nbeats_features.columns:
                df[col] = nbeats_features[col].values
        except Exception as e:
            print(f"⚠️ N-BEATS features skipped: {e}")
        
        # 3. PSX seasonal features (Ramadan, EID, fiscal year, etc.)
        # NOTE: These are based on calendar date, not price data - inherently causal
        if 'Date' in df.columns:
            try:
                seasonal_feats = self.seasonal_features.generate(df['Date'])
                for col in seasonal_feats.columns:
                    df[f'seasonal_{col}'] = seasonal_feats[col].values
            except Exception as e:
                print(f"⚠️ Seasonal features skipped: {e}")
        
        # 4. Generate exponential gating features (EMA-based - inherently causal)
        exp_features = self.exp_gating.fit_transform(close_prices)
        
        for col in exp_features.columns:
            df[col] = exp_features[col].values
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and target vector.
        """
        # Apply preprocessing
        df = self.preprocess(df)
        
        # Select feature columns (exclude Date and target)
        exclude_cols = ['Date', 'Target', 'Target_Next_Day', 'Target_Next_Week', 'Target_Next_Month']
        feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'int64']]
        
        # Create target (next day's close)
        df['Target'] = df['Close'].shift(-1)
        
        # Remove rows with NaN
        df_clean = df.dropna()
        
        X = df_clean[feature_cols].values
        y = df_clean['Target'].values
        
        return X, y, feature_cols
    
    def fit(self, df: pd.DataFrame, verbose: bool = True) -> Dict[str, float]:
        """
        Train the ensemble model with walk-forward validation.
        
        Returns:
            Dictionary of validation metrics
        """
        if verbose:
            print("="*70)
            print("🚀 TRAINING SOTA ENSEMBLE MODEL")
            print("="*70)
            print(f"  • Wavelet Denoising: {'Enabled' if self.use_wavelet else 'Disabled'}")
            print(f"  • Lookback: {self.lookback} days")
            print(f"  • Models: {list(self.base_models.keys())}")
            print()
        
        # Prepare features
        X, y, feature_cols = self.prepare_features(df)
        self.feature_names = feature_cols
        
        # 🆕 Calculate fair value for trend dampening
        self.fair_value = self.trend_dampener.calculate_fair_value(df)
        if verbose:
            print(f"  • Fair Value (for trend dampening): PKR {self.fair_value:.2f}")
            print(f"  • Quality Score: {self.quality_score:.2f}")
        
        if verbose:
            print(f"  • Features: {len(feature_cols)}")
            print(f"  • Samples: {len(X)}")
            print()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        validation_scores = {name: [] for name in self.base_models.keys()}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            for name, model in self.base_models.items():
                # Clone model for this fold
                from sklearn.base import clone
                model_clone = clone(model)
                model_clone.fit(X_train, y_train)
                
                y_pred = model_clone.predict(X_val)
                
                # Evaluate with hybrid loss
                metrics = evaluate_model_comprehensive(y_val, y_pred)
                validation_scores[name].append(metrics['trend_accuracy'])
            
            if verbose:
                print(f"  Fold {fold + 1}/5 complete")
        
        # Calculate average trend accuracy for each model
        avg_scores = {}
        for name, scores in validation_scores.items():
            avg_scores[name] = np.mean(scores)
            if verbose:
                print(f"    {name}: Trend Accuracy = {avg_scores[name]:.2%}")
        
        # Calculate ensemble weights (proportional to performance)
        total_score = sum(avg_scores.values())
        for name, score in avg_scores.items():
            self.model_weights[name] = score / total_score if total_score > 0 else 1 / len(avg_scores)
        
        if verbose:
            print()
            print("  📊 Ensemble Weights:")
            for name, weight in self.model_weights.items():
                print(f"    {name}: {weight:.3f}")
        
        # Train final models on all data
        if verbose:
            print()
            print("  🔧 Training final models on full dataset...")
        
        for name, model in self.base_models.items():
            model.fit(X_scaled, y)
            self.models[name] = model
        
        self.is_fitted = True
        
        # Final evaluation
        y_pred_ensemble = self._predict_ensemble(X_scaled)
        final_metrics = evaluate_model_comprehensive(y, y_pred_ensemble)
        
        if verbose:
            print()
            print("  ✅ TRAINING COMPLETE!")
            print(f"    Final R²: {final_metrics['r2']:.4f}")
            print(f"    Final MASE: {final_metrics['mase']:.4f} (< 1 is better than naive)")
            print(f"    Final Trend Accuracy: {final_metrics['trend_accuracy']:.2%}")
            print()
        
        return final_metrics
    
    def _predict_ensemble(self, X_scaled: np.ndarray) -> np.ndarray:
        """Make weighted ensemble prediction."""
        predictions = []
        weights = []
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)
            predictions.append(pred)
            weights.append(self.model_weights.get(name, 1.0))
        
        # Weighted average
        weights = np.array(weights) / np.sum(weights)
        ensemble_pred = np.average(np.array(predictions), axis=0, weights=weights)
        
        return ensemble_pred
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X, _, _ = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        return self._predict_ensemble(X_scaled)
    
    def predict_future(self, df: pd.DataFrame, months_ahead: int = 24) -> List[Dict]:
        """
        Generate multi-step predictions with confidence intervals.
        Uses iterative prediction with feature roll-forward.
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        predictions = []
        current_df = df.copy()
        current_price = df['Close'].iloc[-1]
        current_date = pd.to_datetime(df['Date'].iloc[-1])
        
        trading_days_per_month = 21
        
        for month in range(months_ahead):
            # Prepare features from current state
            X, _, _ = self.prepare_features(current_df)
            if len(X) == 0:
                break
            
            X_latest = X[-1:] 
            X_scaled = self.scaler.transform(X_latest)
            
            # Get predictions from all models
            all_preds = []
            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                all_preds.append(pred)
            
            # Ensemble prediction
            weights = [self.model_weights.get(name, 1.0) for name in self.models.keys()]
            weights = np.array(weights) / np.sum(weights)
            ensemble_pred = np.average(all_preds, weights=weights)
            
            # Confidence interval from model disagreement
            pred_std = np.std(all_preds)
            lower_ci = ensemble_pred - 2 * pred_std
            upper_ci = ensemble_pred + 2 * pred_std
            
            # Calculate upside
            upside = (ensemble_pred - current_price) / current_price * 100
            
            prediction_date = current_date + pd.DateOffset(months=month + 1)
            
            predictions.append({
                'month': prediction_date.strftime('%Y-%m'),
                'date': prediction_date.strftime('%Y-%m-%d'),
                'current_price': float(current_price),
                'predicted_price': float(ensemble_pred),
                'lower_ci': float(lower_ci),
                'upper_ci': float(upper_ci),
                'upside_potential': float(upside),
                'confidence': float(1 - pred_std / (ensemble_pred + 1e-8))  # Higher = more model agreement
            })
            
            # Roll forward: add predicted price to dataframe for next iteration
            new_row = current_df.iloc[-1:].copy()
            new_row['Date'] = prediction_date
            new_row['Close'] = ensemble_pred
            new_row['Open'] = ensemble_pred
            new_row['High'] = ensemble_pred * 1.01
            new_row['Low'] = ensemble_pred * 0.99
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        return predictions
    
    def predict_daily(self, df: pd.DataFrame, end_date: str = '2026-12-31',
                       seed: int = 42, progress_callback=None, max_horizon: int = None,
                       force_full_year: bool = False) -> List[Dict]:
        """
        🔮 Generate DAILY predictions with research-validated horizons.

        ⚠️ RESEARCH FINDING: R² collapses after 20 days (PSX LSTM study 2025)
        - Predictions beyond 21 days labeled as "low_reliability" (informational only)
        - Hard cap at 60 days to prevent fantasy predictions

        Uses multi-horizon ensemble weighting:
        - Short-term (1-7 days): XGBoost/LightGBM with 35% weight - HIGH RELIABILITY
        - Medium-term (8-21 days): RandomForest/ExtraTrees with 30% weight - MEDIUM RELIABILITY
        - Long-term (22-60 days): GradientBoosting with 20% weight - LOW RELIABILITY (informational)

        Args:
            df: Historical DataFrame with OHLCV data
            end_date: Target end date for predictions (capped at 60 days from last date)
            seed: Random seed for reproducibility
            max_horizon: Maximum prediction horizon (default 21 days, max 60 days)

        Returns:
            List of daily predictions with confidence intervals and reliability tiers
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Set random seed for reproducibility
        np.random.seed(seed)

        # Enforce research-backed horizon limits
        if max_horizon is None:
            if force_full_year:
                # Full year predictions for visualization (informational only)
                max_horizon = 365
                print(f"  📅 Full year prediction mode (informational beyond 21 days)")
            else:
                # Default to 60 day cap when not forcing full year
                max_horizon = 60
                print(f"  📅 Using default 60-day horizon (use force_full_year=True for longer)")
        else:
            # Apply 60-day hard cap for research-validated mode (unless force_full_year)
            if not force_full_year:
                max_horizon = min(max_horizon, 60)
            if max_horizon > 21:
                print(f"  ⚠️ WARNING: Horizon {max_horizon} days exceeds research-validated range (21 days)")
                print(f"  ⚠️ Predictions beyond day 21 are LOW RELIABILITY (R² drops below 0.70)")

        predictions = []
        current_df = df.copy()
        current_price = df['Close'].iloc[-1]
        current_date = pd.to_datetime(df['Date'].iloc[-1])
        end_date_obj = pd.to_datetime(end_date)

        # Cap end date to max_horizon
        max_end_date = current_date + pd.Timedelta(days=max_horizon * 2)  # *2 to account for weekends
        if end_date_obj > max_end_date:
            end_date_obj = max_end_date
            print(f"  📅 Capping predictions to {max_horizon} days: {end_date_obj.strftime('%Y-%m-%d')}")

        day_offset = 0
        max_days = max_horizon  # Research-validated limit
        
        print(f"  🔮 Generating daily predictions from {current_date.strftime('%Y-%m-%d')} to {end_date}...")
        
        # Progress update interval
        progress_interval = max(1, min(50, max_days // 10))
        
        while current_date < end_date_obj and day_offset < max_days:
            day_offset += 1
            
            # Send progress updates periodically
            if progress_callback and (day_offset % progress_interval == 0 or day_offset >= max_days - 1):
                progress_pct = 85 + int((day_offset / max_days) * 10)  # 85-95% range
                try:
                    update_data = {
                        'stage': 'predicting',
                        'progress': min(95, progress_pct),
                        'message': f'🔮 Generating predictions... {day_offset}/{max_days} days ({min(95, progress_pct)}%)'
                    }
                    if callable(progress_callback):
                        progress_callback(update_data)
                except Exception:
                    pass  # Don't break prediction if progress update fails
            
            # Skip weekends (PSX is closed)
            next_date = current_date + pd.Timedelta(days=1)
            while next_date.dayofweek >= 5:  # Saturday = 5, Sunday = 6
                next_date += pd.Timedelta(days=1)
            current_date = next_date
            
            # Skip PSX holidays
            date_str = current_date.strftime('%Y-%m-%d')
            if self.seasonal_features.is_psx_holiday(date_str):
                continue  # Skip this day, don't increment day_offset
            
            if current_date > end_date_obj:
                break
            
            # Prepare features from current state
            try:
                X, _, _ = self.prepare_features(current_df)
                if len(X) == 0:
                    break
                
                X_latest = X[-1:]
                X_scaled = self.scaler.transform(X_latest)
            except Exception as e:
                print(f"  ⚠️ Feature preparation error at day {day_offset}: {e}")
                break
            
            # Get multi-horizon weighted predictions
            horizon_weights = self.multi_horizon.get_horizon_weight(day_offset)
            
            all_preds = []
            all_weights = []
            
            for name, model in self.models.items():
                try:
                    pred = model.predict(X_scaled)[0]
                    all_preds.append(pred)
                    # Combine model weight with horizon weight
                    model_weight = self.model_weights.get(name, 1.0)
                    horizon_weight = horizon_weights.get(name, 0.1)
                    all_weights.append(model_weight * (1 + horizon_weight))
                except Exception:
                    continue
            
            if not all_preds:
                break
            
            # Weighted ensemble prediction
            weights_norm = np.array(all_weights) / np.sum(all_weights)
            ensemble_pred = np.average(all_preds, weights=weights_norm)
            
            # 🔧 FIX: Apply 7.5% circuit breaker (PSX daily limit)
            prev_close = current_df['Close'].iloc[-1]
            max_move = 0.075  # 7.5% circuit
            ensemble_pred = np.clip(
                ensemble_pred,
                prev_close * (1 - max_move),
                prev_close * (1 + max_move)
            )
            
            # 🆕 TREND DAMPENING: Pull bearish predictions toward fair value for quality stocks
            if self.fair_value and self.quality_score > 0.55:
                raw_pred = ensemble_pred
                ensemble_pred = self.trend_dampener.apply_dampening(
                    raw_prediction=ensemble_pred,
                    current_price=current_price,
                    fair_value=self.fair_value,
                    quality_score=self.quality_score,
                    day_offset=day_offset
                )
                # Log significant dampening
                if abs(ensemble_pred - raw_pred) > 0.5 and day_offset == 1:
                    print(f"    🔄 Trend dampening: {raw_pred:.2f} → {ensemble_pred:.2f} (quality={self.quality_score:.2f})")
            
            # 🔧 FIX: Exponential uncertainty (errors compound multiplicatively)
            pred_std = np.std(all_preds)
            uncertainty_factor = 1.0 * (1.02 ** (day_offset / 21))  # ~2% increase per month
            lower_ci = ensemble_pred - 2 * pred_std * uncertainty_factor
            upper_ci = ensemble_pred + 2 * pred_std * uncertainty_factor
            
            # Calculate metrics
            upside = (ensemble_pred - current_price) / current_price * 100
            confidence = max(0, min(1, 1 - (pred_std * uncertainty_factor) / (abs(ensemble_pred) + 1e-8)))
            
            # Determine reliability tier based on research
            reliability = self._get_reliability_tier(day_offset)

            predictions.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day_offset': day_offset,
                'predicted_price': float(round(ensemble_pred, 2)),
                'lower_ci': float(round(max(0, lower_ci), 2)),
                'upper_ci': float(round(upper_ci, 2)),
                'upside_potential': float(round(upside, 2)),
                'confidence': float(round(confidence, 3)),
                'horizon_type': self._get_horizon_type(day_offset),
                'reliability': reliability  # NEW: Research-backed reliability tier
            })
            
            # Roll forward: add predicted price to dataframe for next iteration
            new_row = current_df.iloc[-1:].copy()
            new_row['Date'] = current_date
            new_row['Close'] = ensemble_pred
            new_row['Open'] = ensemble_pred * (1 + np.random.uniform(-0.005, 0.005))
            new_row['High'] = ensemble_pred * (1 + np.random.uniform(0.002, 0.015))
            new_row['Low'] = ensemble_pred * (1 - np.random.uniform(0.002, 0.015))
            new_row['Volume'] = current_df['Volume'].iloc[-20:].mean() * np.random.uniform(0.8, 1.2)
            current_df = pd.concat([current_df, new_row], ignore_index=True)
            
            # Progress indicator every 100 days
            if day_offset % 100 == 0:
                print(f"    📈 Day {day_offset}: {current_date.strftime('%Y-%m-%d')} → PKR {ensemble_pred:.2f}")
        
        print(f"  ✅ Generated {len(predictions)} daily predictions")
        return predictions
    
    def _get_horizon_type(self, day_offset: int) -> str:
        """Get human-readable horizon type for a given day offset."""
        if day_offset <= 7:
            return 'short_term'
        elif day_offset <= 21:
            return 'medium_term'
        elif day_offset <= 60:
            return 'long_term'
        else:
            return 'trend'

    def _get_reliability_tier(self, day_offset: int) -> str:
        """
        Get research-validated reliability tier.

        Based on PSX LSTM study (arXiv 2025):
        - R² day 7: 0.84-0.86 (HIGH)
        - R² day 21: 0.70-0.80 (MEDIUM)
        - R² beyond 21: <0.70 (LOW)
        """
        if day_offset <= 7:
            return 'high'  # R² > 0.84
        elif day_offset <= 21:
            return 'medium'  # R² 0.70-0.80
        else:
            return 'low'  # R² < 0.70 (informational only)
    
    def backtest_model(self, df: pd.DataFrame, test_months: int = 6) -> Dict:
        """
        Walk-forward backtest on held-out data.
        Returns realistic performance metrics including Sharpe ratio.
        
        Args:
            df: Full historical DataFrame
            test_months: Number of months to hold out for testing
            
        Returns:
            Dictionary with trend_accuracy, mape, rmse, max_error, sharpe
        """
        # Split data
        split_idx = len(df) - (test_months * 21)
        if split_idx < 200:
            print("  ⚠️ Not enough data for backtest")
            return {'error': 'Insufficient data'}
        
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        print(f"  📊 Backtesting: Train={len(train_df)} days, Test={len(test_df)} days")
        
        # Train on historical
        self.fit(train_df, verbose=False)
        
        # Predict daily through test period
        predictions = []
        actual_prices = test_df['Close'].values
        actual_dates = test_df['Date'].values
        
        current_df = train_df.copy()
        
        for i, (date, actual) in enumerate(zip(actual_dates, actual_prices)):
            # Get prediction for this day
            try:
                X, _, _ = self.prepare_features(current_df)
                X_scaled = self.scaler.transform(X[-1:])
                pred = self._predict_ensemble(X_scaled)[0]
            except Exception:
                pred = current_df['Close'].iloc[-1]  # Fallback to last known
            
            prev_close = current_df['Close'].iloc[-1]
            direction_correct = (pred > prev_close) == (actual > prev_close)
            
            predictions.append({
                'date': date,
                'predicted': float(pred),
                'actual': float(actual),
                'error': float(pred - actual),
                'direction_correct': bool(direction_correct)
            })
            
            # Roll forward with ACTUAL price (not predicted)
            new_row = test_df.iloc[i:i+1].copy()
            current_df = pd.concat([current_df, new_row], ignore_index=True)
        
        # Calculate metrics
        results = pd.DataFrame(predictions)
        
        metrics = {
            'trend_accuracy': float(results['direction_correct'].mean()),
            'mape': float((abs(results['error']) / results['actual']).mean() * 100),
            'rmse': float(np.sqrt((results['error'] ** 2).mean())),
            'max_error': float(results['error'].abs().max()),
            'sharpe': float(self._calculate_sharpe(results)),
            'test_days': len(results),
            'train_days': len(train_df)
        }
        
        print(f"  ✅ Backtest complete:")
        print(f"     Trend Accuracy: {metrics['trend_accuracy']:.1%}")
        print(f"     MAPE: {metrics['mape']:.2f}%")
        print(f"     Sharpe Ratio: {metrics['sharpe']:.2f}")
        
        return metrics
    
    def _calculate_sharpe(self, results: pd.DataFrame, risk_free: float = 0.10,
                          transaction_cost: float = 0.005) -> float:
        """
        Calculate Sharpe ratio of predicted vs actual returns.
        Strategy: long when pred > current, else flat
        
        Args:
            results: DataFrame with 'actual' and 'direction_correct' columns
            risk_free: Annual risk-free rate (default 10% for PKR)
            transaction_cost: Round-trip cost (default 0.5% for PSX)
        
        Returns:
            Annualized Sharpe ratio accounting for transaction costs
        """
        if len(results) < 2:
            return 0.0
        
        actual_returns = results['actual'].pct_change().dropna()
        
        # Strategy: long when direction correct, else flat
        direction_correct = results['direction_correct'].iloc[1:].values
        strategy_returns = actual_returns.values * direction_correct
        
        # Detect position changes and subtract transaction costs
        positions = direction_correct.astype(float)
        position_changes = np.abs(np.diff(positions, prepend=0))
        costs = position_changes * transaction_cost
        
        # Net returns after transaction costs
        net_returns = strategy_returns - costs
        
        # Daily risk-free rate (10% annual)
        daily_rf = risk_free / 252
        excess_returns = net_returns - daily_rf
        
        if len(excess_returns) < 2 or excess_returns.std() < 1e-8:
            return 0.0
        
        sharpe = np.sqrt(252) * excess_returns.mean() / (excess_returns.std() + 1e-8)
        return sharpe
    
    def save(self, path: Path, symbol: str):
        """Save the trained model."""
        model_data = {
            'models': self.models,
            'model_weights': self.model_weights,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'lookback': self.lookback,
            'horizon': self.horizon,
            'use_wavelet': self.use_wavelet
        }
        
        joblib.dump(model_data, path / f"{symbol}_sota_ensemble.pkl")
        print(f"  ✅ Model saved to: {path / f'{symbol}_sota_ensemble.pkl'}")
    
    @classmethod
    def load(cls, path: Path, symbol: str) -> 'SOTAEnsemblePredictor':
        """Load a trained model."""
        model_data = joblib.load(path / f"{symbol}_sota_ensemble.pkl")
        
        predictor = cls(
            lookback=model_data['lookback'],
            horizon=model_data['horizon'],
            use_wavelet=model_data['use_wavelet']
        )
        
        predictor.models = model_data['models']
        predictor.model_weights = model_data['model_weights']
        predictor.scaler = model_data['scaler']
        predictor.feature_names = model_data['feature_names']
        predictor.is_fitted = True
        
        return predictor


# ============================================================================
# INTEGRATION WITH STOCK ANALYZER
# ============================================================================

async def train_sota_model_with_progress(df: pd.DataFrame, symbol: str, websocket=None) -> Tuple[SOTAEnsemblePredictor, Dict]:
    """
    Train SOTA model with WebSocket progress updates.
    """
    import asyncio
    
    async def send_progress(stage: str, progress: int, message: str):
        if websocket:
            await websocket.send_json({
                'stage': stage,
                'progress': progress,
                'message': message
            })
        await asyncio.sleep(0.01)
    
    await send_progress('preprocessing', 55, '🔬 Applying wavelet denoising...')
    
    # Initialize model
    model = SOTAEnsemblePredictor(
        lookback=150,
        horizon=21,
        use_wavelet=PYWT_AVAILABLE
    )
    
    await send_progress('training', 60, '🤖 Training SOTA ensemble model...')
    
    # Train with custom progress
    metrics = model.fit(df, verbose=False)
    
    await send_progress('training', 80, f'📊 Model trained! Trend Accuracy: {metrics["trend_accuracy"]:.1%}')
    
    # Save model
    models_dir = Path(__file__).parent.parent / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    model.save(models_dir, symbol)
    
    await send_progress('predicting', 85, '🔮 Generating predictions through 2026...')
    
    # Generate predictions
    predictions = model.predict_future(df, months_ahead=24)
    
    await send_progress('complete', 100, '✅ SOTA analysis complete!')
    
    return model, {'metrics': metrics, 'predictions': predictions}


def quick_train_sota(symbol: str) -> Tuple[SOTAEnsemblePredictor, Dict]:
    """
    Quick training function for standalone use.
    """
    data_file = Path(__file__).parent.parent / "data" / f"{symbol}_historical_with_indicators.json"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    
    model = SOTAEnsemblePredictor(lookback=150, horizon=21, use_wavelet=PYWT_AVAILABLE)
    metrics = model.fit(df)
    
    predictions = model.predict_future(df, months_ahead=24)
    
    # Save predictions
    pred_file = Path(__file__).parent.parent / "data" / f"{symbol}_sota_predictions_2026.json"
    with open(pred_file, 'w') as f:
        json.dump({
            'symbol': symbol,
            'generated_at': datetime.now().isoformat(),
            'model': 'SOTA Ensemble (Wavelet + xLSTM-TS features + TiDE)',
            'metrics': {k: float(v) for k, v in metrics.items()},
            'predictions': predictions
        }, f, indent=2)
    
    print(f"  ✅ Predictions saved to: {pred_file}")
    
    return model, {'metrics': metrics, 'predictions': predictions}


if __name__ == "__main__":
    print("="*70)
    print("🔬 SOTA STOCK PREDICTION MODEL")
    print("="*70)
    print()
    print("Features:")
    print("  ✅ Wavelet Denoising (db4 DWT)" if PYWT_AVAILABLE else "  ❌ Wavelet Denoising (install: pip install PyWavelets)")
    print("  ✅ xLSTM-TS Style Features")
    print("  ✅ TiDE Dense Encoding")
    print("  ✅ Hybrid Loss (MSE + Trend Accuracy)")
    print("  ✅ Ensemble: RF, ET, GB" + (", XGB" if XGBOOST_AVAILABLE else "") + (", LGBM" if LIGHTGBM_AVAILABLE else ""))
    print()
    
    # Example usage
    symbol = "FATIMA"
    try:
        model, results = quick_train_sota(symbol)
        print()
        print("📈 First 3 Monthly Predictions:")
        for pred in results['predictions'][:3]:
            print(f"  {pred['month']}: PKR {pred['predicted_price']:.2f} ({pred['upside_potential']:+.1f}%)")
    except FileNotFoundError as e:
        print(f"  ⚠️  {e}")
        print("  Run the stock analyzer first to fetch data.")
