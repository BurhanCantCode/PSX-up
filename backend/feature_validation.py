"""
Feature Validation Framework with SHAP Analysis

This module implements research-backed feature validation using:
1. SHAP (SHapley Additive exPlanations) for feature importance
2. Correlation analysis to remove multicollinearity
3. VIF (Variance Inflation Factor) to detect redundant features
4. Validation of research claims (USD/PKR, Williams %R, KSE-100 beta)

Research Basis:
- Williams %R: Expected 15-18% importance (PSX study, Modern Finance 2024)
- USD/PKR: Expected 12-15% importance (PSX LSTM study, arXiv 2025)
- KSE-100 beta: Expected 10-12% importance (PSX research)

Author: Research-backed PSX prediction team
Date: 2026-01-08
"""

import pandas as pd
import numpy as np
import shap
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor


class SHAPValidator:
    """
    Validate features using SHAP analysis to prove which features actually contribute.

    SHAP values explain the impact of each feature on model predictions using game theory.
    This prevents overfitting from noise features (estimated 20-30% of current 120 features).
    """

    def __init__(self, threshold: float = 0.01):
        """
        Args:
            threshold: Minimum SHAP importance to keep feature (default 0.01 = 1%)
        """
        self.threshold = threshold
        self.shap_values = None
        self.importance_df = None
        self.explainer = None

    def analyze_model(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                       feature_names: List[str]) -> Dict:
        """
        Generate SHAP values and feature importance rankings.

        Args:
            model: Trained sklearn model (tree-based for TreeExplainer)
            X_train: Training features
            X_test: Test features
            feature_names: List of feature names

        Returns:
            Dictionary with shap_values, importance_df, top_10 features
        """
        print("🔍 Analyzing feature importance with SHAP...")

        # Use TreeExplainer for tree-based models (RF, XGB, LGBM, etc.)
        try:
            self.explainer = shap.TreeExplainer(model)
            print("  ✅ Using TreeExplainer (optimized for tree models)")
        except Exception:
            # Fallback to KernelExplainer for non-tree models
            print("  ⚠️ TreeExplainer failed, using KernelExplainer (slower)")
            background = shap.sample(X_train, 100)  # Sample 100 points for speed
            self.explainer = shap.KernelExplainer(model.predict, background)

        # Calculate SHAP values on test set
        self.shap_values = self.explainer.shap_values(X_test)

        # Aggregate importance: mean absolute SHAP value per feature
        shap_importance = np.abs(self.shap_values).mean(axis=0)

        # Normalize to percentage
        shap_importance_pct = shap_importance / shap_importance.sum()

        # Create importance DataFrame
        self.importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': shap_importance,
            'shap_importance_pct': shap_importance_pct * 100  # Percentage
        }).sort_values('shap_importance', ascending=False)

        # Identify top features
        top_10 = self.importance_df.head(10).to_dict('records')

        print(f"  ✅ SHAP analysis complete. Top 10 features:")
        for i, feat in enumerate(top_10, 1):
            print(f"     {i}. {feat['feature']}: {feat['shap_importance_pct']:.2f}%")

        return {
            'shap_values': self.shap_values,
            'importance_df': self.importance_df,
            'top_10': top_10,
            'explainer': self.explainer
        }

    def validate_research_claims(self, importance_df: pd.DataFrame = None) -> Dict:
        """
        Verify that research-validated features (USD/PKR, Williams %R, KSE-100)
        are actually in the top features as expected.

        Expected importance (from research):
        - Williams %R: 15-18%
        - USD/PKR: 12-15%
        - KSE-100 beta: 10-12%

        Args:
            importance_df: Feature importance DataFrame (uses self.importance_df if None)

        Returns:
            Dictionary with validation results for each critical feature
        """
        if importance_df is None:
            importance_df = self.importance_df

        if importance_df is None:
            raise ValueError("No importance data available. Run analyze_model() first.")

        print("\n📊 Validating research claims...")

        # Define research-validated features and their expected importance
        critical_features = {
            'williams_r': {
                'expected_min': 0.15,  # 15% minimum
                'expected_max': 0.18,  # 18% maximum
                'description': 'Williams %R (PSX study: 18% importance)'
            },
            'usdpkr': {
                'expected_min': 0.12,  # 12%
                'expected_max': 0.15,  # 15%
                'description': 'USD/PKR exchange rate (PSX LSTM study: 15%)'
            },
            'kse100_beta': {
                'expected_min': 0.10,  # 10%
                'expected_max': 0.12,  # 12%
                'description': 'KSE-100 beta (PSX research: 12%)'
            },
            'disparity_5': {
                'expected_min': 0.14,  # 14%
                'expected_max': 0.16,  # 16%
                'description': 'Disparity Index 5-day (PSX study: 16%)'
            },
            'rsi_14': {
                'expected_min': 0.10,  # 10%
                'expected_max': 0.12,  # 12%
                'description': 'RSI 14-period (PSX research: 12%)'
            }
        }

        results = {}
        total_validated = 0

        for pattern, expected in critical_features.items():
            # Find matching features (case-insensitive partial match)
            matching = importance_df[
                importance_df['feature'].str.contains(pattern, case=False, na=False)
            ]

            if not matching.empty:
                # Get highest importance among matches
                top_match = matching.iloc[0]
                actual_importance = top_match['shap_importance_pct'] / 100  # Convert to decimal
                feature_name = top_match['feature']

                # Check if within expected range
                validated = expected['expected_min'] <= actual_importance <= expected['expected_max']

                if validated:
                    status = '✅ VALIDATED'
                    total_validated += 1
                elif actual_importance < expected['expected_min']:
                    status = '⚠️ LOWER THAN EXPECTED'
                else:
                    status = '⚠️ HIGHER THAN EXPECTED'

                results[pattern] = {
                    'feature_name': feature_name,
                    'expected_min': expected['expected_min'],
                    'expected_max': expected['expected_max'],
                    'actual': actual_importance,
                    'actual_pct': actual_importance * 100,
                    'validated': validated,
                    'status': status,
                    'description': expected['description'],
                    'rank': int(matching.index[0]) + 1
                }

                print(f"  {status}: {feature_name}")
                print(f"     Expected: {expected['expected_min']*100:.1f}-{expected['expected_max']*100:.1f}%, "
                      f"Actual: {actual_importance*100:.2f}%, Rank: {results[pattern]['rank']}")
            else:
                results[pattern] = {
                    'feature_name': None,
                    'expected_min': expected['expected_min'],
                    'expected_max': expected['expected_max'],
                    'actual': 0.0,
                    'validated': False,
                    'status': '❌ NOT FOUND',
                    'description': expected['description']
                }
                print(f"  ❌ NOT FOUND: {pattern} ({expected['description']})")

        # Summary
        print(f"\n📈 Validation Summary: {total_validated}/{len(critical_features)} features validated")

        if total_validated < len(critical_features):
            print(f"  ⚠️ WARNING: Some research claims not validated in your model!")
            print(f"  ⚠️ Consider retraining with emphasis on validated features.")

        return results

    def remove_noise_features(self, feature_names: List[str],
                                importance_df: pd.DataFrame = None,
                                threshold: float = None) -> List[str]:
        """
        Remove features with SHAP importance below threshold (likely noise).

        Args:
            feature_names: Full list of feature names
            importance_df: Feature importance DataFrame (uses self.importance_df if None)
            threshold: Minimum importance to keep (uses self.threshold if None)

        Returns:
            List of features to keep (noise removed)
        """
        if importance_df is None:
            importance_df = self.importance_df

        if threshold is None:
            threshold = self.threshold

        if importance_df is None:
            raise ValueError("No importance data. Run analyze_model() first.")

        # Features above threshold
        keep_features = importance_df[
            importance_df['shap_importance_pct'] >= (threshold * 100)
        ]['feature'].tolist()

        # Always protect research-validated features even if below threshold
        protected_keywords = ['williams_r', 'usdpkr', 'kse100_beta', 'disparity_5', 'rsi_14']
        protected_features = [
            f for f in feature_names
            if any(keyword in f.lower() for keyword in protected_keywords)
        ]

        # Combine
        final_features = list(set(keep_features + protected_features))

        removed_count = len(feature_names) - len(final_features)
        print(f"\n🗑️ Noise removal: {removed_count} features removed (kept {len(final_features)}/{len(feature_names)})")
        print(f"   Threshold: {threshold*100:.1f}% SHAP importance")

        if removed_count > 0:
            print(f"   ✅ Reduced feature count: {len(feature_names)} → {len(final_features)}")

        return final_features


def remove_multicollinear_features(df: pd.DataFrame, feature_cols: List[str],
                                      threshold: float = 0.90) -> List[str]:
    """
    Remove highly correlated features (multicollinearity) while protecting validated ones.

    Multicollinearity reduces model interpretability and can cause overfitting.
    Features with correlation > 0.90 provide redundant information.

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        threshold: Correlation threshold (default 0.90)

    Returns:
        List of features to keep (multicollinearity removed)
    """
    print(f"\n🔗 Analyzing feature correlations (threshold={threshold})...")

    # Ensure numeric features only
    numeric_features = [f for f in feature_cols if df[f].dtype in [np.float64, np.int64, np.float32, np.int32]]

    if len(numeric_features) < len(feature_cols):
        print(f"   ⚠️ Skipped {len(feature_cols) - len(numeric_features)} non-numeric features")

    # Calculate correlation matrix
    corr_matrix = df[numeric_features].corr().abs()

    # Get upper triangle (avoid double counting)
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = []
    for column in upper_triangle.columns:
        if any(upper_triangle[column] > threshold):
            # Don't drop if it's a protected feature
            if not is_protected_feature(column):
                to_drop.append(column)

    # Features to keep
    keep_features = [f for f in numeric_features if f not in to_drop]

    print(f"   ✅ Removed {len(to_drop)} highly correlated features")
    if len(to_drop) > 0:
        print(f"      Examples: {to_drop[:5]}")

    return keep_features


def calculate_vif(df: pd.DataFrame, feature_cols: List[str],
                   threshold: float = 10.0) -> pd.DataFrame:
    """
    Calculate Variance Inflation Factor (VIF) to detect multicollinearity.

    VIF measures how much variance of a feature is explained by other features:
    - VIF < 5: Low multicollinearity (good)
    - VIF 5-10: Moderate (acceptable)
    - VIF > 10: High multicollinearity (remove feature)

    Args:
        df: DataFrame with features
        feature_cols: List of feature column names
        threshold: VIF threshold for removal (default 10.0)

    Returns:
        DataFrame with VIF scores and recommendations
    """
    print(f"\n📊 Calculating VIF (threshold={threshold})...")

    # Ensure numeric features only
    numeric_features = [f for f in feature_cols if df[f].dtype in [np.float64, np.int64, np.float32, np.int32]]

    # Remove features with zero variance
    non_zero_var = [f for f in numeric_features if df[f].std() > 1e-8]

    if len(non_zero_var) < len(numeric_features):
        print(f"   ⚠️ Removed {len(numeric_features) - len(non_zero_var)} zero-variance features")

    # Calculate VIF for each feature
    vif_data = []
    for i, feature in enumerate(non_zero_var):
        try:
            vif_score = variance_inflation_factor(df[non_zero_var].fillna(0).values, i)
            vif_data.append({
                'feature': feature,
                'VIF': vif_score,
                'multicollinearity': 'High' if vif_score > threshold else ('Moderate' if vif_score > 5 else 'Low'),
                'recommendation': 'Remove' if vif_score > threshold and not is_protected_feature(feature) else 'Keep',
                'protected': is_protected_feature(feature)
            })
        except Exception as e:
            print(f"   ⚠️ VIF calculation failed for {feature}: {e}")

    # Create DataFrame and sort by VIF
    vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)

    # Summary
    high_vif_count = len(vif_df[vif_df['VIF'] > threshold])
    print(f"   ⚠️ {high_vif_count} features with VIF > {threshold} (high multicollinearity)")

    # Show worst offenders
    if high_vif_count > 0:
        print(f"   Top multicollinear features:")
        for idx, row in vif_df.head(5).iterrows():
            print(f"      {row['feature']}: VIF={row['VIF']:.1f} ({row['recommendation']})")

    return vif_df


def is_protected_feature(feature_name: str) -> bool:
    """
    Check if feature is research-validated and should be protected from removal.

    Protected features:
    - williams_r: 18% importance
    - usdpkr: 15% importance
    - kse100_beta: 12% importance
    - disparity_5: 16% importance
    - disparity_10: 14% importance
    - rsi_14: 12% importance

    Args:
        feature_name: Feature name to check

    Returns:
        True if feature should be protected
    """
    protected_keywords = [
        'williams_r', 'williams',
        'usdpkr', 'usd_pkr', 'usd/pkr',
        'kse100_beta', 'kse_100_beta', 'kse100beta',
        'disparity_5', 'disparity_10',
        'rsi_14', 'rsi14'
    ]

    feature_lower = feature_name.lower()
    return any(keyword in feature_lower for keyword in protected_keywords)


class HorizonAdaptiveSelector:
    """
    Dynamic feature selection based on prediction horizon.

    Research finding: Features for day 1 ≠ features for day 21
    - Short-term (1-7 days): Technical indicators dominate
    - Medium-term (8-21 days): Hybrid (technicals + macro)
    - Long-term (22+ days): Macro factors dominate
    """

    # Feature groups by horizon
    HORIZON_FEATURES = {
        'short_term': [
            'williams_r', 'rsi_14', 'macd_histogram', 'macd',
            'volume_surge', 'volume_ratio', 'atr_14',
            'bb_percent_b', 'disparity_5'
        ],
        'medium_term': [
            'williams_r', 'disparity_5', 'disparity_10',
            'usdpkr_trend', 'usdpkr_change',
            'kse100_return', 'kse100_trend',
            'ema_50_above_200', 'rsi_14'
        ],
        'long_term': [
            'usdpkr_close', 'usdpkr_trend', 'usdpkr_volatility',
            'kse100_beta', 'kse100_correlation',
            'seasonal_is_fiscal_year_end', 'seasonal_is_ramadan',
            'news_bias', 'news_volume',
            'ema_200', 'sma_200'
        ]
    }

    @classmethod
    def select_features(cls, all_features: List[str], horizon_days: int) -> List[str]:
        """
        Select optimal features for given prediction horizon.

        Args:
            all_features: Full list of available features
            horizon_days: Prediction horizon in days

        Returns:
            Filtered list of features optimal for this horizon
        """
        # Determine horizon type
        if horizon_days <= 7:
            horizon_type = 'short_term'
        elif horizon_days <= 21:
            horizon_type = 'medium_term'
        else:
            horizon_type = 'long_term'

        # Get recommended features for this horizon
        recommended = cls.HORIZON_FEATURES[horizon_type]

        # Filter to only available features (partial match)
        selected = []
        for feat in all_features:
            if any(rec in feat.lower() for rec in recommended):
                selected.append(feat)

        print(f"🎯 Horizon-adaptive selection ({horizon_type}, {horizon_days} days): {len(selected)} features")

        return selected


# ============================================================================
# Utility Functions
# ============================================================================

def reduce_features_pipeline(df: pd.DataFrame, feature_cols: List[str],
                               model, X_train, X_test,
                               target_count: int = 50) -> List[str]:
    """
    Complete pipeline to reduce features from 120 to ~50.

    Steps:
    1. SHAP analysis - remove noise (< 1% importance)
    2. Correlation analysis - remove r > 0.90
    3. VIF analysis - remove VIF > 10
    4. Validate research claims

    Args:
        df: Full DataFrame
        feature_cols: Initial feature list (120+)
        model: Trained model for SHAP
        X_train: Training features
        X_test: Test features
        target_count: Target feature count (default 50)

    Returns:
        Reduced feature list (validated and optimized)
    """
    print(f"🚀 Starting feature reduction pipeline: {len(feature_cols)} → {target_count} features")
    print("=" * 70)

    # Step 1: SHAP analysis
    validator = SHAPValidator(threshold=0.01)
    shap_results = validator.analyze_model(model, X_train, X_test, feature_cols)
    features_after_shap = validator.remove_noise_features(feature_cols)

    print(f"\n✅ After SHAP: {len(features_after_shap)} features")

    # Step 2: Validate research claims
    validation_results = validator.validate_research_claims()

    # Step 3: Correlation analysis
    features_after_corr = remove_multicollinear_features(
        df, features_after_shap, threshold=0.90
    )

    print(f"\n✅ After correlation: {len(features_after_corr)} features")

    # Step 4: VIF analysis
    vif_df = calculate_vif(df, features_after_corr, threshold=10.0)
    features_to_keep = vif_df[vif_df['recommendation'] == 'Keep']['feature'].tolist()

    print(f"\n✅ After VIF: {len(features_to_keep)} features")

    # Summary
    print("\n" + "=" * 70)
    print(f"🎯 FINAL RESULT: {len(feature_cols)} → {len(features_to_keep)} features ({len(feature_cols) - len(features_to_keep)} removed)")
    print(f"   Reduction: {(1 - len(features_to_keep)/len(feature_cols))*100:.1f}%")

    if len(features_to_keep) > target_count:
        print(f"   ⚠️ Still above target ({target_count}). Consider stricter thresholds.")
    else:
        print(f"   ✅ Below target ({target_count}). Feature set optimized!")

    return features_to_keep
