#!/usr/bin/env python3
"""
Geopolitical feature scaffolding for flagged shadow rollouts.

This module is intentionally low-risk:
- Reads existing sentiment/news output
- Computes compact risk signals
- Defaults to neutral values on any failure
"""

from __future__ import annotations

from typing import Dict, List


GEO_TERMS = {
    "conflict": ["war", "conflict", "attack", "strike", "escalation", "border tension"],
    "energy_supply": ["oil", "gas", "shipping lane", "strait", "supply disruption"],
    "regional": ["iran", "israel", "gulf", "middle east", "afghanistan", "pakistan border"],
    "risk_off": ["sanctions", "safe haven", "global risk", "sell-off", "volatility spike"],
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _term_score(text: str, terms: List[str]) -> float:
    hits = sum(1 for t in terms if t in text)
    return _clamp01(hits / max(1, len(terms) * 0.35))


def neutral_geopolitical_features() -> Dict[str, float]:
    return {
        "geo_conflict_risk": 0.0,
        "geo_energy_supply_risk": 0.0,
        "geo_regional_tension": 0.0,
        "geo_global_risk_off": 0.0,
        "geo_news_volume": 0.0,
    }


def get_geopolitical_features_from_news(news_items: List[Dict]) -> Dict[str, float]:
    if not news_items:
        return neutral_geopolitical_features()

    text = " ".join((item.get("title") or "").lower() for item in news_items[:30])
    volume = _clamp01(len(news_items) / 20.0)

    features = {
        "geo_conflict_risk": _term_score(text, GEO_TERMS["conflict"]),
        "geo_energy_supply_risk": _term_score(text, GEO_TERMS["energy_supply"]),
        "geo_regional_tension": _term_score(text, GEO_TERMS["regional"]),
        "geo_global_risk_off": _term_score(text, GEO_TERMS["risk_off"]),
        "geo_news_volume": volume,
    }
    return features


def get_geopolitical_features_for_symbol(symbol: str, use_cache: bool = True) -> Dict[str, float]:
    """
    Pull minimal geopolitical signals from existing sentiment pipeline.
    Returns neutral values on failures by design.
    """
    try:
        from backend.sentiment_analyzer import get_stock_sentiment
    except Exception:
        return neutral_geopolitical_features()

    try:
        sentiment = get_stock_sentiment(symbol, use_cache=use_cache)
        news_items = sentiment.get("news_items", [])
        return get_geopolitical_features_from_news(news_items)
    except Exception:
        return neutral_geopolitical_features()
