#!/usr/bin/env python3
"""
Helpers for energy-shock news parsing and feature extraction.

This module centralizes:
  - domestic fuel-price headline parsing
  - circular-debt signal detection
  - energy supply shock detection
  - date-aware projection of news events onto trading dates
"""

from __future__ import annotations

import bisect
import re
from datetime import date, datetime
from typing import Dict, Iterable, List, Optional

ENERGY_SHOCK_SYMBOLS = frozenset({"OGDC", "PPL", "POL", "MARI", "PSO"})
ENERGY_SHOCK_SECTORS = frozenset({"exploration_production", "oil_marketing"})

FUEL_CONTEXT_TERMS = (
    "petrol",
    "diesel",
    "hsd",
    "fuel",
    "petroleum product",
    "petroleum products",
)
FUEL_UP_TERMS = (
    "hike",
    "hiked",
    "increase",
    "increases",
    "increased",
    "raise",
    "raises",
    "raised",
    "surge",
    "surges",
    "higher",
    "up by",
)
FUEL_DOWN_TERMS = (
    "cut",
    "cuts",
    "cutting",
    "reduce",
    "reduces",
    "reduced",
    "slash",
    "slashes",
    "slashed",
    "decline",
    "declines",
    "declined",
    "lower",
    "down by",
)

CIRCULAR_DEBT_CONTEXT_TERMS = (
    "circular debt",
    "receivable",
    "receivables",
    "payment release",
    "payment releases",
    "settlement",
    "settles",
    "arrears",
    "dues",
)
CIRCULAR_DEBT_POSITIVE_TERMS = (
    "relief",
    "settlement",
    "settled",
    "settles",
    "payment release",
    "payment released",
    "payments released",
    "cleared",
    "clearance",
    "resolved",
    "resolve",
    "equity swap",
    "cash injection",
    "recovery plan",
)
CIRCULAR_DEBT_NEGATIVE_TERMS = (
    "worsens",
    "worsening",
    "surges",
    "rises",
    "mounts",
    "delay",
    "delayed",
    "stalled",
    "unpaid",
    "backlog",
    "arrears rise",
    "receivables surge",
)

ENERGY_SUPPLY_SHOCK_TERMS = (
    "blockade",
    "blocked shipping lane",
    "shipping disruption",
    "shipping disrupted",
    "strait of hormuz",
    "hormuz closure",
    "hormuz close",
    "hormuz shut",
    "closure threat",
    "middle east war pushes energy prices higher",
    "energy prices higher",
    "oil prices surge amid war",
    "war pushes energy prices higher",
    "supply disruption",
    "energy embargo",
)
REGIONAL_WAR_TERMS = (
    "war",
    "conflict",
    "middle east",
    "iran",
    "israel",
    "tehran",
    "gulf",
    "strike",
    "airstrike",
    "missile",
    "retaliation",
    "military",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def extract_event_text(item: Dict) -> str:
    return normalize_text(
        " ".join(
            [
                item.get("title", ""),
                item.get("description", ""),
                item.get("summary", ""),
            ]
        )
    )


def parse_event_date(date_str: str) -> Optional[date]:
    if not date_str:
        return None
    raw = str(date_str).strip()
    for fmt in ("%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"):
        try:
            return datetime.strptime(raw[: len(datetime.now().strftime(fmt))], fmt).date()
        except Exception:
            continue
    try:
        return datetime.strptime(raw[:10], "%Y-%m-%d").date()
    except Exception:
        return None


def _extract_rupee_values(text: str) -> List[float]:
    matches = re.findall(r"(?:rs\.?|rupees?)\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if not matches:
        matches = re.findall(r"(\d+(?:\.\d+)?)\s*(?:rs\.?|rupees?)", text, flags=re.IGNORECASE)
    values: List[float] = []
    for match in matches:
        try:
            values.append(float(match))
        except Exception:
            continue
    return values


def parse_local_fuel_price_delta(text: str) -> Optional[float]:
    text_norm = normalize_text(text)
    if not any(term in text_norm for term in FUEL_CONTEXT_TERMS):
        return None

    direction = 0
    if any(term in text_norm for term in FUEL_UP_TERMS):
        direction = 1
    elif any(term in text_norm for term in FUEL_DOWN_TERMS):
        direction = -1

    if direction == 0:
        return None

    values = _extract_rupee_values(text_norm)
    if not values:
        return None

    magnitude = max(values)
    return round(direction * magnitude, 2)


def score_circular_debt_signal(text: str) -> int:
    text_norm = normalize_text(text)
    if not any(term in text_norm for term in CIRCULAR_DEBT_CONTEXT_TERMS):
        return 0

    positive = any(term in text_norm for term in CIRCULAR_DEBT_POSITIVE_TERMS)
    negative = any(term in text_norm for term in CIRCULAR_DEBT_NEGATIVE_TERMS)
    if positive and not negative:
        return 1
    if negative and not positive:
        return -1
    return 0


def is_energy_supply_shock_text(text: str) -> bool:
    text_norm = normalize_text(text)
    return any(term in text_norm for term in ENERGY_SUPPLY_SHOCK_TERMS)


def has_regional_war_terms(text: str) -> bool:
    text_norm = normalize_text(text)
    return any(term in text_norm for term in REGIONAL_WAR_TERMS)


def map_event_date_to_trading_date(event_date: Optional[date], trading_dates: Iterable[date]) -> Optional[date]:
    ordered = sorted(trading_dates)
    if not ordered:
        return None
    if event_date is None:
        return ordered[-1]
    if event_date <= ordered[0]:
        return ordered[0]
    if event_date >= ordered[-1]:
        return ordered[-1]

    idx = bisect.bisect_right(ordered, event_date) - 1
    idx = max(0, min(idx, len(ordered) - 1))
    return ordered[idx]


def build_energy_event_feature_frame(stock_df: pd.DataFrame, news_items: List[Dict], symbol: Optional[str] = None) -> pd.DataFrame:
    """
    Project energy-relevant news events onto the stock's trading calendar.

    News newer than the latest trading date is attached to the last available
    trading session so weekend/news-offday events still influence the run.
    """
    import pandas as pd

    if "Date" not in stock_df.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(stock_df["Date"], errors="coerce").dt.normalize()
    if dates.isna().all():
        return pd.DataFrame()

    frame = pd.DataFrame({"Date": dates})
    frame["local_fuel_price_delta_rs"] = 0.0
    frame["local_fuel_price_shock"] = 0
    frame["circular_debt_signal"] = 0.0
    frame["geo_shock_signal"] = 0

    trading_dates = [d.date() for d in frame["Date"].dropna().tolist()]
    date_to_index = {d.date(): idx for idx, d in enumerate(frame["Date"].tolist()) if pd.notna(d)}

    for item in news_items or []:
        event_text = extract_event_text(item)
        mapped_date = map_event_date_to_trading_date(
            parse_event_date(item.get("date") or item.get("published", "")),
            trading_dates,
        )
        if mapped_date is None or mapped_date not in date_to_index:
            continue

        idx = date_to_index[mapped_date]
        delta = parse_local_fuel_price_delta(event_text)
        if delta is not None and abs(delta) > abs(float(frame.at[idx, "local_fuel_price_delta_rs"])):
            frame.at[idx, "local_fuel_price_delta_rs"] = delta

        circular_signal = score_circular_debt_signal(event_text)
        if circular_signal:
            current = float(frame.at[idx, "circular_debt_signal"])
            frame.at[idx, "circular_debt_signal"] = max(-1.0, min(1.0, current + circular_signal))

        if is_energy_supply_shock_text(event_text):
            frame.at[idx, "geo_shock_signal"] = 1

    frame["local_fuel_price_shock"] = (
        frame["local_fuel_price_delta_rs"].abs() >= 10.0
    ).astype(int)

    oil_change = pd.to_numeric(stock_df.get("oil_change", 0), errors="coerce").fillna(0.0)
    kse_return = pd.to_numeric(stock_df.get("kse100_return", 0), errors="coerce").fillna(0.0)

    frame["energy_shock_regime"] = (
        (frame["geo_shock_signal"] == 1)
        | (frame["local_fuel_price_shock"] == 1)
        | (oil_change.abs() >= 0.05)
    ).astype(int)
    frame["kse_oil_interaction"] = kse_return * oil_change
    frame["kse_energy_shock_interaction"] = kse_return * frame["energy_shock_regime"]

    if symbol and symbol.upper() not in ENERGY_SHOCK_SYMBOLS:
        frame["local_fuel_price_delta_rs"] = 0.0
        frame["local_fuel_price_shock"] = 0
        frame["circular_debt_signal"] = 0.0
        frame["geo_shock_signal"] = 0
        frame["energy_shock_regime"] = (oil_change.abs() >= 0.05).astype(int)
        frame["kse_energy_shock_interaction"] = kse_return * frame["energy_shock_regime"]

    return frame
