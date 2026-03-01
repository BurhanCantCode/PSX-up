#!/usr/bin/env python3
"""
Baseline freeze + drift check utility for prediction regression safety.

Usage:
  python backend/prediction_regression_check.py freeze
  python backend/prediction_regression_check.py compare
  python backend/prediction_regression_check.py compare --strict
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "prediction_logs"
BASELINE_FILE = LOG_DIR / "baseline_snapshot.json"
DRIFT_FILE = LOG_DIR / "drift_report.json"

DEFAULT_SYMBOLS = ["KSE100", "OGDC", "PPL", "LUCK", "HBL", "SYS"]


@dataclass
class Snapshot:
    symbol: str
    current_price: Optional[float]
    model: str
    generated_at: str
    day_1: Dict
    day_7: Dict
    day_21: Dict
    source_file: str

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "model": self.model,
            "generated_at": self.generated_at,
            "day_1": self.day_1,
            "day_7": self.day_7,
            "day_21": self.day_21,
            "source_file": self.source_file,
        }


def _load_json(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _get_prediction_file(symbol: str) -> Optional[Path]:
    candidates = [
        DATA_DIR / f"{symbol}_research_predictions_2026.json",
        DATA_DIR / f"{symbol}_sota_predictions_2026.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _direction_from_pred(pred: Dict) -> str:
    upside = float(pred.get("upside_potential", 0) or 0)
    if upside > 0:
        return "BULLISH"
    if upside < 0:
        return "BEARISH"
    return "NEUTRAL"


def _select_day(preds: List[Dict], day: int) -> Dict:
    if not preds:
        return {}
    for p in preds:
        if int(p.get("day", -1)) == day:
            return p
    idx = day - 1
    if 0 <= idx < len(preds):
        return preds[idx]
    return preds[-1]


def _normalize_point(pred: Dict) -> Dict:
    if not pred:
        return {}
    return {
        "date": pred.get("date"),
        "predicted_price": float(pred.get("predicted_price", 0) or 0),
        "upside_potential": float(pred.get("upside_potential", 0) or 0),
        "confidence": float(pred.get("confidence", 0) or 0),
        "raw_direction": _direction_from_pred(pred),
        "stable_direction": pred.get("stable_direction", _direction_from_pred(pred)),
    }


def build_snapshot(symbol: str) -> Optional[Snapshot]:
    pred_file = _get_prediction_file(symbol)
    if pred_file is None:
        return None

    payload = _load_json(pred_file)
    preds = payload.get("daily_predictions", [])
    if not preds:
        return None

    day_1 = _normalize_point(_select_day(preds, 1))
    day_7 = _normalize_point(_select_day(preds, 7))
    day_21 = _normalize_point(_select_day(preds, 21))

    return Snapshot(
        symbol=symbol,
        current_price=float(payload.get("current_price", 0) or 0) if "current_price" in payload else None,
        model=payload.get("model", "unknown"),
        generated_at=payload.get("generated_at", datetime.now().isoformat()),
        day_1=day_1,
        day_7=day_7,
        day_21=day_21,
        source_file=str(pred_file),
    )


def freeze(symbols: List[str]) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    snapshots: Dict[str, Dict] = {}
    missing: List[str] = []

    for symbol in symbols:
        snap = build_snapshot(symbol)
        if snap is None:
            missing.append(symbol)
            continue
        snapshots[symbol] = snap.to_dict()

    payload = {
        "frozen_at": datetime.now().isoformat(),
        "symbols_requested": symbols,
        "symbols_frozen": sorted(list(snapshots.keys())),
        "symbols_missing": missing,
        "snapshots": snapshots,
    }

    with open(BASELINE_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved baseline snapshot: {BASELINE_FILE}")
    print(f"Frozen: {len(snapshots)} | Missing: {len(missing)}")
    return 0 if snapshots else 1


def _pct_diff(old: float, new: float) -> float:
    if abs(old) < 1e-9:
        return 0.0 if abs(new) < 1e-9 else 100.0
    return ((new - old) / old) * 100.0


def compare(symbols: List[str], strict: bool, drift_threshold_pct: float = 0.5) -> int:
    if not BASELINE_FILE.exists():
        print(f"Baseline not found: {BASELINE_FILE}")
        return 1

    baseline = _load_json(BASELINE_FILE)
    baseline_snaps = baseline.get("snapshots", {})
    report: Dict[str, Dict] = {}
    failures: List[str] = []

    for symbol in symbols:
        old = baseline_snaps.get(symbol)
        new_snap = build_snapshot(symbol)
        if old is None or new_snap is None:
            report[symbol] = {"status": "missing"}
            failures.append(symbol)
            continue

        new = new_snap.to_dict()
        symbol_report: Dict[str, Dict] = {"status": "ok", "points": {}}
        for key in ["day_1", "day_7", "day_21"]:
            old_point = old.get(key, {})
            new_point = new.get(key, {})
            old_price = float(old_point.get("predicted_price", 0) or 0)
            new_price = float(new_point.get("predicted_price", 0) or 0)
            drift = _pct_diff(old_price, new_price)
            direction_match = old_point.get("raw_direction") == new_point.get("raw_direction")
            stable_match = old_point.get("stable_direction") == new_point.get("stable_direction")

            symbol_report["points"][key] = {
                "old_price": old_price,
                "new_price": new_price,
                "drift_pct": round(drift, 4),
                "direction_match": direction_match,
                "stable_direction_match": stable_match,
            }
            if abs(drift) > drift_threshold_pct:
                symbol_report["status"] = "drift_exceeded"

        if symbol_report["status"] != "ok":
            failures.append(symbol)
        report[symbol] = symbol_report

    output = {
        "compared_at": datetime.now().isoformat(),
        "baseline_file": str(BASELINE_FILE),
        "drift_threshold_pct": drift_threshold_pct,
        "strict": strict,
        "symbols": report,
        "failures": failures,
    }

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Saved drift report: {DRIFT_FILE}")
    print(f"Failures: {len(failures)}")

    if strict and failures:
        return 2
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prediction baseline freeze/drift checker")
    parser.add_argument("command", choices=["freeze", "compare"])
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_SYMBOLS),
        help="Comma-separated symbol list",
    )
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on drift/missing")
    parser.add_argument("--drift-threshold-pct", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.command == "freeze":
        return freeze(symbols)
    return compare(symbols, strict=args.strict, drift_threshold_pct=args.drift_threshold_pct)


if __name__ == "__main__":
    raise SystemExit(main())
