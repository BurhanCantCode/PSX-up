#!/usr/bin/env python3
"""
Live smoke test for KSE100/index sentiment retrieval.

Usage:
  source venv/bin/activate
  python3 tests/live_kse100_sentiment_smoke.py --symbol KSE100 --min-news 1
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.sentiment_analyzer import get_stock_sentiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live KSE100 sentiment smoke test")
    parser.add_argument("--symbol", default="KSE100", help="Symbol to test (default: KSE100)")
    parser.add_argument(
        "--min-news",
        type=int,
        default=1,
        help="Minimum number of articles required to pass (default: 1)",
    )
    parser.add_argument(
        "--no-assert",
        action="store_true",
        help="Run probe without failing when result is below threshold",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    symbol = args.symbol.upper().strip()

    result = get_stock_sentiment(symbol, use_cache=False)

    news_count = int(result.get("news_count", 0))
    retrieval_mode = result.get("retrieval_mode", "unknown")
    sources_successful = int(result.get("sources_successful", 0))
    filtered_count = int(result.get("filtered_count", 0))

    print("\n" + "=" * 80)
    print("LIVE SENTIMENT SMOKE RESULT")
    print("=" * 80)
    print(f"Symbol: {symbol}")
    print(f"News count: {news_count}")
    print(f"Retrieval mode: {retrieval_mode}")
    print(f"Sources successful: {sources_successful}")
    print(f"Filtered count: {filtered_count}")
    print("Top titles:")
    for item in result.get("news_items", [])[:5]:
        source = item.get("source_name", item.get("source", "Unknown"))
        title = item.get("title", "")
        print(f"- [{source}] {title[:140]}")

    if args.no_assert:
        return 0

    if news_count < args.min_news:
        print(
            f"\nFAIL: expected at least {args.min_news} news items, got {news_count}."
        )
        return 1

    print(f"\nPASS: news_count ({news_count}) >= min_news ({args.min_news})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
