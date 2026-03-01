#!/usr/bin/env python3
"""
Live scraper probe for debugging extraction quality per source/query.

Usage:
  python3 tests/live_scrape_probe.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from pprint import pprint

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.enhanced_news_fetcher import (
    NEWS_SOURCES,
    fetch_news_curl_with_status,
    extract_articles_from_html,
    score_index_relevance,
)


QUERIES = [
    "PSX market today",
    "KSE-100 index",
    "Pakistan stocks",
    "SBP policy rate",
]


def snippet(s: str, n: int = 200) -> str:
    s = re.sub(r"\s+", " ", s)
    return s[:n]


def main() -> None:
    print("=" * 80)
    print("LIVE SCRAPE PROBE")
    print("=" * 80)

    for source, cfg in NEWS_SOURCES.items():
        print(f"\n\nSOURCE: {source}")
        print("-" * 80)

        for q in QUERIES:
            url = cfg["search_url"].format(q.replace(" ", "+"))
            fetched = fetch_news_curl_with_status(url, timeout=12)
            html = fetched.get("html", "")
            status = fetched.get("status")
            anchors = len(re.findall(r"<a[^>]+href=", html, re.IGNORECASE))
            parsed = extract_articles_from_html(html, source.replace("_", " ").title())

            print(f"\nQuery: {q}")
            print(f"Status: {status} | HTML bytes: {len(html)} | anchors: {anchors} | parsed: {len(parsed)}")

            if len(parsed) == 0:
                # print diagnostic snippets
                title_tags = re.findall(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
                if title_tags:
                    print(f"Page title: {snippet(title_tags[0], 120)}")
                # first few raw href text snippets
                raw_link_pairs = re.findall(
                    r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>',
                    html,
                    re.IGNORECASE | re.DOTALL,
                )
                print(f"Raw links sampled: {len(raw_link_pairs[:5])}")
                for i, (href, text) in enumerate(raw_link_pairs[:5], start=1):
                    clean_text = re.sub(r"<[^>]+>", " ", text)
                    clean_text = re.sub(r"\s+", " ", clean_text).strip()
                    print(f"  {i}. href={href[:120]} | text={snippet(clean_text, 120)}")
            else:
                top = []
                for art in parsed[:5]:
                    sc = score_index_relevance(art.get("title", ""), art.get("url", ""))
                    top.append(
                        {
                            "title": art.get("title", "")[:90],
                            "url": art.get("url", "")[:110],
                            "score": sc,
                        }
                    )
                pprint(top)

        # Also probe fallback URL
        fallback = cfg.get("fallback_url")
        if fallback:
            fetched = fetch_news_curl_with_status(fallback, timeout=12)
            html = fetched.get("html", "")
            parsed = extract_articles_from_html(html, source.replace("_", " ").title())
            print(f"\nFallback URL: {fallback}")
            print(
                f"Status: {fetched.get('status')} | HTML bytes: {len(html)} | "
                f"anchors: {len(re.findall(r'<a[^>]+href=', html, re.IGNORECASE))} | parsed: {len(parsed)}"
            )
            if parsed:
                for art in parsed[:3]:
                    print(f"  - {art.get('title', '')[:110]}")


if __name__ == "__main__":
    main()
