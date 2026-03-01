#!/usr/bin/env python3
"""
Iterative parser tuning on live pages before production changes.

Run:
  python3 tests/live_parser_iteration.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.enhanced_news_fetcher import (  # noqa: E402
    NEWS_SOURCES,
    fetch_news_curl_with_status,
    extract_articles_from_html,
    score_index_relevance,
)


QUERY = "PSX market today"

SOURCE_BASE = {
    "business_recorder": "https://www.brecorder.com",
    "dawn": "https://www.dawn.com",
    "pakistan_today": "https://www.pakistantoday.com.pk",
    "express_tribune": "https://tribune.com.pk",
    "geo_news": "https://www.geo.tv",
    "minute_mirror": "https://minutemirror.com.pk",
}


def clean_text(x: str) -> str:
    x = re.sub(r"<script.*?</script>", " ", x, flags=re.I | re.S)
    x = re.sub(r"<style.*?</style>", " ", x, flags=re.I | re.S)
    x = re.sub(r"<[^>]+>", " ", x)
    x = html.unescape(x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def normalize_href(source: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return f"https:{href}"
    if href.startswith("/"):
        return SOURCE_BASE.get(source, "") + href
    return href


def likely_article_link(source: str, url: str, title: str) -> bool:
    u = url.lower()
    t = title.lower()
    p = urlparse(url)
    path = p.path.lower()

    if not url or url.startswith("#"):
        return False
    if any(x in u for x in ["login", "signup", "privacy", "about", "contact", "profile"]):
        return False
    if any(x in t for x in ["ramazan calendar", "weather forecast", "geo tv satellite parameters"]):
        return False

    # Source-specific article path hints
    if source == "dawn":
        return "/news/" in path
    if source == "express_tribune":
        return "/story/" in path
    if source == "business_recorder":
        return "/news/" in path or "/company/" in path or "/markets/" in path
    if source == "geo_news":
        return "/latest/" in path or "/category/" in path
    if source == "minute_mirror":
        # most article URLs end with trailing numeric id
        if re.search(r"-\d+/?$", path):
            return True
    if source == "pakistan_today":
        return re.search(r"/\d{4}/\d{2}/\d{2}/", path) is not None or "/category/business" in path

    # generic fallback
    return len(path) > 8 and not path.endswith("/")


def candidate_extract(html_text: str, source: str):
    out = []
    seen = set()
    matches = re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>.*?</a>', html_text, flags=re.I | re.S)
    for m in matches:
        whole = m.group(0)
        href = m.group(1)
        inner = re.sub(r'^<a[^>]*>|</a>$', '', whole, flags=re.I | re.S)
        title = clean_text(inner)
        if not title:
            m_title = re.search(r'\btitle="([^"]+)"', whole, flags=re.I | re.S)
            if m_title:
                title = clean_text(m_title.group(1))
        if not title:
            m_alt = re.search(r'\balt="([^"]+)"', whole, flags=re.I | re.S)
            if m_alt:
                title = clean_text(m_alt.group(1))
        href_norm = normalize_href(source, href.strip())
        if len(title) < 25 or len(title) > 220:
            continue
        if title.lower() in seen:
            continue
        if not likely_article_link(source, href_norm, title):
            continue
        score = score_index_relevance(title, href_norm)
        out.append(
            {
                "title": title,
                "url": href_norm,
                "score": score,
            }
        )
        seen.add(title.lower())
        if len(out) >= 20:
            break
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def main():
    print("=" * 100)
    print("LIVE PARSER ITERATION")
    print("=" * 100)

    for source, cfg in NEWS_SOURCES.items():
        search_url = cfg["search_url"].format(QUERY.replace(" ", "+"))
        fetched = fetch_news_curl_with_status(search_url, timeout=12)
        html_text = fetched.get("html", "")
        status = fetched.get("status")

        cur = extract_articles_from_html(html_text, source.replace("_", " ").title())
        cand = candidate_extract(html_text, source)

        print(f"\nSOURCE={source} status={status} bytes={len(html_text)} current={len(cur)} candidate={len(cand)}")
        print("Top candidate:")
        for row in cand[:8]:
            print(f"  - [{row['score']:.1f}] {row['title'][:95]}")

        fallback = cfg.get("fallback_url")
        if fallback:
            if "{}" in fallback:
                fallback = fallback.format(QUERY.replace(" ", "+"))
            fetched_fb = fetch_news_curl_with_status(fallback, timeout=12)
            html_fb = fetched_fb.get("html", "")
            cand_fb = candidate_extract(html_fb, source)
            print(
                f"  fallback status={fetched_fb.get('status')} bytes={len(html_fb)} "
                f"candidate={len(cand_fb)} url={fallback}"
            )
            for row in cand_fb[:5]:
                print(f"    * [{row['score']:.1f}] {row['title'][:90]}")


if __name__ == "__main__":
    main()
