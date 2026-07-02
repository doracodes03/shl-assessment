"""
Fetches the SHL product catalog JSON and writes it to data/catalog.json.

Run this at build/deploy time (and on a schedule, e.g. nightly) so the
running service never makes a network call mid-conversation -- it only
reads the cached file. This keeps /chat fast and within the 30s-per-call
timeout regardless of upstream availability.

Usage:
    python scripts/fetch_catalog.py [--url URL] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import requests

DEFAULT_URL = "https://tcp-us-prod-rnd.shl.com/voiceRater/shl-ai-hiring/shl_product_catalog.json"
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "shl_product_catalog.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Fetching catalog from {args.url} ...", file=sys.stderr)
    resp = requests.get(args.url, timeout=60)
    resp.raise_for_status()

    # Some exported SHL catalog files contain control characters inside
    # string values which cause `response.json()` to raise. Save the raw
    # text and let the catalog loader (`app.catalog`) sanitize on read.
    raw_text = resp.text

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(raw_text)

    # Attempt to estimate number of top-level items for informational output
    try:
        items = json.loads(raw_text)
        count = len(items) if isinstance(items, list) else 0
    except Exception:
        count = 0

    print(f"Wrote raw catalog to {args.out} (items_estimate={count})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
