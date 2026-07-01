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
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print(f"Fetching catalog from {args.url} ...", file=sys.stderr)
    resp = requests.get(args.url, timeout=60)
    resp.raise_for_status()
    items = resp.json()

    if not isinstance(items, list) or not items:
        print("ERROR: fetched payload doesn't look like a non-empty list", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(items)} raw catalog items to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
