#!/usr/bin/env python3
"""
Test script for the Gainers logic. Run from project root:
  python3 test_gainers.py          # full S&P 500 (slow, 1–2 min)
  python3 test_gainers.py --fast   # small universe (quick, ~15 sec)

Checks:
  1. What get_top_gainers_sp500() returns (count, keys, sample row)
  2. Why rows might be skipped (symbol, price, filters)
  3. How many rows pass the same filters as pages/Gainers.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd

FAST = "--fast" in sys.argv
if FAST:
    # Use a small ticker list so the test finishes in ~15 seconds
    _ORIGINAL_GET_SP500 = None
    _FAST_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "JPM", "JNJ", "V", "WMT", "PG", "HD", "MA", "DIS"]


def _fmt_vol(v):
    if v is None or (isinstance(v, (int, float)) and (v != v or v < 0)):
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.2f}K"
    return str(int(v))


def main():
    import gainers_tv as gtv
    from gainers_tv import get_sp500_tickers

    if FAST:
        # Patch S&P 500 list to a small set for fast run
        gtv._SP500_TICKERS_CACHE = _FAST_TICKERS
        print("(Fast mode: using 15 tickers only)")
    else:
        # Ensure cache is loaded (or use fallback)
        get_sp500_tickers()

    get_top_gainers_sp500 = gtv.get_top_gainers_sp500

    print("=" * 60)
    print("GAINERS TEST — same logic as pages/Gainers.py")
    print("=" * 60)

    # 1) Backend: fetch gainers (no cache, direct call)
    limit = 20 if FAST else 30
    print(f"\n1) Calling get_top_gainers_sp500(limit={limit}, timeframe='1d') ...")
    gainers = get_top_gainers_sp500(limit=limit, timeframe="1d")
    print(f"   Returned: {len(gainers)} items")

    if not gainers:
        print("\n   >>> No gainers from backend. Possible causes:")
        print("      - yfinance not installed (pip install yfinance)")
        print("      - Network/rate limit (try again later)")
        print("      - get_sp500_tickers() returned empty")
        spx = get_sp500_tickers()
        print(f"      - get_sp500_tickers() has {len(spx)} tickers")
        return

    # 2) Inspect first item keys and sample values
    first = gainers[0]
    keys = list(first.keys())
    print(f"\n2) Keys in each item: {keys}")
    print(f"   Sample (first item): symbol={first.get('symbol')!r} price={first.get('price')!r} "
          f"change_pct={first.get('change_pct')!r} gap_pct={first.get('gap_pct')!r} "
          f"volume_ratio={first.get('volume_ratio')!r} scanner_score={first.get('scanner_score')!r}")

    # 3) Apply same row-building + filter logic as Gainers.py (min filters: 0, 1, 0)
    min_gap_filter = 0.0
    min_vol_ratio_filter = 1.0
    min_score_filter = 0.0

    rows = []
    skip_reasons = {"symbol": 0, "price": 0, "gap": 0, "vol_ratio": 0, "score": 0}

    for g in gainers:
        change_pct = g.get("change_pct")
        symbol = (g.get("symbol") or "—").strip()
        if not symbol or str(symbol).lower() in ("none", "nan"):
            skip_reasons["symbol"] += 1
            continue
        price_val = g.get("price")
        if price_val is None or (isinstance(price_val, (int, float)) and (price_val != price_val or price_val <= 0)):
            skip_reasons["price"] += 1
            continue
        gap_pct = g.get("gap_pct")
        gap_val = float(gap_pct) if gap_pct is not None and not (isinstance(gap_pct, float) and pd.isna(gap_pct)) else 0.0
        vol_ratio = g.get("volume_ratio")
        vol_ratio_val = float(vol_ratio) if vol_ratio is not None and not (isinstance(vol_ratio, float) and pd.isna(vol_ratio)) else None
        score_val = g.get("scanner_score")
        score_val = float(score_val) if score_val is not None and not (isinstance(score_val, float) and pd.isna(score_val)) else None

        if gap_val < min_gap_filter:
            skip_reasons["gap"] += 1
            continue
        if vol_ratio_val is not None and vol_ratio_val < min_vol_ratio_filter:
            skip_reasons["vol_ratio"] += 1
            continue
        if score_val is not None and score_val < min_score_filter:
            skip_reasons["score"] += 1
            continue

        intraday = g.get("intraday_move_pct")
        intraday_val = intraday if intraday is not None else 0.0
        rows.append({
            "Change From Close(%)": change_pct if change_pct is not None else 0.0,
            "Symbol": symbol,
            "Price": price_val,
            "Volume": _fmt_vol(g.get("volume")),
            "Volume Ratio": vol_ratio_val if vol_ratio_val is not None else 0.0,
            "Intraday Move %": intraday_val,
            "Score": score_val if score_val is not None else 0.0,
            "Gap(%)": gap_pct if gap_pct is not None else (change_pct if change_pct is not None else "—"),
        })

    print(f"\n3) Filter (Min Gap={min_gap_filter}, Min Vol Ratio={min_vol_ratio_filter}, Min Score={min_score_filter})")
    print(f"   Rows passed: {len(rows)}")
    print(f"   Skipped: symbol={skip_reasons['symbol']} price={skip_reasons['price']} "
          f"gap={skip_reasons['gap']} vol_ratio={skip_reasons['vol_ratio']} score={skip_reasons['score']}")

    # 4) Show first few passed rows
    if rows:
        print("\n4) First 5 rows that passed:")
        for i, r in enumerate(rows[:5]):
            print(f"   {i+1}. {r['Symbol']}  price={r['Price']}  change%={r['Change From Close(%)']}  "
                  f"gap%={r['Gap(%)']}  vol_ratio={r['Volume Ratio']}  score={r['Score']}")
    else:
        print("\n4) No rows passed. To show gainers in the app:")
        if skip_reasons["vol_ratio"]:
            print("   - All skipped by Min Volume Ratio: set Min Volume Ratio to 0 in the UI (common when market is closed).")
        if skip_reasons["score"]:
            print("   - All skipped by Min Score: set Min Score to 0 or lower.")
        if not (skip_reasons["vol_ratio"] or skip_reasons["score"]):
            print("   - Check symbol/price skip reasons above.")

    # 5) Count how many have volume_ratio / scanner_score missing
    missing_vr = sum(1 for g in gainers if g.get("volume_ratio") is None)
    missing_sc = sum(1 for g in gainers if g.get("scanner_score") is None)
    print(f"\n5) Backend data: items with volume_ratio missing={missing_vr}, scanner_score missing={missing_sc}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
