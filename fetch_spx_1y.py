#!/usr/bin/env python3
"""
Fetch SPX (^GSPC) intraday 5-minute OHLCV and save to CSV.
Note: Yahoo only provides 5m data for the last 60 days. Older history is not available for intraday.
Run: python3 fetch_spx_1y.py          (last 60 days of 5m bars)
     python3 fetch_spx_1y.py 30       (last 30 days)
Output: spx_5m_60d.csv — one row per 5-min bar; columns include Datetime, Open, High, Low, Close, Volume.
"""
import sys
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd

TICKER = "^GSPC"
DEFAULT_DAYS = 60  # Yahoo intraday 5m limit is ~60 days; use single request
OUTPUT_PREFIX = "spx_5m"


def fetch_5m_chunk(start: datetime, end: datetime) -> pd.DataFrame:
    """Fetch one chunk of 5m data. yfinance allows ~60 days per request for intraday."""
    d = yf.download(
        TICKER,
        start=start,
        end=end,
        interval="5m",
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if d.empty or len(d) < 2:
        return pd.DataFrame()
    if isinstance(d.columns, pd.MultiIndex):
        d = d.copy()
        d.columns = d.columns.get_level_values(0)
    if isinstance(d.index, pd.MultiIndex):
        d.index = d.index.get_level_values(0)
    return d


def fetch_5m_spx(days: int = 60) -> pd.DataFrame:
    """Fetch 5-minute SPX for the last `days` days. Yahoo allows max ~60 days for 5m."""
    if days > 60:
        print("Note: Yahoo limits 5m data to last 60 days. Using 60 days.")
        days = 60
    end = datetime.now()
    start = end - timedelta(days=days)
    print(f"  Requesting {start.date()} to {end.date()}...")
    return fetch_5m_chunk(start, end)


def main():
    days = DEFAULT_DAYS
    if len(sys.argv) > 1:
        try:
            days = int(sys.argv[1])
            if days < 1:
                days = DEFAULT_DAYS
        except ValueError:
            days = DEFAULT_DAYS

    print(f"Fetching SPX 5-minute data for the last {days} day(s)...")
    df = fetch_5m_spx(days)
    if df.empty:
        print("No data returned. Check connection or ticker.")
        sys.exit(1)

    # Ensure index has a name for CSV (datetime)
    df.index.name = "Datetime"
    out_file = f"{OUTPUT_PREFIX}_{days}d.csv"
    df.to_csv(out_file)
    print(f"Saved {len(df)} rows (5-min bars) to {out_file}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
