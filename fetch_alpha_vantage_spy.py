"""
Fetch SPY data from Alpha Vantage and save to CSV.

Default: 5-minute intraday for the last 1–2 months (INTRADAY_EXTENDED, 2 slices).
Use --daily for daily data (last 2 months).

Usage:
  export ALPHA_VANTAGE_API_KEY=your_key
  python fetch_alpha_vantage_spy.py

  python fetch_alpha_vantage_spy.py --apikey YOUR_API_KEY
  python fetch_alpha_vantage_spy.py --daily   # daily instead of 5min

Output: spy_alpha_vantage_5m_2m.csv (5min) or spy_alpha_vantage_2m.csv (daily)
"""
import argparse
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests


def get_api_key():
    parser = argparse.ArgumentParser(description="Download SPY data from Alpha Vantage (default: 5min intraday, last 2 months).")
    parser.add_argument("--apikey", type=str, help="Alpha Vantage API key (or set ALPHA_VANTAGE_API_KEY)")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (default: spy_alpha_vantage_5m_2m.csv or ..._2m.csv for daily)")
    parser.add_argument("--symbol", type=str, default="SPY", help="Symbol (default: SPY)")
    parser.add_argument("--daily", action="store_true", help="Fetch daily data instead of 5-minute intraday")
    parser.add_argument("--adjusted", action="store_true", help="[Daily only] TIME_SERIES_DAILY_ADJUSTED (premium)")
    parser.add_argument("--full", action="store_true", help="[Daily only] Full history (premium)")
    parser.add_argument("--months", type=int, default=2, help="For 5min: number of months (1 or 2, default 2)")
    args = parser.parse_args()
    key = args.apikey or os.environ.get("ALPHA_VANTAGE_API_KEY")
    if not key:
        print("Error: Set ALPHA_VANTAGE_API_KEY or pass --apikey YOUR_KEY", file=sys.stderr)
        sys.exit(1)
    out = args.output or ("spy_alpha_vantage_2m.csv" if args.daily else "spy_alpha_vantage_5m_2m.csv")
    return key, out, args.symbol, args.daily, args.adjusted, args.full, args.months


def fetch_intraday_extended(apikey: str, symbol: str, interval: str = "5min", slices: list = None) -> pd.DataFrame:
    """Fetch intraday extended CSV (one slice per request). slice: year1month1 = latest, year1month2 = previous, ..."""
    if slices is None:
        slices = ["year1month1", "year1month2"]
    all_dfs = []
    for slice_name in slices:
        url = (
            "https://www.alphavantage.co/query"
            "?function=TIME_SERIES_INTRADAY_EXTENDED"
            f"&symbol={symbol}"
            f"&interval={interval}"
            f"&slice={slice_name}"
            f"&apikey={apikey}"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        if r.text.strip().startswith("{"):
            j = r.json()
            msg = j.get("Information", j.get("Error Message", str(j)))
            raise RuntimeError(msg)
        df = pd.read_csv(pd.io.common.BytesIO(r.content))
        if df.empty:
            continue
        all_dfs.append(df)
    if not all_dfs:
        raise RuntimeError("No intraday data returned")
    return pd.concat(all_dfs, ignore_index=True)


def fetch_intraday(apikey: str, symbol: str, interval: str = "5min", full: bool = True) -> pd.DataFrame:
    """Single-call intraday. full=True => ~1 month for 5min (may be premium); full=False => compact = last 100 bars."""
    outputsize = "full" if full else "compact"
    url = (
        "https://www.alphavantage.co/query"
        "?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}"
        f"&interval={interval}"
        f"&outputsize={outputsize}"
        "&datatype=csv"
        f"&apikey={apikey}"
    )
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    if r.text.strip().startswith("{"):
        j = r.json()
        msg = j.get("Information", j.get("Error Message", str(j)))
        raise RuntimeError(msg)
    return pd.read_csv(pd.io.common.BytesIO(r.content))


def fetch_polygon_5m(symbol: str, months: int = 2) -> pd.DataFrame:
    """Fetch 5-min bars from Polygon.io (free tier). Ticker SPY or I:SPX. Returns DataFrame with Datetime index."""
    key = os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY") or ""
    if not key:
        return pd.DataFrame()
    # Polygon: SPY or I:SPX
    ticker = "I:SPX" if symbol.upper() in ("SPX", "I:SPX") else symbol.upper()
    to_d = datetime.now()
    from_d = to_d - timedelta(days=min(months * 31, 365))
    from_s = from_d.strftime("%Y-%m-%d")
    to_s = to_d.strftime("%Y-%m-%d")
    url = f"https://api.polygon.io/v2/aggs/ticker/{ticker}/range/5/minute/{from_s}/{to_s}?apiKey={key}&limit=50000"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    results = data.get("results") or []
    if not results:
        return pd.DataFrame()
    rows = []
    for res in results:
        t_ms = res.get("t")
        if t_ms is None:
            continue
        ts = pd.Timestamp(t_ms, unit="ms")
        rows.append({
            "Datetime": ts,
            "Open": res.get("o"),
            "High": res.get("h"),
            "Low": res.get("l"),
            "Close": res.get("c"),
            "Volume": res.get("v", 0),
        })
    df = pd.DataFrame(rows)
    df = df.sort_values("Datetime").reset_index(drop=True)
    return df


def fetch_daily(apikey: str, symbol: str = "SPY", adjusted: bool = False, full: bool = False) -> pd.DataFrame:
    """Fetch daily (or daily adjusted) CSV. full=True requires premium; default compact = last 100 points."""
    func = "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY"
    outputsize = "full" if full else "compact"
    url = (
        "https://www.alphavantage.co/query"
        f"?function={func}"
        f"&symbol={symbol}"
        f"&outputsize={outputsize}"
        "&datatype=csv"
        f"&apikey={apikey}"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    # API returns CSV when datatype=csv
    if r.text.strip().startswith("{"):
        try:
            j = r.json()
            if "Information" in j:
                raise RuntimeError(j["Information"])
            if "Error Message" in j:
                raise RuntimeError(j["Error Message"])
        except Exception as e:
            raise RuntimeError(f"API returned JSON error: {e}") from e
    df = pd.read_csv(pd.io.common.BytesIO(r.content))
    return df


def _normalize_columns(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    """Alpha Vantage CSV: 'timestamp'/'date'/'time', '1. open' or 'open', etc. Intraday extended uses 'time'."""
    rename = {}
    for c in df.columns:
        s = c.strip().lower()
        if s in ("timestamp", "date", "time"):
            rename[c] = "Datetime"
        elif s.startswith("1.") or s == "open":
            rename[c] = "Open"
        elif s.startswith("2.") or s == "high":
            rename[c] = "High"
        elif s.startswith("3.") or s == "low":
            rename[c] = "Low"
        elif s.startswith("4.") or s == "close":
            rename[c] = "Close"
        elif "adjusted" in s or (s.startswith("5.") and "adjusted" in c):
            rename[c] = "Adjusted Close"
        elif s.startswith("6.") or s == "volume" or (s.startswith("5.") and "volume" in c.lower()):
            rename[c] = "Volume"
    df = df.rename(columns={k: v for k, v in rename.items()})
    if intraday and "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"])
    return df


def main() -> None:
    apikey, output_path, symbol, daily, adjusted, full, months = get_api_key()

    if daily:
        print(f"Fetching {symbol} daily {'adjusted ' if adjusted else ''}from Alpha Vantage...")
        df = fetch_daily(apikey, symbol, adjusted=adjusted, full=full)
        df = _normalize_columns(df, intraday=False)
        date_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        cutoff = pd.Timestamp.now().normalize() - timedelta(days=60)
        ser = df[date_col]
        if hasattr(ser.dt, "tz") and ser.dt.tz is not None:
            cutoff = cutoff.tz_localize(ser.dt.tz)
        df = df.loc[df[date_col] >= cutoff].copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        out_cols = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        if "Adjusted Close" in df.columns and "Close" in out_cols:
            df["Close"] = df["Adjusted Close"]
        out = df[[c for c in out_cols if c in df.columns]].copy()
    else:
        # 5-minute intraday: try Alpha Vantage (full then compact), then Polygon.io if available
        df = None
        av_error = None
        print(f"Fetching {symbol} 5-minute intraday (target: last {months} month(s))...")
        try:
            df = fetch_intraday(apikey, symbol, interval="5min", full=True)
            print("  Using Alpha Vantage (full).")
        except RuntimeError as e:
            av_error = e
            err = str(e).lower()
            if "premium" in err or "full" in err or "rate" in err or "spreading" in err:
                time.sleep(2)
                try:
                    df = fetch_intraday(apikey, symbol, interval="5min", full=False)
                    print("  Using Alpha Vantage (compact, last 100 bars).")
                except RuntimeError:
                    pass
        if df is None or df.empty:
            print("  Alpha Vantage intraday not available (premium/rate limit). Trying Polygon.io...")
            df = fetch_polygon_5m(symbol, months=months)
            if df is not None and not df.empty:
                print(f"  Using Polygon.io ({len(df)} bars).")
        if df is None or df.empty:
            print("  No 5-min source available. Use --daily for free daily data, or set POLYGON_API_KEY for 5min via Polygon.")
            sys.exit(1)
        df = _normalize_columns(df, intraday=True)
        date_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
        if date_col not in df.columns:
            date_col = df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        # Filter to last N days if we have more (e.g. from Polygon)
        days = 60 if months >= 2 else 31
        cutoff = pd.Timestamp.now() - timedelta(days=days)
        ser = df[date_col]
        if hasattr(ser.dt, "tz") and ser.dt.tz is not None:
            cutoff = cutoff.tz_localize(ser.dt.tz) if getattr(cutoff, "tzinfo", None) is None else cutoff
        else:
            cutoff = cutoff.tz_localize(None) if hasattr(cutoff, "tz_localize") else cutoff
        df = df.loc[df[date_col] >= cutoff].copy()
        df = df.sort_values(date_col).reset_index(drop=True)
        out_cols = [c for c in ["Datetime", "Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[[c for c in out_cols if c in df.columns]].copy()

    out.to_csv(output_path, index=False)
    print(f"Saved ({len(out)} rows) to {output_path}")
    return


if __name__ == "__main__":
    main()
