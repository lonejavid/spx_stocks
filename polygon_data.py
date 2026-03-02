"""
Polygon.io data fetcher for SPX 5-minute bars.
Uses REST API: /v2/aggs/ticker/I:SPX/range/5/minute/{from}/{to}
"""
import os
import time
from datetime import datetime, timedelta
import pandas as pd
import pytz

EST = pytz.timezone("America/New_York")
POLYGON_BASE = "https://api.polygon.io"
SPX_TICKER = "I:SPX"


def get_polygon_api_key() -> str:
    """Read API key from environment. Prefer POLYGON_API_KEY; accept MASSIVE key env if set."""
    return (os.environ.get("POLYGON_API_KEY") or os.environ.get("MASSIVE_API_KEY") or "").strip()


def fetch_spx_5m(from_date: str, to_date: str, api_key: str = None) -> pd.DataFrame:
    """
    Fetch 5-minute OHLCV bars for SPX from Polygon.io.
    from_date / to_date: YYYY-MM-DD (inclusive of trading session dates).
    Returns DataFrame with DatetimeIndex in ET and columns Open, High, Low, Close, Volume.
    Empty DataFrame on error or no data.
    """
    key = api_key or get_polygon_api_key()
    if not key:
        return pd.DataFrame()
    url = f"{POLYGON_BASE}/v2/aggs/ticker/{SPX_TICKER}/range/5/minute/{from_date}/{to_date}"
    try:
        from urllib.request import Request, urlopen
        from urllib.error import URLError, HTTPError
        req = Request(f"{url}?apiKey={key}", headers={"User-Agent": "SPX-Dashboard/1.0"})
        with urlopen(req, timeout=15) as resp:
            data = __import__("json").loads(resp.read().decode())
    except Exception:
        return pd.DataFrame()
    results = data.get("results") or []
    if not results:
        return pd.DataFrame()
    rows = []
    index_vals = []
    for r in results:
        t_ms = r.get("t")
        if t_ms is None:
            continue
        ts = pd.Timestamp(t_ms, unit="ms")
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC").tz_convert(EST)
        index_vals.append(ts)
        rows.append({
            "Open": r.get("o"),
            "High": r.get("h"),
            "Low": r.get("l"),
            "Close": r.get("c"),
            "Volume": r.get("v", 0),
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(index_vals))
    df = df.sort_index()
    return df


def fetch_spx_5m_with_retry(from_date: str, to_date: str, api_key: str = None, max_retries: int = 2) -> pd.DataFrame:
    """Fetch SPX 5m with one retry after 2s on failure."""
    key = api_key or get_polygon_api_key()
    for attempt in range(max_retries):
        df = fetch_spx_5m(from_date, to_date, key)
        if not df.empty and len(df) >= 2:
            return df
        if attempt < max_retries - 1:
            time.sleep(2)
    return df if not df.empty else pd.DataFrame()
