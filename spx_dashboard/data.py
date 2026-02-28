"""
Data fetching for SPX Day Trading Dashboard.
Uses yfinance. Handles market hours, previous trading day, and regular session filter.
"""
from datetime import datetime, timedelta, time as dt_time
import pandas as pd
import pytz
import yfinance as yf

EST = pytz.timezone("America/New_York")

TICKER_SPX = "^GSPC"
TICKER_VIX = "^VIX"
CONFIRMATION_TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "QQQ", "SPY", "^TNX"]


def _ensure_single_column(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    return df


def is_market_open() -> bool:
    """True if current ET is Mon–Fri 9:30 AM–4:00 PM."""
    now = datetime.now(EST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt_time(9, 30) <= t < dt_time(16, 0)


def get_display_date():
    """Return (display_date, label). When closed, display_date = previous full trading day."""
    now = datetime.now(EST)
    today = now.date()
    t = now.time()
    if today.weekday() >= 5:
        days_back = today.weekday() - 4
        prev = today - timedelta(days=days_back)
        return prev, f"Previous trading day ({prev.isoformat()})"
    if t < dt_time(9, 30):
        prev = today - timedelta(days=3) if today.weekday() == 0 else today - timedelta(days=1)
        return prev, f"Previous trading day ({prev.isoformat()})"
    if t >= dt_time(16, 0):
        return today, f"Previous trading day ({today.isoformat()})"
    return today, "Today"


def _filter_regular_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 9:30–16:00 ET."""
    if df.empty:
        return df
    idx = pd.DatetimeIndex(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize("America/New_York", ambiguous="infer")
    else:
        idx = idx.tz_convert("America/New_York")
    start, end = dt_time(9, 30), dt_time(16, 0)
    keep = [start <= t <= end for t in idx.time]
    return df.iloc[keep].copy()


def fetch_intraday_5m(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch 5m data for one ticker. Optional start_date/end_date for specific day."""
    try:
        if start_date and end_date:
            data = yf.download(
                ticker, start=start_date, end=end_date, interval="5m",
                progress=False, auto_adjust=True, threads=False
            )
        else:
            data = yf.download(
                ticker, period="1d", interval="5m",
                progress=False, auto_adjust=True, threads=False
            )
    except Exception:
        return pd.DataFrame()
    if data.empty or len(data) < 2:
        return pd.DataFrame()
    df = _ensure_single_column(data)
    return df


def fetch_all_dashboard_data(show_date_str: str = None) -> dict:
    """
    Fetch SPX 5m, VIX, and all confirmation tickers.
    show_date_str = YYYY-MM-DD when market closed; None for live today.
    Returns dict: spx, vix_df, vix_value, and keyed by ticker for each confirmation.
    """
    out = {}
    if show_date_str:
        start = show_date_str
        end_d = datetime.strptime(show_date_str, "%Y-%m-%d").date() + timedelta(days=1)
        end = end_d.isoformat()
    else:
        start, end = None, None

    spx = fetch_intraday_5m(TICKER_SPX, start, end)
    if not spx.empty:
        spx = _filter_regular_session(spx)
    out["spx"] = spx

    vix_df = fetch_intraday_5m(TICKER_VIX, start, end)
    if not vix_df.empty:
        vix_df = _filter_regular_session(vix_df)
    out["vix"] = vix_df
    vix_value = None
    if not vix_df.empty and "Close" in vix_df.columns:
        try:
            vix_value = float(vix_df["Close"].iloc[-1])
        except Exception:
            pass
    out["vix_value"] = vix_value

    for t in CONFIRMATION_TICKERS:
        df_t = fetch_intraday_5m(t, start, end)
        if not df_t.empty:
            df_t = _filter_regular_session(df_t)
        out[t] = df_t

    return out


def fetch_daily_for_backtest(ticker: str = "^GSPC", years: int = 1) -> pd.DataFrame:
    """Fetch daily OHLCV for backtesting (e.g. last 1 year)."""
    try:
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        data = yf.download(
            ticker, start=start, end=end, interval="1d",
            progress=False, auto_adjust=True, threads=False
        )
    except Exception:
        return pd.DataFrame()
    if data.empty or len(data) < 50:
        return pd.DataFrame()
    return _ensure_single_column(data)
