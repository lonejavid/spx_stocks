"""
SPX Day Trading Dashboard — 5-min intraday, RSI/MACD/VWAP/EMA, BUY/SELL signals, dark theme.
Auto-refresh every 5 minutes. Best windows: 10:00–11:30 AM & 2:30–3:30 PM EST.
"""
import json
import os
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta, time as dt_time
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import pytz

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False

st.set_page_config(page_title="SPX Day Trading", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# Theme + responsive (mobile & web)
DARK_CSS = """
<style>
/* Base */
.block-container { padding: 1rem; max-width: 1600px; margin-left: auto; margin-right: auto; width: 100%; box-sizing: border-box; }
h1, h2, h3 { color: #f0f2f6 !important; }
h1 { font-size: 1.75rem !important; border-bottom: 2px solid #00d4aa; padding-bottom: 0.4rem; }
h2 { font-size: 1.1rem !important; color: #b0b4bc !important; margin-top: 1.2rem !important; }
[data-testid="stMetricValue"] { font-size: 1.4rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8892a0 !important; font-size: 0.75rem !important; text-transform: uppercase !important; }
p { color: #8892a0 !important; }
#MainMenu, footer { visibility: hidden; }
/* Tables & charts: prevent overflow */
[data-testid="stDataFrame"], .stPlotlyChart { overflow-x: auto !important; -webkit-overflow-scrolling: touch; max-width: 100%; }
.js-plotly-plot { max-width: 100% !important; }
/* Tablet */
@media (max-width: 992px) {
  .block-container { padding: 0.75rem; }
  h1 { font-size: 1.5rem !important; }
  h2 { font-size: 1rem !important; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap !important; }
  [data-testid="column"] { min-width: 33.33% !important; flex: 1 1 33.33% !important; }
}
/* Mobile */
@media (max-width: 768px) {
  .block-container { padding: 0.5rem 0.75rem; padding-top: 0.75rem; }
  h1 { font-size: 1.35rem !important; }
  h2 { font-size: 0.95rem !important; margin-top: 0.9rem !important; }
  [data-testid="stHorizontalBlock"] { flex-direction: column !important; }
  [data-testid="column"] { min-width: 100% !important; width: 100% !important; max-width: 100% !important; }
  [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
  [data-testid="stMetricLabel"] { font-size: 0.7rem !important; }
  .stCaption { font-size: 0.8rem !important; }
  div[data-testid="stVerticalBlock"] > div { padding: 0.25rem 0 !important; }
}
/* Small mobile */
@media (max-width: 480px) {
  .block-container { padding: 0.4rem 0.5rem; }
  h1 { font-size: 1.2rem !important; }
  h2 { font-size: 0.9rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
}
/* Banner & signal on mobile */
.market-banner { font-size: 1rem; padding: 12px 16px !important; }
.signal-value { font-size: 1.3rem; }
@media (max-width: 768px) {
  .market-banner { font-size: 0.9rem !important; padding: 10px 12px !important; }
  .signal-value { font-size: 1.1rem !important; }
}
@media (max-width: 480px) {
  .market-banner { font-size: 0.85rem !important; padding: 8px 10px !important; }
  .signal-value { font-size: 1rem !important; }
}
/* Ensure viewport */
@viewport { width: device-width; initial-scale: 1; }
</style>
"""

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,36,0.8)",
    font=dict(color="#e0e2e6", size=11),
    margin=dict(t=40, b=40, l=50, r=25),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b0b4bc")),
)
AXIS = dict(showgrid=True, gridcolor="rgba(80,85,95,0.5)", zeroline=False)

TICKERS_MAIN = "^GSPC"
TICKER_VIX = "^VIX"
TICKERS_STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "QQQ", "SPY"]
EST = pytz.timezone("America/New_York")

# 8 conditions for signal strength score (labels for display)
CONDITION_LABELS = [
    "Price above VWAP",
    "RSI below 40 (buy) or above 60 (sell)",
    "EMA 9 crossed EMA 21",
    "MACD crossed signal line",
    "VIX below 20",
    "AAPL green today",
    "MSFT green today",
    "QQQ above VWAP",
]

# Trading windows EST (green zones)
WINDOW1 = (10, 0), (11, 30)   # 10:00 - 11:30
WINDOW2 = (14, 30), (15, 30)  # 2:30 PM - 3:30 PM

BUY_RSI_MAX = 40
SELL_RSI_MIN = 60
VIX_CALM = 20

# Load from environment so the secret is never committed (use .env or export SLACK_WEBHOOK_URL)
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
SLACK_COOLDOWN_MINUTES = 30


def send_slack_alert(message: str) -> None:
    """Send a POST request to the Slack webhook. All errors handled silently."""
    if not message or not SLACK_WEBHOOK_URL.strip():
        return
    try:
        body = json.dumps({"text": message}).encode("utf-8")
        req = Request(SLACK_WEBHOOK_URL, data=body, method="POST", headers={"Content-Type": "application/json"})
        urlopen(req, timeout=10)
    except (URLError, HTTPError, OSError, ValueError, Exception):
        pass


def _should_send_signal_alert(signal_type: str) -> bool:
    """True if we should send (no duplicate in last 30 min)."""
    key_when = "slack_last_signal_time"
    key_type = "slack_last_signal_type"
    now = datetime.now(EST)
    last_time = st.session_state.get(key_when)
    last_type = st.session_state.get(key_type)
    if last_type == signal_type and last_time is not None:
        if (now - last_time).total_seconds() < SLACK_COOLDOWN_MINUTES * 60:
            return False
    return True


def _record_signal_sent(signal_type: str) -> None:
    st.session_state["slack_last_signal_time"] = datetime.now(EST)
    st.session_state["slack_last_signal_type"] = signal_type


def _should_send_daily_summary() -> bool:
    """True if current time is 4:05–4:20 PM ET and we haven't sent today."""
    now = datetime.now(EST)
    t = now.time()
    if not (dt_time(16, 5) <= t <= dt_time(16, 20)):
        return False
    today = now.date()
    if st.session_state.get("slack_last_daily_summary_date") == today:
        return False
    return True


def _record_daily_summary_sent() -> None:
    st.session_state["slack_last_daily_summary_date"] = datetime.now(EST).date()


def _ensure_single_column(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """If yfinance returns MultiIndex columns, flatten to single level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    return df


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1 / length, adjust=False).mean()
    al = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = ag / al.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)


def _macd_df(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    e1 = close.ewm(span=fast, adjust=False).mean()
    e2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = e1 - e2
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return pd.DataFrame({"MACD": macd_line, "MACD_signal": sig_line, "MACD_hist": hist})


def _vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP = cumsum(typical_price * volume) / cumsum(volume). Typical price = (H+L+C)/3."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, pd.NA).ffill()


def _is_market_open():
    """True if current ET is Mon–Fri 9:30 AM–4:00 PM."""
    now = datetime.now(EST)
    if now.weekday() >= 5:
        return False
    t = now.time()
    return dt_time(9, 30) <= t < dt_time(16, 0)


def _market_open_and_display_date():
    """Return (is_market_open, display_date, label). When closed, display_date = previous full trading day."""
    now = datetime.now(EST)
    today = now.date()
    t = now.time()
    # Weekend
    if today.weekday() >= 5:
        days_back = today.weekday() - 4
        prev = today - timedelta(days=days_back)
        return False, prev, f"Previous trading day ({prev.isoformat()})"
    # Before 9:30 ET
    if t < dt_time(9, 30):
        if today.weekday() == 0:
            prev = today - timedelta(days=3)
        else:
            prev = today - timedelta(days=1)
        return False, prev, f"Previous trading day ({prev.isoformat()})"
    # After 4:00 PM ET (16:00)
    if t >= dt_time(16, 0):
        return False, today, f"Previous trading day ({today.isoformat()})"
    return True, today, "Today"


def _next_market_open_et():
    """Return datetime (ET) of next market open (9:30 AM ET, Mon–Fri)."""
    now = datetime.now(EST)
    today = now.date()
    open_today = EST.localize(datetime.combine(today, dt_time(9, 30)))
    if now.weekday() < 5 and now < open_today:
        return open_today
    days = 1
    if now.weekday() == 4:
        days = 3
    elif now.weekday() == 5:
        days = 2
    elif now.weekday() == 6:
        days = 1
    next_day = today + timedelta(days=days)
    return EST.localize(datetime.combine(next_day, dt_time(9, 30)))


def _countdown_to_next_open():
    """Return string like '2d 14h 22m' until next 9:30 AM ET."""
    now = datetime.now(EST)
    nxt = _next_market_open_et()
    delta = nxt - now
    if delta.total_seconds() <= 0:
        return "Opening soon"
    total_secs = int(delta.total_seconds())
    d, r = divmod(total_secs, 86400)
    h, r = divmod(r, 3600)
    m, _ = divmod(r, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    parts.append(f"{h}h")
    parts.append(f"{m}m")
    return " ".join(parts)


@st.cache_data(ttl=120)
def fetch_intraday_5m(ticker: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch 5m data. If start_date/end_date given, fetch that range (for previous trading day)."""
    try:
        if start_date and end_date:
            data = yf.download(ticker, start=start_date, end=end_date, interval="5m", progress=False, auto_adjust=True, threads=False)
        else:
            data = yf.download(ticker, period="1d", interval="5m", progress=False, auto_adjust=True, threads=False)
    except Exception:
        return pd.DataFrame()
    if data.empty or len(data) < 2:
        return pd.DataFrame()
    df = _ensure_single_column(data, ticker)
    return df


def _filter_session(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 9:30 AM–4:00 PM ET (regular session). No pre-market or after-hours."""
    if df.empty:
        return df
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    idx = pd.DatetimeIndex(df.index)
    if idx.tzinfo is None:
        idx = idx.tz_localize("America/New_York", ambiguous="infer")
    else:
        idx = idx.tz_convert("America/New_York")
    start, end = dt_time(9, 30), dt_time(16, 0)
    keep = [start <= t <= end for t in idx.time]
    out = df.iloc[keep].copy()
    return out[~out.index.duplicated(keep="first")].sort_index()


@st.cache_data(ttl=120)
def fetch_all_intraday(show_date_str: str = None):
    """Fetch 5m data. show_date_str = YYYY-MM-DD for that day; when None, use period=1d."""
    out = {}
    if show_date_str:
        start = show_date_str
        end_d = datetime.strptime(show_date_str, "%Y-%m-%d").date() + timedelta(days=1)
        end = end_d.isoformat()
    else:
        start, end = None, None
    spx = fetch_intraday_5m(TICKERS_MAIN, start, end)
    if not spx.empty:
        spx = _filter_session(spx)
    out["spx"] = spx
    vix_df = fetch_intraday_5m(TICKER_VIX, start, end)
    if not vix_df.empty:
        vix_df = _filter_session(vix_df)
    out["vix"] = vix_df
    vix_current = float(vix_df["Close"].iloc[-1]) if not vix_df.empty and "Close" in vix_df.columns else None
    out["vix_value"] = vix_current
    for t in TICKERS_STOCKS:
        dt = fetch_intraday_5m(t, start, end)
        if not dt.empty:
            dt = _filter_session(dt)
        out[t] = dt
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14), MACD(12,26,9), VWAP, EMA9, EMA21 to OHLCV DataFrame."""
    if df.empty or "Close" not in df.columns:
        return df
    df = df.copy()
    close = df["Close"]
    if HAS_PANDAS_TA:
        df["RSI"] = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            for c in macd.columns:
                df[c] = macd[c]
    else:
        df["RSI"] = _rsi(close, 14)
        m = _macd_df(close, 12, 26, 9)
        for c in m.columns:
            df[c] = m[c]
    if "High" in df.columns and "Low" in df.columns and "Volume" in df.columns:
        df["VWAP"] = _vwap(df)
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    return df



def _macd_signal_cols(df: pd.DataFrame):
    """Return (macd_line_col, signal_line_col)."""
    sig = [c for c in df.columns if "MACDs" in str(c) or ("MACD" in str(c) and "signal" in str(c).lower() and "hist" not in str(c).lower())][:1]
    macd = [c for c in df.columns if c == "MACD" or (str(c).startswith("MACD_") and "MACDh" not in str(c) and "MACDs" not in str(c))][:1]
    if not macd:
        macd = [c for c in df.columns if "MACD" in str(c) and "MACDh" not in str(c) and "MACDs" not in str(c)][:1]
    return (macd[0] if macd else None, sig[0] if sig else None)


def macd_cross_above(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    mc, sc = _macd_signal_cols(df)
    if mc is None or sc is None:
        return False
    return df[mc].iloc[idx] > df[sc].iloc[idx] and df[mc].iloc[idx - 1] <= df[sc].iloc[idx - 1]

def macd_cross_below(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    mc, sc = _macd_signal_cols(df)
    if mc is None or sc is None:
        return False
    return df[mc].iloc[idx] < df[sc].iloc[idx] and df[mc].iloc[idx - 1] >= df[sc].iloc[idx - 1]


def _daily_signal_stats(df_spx: pd.DataFrame, vix_value: float):
    """For displayed day: close, pct_change, n_buy, n_sell, first_buy_time, first_sell_time."""
    if df_spx.empty or len(df_spx) < 2:
        return None
    close_final = df_spx["Close"].iloc[-1]
    open_first = df_spx["Open"].iloc[0] if "Open" in df_spx.columns else df_spx["Close"].iloc[0]
    pct = ((close_final - open_first) / open_first * 100) if open_first and open_first != 0 else 0
    n_buy = n_sell = 0
    first_buy_time = first_sell_time = None
    for i in range(1, len(df_spx)):
        s = get_signal(df_spx.iloc[: i + 1], vix_value)
        if s == "BUY":
            n_buy += 1
            if first_buy_time is None:
                ts = df_spx.index[i]
                first_buy_time = pd.Timestamp(ts).strftime("%I:%M %p") if hasattr(pd.Timestamp(ts), "strftime") else str(ts)
        elif s == "SELL":
            n_sell += 1
            if first_sell_time is None:
                ts = df_spx.index[i]
                first_sell_time = pd.Timestamp(ts).strftime("%I:%M %p") if hasattr(pd.Timestamp(ts), "strftime") else str(ts)
    best = []
    if first_buy_time:
        best.append(f"First BUY {first_buy_time}")
    if first_sell_time:
        best.append(f"First SELL {first_sell_time}")
    best_str = " · ".join(best) if best else "No signals"
    return {"close": close_final, "pct": pct, "n_buy": n_buy, "n_sell": n_sell, "best": best_str}


def get_signal(df_spx: pd.DataFrame, vix_value: float) -> str:
    """BUY / SELL / WAIT from latest bar."""
    if df_spx.empty or len(df_spx) < 2:
        return "WAIT"
    i = len(df_spx) - 1
    row = df_spx.iloc[i]
    rsi = row.get("RSI")
    close = row["Close"]
    vwap = row.get("VWAP")
    if pd.isna(rsi) or pd.isna(vwap) or vix_value is None:
        return "WAIT"
    # VIX < 20 for BUY; VIX >= 20 for SELL (20.0 or above = ❌ for BUY)
    vix_ok_buy = vix_value < VIX_CALM
    vix_ok_sell = vix_value >= VIX_CALM
    above_vwap = close > vwap
    below_vwap = close < vwap
    cross_up = macd_cross_above(df_spx, i)
    cross_dn = macd_cross_below(df_spx, i)
    if rsi < BUY_RSI_MAX and above_vwap and cross_up and vix_ok_buy:
        return "BUY"
    if rsi > SELL_RSI_MIN and below_vwap and cross_dn and vix_ok_sell:
        return "SELL"
    return "WAIT"


def _qwap(df: pd.DataFrame) -> pd.Series:
    """VWAP for any OHLCV DataFrame."""
    if df.empty or "High" not in df.columns or "Volume" not in df.columns:
        return pd.Series(dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, pd.NA).ffill()


def get_signal_strength_conditions(df_spx: pd.DataFrame, vix_value: float, data: dict) -> tuple:
    """
    Return (score 0-8, list of 8 bools for BUY, list of 8 bools for SELL).
    Order: Price vs VWAP, RSI, EMA cross, MACD cross, VIX, AAPL green, MSFT green, QQQ above VWAP.
    """
    buy_bools = [False] * 8
    sell_bools = [False] * 8
    if df_spx.empty or len(df_spx) < 2:
        return 0, buy_bools, sell_bools
    i = len(df_spx) - 1
    row = df_spx.iloc[i]
    close = row["Close"]
    vwap = row.get("VWAP")
    rsi_val = row.get("RSI")
    # 1. Price above/below VWAP
    if vwap is not None and not pd.isna(vwap):
        buy_bools[0] = close > vwap
        sell_bools[0] = close < vwap
    # 2. RSI below 40 / above 60
    if rsi_val is not None and not pd.isna(rsi_val):
        buy_bools[1] = rsi_val < 40
        sell_bools[1] = rsi_val > 60
    # 3. EMA 9 crossed EMA 21
    buy_bools[2] = _ema_cross_above(df_spx, i)
    sell_bools[2] = _ema_cross_below(df_spx, i)
    # 4. MACD crossed signal
    buy_bools[3] = macd_cross_above(df_spx, i)
    sell_bools[3] = macd_cross_below(df_spx, i)
    # 5. VIX strictly below 20 for BUY; 20.0 or above for SELL (so 20.0 shows ❌ BUY, ✅ SELL)
    if vix_value is not None:
        buy_bools[4] = vix_value < 20
        sell_bools[4] = vix_value >= 20
    # 6. AAPL green today
    aapl = data.get("AAPL")
    if aapl is not None and not aapl.empty and "Open" in aapl.columns and "Close" in aapl.columns:
        aapl_green = aapl["Close"].iloc[-1] > aapl["Open"].iloc[0]
        buy_bools[5] = aapl_green
        sell_bools[5] = not aapl_green
    # 7. MSFT green today
    msft = data.get("MSFT")
    if msft is not None and not msft.empty and "Open" in msft.columns and "Close" in msft.columns:
        msft_green = msft["Close"].iloc[-1] > msft["Open"].iloc[0]
        buy_bools[6] = msft_green
        sell_bools[6] = not msft_green
    # 8. QQQ above VWAP
    qqq = data.get("QQQ")
    if qqq is not None and not qqq.empty and "Close" in qqq.columns:
        qqq_vwap = _qwap(qqq)
        if not qqq_vwap.empty:
            qqq_above = qqq["Close"].iloc[-1] > qqq_vwap.iloc[-1]
            buy_bools[7] = qqq_above
            sell_bools[7] = not qqq_above
    buy_score = sum(buy_bools)
    sell_score = sum(sell_bools)
    return max(buy_score, sell_score), buy_bools, sell_bools


def _ema_cross_above(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1 or "EMA9" not in df.columns or "EMA21" not in df.columns:
        return False
    return df["EMA9"].iloc[idx] > df["EMA21"].iloc[idx] and df["EMA9"].iloc[idx - 1] <= df["EMA21"].iloc[idx - 1]


def _ema_cross_below(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1 or "EMA9" not in df.columns or "EMA21" not in df.columns:
        return False
    return df["EMA9"].iloc[idx] < df["EMA21"].iloc[idx] and df["EMA9"].iloc[idx - 1] >= df["EMA21"].iloc[idx - 1]


def _fetch_daily_spx(years: int = 1) -> pd.DataFrame:
    """Fetch SPX daily OHLCV for backtesting."""
    try:
        end = datetime.now()
        start = end - timedelta(days=365 * years)
        d = yf.download(TICKERS_MAIN, start=start, end=end, interval="1d", progress=False, auto_adjust=True, threads=False)
    except Exception:
        return pd.DataFrame()
    if d.empty or len(d) < 50:
        return pd.DataFrame()
    return _ensure_single_column(d, TICKERS_MAIN)


def _backtest_rsi(daily: pd.DataFrame) -> dict:
    """BUY RSI < 30, SELL RSI > 70."""
    df = daily.copy()
    df["RSI"] = _rsi(df["Close"], 14)
    trades = []
    pos = None
    entry = None
    for i in range(14, len(df)):
        r = df["RSI"].iloc[i]
        c = df["Close"].iloc[i]
        if pos is None and r < 30:
            pos = "long"
            entry = c
        elif pos and r > 70:
            pnl = (c - entry) / entry * 100
            trades.append({"pnl_pct": pnl})
            pos = None
    return _backtest_result(trades, "RSI (buy <30, sell >70)")


def _backtest_macd(daily: pd.DataFrame) -> dict:
    """BUY MACD cross above signal, SELL MACD cross below."""
    df = daily.copy()
    m = _macd_df(df["Close"], 12, 26, 9)
    df["MACD"] = m["MACD"]
    df["MACD_signal"] = m["MACD_signal"]
    trades = []
    pos = None
    entry = None
    for i in range(1, len(df)):
        cross_up = df["MACD"].iloc[i] > df["MACD_signal"].iloc[i] and df["MACD"].iloc[i - 1] <= df["MACD_signal"].iloc[i - 1]
        cross_dn = df["MACD"].iloc[i] < df["MACD_signal"].iloc[i] and df["MACD"].iloc[i - 1] >= df["MACD_signal"].iloc[i - 1]
        c = df["Close"].iloc[i]
        if pos is None and cross_up:
            pos = "long"
            entry = c
        elif pos and cross_dn:
            pnl = (c - entry) / entry * 100
            trades.append({"pnl_pct": pnl})
            pos = None
    return _backtest_result(trades, "MACD crossover")


def _backtest_ema(daily: pd.DataFrame) -> dict:
    """BUY EMA9 cross above EMA21, SELL EMA9 cross below EMA21."""
    df = daily.copy()
    df["EMA9"] = df["Close"].ewm(span=9, adjust=False).mean()
    df["EMA21"] = df["Close"].ewm(span=21, adjust=False).mean()
    trades = []
    pos = None
    entry = None
    for i in range(1, len(df)):
        cross_up = df["EMA9"].iloc[i] > df["EMA21"].iloc[i] and df["EMA9"].iloc[i - 1] <= df["EMA21"].iloc[i - 1]
        cross_dn = df["EMA9"].iloc[i] < df["EMA21"].iloc[i] and df["EMA9"].iloc[i - 1] >= df["EMA21"].iloc[i - 1]
        c = df["Close"].iloc[i]
        if pos is None and cross_up:
            pos = "long"
            entry = c
        elif pos and cross_dn:
            pnl = (c - entry) / entry * 100
            trades.append({"pnl_pct": pnl})
            pos = None
    return _backtest_result(trades, "EMA 9/21 crossover")


def _backtest_result(trades: list, name: str) -> dict:
    if not trades:
        return {"name": name, "total_trades": 0, "win_rate_pct": 0, "total_return_pct": 0, "best_pct": 0, "worst_pct": 0}
    t = pd.DataFrame(trades)
    wins = t[t["pnl_pct"] > 0]
    total_return = (1 + t["pnl_pct"] / 100).prod() - 1
    return {
        "name": name,
        "total_trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100,
        "total_return_pct": total_return * 100,
        "best_pct": t["pnl_pct"].max(),
        "worst_pct": t["pnl_pct"].min(),
    }


def trading_window_shapes(df: pd.DataFrame):
    """Return Plotly shapes for 10:00–11:30 and 14:30–15:30 EST on the chart's date."""
    if df.empty or not hasattr(df.index, 'min'):
        return []
    try:
        ts = pd.Timestamp(df.index.min())
        if ts.tzinfo is None:
            ts = ts.tz_localize(EST)
        else:
            ts = ts.tz_convert(EST)
        d = ts.date()
    except Exception:
        d = datetime.now(EST).date()
    shapes = []
    for (h1, m1), (h2, m2) in [WINDOW1, WINDOW2]:
        x0 = pd.Timestamp(datetime(d.year, d.month, d.day, h1, m1), tz=EST)
        x1 = pd.Timestamp(datetime(d.year, d.month, d.day, h2, m2), tz=EST)
        shapes.append(dict(type="rect", x0=x0, x1=x1, y0=0, y1=1, yref="paper", fillcolor="rgba(0,212,170,0.15)", line=dict(width=0)))
    return shapes


def run_dashboard():
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    st.title("SPX Day Trading Dashboard")
    st.caption("5-min intraday · RSI · MACD · VWAP · EMA · BUY/SELL when RSI/VWAP/MACD cross + VIX filter · Best windows 10:00–11:30 & 14:30–15:30 EST")

    is_market_open, show_date, data_label = _market_open_and_display_date()
    show_date_str = show_date.isoformat() if hasattr(show_date, "isoformat") else str(show_date)
    if not is_market_open:
        data = fetch_all_intraday(show_date_str)
    else:
        data = fetch_all_intraday(None)
    spx = data["spx"]
    vix_df = data["vix"]
    vix_value = data.get("vix_value")
    if spx.empty or len(spx) < 5:
        st.warning("Not enough SPX 5-minute data for this session. Try again when the market is open (9:30–16:00 ET) or check back for previous trading day.")
        return
    spx = add_indicators(spx)
    signal = get_signal(spx, vix_value)
    now_est = datetime.now(EST).strftime("%Y-%m-%d %H:%M ET")
    rsi_val = spx["RSI"].iloc[-1] if "RSI" in spx.columns else None

    # Market status banner
    if is_market_open:
        st.markdown(
            '<div class="market-banner" style="background: linear-gradient(90deg, rgba(46,204,113,0.25), rgba(46,204,113,0.1)); border: 1px solid #2ecc71; border-radius: 8px; margin-bottom: 1rem;">'
            '<span>🟢 <strong>Market Open</strong> — Live data updating every 5 min</span></div>',
            unsafe_allow_html=True,
        )
    else:
        countdown = _countdown_to_next_open()
        st.markdown(
            f'<div class="market-banner" style="background: linear-gradient(90deg, rgba(231,76,60,0.25), rgba(231,76,60,0.1)); border: 1px solid #e74c3c; border-radius: 8px; margin-bottom: 0.5rem;">'
            f'<span>🔴 <strong>Market Closed</strong> — Opens 9:30 AM EST</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Next open in:** {countdown}")
        st.markdown("")

    # Always: Top row — SPX, VIX, RSI, Signal, Time (ET)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("SPX", f"${spx['Close'].iloc[-1]:,.2f}")
    with c2:
        vix_str = f"{vix_value:.1f}" if vix_value is not None else "—"
        st.metric("VIX", vix_str)
    with c3:
        rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "—"
        st.metric("RSI", rsi_str)
    with c4:
        color = "#2ecc71" if signal == "BUY" else "#e74c3c" if signal == "SELL" else "#8892a0"
        st.markdown(f"**Signal**<br><span class=\"signal-value\" style='color:{color};'>{signal}</span>", unsafe_allow_html=True)
    with c5:
        st.metric("Time (ET)", now_est)

    # Signal strength score (4/8) and condition breakdown
    score_val, buy_bools, sell_bools = get_signal_strength_conditions(spx, vix_value, data)
    show_buy = signal == "BUY" or (signal == "WAIT" and sum(buy_bools) >= sum(sell_bools))
    active_bools = buy_bools if show_buy else sell_bools
    active_score = sum(buy_bools) if show_buy else sum(sell_bools)
    st.markdown(f"**Signal strength:** {active_score}/8")
    for j, label in enumerate(CONDITION_LABELS):
        st.write("✅" if active_bools[j] else "❌", label)

    # Slack: signal alert when score >= 5 (no duplicate within 30 min)
    if active_score >= 5 and signal in ("BUY", "SELL") and _should_send_signal_alert(signal):
        try:
            spx_p = spx["Close"].iloc[-1]
            r_str = f"{rsi_val:.1f}" if rsi_val is not None else "—"
            v_str = f"{vix_value:.1f}" if vix_value is not None else "—"
            time_short = datetime.now(EST).strftime("%I:%M %p ET").lstrip("0")
            conditions_met = [CONDITION_LABELS[k] for k in range(8) if active_bools[k]]
            cond_str = ", ".join(conditions_met) if conditions_met else "—"
            if signal == "BUY":
                msg = (
                    f"🟢 *BUY SIGNAL — SPX Day Trading*\n"
                    f"💰 *SPX Price:* ${spx_p:,.2f}\n"
                    f"📊 *Signal Score:* {active_score}/8\n"
                    f"⏰ *Time:* {time_short}\n"
                    f"📈 *RSI:* {r_str}\n"
                    f"😰 *VIX:* {v_str}\n"
                    f"✅ Conditions met: {cond_str}\n"
                    f"⚠️ Not financial advice."
                )
            else:
                msg = (
                    f"🔴 *SELL SIGNAL — SPX Day Trading*\n"
                    f"💰 *SPX Price:* ${spx_p:,.2f}\n"
                    f"📊 *Signal Score:* {active_score}/8\n"
                    f"⏰ *Time:* {time_short}\n"
                    f"📉 *RSI:* {r_str}\n"
                    f"😰 *VIX:* {v_str}\n"
                    f"✅ Conditions met: {cond_str}\n"
                    f"⚠️ Not financial advice."
                )
            send_slack_alert(msg)
            _record_signal_sent(signal)
        except Exception:
            pass

    # Slack: daily summary at 4:05–4:20 PM ET (once per day)
    if _should_send_daily_summary() and not spx.empty and len(spx) >= 5:
        try:
            stats = _daily_signal_stats(spx, vix_value)
            if stats is not None:
                today_str = datetime.now(EST).strftime("%Y-%m-%d")
                r_close = f"{rsi_val:.1f}" if rsi_val is not None else "—"
                v_close = f"{vix_value:.1f}" if vix_value is not None else "—"
                msg = (
                    f"📊 *Daily Summary — SPX*\n"
                    f"📅 Date: {today_str}\n"
                    f"💰 SPX Closed: ${stats['close']:,.2f} ({stats['pct']:+.2f}%)\n"
                    f"🟢 BUY Signals: {stats['n_buy']}\n"
                    f"🔴 SELL Signals: {stats['n_sell']}\n"
                    f"📈 RSI at close: {r_close}\n"
                    f"😰 VIX at close: {v_close}"
                )
                send_slack_alert(msg)
                _record_daily_summary_sent()
        except Exception:
            pass

    st.markdown("---")

    # When closed: summary card (SPX close, % change, BUY/SELL counts, best signal)
    if not is_market_open:
        stats = _daily_signal_stats(spx, vix_value)
        if stats:
            st.subheader(f"Session summary — {show_date_str}")
            s1, s2, s3, s4, s5 = st.columns(5)
            with s1:
                st.metric("SPX closed", f"${stats['close']:,.2f}")
            with s2:
                st.metric("% change", f"{stats['pct']:+.2f}%")
            with s3:
                st.metric("BUY signals", stats["n_buy"])
            with s4:
                st.metric("SELL signals", stats["n_sell"])
            with s5:
                st.markdown("**Best signal**<br><span class=\"signal-value\" style='color:#b0b4bc;'>" + stats["best"] + "</span>", unsafe_allow_html=True)
            st.markdown("---")

    # SPX intraday: clean line chart only (9:30 AM–4:00 PM ET data)
    chart_spx = spx[~spx.index.duplicated(keep="first")].sort_index()
    st.subheader("SPX 5-min price & VWAP")
    chart_title = f"SPX 5-min with VWAP & EMAs — {data_label} (9:30 AM–4:00 PM ET only)"
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=chart_spx.index, y=chart_spx["Close"], name="SPX",
        mode="lines", line=dict(color="#00d4aa", width=2),
    ))
    if "VWAP" in chart_spx.columns:
        fig_price.add_trace(go.Scatter(x=chart_spx.index, y=chart_spx["VWAP"], name="VWAP", mode="lines", line=dict(color="#f39c12", width=1.5, dash="dash")))
    if "EMA9" in chart_spx.columns:
        fig_price.add_trace(go.Scatter(x=chart_spx.index, y=chart_spx["EMA9"], name="EMA 9", mode="lines", line=dict(color="#9b59b6", width=1.2)))
    if "EMA21" in chart_spx.columns:
        fig_price.add_trace(go.Scatter(x=chart_spx.index, y=chart_spx["EMA21"], name="EMA 21", mode="lines", line=dict(color="#3498db", width=1.2)))
    fig_price.update_layout(**PLOTLY_LAYOUT, title=chart_title, height=380, xaxis=AXIS, yaxis=AXIS)
    for s in trading_window_shapes(chart_spx):
        fig_price.add_shape(s)
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI
    st.subheader("RSI (14)")
    rsi_col = "RSI" if "RSI" in spx.columns else None
    if rsi_col:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=spx.index, y=spx[rsi_col], name="RSI", line=dict(color="#00d4aa", width=2)))
        fig_rsi.add_hline(y=BUY_RSI_MAX, line_dash="dash", line_color="#2ecc71", annotation_text="40 (buy zone)")
        fig_rsi.add_hline(y=SELL_RSI_MIN, line_dash="dash", line_color="#e74c3c", annotation_text="60 (sell zone)")
        fig_rsi.update_layout(**PLOTLY_LAYOUT, title="RSI (14)", height=280, xaxis=AXIS, yaxis=dict(**AXIS, range=[0, 100]))
        st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    mc, sc = _macd_signal_cols(spx)
    hist_col = [c for c in spx.columns if "MACDh" in str(c) or (isinstance(c, str) and "MACD" in c and "hist" in c.lower())][:1]
    if mc and sc:
        st.subheader("MACD (12, 26, 9)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=spx.index, y=spx[mc], name="MACD", line=dict(color="#00d4aa", width=2)))
        fig_macd.add_trace(go.Scatter(x=spx.index, y=spx[sc], name="Signal", line=dict(color="#f39c12", width=1.5)))
        if hist_col:
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in spx[hist_col[0]]]
            fig_macd.add_trace(go.Bar(x=spx.index, y=spx[hist_col[0]], name="Histogram", marker_color=colors, opacity=0.6))
        fig_macd.update_layout(**PLOTLY_LAYOUT, title="MACD (12, 26, 9)", height=280, xaxis=AXIS, yaxis=AXIS)
        st.plotly_chart(fig_macd, use_container_width=True)

    # Backtesting section (3 strategies on 1 year SPX daily)
    st.markdown("---")
    st.subheader("Backtesting (1 year SPX daily)")
    with st.spinner("Running backtests..."):
        try:
            daily = _fetch_daily_spx(1)
            if daily is not None and len(daily) >= 50:
                r1 = _backtest_rsi(daily)
                r2 = _backtest_macd(daily)
                r3 = _backtest_ema(daily)
                bt1, bt2, bt3 = st.columns(3)
                for col, r in [(bt1, r1), (bt2, r2), (bt3, r3)]:
                    with col:
                        st.markdown(f"**{r['name']}**")
                        st.metric("Total trades", r["total_trades"])
                        st.metric("Win rate", f"{r['win_rate_pct']:.1f}%")
                        st.metric("Total return", f"{r['total_return_pct']:.1f}%")
                        st.metric("Best trade", f"{r['best_pct']:+.2f}%")
                        st.metric("Worst trade", f"{r['worst_pct']:+.2f}%")
            else:
                st.info("Not enough daily data for backtest.")
        except Exception as e:
            st.warning(f"Backtest error: {e}")

    # Table: AAPL, MSFT, NVDA, AMZN, QQQ, SPY (when closed = previous day closing prices)
    table_label = "Previous day closing prices" if not is_market_open else f"Confirmation stocks — {data_label}"
    st.subheader(table_label)
    rows = []
    for t in TICKERS_STOCKS:
        df_t = data.get(t)
        if df_t is None or df_t.empty or "Close" not in df_t.columns:
            rows.append({"Ticker": t, "Price": "—", "% Change": "—"})
            continue
        close_now = df_t["Close"].iloc[-1]
        open_today = df_t["Open"].iloc[0] if "Open" in df_t.columns else close_now
        pct = ((close_now - open_today) / open_today * 100) if open_today and open_today != 0 else 0
        rows.append({"Ticker": t, "Price": f"${close_now:,.2f}", "% Change": f"{pct:+.2f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Signals log (session state)
    if "signals_log" not in st.session_state:
        st.session_state.signals_log = []
    if signal in ("BUY", "SELL"):
        st.session_state.signals_log.append({"Time (ET)": now_est, "Signal": signal, "SPX": f"${spx['Close'].iloc[-1]:,.2f}", "VIX": f"{vix_value:.1f}" if vix_value else "—"})
        st.session_state.signals_log = st.session_state.signals_log[-50:]
    st.subheader("Trading signals log")
    if st.session_state.signals_log:
        st.dataframe(pd.DataFrame(st.session_state.signals_log).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No BUY/SELL signals yet. Conditions: BUY = RSI<40, Price>VWAP, MACD cross up, VIX<20. SELL = RSI>60, Price<VWAP, MACD cross down, VIX>20.")

    # Auto-refresh note
    st.markdown("---")
    st.caption("Data refreshes every 5 minutes automatically. Green zones on chart: 10:00–11:30 AM & 2:30–3:30 PM EST.")


# Auto-refresh every 5 minutes when Streamlit supports run_every
try:
    use_fragment = hasattr(st, "fragment") and callable(getattr(st.fragment, "__call__", None))
    run_every_available = use_fragment
except Exception:
    run_every_available = False

if run_every_available:
    try:
        @st.fragment(run_every=timedelta(seconds=300))
        def _auto_refresh():
            run_dashboard()
        _auto_refresh()
    except Exception:
        run_dashboard()
else:
    run_dashboard()
