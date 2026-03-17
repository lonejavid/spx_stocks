"""
SPX Day Trading Dashboard — 5-min intraday, RSI/MACD/VWAP/EMA, BUY/SELL signals, dark theme.
Auto-refresh every 5 minutes. Best windows: 10:00–11:30 AM & 2:30–3:30 PM EST.
"""
import contextlib
import io
import json
import math
import os

# Load .env so SLACK_WEBHOOK_URL and POLYGON_API_KEY are available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import yfinance as yf
import pandas as pd

try:
    from yfinance import YFRateLimitError
except ImportError:
    YFRateLimitError = type("YFRateLimitError", (Exception,), {})  # no-op if old yfinance
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

st.set_page_config(page_title="SPX Day Trading", page_icon="📊", layout="wide", initial_sidebar_state="expanded")

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
/* BUY/SELL condition panels: show all rows (no clipping) */
.buy-sell-panel { overflow: visible !important; min-height: 1px; }
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

# Timeframe options: (yfinance interval, display label)
INTERVAL_OPTIONS = [
    ("1m", "1 min"),
    ("2m", "2 min"),
    ("5m", "5 min"),
    ("15m", "15 min"),
    ("30m", "30 min"),
    ("1h", "1 hour"),
]
DEFAULT_INTERVAL_INDEX = 2  # 5 min

# 8 conditions for signal strength score (labels for display)
CONDITION_LABELS = [
    "Price above/below VWAP",
    "RSI in BUY zone (<45) or SELL zone (>55)",
    "EMA 9 crossed EMA 21 (current bar)",
    "MACD above/below signal line",
    "VIX < 25 (BUY) / >= 22 (SELL)",
    "AAPL green/red today",
    "MSFT green/red today",
    "QQQ above/below VWAP",
]

# Two-column display: separate labels per side (more descriptive)
BUY_LABELS = [
    "Price above VWAP",
    "RSI < 45 (oversold)",
    "EMA 9 above EMA 21",
    "MACD above signal line",
    "VIX < 25 (calm market)",
    "AAPL green today",
    "MSFT green today",
    "QQQ above VWAP",
]
SELL_LABELS = [
    "Price below VWAP",
    "RSI > 55 (overbought)",
    "EMA 9 below EMA 21",
    "MACD below signal line",
    "VIX ≥ 22 (elevated fear)",
    "AAPL red today",
    "MSFT red today",
    "QQQ below VWAP",
]

# Persist condition checkboxes across refreshes (file-based; no localStorage in Streamlit)
CONDITION_PERSISTENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".dashboard_conditions.json")


def _load_condition_flags() -> tuple:
    """Return (enabled_buy: list of 8 bools, enabled_sell: list of 8 bools). Default all True."""
    try:
        if os.path.isfile(CONDITION_PERSISTENCE_PATH):
            with open(CONDITION_PERSISTENCE_PATH, "r") as f:
                d = json.load(f)
            buy = d.get("enabled_buy", [True] * 8)
            sell = d.get("enabled_sell", [True] * 8)
            return (([bool(b) for b in buy] + [True] * 8)[:8], ([bool(s) for s in sell] + [True] * 8)[:8])
    except Exception:
        pass
    return ([True] * 8, [True] * 8)


def _save_condition_flags(enabled_buy: list, enabled_sell: list) -> None:
    """Persist checkbox state to JSON file."""
    try:
        with open(CONDITION_PERSISTENCE_PATH, "w") as f:
            json.dump({"enabled_buy": enabled_buy, "enabled_sell": enabled_sell}, f, indent=0)
    except Exception:
        pass


def _condition_row_html(label: str, is_met: bool, side: str = "buy") -> str:
    """Return HTML for one condition row: icon + label only (no colored bar). Used in BUY/SELL columns."""
    icon = "✅" if is_met else "❌"
    text_color = "#e0e0e0" if is_met else "#888888"
    label_escaped = label.replace("<", "&lt;").replace(">", "&gt;")
    return f"<div style='font-size: 13px; color: {text_color}; padding: 2px 0 6px 0;'>{icon} {label_escaped}</div>"


# Trading windows EST (green zones)
WINDOW1 = (10, 0), (11, 30)   # 10:00 - 11:30
WINDOW2 = (14, 30), (15, 30)  # 2:30 PM - 3:30 PM

# Simulated intraday trading (fake capital for backtest-style P&L)
SIM_STARTING_CAPITAL = 50000
SIM_POSITION_SIZE = 10000

# Relaxed thresholds for professional intraday (was 40/60/20)
BUY_RSI_MAX = 45
SELL_RSI_MIN = 55
VIX_BUY_MAX = 25
VIX_SELL_MIN = 22

# -----------------------------------------------------------------------------
# VERIFICATION MOCK TESTS (commented — uncomment to verify signal logic)
# -----------------------------------------------------------------------------
# Mock test — this should produce a BUY signal:
#   RSI = 43, MACD_line = 2.1, MACD_signal = 1.8, EMA9 = 6850, EMA21 = 6845
#   VIX = 23, AAPL_green = True, QQQ_above_vwap = False, MSFT_green = True
#   Expected: BUY (3/4 core conditions + 2/3 confirmations)
#
# Mock test — this should produce a SELL signal:
#   RSI = 58, MACD_line = -1.2, MACD_signal = 0.3, EMA9 = 6830, EMA21 = 6840
#   VIX = 24, AAPL_green = False, QQQ_above_vwap = False, MSFT_green = False
#   Expected: SELL (3/4 core conditions + 3/3 confirmations)
# -----------------------------------------------------------------------------

# Slack webhook: on Streamlit Cloud use Secrets (st.secrets); locally use .env (os.environ)
def _get_slack_webhook_url():
    url = ""
    try:
        if hasattr(st, "secrets") and st.secrets:
            url = st.secrets.get("SLACK_WEBHOOK_URL", "") or getattr(st.secrets, "SLACK_WEBHOOK_URL", "")
    except Exception:
        pass
    if not (url and str(url).strip()):
        url = os.environ.get("SLACK_WEBHOOK_URL", "").strip()
    return str(url).strip() if url else ""


SLACK_WEBHOOK_URL = _get_slack_webhook_url()
SLACK_COOLDOWN_MINUTES = 30


def send_slack_alert(message: str) -> bool:
    """Send a POST request to the Slack webhook. Returns True if sent, False if skipped or failed."""
    if not message or not SLACK_WEBHOOK_URL.strip():
        return False
    try:
        body = json.dumps({"text": message}).encode("utf-8")
        req = Request(SLACK_WEBHOOK_URL, data=body, method="POST", headers={"Content-Type": "application/json"})
        urlopen(req, timeout=10)
        return True
    except (URLError, HTTPError, OSError, ValueError, Exception):
        return False


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


def _signal_base_type(signal: str) -> str:
    """Return 'BUY' or 'SELL' for cooldown/key; STRONG BUY/SELL share cooldown with BUY/SELL."""
    if signal in ("BUY", "STRONG BUY"):
        return "BUY"
    if signal in ("SELL", "STRONG SELL"):
        return "SELL"
    return signal


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
    def _col(x):
        c = df[x] if x in df.columns else None
        if c is None:
            return None
        return c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c
    high, low, close, vol = _col("High"), _col("Low"), _col("Close"), _col("Volume")
    if high is None or low is None or close is None or vol is None:
        return pd.Series(dtype=float)
    tp = (high + low + close) / 3
    out = (tp * vol).cumsum() / vol.cumsum().replace(0, pd.NA).ffill()
    return out.iloc[:, 0] if isinstance(out, pd.DataFrame) else out


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


# Set when a fetch fails due to Yahoo rate limit (so UI can show a specific message)
_rate_limited = False

@st.cache_data(ttl=300)
def fetch_intraday(ticker: str, interval: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Fetch intraday data for given interval (e.g. 1m, 2m, 5m, 15m, 30m, 1h)."""
    global _rate_limited
    _rate_limited = False
    try:
        with contextlib.redirect_stderr(io.StringIO()):  # hide yfinance "1 Failed download" in terminal
            if start_date and end_date:
                data = yf.download(ticker, start=start_date, end=end_date, interval=interval, progress=False, auto_adjust=True, threads=False)
            else:
                data = yf.download(ticker, period="1d", interval=interval, progress=False, auto_adjust=True, threads=False)
    except YFRateLimitError:
        _rate_limited = True
        return pd.DataFrame()
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
    # yfinance intraday often returns UTC; convert to ET for session filter
    if idx.tzinfo is None:
        idx = idx.tz_localize("UTC", ambiguous="infer").tz_convert("America/New_York")
    else:
        idx = idx.tz_convert("America/New_York")
    df.index = idx
    start, end = dt_time(9, 30), dt_time(16, 0)
    keep = [start <= t <= end for t in idx.time]
    out = df.iloc[keep].copy()
    return out[~out.index.duplicated(keep="first")].sort_index()


@st.cache_data(ttl=300)
def fetch_all_intraday(show_date_str: str, interval: str):
    """Fetch intraday data for given interval. show_date_str = YYYY-MM-DD for that day; when None, use period=1d."""
    import time
    out = {}
    if show_date_str:
        start = show_date_str
        end_d = datetime.strptime(show_date_str, "%Y-%m-%d").date() + timedelta(days=1)
        end = end_d.isoformat()
    else:
        start, end = None, None
    spx = fetch_intraday(TICKERS_MAIN, interval, start, end)
    time.sleep(0.3)  # reduce chance of Yahoo rate limit
    if not spx.empty:
        spx = _filter_session(spx)
    out["spx"] = spx
    out["rate_limited"] = _rate_limited
    vix_df = fetch_intraday(TICKER_VIX, interval, start, end)
    time.sleep(0.3)
    if not vix_df.empty:
        vix_df = _filter_session(vix_df)
    out["vix"] = vix_df
    if not vix_df.empty and "Close" in vix_df.columns:
        last = vix_df["Close"].iloc[-1]
        try:
            vix_current = float(last.squeeze()) if hasattr(last, "squeeze") else float(last)
        except (TypeError, ValueError):
            vix_current = None
    else:
        vix_current = None
    out["vix_value"] = vix_current
    for t in TICKERS_STOCKS:
        time.sleep(0.2)
        dt = fetch_intraday(t, interval, start, end)
        if not dt.empty:
            dt = _filter_session(dt)
        out[t] = dt
    out["rate_limited"] = out.get("rate_limited", False) or _rate_limited
    return out


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14), MACD(12,26,9), VWAP, EMA9, EMA21 to OHLCV DataFrame."""
    if df.empty or "Close" not in df.columns:
        return df
    df = df.copy()
    close = df["Close"]
    # Ensure Series (e.g. from MultiIndex or single-column DataFrame)
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.astype(float)
    if HAS_PANDAS_TA:
        rsi_out = ta.rsi(close, length=14)
        df["RSI"] = rsi_out.iloc[:, 0] if isinstance(rsi_out, pd.DataFrame) else rsi_out
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
        vwap_ser = _vwap(df)
        df["VWAP"] = vwap_ser.iloc[:, 0] if isinstance(vwap_ser, pd.DataFrame) else vwap_ser
    df["EMA9"] = close.ewm(span=9, adjust=False).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()
    return df



def _rsi_scalar_at_row(df: pd.DataFrame, row_idx: int):
    """
    Return RSI as a plain float at row row_idx, or None if missing/NaN.
    Tries column 'RSI' (built-in _rsi) then 'RSI_14' (pandas_ta). Coerces Series to scalar.
    """
    for col in ("RSI", "RSI_14"):
        if col not in df.columns:
            continue
        val = df[col].iloc[row_idx]
        if pd.isna(val):
            return None
        if hasattr(val, "item"):
            return float(val.item())
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


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


def _get_confirmation_bools(data: dict) -> tuple:
    """Return (aapl_green, msft_green, qqq_above_vwap). Used for BUY/SELL confirmations."""
    aapl_green = msft_green = qqq_above_vwap = False
    aapl = data.get("AAPL") if data else None
    if aapl is not None and not aapl.empty and "Open" in aapl.columns and "Close" in aapl.columns:
        aapl_green = aapl["Close"].iloc[-1] > aapl["Open"].iloc[0]
    msft = data.get("MSFT") if data else None
    if msft is not None and not msft.empty and "Open" in msft.columns and "Close" in msft.columns:
        msft_green = msft["Close"].iloc[-1] > msft["Open"].iloc[0]
    qqq = data.get("QQQ") if data else None
    if qqq is not None and not qqq.empty and "Close" in qqq.columns:
        qqq_vwap = _qwap(qqq)
        if not qqq_vwap.empty:
            qqq_above_vwap = qqq["Close"].iloc[-1] > qqq_vwap.iloc[-1]
    return aapl_green, msft_green, qqq_above_vwap


def _daily_signal_stats(df_spx: pd.DataFrame, vix_value: float, data: dict = None, enabled_buy: list = None, enabled_sell: list = None):
    """For displayed day: close, pct_change, n_buy, n_sell, first_buy_time, first_sell_time. Uses enabled_buy/enabled_sell when provided."""
    if df_spx.empty or len(df_spx) < 2:
        return None
    close_final = df_spx["Close"].iloc[-1]
    open_first = df_spx["Open"].iloc[0] if "Open" in df_spx.columns else df_spx["Close"].iloc[0]
    pct = ((close_final - open_first) / open_first * 100) if open_first and open_first != 0 else 0
    n_buy = n_sell = 0
    first_buy_time = first_sell_time = None
    for i in range(1, len(df_spx)):
        s = get_signal(df_spx.iloc[: i + 1], vix_value, data or {}, enabled_buy, enabled_sell)[0]
        if s in ("BUY", "STRONG BUY"):
            n_buy += 1
            if first_buy_time is None:
                ts = df_spx.index[i]
                first_buy_time = pd.Timestamp(ts).strftime("%I:%M %p") if hasattr(pd.Timestamp(ts), "strftime") else str(ts)
        elif s in ("SELL", "STRONG SELL"):
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


def get_signal(df_spx: pd.DataFrame, vix_value: float, data: dict = None, enabled_buy: list = None, enabled_sell: list = None):
    """
    BUY/SELL using only enabled conditions. When enabled_* is None, all 8 conditions are used (original logic).
    Dynamic thresholds: core need = ceil(0.75 * n_core_enabled), conf need = 1 if any conf enabled else 0.
    If exactly one condition is enabled and it is met, signal fires (BUY or SELL).
    Returns (signal_str, debug_dict).
    """
    if df_spx.empty or len(df_spx) < 2:
        return "WAIT", None
    data = data or {}
    enabled_buy = enabled_buy if enabled_buy is not None else [True] * 8
    enabled_sell = enabled_sell if enabled_sell is not None else [True] * 8
    enabled_buy = ([bool(x) for x in enabled_buy] + [True] * 8)[:8]
    enabled_sell = ([bool(x) for x in enabled_sell] + [True] * 8)[:8]

    i = len(df_spx) - 1
    row = df_spx.iloc[i]
    rsi = _rsi_scalar_at_row(df_spx, i)
    close = row["Close"]
    vwap = row.get("VWAP")
    ema9 = row.get("EMA9")
    ema21 = row.get("EMA21")
    if rsi is None or vix_value is None:
        return "WAIT", None
    mc, sc = _macd_signal_cols(df_spx)
    macd_above = (mc and sc and df_spx[mc].iloc[i] > df_spx[sc].iloc[i])
    macd_below = (mc and sc and df_spx[mc].iloc[i] < df_spx[sc].iloc[i])
    ema_ok_buy = ("EMA9" in df_spx.columns and "EMA21" in df_spx.columns and
                  pd.notna(df_spx["EMA9"].iloc[i]) and pd.notna(df_spx["EMA21"].iloc[i]) and
                  df_spx["EMA9"].iloc[i] > df_spx["EMA21"].iloc[i])
    ema_ok_sell = ("EMA9" in df_spx.columns and "EMA21" in df_spx.columns and
                   pd.notna(df_spx["EMA9"].iloc[i]) and pd.notna(df_spx["EMA21"].iloc[i]) and
                   df_spx["EMA9"].iloc[i] < df_spx["EMA21"].iloc[i])
    rsi_ok_buy = rsi < BUY_RSI_MAX
    rsi_ok_sell = rsi > SELL_RSI_MIN
    vix_ok_buy = vix_value < VIX_BUY_MAX
    vix_ok_sell = vix_value >= VIX_SELL_MIN
    aapl_green, msft_green, qqq_above_vwap = _get_confirmation_bools(data)
    price_above_vwap = (vwap is not None and not pd.isna(vwap) and float(close) > float(vwap))
    price_below_vwap = (vwap is not None and not pd.isna(vwap) and float(close) < float(vwap))

    # Per-condition bools: BUY [0]=VWAP, [1]=RSI, [2]=EMA above/below (sustained), [3]=MACD, [4]=VIX, [5]=AAPL, [6]=MSFT, [7]=QQQ
    buy_bools = [price_above_vwap, rsi_ok_buy, ema_ok_buy, macd_above, vix_ok_buy, aapl_green, msft_green, qqq_above_vwap]
    sell_bools = [price_below_vwap, rsi_ok_sell, ema_ok_sell, macd_below, vix_ok_sell, not aapl_green, not msft_green, not qqq_above_vwap]

    n_buy_enabled = sum(enabled_buy)
    n_sell_enabled = sum(enabled_sell)
    # Single condition: if only one enabled and it's met, fire
    if n_buy_enabled == 1:
        idx = next((j for j in range(8) if enabled_buy[j]), 0)
        if buy_bools[idx]:
            return "BUY", {"rsi_ok": rsi_ok_buy, "macd_ok": macd_above, "vix_ok": vix_ok_buy, "ema_ok": ema_ok_buy}
    if n_sell_enabled == 1:
        idx = next((j for j in range(8) if enabled_sell[j]), 0)
        if sell_bools[idx]:
            return "SELL", {"rsi_ok": rsi_ok_sell, "macd_ok": macd_below, "vix_ok": vix_ok_sell, "ema_ok": ema_ok_sell}

    if n_buy_enabled == 0 and n_sell_enabled == 0:
        return "WAIT", None

    # Core = indices 1..4 (RSI, EMA cross, MACD, VIX), Conf = 5..7
    n_core_buy = sum(enabled_buy[1:5])
    n_conf_buy = sum(enabled_buy[5:8])
    need_core_buy = math.ceil(0.75 * n_core_buy) if n_core_buy else 0
    need_conf_buy = 1 if n_conf_buy else 0
    core_buy_met = sum(buy_bools[j] for j in range(1, 5) if enabled_buy[j])
    conf_buy_met = sum(buy_bools[j] for j in range(5, 8) if enabled_buy[j])
    buy_ok = (need_core_buy <= core_buy_met and need_conf_buy <= conf_buy_met)
    if enabled_buy[1] and not rsi_ok_buy:
        buy_ok = False

    n_core_sell = sum(enabled_sell[1:5])
    n_conf_sell = sum(enabled_sell[5:8])
    need_core_sell = math.ceil(0.75 * n_core_sell) if n_core_sell else 0
    need_conf_sell = 1 if n_conf_sell else 0
    core_sell_met = sum(sell_bools[j] for j in range(1, 5) if enabled_sell[j])
    conf_sell_met = sum(sell_bools[j] for j in range(5, 8) if enabled_sell[j])
    sell_ok = (need_core_sell <= core_sell_met and need_conf_sell <= conf_sell_met)
    if enabled_sell[1] and not rsi_ok_sell:
        sell_ok = False

    if buy_ok and (not sell_ok or (core_buy_met == n_core_buy and conf_buy_met == n_conf_buy)):
        debug = {"rsi_ok": rsi_ok_buy, "macd_ok": macd_above, "vix_ok": vix_ok_buy, "ema_ok": ema_ok_buy}
        if n_core_buy and n_conf_buy and core_buy_met == n_core_buy and conf_buy_met == n_conf_buy:
            return "STRONG BUY", debug
        return "BUY", debug
    if sell_ok:
        debug = {"rsi_ok": rsi_ok_sell, "macd_ok": macd_below, "vix_ok": vix_ok_sell, "ema_ok": ema_ok_sell}
        if n_core_sell and n_conf_sell and core_sell_met == n_core_sell and conf_sell_met == n_conf_sell:
            return "STRONG SELL", debug
        return "SELL", debug
    return "WAIT", None


def _qwap(df: pd.DataFrame) -> pd.Series:
    """VWAP for any OHLCV DataFrame."""
    if df.empty or "High" not in df.columns or "Volume" not in df.columns:
        return pd.Series(dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, pd.NA).ffill()


def get_signal_strength_conditions(df_spx: pd.DataFrame, vix_value: float, data: dict) -> tuple:
    """
    Return (score 0-8, list of 8 bools for BUY, list of 8 bools for SELL).
    Order: Price vs VWAP, RSI zone 45/55, EMA cross (current bar), MACD above/below signal, VIX 25/22, AAPL, MSFT, QQQ.
    """
    buy_bools = [False] * 8
    sell_bools = [False] * 8
    if df_spx.empty or len(df_spx) < 2:
        return 0, buy_bools, sell_bools
    i = len(df_spx) - 1
    row = df_spx.iloc[i]
    close = row["Close"]
    vwap = row.get("VWAP")
    # RSI: resolve column (RSI or RSI_14) and coerce to scalar float so comparison is bool, not Series
    rsi_val = _rsi_scalar_at_row(df_spx, i)
    # 1. Price above/below VWAP (coerce to scalar so we get bool, not Series)
    if vwap is not None and not pd.isna(vwap):
        try:
            c, v = float(close), float(vwap)
            buy_bools[0] = bool(c > v)
            sell_bools[0] = bool(c < v)
        except (TypeError, ValueError):
            pass
    # 2. RSI in BUY zone (<45) or SELL zone (>55)
    if rsi_val is not None:
        buy_bools[1] = bool(rsi_val < BUY_RSI_MAX)
        sell_bools[1] = bool(rsi_val > SELL_RSI_MIN)
    # 3. EMA 9 above/below EMA 21 (sustained on current bar)
    if "EMA9" in df_spx.columns and "EMA21" in df_spx.columns:
        ema9_i = df_spx["EMA9"].iloc[i]
        ema21_i = df_spx["EMA21"].iloc[i]
        if pd.notna(ema9_i) and pd.notna(ema21_i):
            buy_bools[2] = bool(float(ema9_i) > float(ema21_i))
            sell_bools[2] = bool(float(ema9_i) < float(ema21_i))
    # 4. MACD above/below signal (sustained, not just cross)
    mc, sc = _macd_signal_cols(df_spx)
    if mc and sc:
        buy_bools[3] = bool(df_spx[mc].iloc[i] > df_spx[sc].iloc[i])
        sell_bools[3] = bool(df_spx[mc].iloc[i] < df_spx[sc].iloc[i])
    # 5. VIX < 25 (BUY) / >= 22 (SELL)
    if vix_value is not None:
        buy_bools[4] = bool(vix_value < VIX_BUY_MAX)
        sell_bools[4] = bool(vix_value >= VIX_SELL_MIN)
    # 6–8. AAPL, MSFT, QQQ
    aapl_green, msft_green, qqq_above_vwap = _get_confirmation_bools(data)
    buy_bools[5] = bool(aapl_green)
    sell_bools[5] = bool(not aapl_green)
    buy_bools[6] = bool(msft_green)
    sell_bools[6] = bool(not msft_green)
    buy_bools[7] = bool(qqq_above_vwap)
    sell_bools[7] = bool(not qqq_above_vwap)
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


def _dashboard_header():
    """Title + Backtests hint. No st.page_link or any link to Backtests here.
    The top-bar 'Analytics' on Streamlit Cloud is Streamlit's own UI and causes 404; navigate via sidebar only."""
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    with st.sidebar:
        if "dashboard_interval" not in st.session_state:
            st.session_state.dashboard_interval = INTERVAL_OPTIONS[DEFAULT_INTERVAL_INDEX][0]
            st.session_state.dashboard_interval_label = INTERVAL_OPTIONS[DEFAULT_INTERVAL_INDEX][1]
        interval_labels = [opt[1] for opt in INTERVAL_OPTIONS]
        interval_values = [opt[0] for opt in INTERVAL_OPTIONS]
        chosen_idx = st.selectbox(
            "Timeframe",
            range(len(INTERVAL_OPTIONS)),
            format_func=lambda i: interval_labels[i],
            index=interval_values.index(st.session_state.dashboard_interval) if st.session_state.dashboard_interval in interval_values else DEFAULT_INTERVAL_INDEX,
            key="dashboard_interval_select",
        )
        st.session_state.dashboard_interval = interval_values[chosen_idx]
        st.session_state.dashboard_interval_label = interval_labels[chosen_idx]
        st.caption("📊 **Backtests & charts:** click **Backtests** in the sidebar list.")
    head_col1, head_col2 = st.columns([4, 1])
    with head_col1:
        st.title("SPX Day Trading Dashboard")
        interval_label = st.session_state.get("dashboard_interval_label", "5 min")
        st.caption(f"{interval_label} intraday · RSI · MACD · VWAP · EMA · BUY/SELL when RSI/VWAP/MACD cross + VIX filter · Best windows 10:00–11:30 & 14:30–15:30 EST")
    with head_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.caption("📊 **Backtests** (backtests & charts): open the **sidebar** (←) and click **Backtests** in the list.")


def run_dashboard():
    _dashboard_header()
    _run_dashboard_body()


def _run_dashboard_body():
    """Dashboard content (metrics, panels, charts). Can run inside @st.fragment for auto-refresh."""
    is_market_open, show_date, data_label = _market_open_and_display_date()
    show_date_str = show_date.isoformat() if hasattr(show_date, "isoformat") else str(show_date)
    today_et = datetime.now(EST).date()
    # Simulated intraday: one position at a time; reset state when date changes
    if "sim_trades" not in st.session_state:
        st.session_state.sim_trades = []
    if "sim_position" not in st.session_state:
        st.session_state.sim_position = None
    if "sim_date" not in st.session_state:
        st.session_state.sim_date = None
    if st.session_state.sim_date != today_et:
        st.session_state.sim_trades = []
        st.session_state.sim_position = None
        st.session_state.sim_date = today_et
    interval = st.session_state.get("dashboard_interval", "5m")
    interval_label = st.session_state.get("dashboard_interval_label", "5 min")
    if not is_market_open:
        data = fetch_all_intraday(show_date_str, interval)
    else:
        data = fetch_all_intraday(None, interval)
    spx = data["spx"]
    vix_df = data["vix"]
    vix_value = data.get("vix_value")
    rate_limited = data.get("rate_limited", False)
    if spx.empty or len(spx) < 2:
        now_et_str = datetime.now(EST).strftime("%I:%M %p ET")
        if rate_limited:
            st.error(
                "**Yahoo Finance rate limit** — Too many requests. Wait 1–2 minutes, then click **Retry fetch** below. "
                "The app caches data for 5 minutes to reduce how often we ask for new data."
            )
        else:
            st.warning(
                f"Not enough SPX {interval_label} data for this session. "
                f"**Current time: {now_et_str}** — Market hours are 9:30 AM–4:00 PM ET. "
                f"If the market is open, data may be delayed; otherwise try again during market hours or check back for the previous trading day."
            )
        if st.button("Retry fetch", key="retry_spx_fetch"):
            fetch_all_intraday.clear()
            fetch_intraday.clear()
            st.rerun()
        return
    spx = add_indicators(spx)
    # Rehydrate condition checkboxes from file when any key is missing (e.g. after navigating back from another page)
    need_load = any(
        "buy_enabled_%d" % j not in st.session_state or "sell_enabled_%d" % j not in st.session_state
        for j in range(8)
    )
    if need_load:
        eb, es = _load_condition_flags()
        for j in range(8):
            st.session_state["buy_enabled_%d" % j] = eb[j]
            st.session_state["sell_enabled_%d" % j] = es[j]
    enabled_buy = [st.session_state.get("buy_enabled_%d" % j, True) for j in range(8)]
    enabled_sell = [st.session_state.get("sell_enabled_%d" % j, True) for j in range(8)]
    signal, signal_debug = get_signal(spx, vix_value, data, enabled_buy, enabled_sell)
    now_est = datetime.now(EST).strftime("%Y-%m-%d %H:%M ET")
    rsi_val = _rsi_scalar_at_row(spx, len(spx) - 1)
    if len(spx) < 10:
        st.info(f"⏳ Market just opened — {len(spx)} bars so far.")

    # Market status banner
    if is_market_open:
        st.markdown(
            f'<div class="market-banner" style="background: linear-gradient(90deg, rgba(46,204,113,0.25), rgba(46,204,113,0.1)); border: 1px solid #2ecc71; border-radius: 8px; margin-bottom: 1rem;">'
            f'<span>🟢 <strong>Market Open</strong> — Live data updating every {interval_label}</span></div>',
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

    # Best trading windows reminder (highlight current time if inside)
    now_et = datetime.now(EST)
    t = now_et.time()
    in_w1 = dt_time(10, 0) <= t <= dt_time(11, 30)
    in_w2 = dt_time(14, 30) <= t <= dt_time(15, 30)
    in_window = in_w1 or in_w2
    window_note = "10:00–11:30 AM ET and 2:30–3:30 PM ET"
    if in_window:
        st.markdown(f'<div style="background:rgba(0,212,170,0.2); border:1px solid #00d4aa; border-radius:8px; padding:8px 12px; margin-bottom:0.5rem;">'
                    f'✅ <strong>Best trading windows:</strong> {window_note} — <span style="color:#00d4aa;">You are in a preferred window.</span></div>', unsafe_allow_html=True)
    else:
        st.caption(f"Best trading windows: {window_note}")

    # Always: Top row — SPX, VIX, RSI, Signal, Time (ET)
    last_close = spx["Close"].iloc[-1]
    spx_price = float(last_close.squeeze()) if hasattr(last_close, "squeeze") else float(last_close)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("SPX", f"${spx_price:,.2f}")
    with c2:
        vix_str = f"{vix_value:.1f}" if vix_value is not None else "—"
        st.metric("VIX", vix_str)
    with c3:
        rsi_str = f"{rsi_val:.1f}" if rsi_val is not None else "—"
        st.metric("RSI", rsi_str)
    with c4:
        # Signal box colors: STRONG BUY bright green, BUY green, SELL red, STRONG SELL dark red, WAIT gray
        if signal == "STRONG BUY":
            bg, border = "rgba(0,255,127,0.35)", "#00ff7f"
        elif signal == "BUY":
            bg, border = "rgba(46,204,113,0.35)", "#2ecc71"
        elif signal == "STRONG SELL":
            bg, border = "rgba(139,0,0,0.4)", "#8b0000"
        elif signal == "SELL":
            bg, border = "rgba(231,76,60,0.35)", "#e74c3c"
        else:
            bg, border = "rgba(136,146,160,0.25)", "#8892a0"
        st.markdown(
            f'<div style="background:{bg}; border:1px solid {border}; border-radius:8px; padding:8px 12px; text-align:center;">'
            f'<span style="font-size:0.75rem; color:#8892a0;">SIGNAL</span><br>'
            f'<span class="signal-value" style="color:{border}; font-weight:700;">{signal}</span></div>',
            unsafe_allow_html=True,
        )
    with c5:
        st.metric("Time (ET)", now_est)

    # Signal strength: only count enabled conditions
    score_val, buy_bools, sell_bools = get_signal_strength_conditions(spx, vix_value, data)
    n_buy_enabled = sum(enabled_buy)
    n_sell_enabled = sum(enabled_sell)
    buy_count = sum(buy_bools[j] for j in range(8) if enabled_buy[j])
    sell_count = sum(sell_bools[j] for j in range(8) if enabled_sell[j])
    core_buy_met = sum(buy_bools[j] for j in range(1, 5) if enabled_buy[j])
    conf_buy_met = sum(buy_bools[j] for j in range(5, 8) if enabled_buy[j])
    n_core_buy = sum(enabled_buy[1:5])
    n_conf_buy = sum(enabled_buy[5:8])
    need_core_buy = math.ceil(0.75 * n_core_buy) if n_core_buy else 0
    need_conf_buy = 1 if n_conf_buy else 0
    core_sell_met = sum(sell_bools[j] for j in range(1, 5) if enabled_sell[j])
    conf_sell_met = sum(sell_bools[j] for j in range(5, 8) if enabled_sell[j])
    n_core_sell = sum(enabled_sell[1:5])
    n_conf_sell = sum(enabled_sell[5:8])
    need_core_sell = math.ceil(0.75 * n_core_sell) if n_core_sell else 0
    need_conf_sell = 1 if n_conf_sell else 0
    show_buy = signal in ("BUY", "STRONG BUY") or (signal == "WAIT" and buy_count >= sell_count)
    buy_ready = "🔥 READY TO FIRE" if (need_core_buy <= core_buy_met and need_conf_buy <= conf_buy_met) else f"Core: {core_buy_met}/{n_core_buy} needed | Conf: {conf_buy_met}/{n_conf_buy} needed"
    sell_ready = "🔥 READY TO FIRE" if (need_core_sell <= core_sell_met and need_conf_sell <= conf_sell_met) else f"Core: {core_sell_met}/{n_core_sell} needed | Conf: {conf_sell_met}/{n_conf_sell} needed"
    buy_denom = max(1, n_buy_enabled)
    sell_denom = max(1, n_sell_enabled)
    display_score = buy_count if show_buy else sell_count
    display_score_denom = n_buy_enabled if show_buy else n_sell_enabled
    buy_pct = int(buy_count / buy_denom * 100)
    # BUY bar: always green shades (weak → strong)
    buy_bar_color = (
        "#1a5e35" if buy_count <= 2 else
        "#2e7d32" if buy_count <= 4 else
        "#00c853" if buy_count <= 6 else
        "#00e676"
    )
    sell_pct = int(sell_count / sell_denom * 100)
    # SELL bar: always red shades (weak → strong)
    sell_bar_color = (
        "#5e1a1a" if sell_count <= 2 else
        "#7d2e2e" if sell_count <= 4 else
        "#d32f2f" if sell_count <= 6 else
        "#ff3d57"
    )
    bar_col1, bar_col2 = st.columns(2)
    with bar_col1:
        st.markdown(
            f"""
            <div style='margin-bottom:4px'>
                <span style='color:#aaa; font-size:13px'>🟢 BUY strength: {buy_count}/{n_buy_enabled}</span>
            </div>
            <div style='background:#1a1a1a; border-radius:6px; height:14px; width:100%; margin-bottom:6px'>
                <div style='background:{buy_bar_color}; width:{buy_pct}%; height:14px; border-radius:6px'></div>
            </div>
            <div style='margin-bottom:12px'>
                <small style='color:#888'>{buy_ready}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with bar_col2:
        st.markdown(
            f"""
            <div style='margin-bottom:4px'>
                <span style='color:#aaa; font-size:13px'>🔴 SELL strength: {sell_count}/{n_sell_enabled}</span>
            </div>
            <div style='background:#1a1a1a; border-radius:6px; height:14px; width:100%; margin-bottom:6px'>
                <div style='background:{sell_bar_color}; width:{sell_pct}%; height:14px; border-radius:6px'></div>
            </div>
            <div style='margin-bottom:12px'>
                <small style='color:#888'>{sell_ready}</small>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("<hr style='border:1px solid #333; margin: 8px 0'>", unsafe_allow_html=True)
    # Sustained EMA for 9th row in both columns (display only)
    i_last = len(spx) - 1
    sustained_ema_buy = sustained_ema_sell = False
    if i_last >= 0 and "EMA9" in spx.columns and "EMA21" in spx.columns:
        ema9 = spx["EMA9"].iloc[i_last]
        ema21 = spx["EMA21"].iloc[i_last]
        if pd.notna(ema9) and pd.notna(ema21):
            sustained_ema_buy = float(ema9) > float(ema21)
            sustained_ema_sell = float(ema9) < float(ema21)
    # BUY/SELL panels with checkboxes — only checked conditions count toward signal
    col_buy, col_sell = st.columns(2)
    with col_buy:
        bg = "rgba(46,204,113,0.12)" if (need_core_buy <= core_buy_met and need_conf_buy <= conf_buy_met) else "transparent"
        st.markdown(f"<div class=\"buy-sell-panel\" style=\"background:{bg}; border-radius:8px; padding:10px 12px; border:1px solid #1e2a35;\"><p style=\"margin:0 0 8px 0;\"><strong>🟢 BUY Conditions</strong></p></div>", unsafe_allow_html=True)
        if st.button("Reset to default", key="reset_buy_conditions", use_container_width=True):
            for j in range(8):
                st.session_state["buy_enabled_%d" % j] = True
            _save_condition_flags([True] * 8, [st.session_state.get("sell_enabled_%d" % j, True) for j in range(8)])
            st.rerun()
        for j in range(8):
            if j == 5:
                st.markdown("<div style='height:1px; background:#1e2a35; margin:6px 0;'></div>", unsafe_allow_html=True)
            cb_col, label_col = st.columns([1, 8])
            with cb_col:
                st.checkbox(" ", key="buy_enabled_%d" % j, label_visibility="collapsed")
            with label_col:
                icon = "✅" if buy_bools[j] else "❌"
                enabled = st.session_state.get("buy_enabled_%d" % j, True)
                opacity = "1" if enabled else "0.45"
                st.markdown(f"<div style='font-size:13px; color:#e0e0e0; padding:2px 0 6px 0; opacity:{opacity};'>{icon} {BUY_LABELS[j].replace('<', '&lt;').replace('>', '&gt;')}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1px; background:#1e2a35; margin:6px 0;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px; color:#e0e0e0; padding:2px 0 6px 0;'>✅ EMA 9 above EMA 21 (sustained, info only)</div>" if sustained_ema_buy else "<div style='font-size:13px; color:#888; padding:2px 0 6px 0;'>❌ EMA 9 above EMA 21 (sustained, info only)</div>", unsafe_allow_html=True)
        st.markdown(f"<p style=\"margin:12px 0 0 0; color:#8892a0; font-size:0.85em;\">→ BUY needs {need_core_buy}+ core + {need_conf_buy} confirmation (of {n_core_buy} core, {n_conf_buy} conf enabled)</p>", unsafe_allow_html=True)
    with col_sell:
        bg = "rgba(231,76,60,0.12)" if (need_core_sell <= core_sell_met and need_conf_sell <= conf_sell_met) else "transparent"
        st.markdown(f"<div class=\"buy-sell-panel\" style=\"background:{bg}; border-radius:8px; padding:10px 12px; border:1px solid #1e2a35;\"><p style=\"margin:0 0 8px 0;\"><strong>🔴 SELL Conditions</strong></p></div>", unsafe_allow_html=True)
        if st.button("Reset to default", key="reset_sell_conditions", use_container_width=True):
            for j in range(8):
                st.session_state["sell_enabled_%d" % j] = True
            _save_condition_flags([st.session_state.get("buy_enabled_%d" % j, True) for j in range(8)], [True] * 8)
            st.rerun()
        for j in range(8):
            if j == 5:
                st.markdown("<div style='height:1px; background:#1e2a35; margin:6px 0;'></div>", unsafe_allow_html=True)
            cb_col, label_col = st.columns([1, 8])
            with cb_col:
                st.checkbox(" ", key="sell_enabled_%d" % j, label_visibility="collapsed")
            with label_col:
                icon = "✅" if sell_bools[j] else "❌"
                enabled = st.session_state.get("sell_enabled_%d" % j, True)
                opacity = "1" if enabled else "0.45"
                st.markdown(f"<div style='font-size:13px; color:#e0e0e0; padding:2px 0 6px 0; opacity:{opacity};'>{icon} {SELL_LABELS[j].replace('<', '&lt;').replace('>', '&gt;')}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:1px; background:#1e2a35; margin:6px 0;'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:13px; color:#e0e0e0; padding:2px 0 6px 0;'>✅ EMA 9 below EMA 21 (sustained, info only)</div>" if sustained_ema_sell else "<div style='font-size:13px; color:#888; padding:2px 0 6px 0;'>❌ EMA 9 below EMA 21 (sustained, info only)</div>", unsafe_allow_html=True)
        st.markdown(f"<p style=\"margin:12px 0 0 0; color:#8892a0; font-size:0.85em;\">→ SELL needs {need_core_sell}+ core + {need_conf_sell} confirmation (of {n_core_sell} core, {n_conf_sell} conf enabled)</p>", unsafe_allow_html=True)
    # Persist checkbox state to file (survives refresh)
    _save_condition_flags(
        [st.session_state.get("buy_enabled_%d" % j, True) for j in range(8)],
        [st.session_state.get("sell_enabled_%d" % j, True) for j in range(8)],
    )

    # Last signal fired at (for alerts)
    last_fired = st.session_state.get("slack_last_signal_time")
    if last_fired is not None:
        st.caption(f"Last signal fired at: {last_fired.strftime('%H:%M ET')}")

    # Debug: show core condition values when BUY or SELL fires (signal score is never used for signal)
    if signal in ("BUY", "STRONG BUY", "SELL", "STRONG SELL") and signal_debug is not None:
        with st.expander("Debug: core conditions (used for signal only)"):
            d = signal_debug
            st.write(f"**rsi_ok:** {d.get('rsi_ok', '—')} (BUY: RSI<45; SELL: RSI>55)")
            st.write(f"**macd_ok:** {d.get('macd_ok', '—')} (BUY: MACD>Signal; SELL: MACD<Signal)")
            st.write(f"**vix_ok:** {d.get('vix_ok', '—')} (BUY: VIX<25; SELL: VIX≥22)")
            st.write(f"**ema_ok:** {d.get('ema_ok', '—')} (BUY: EMA9>EMA21; SELL: EMA9<EMA21)")
            st.caption("Signal is determined only by these 4 core conditions + confirmations; the 6/8 score is for display only.")

    # Slack: send notification once when the selected (enabled) BUY or SELL strategy is met
    # Signal is already computed from enabled_buy/enabled_sell in get_signal(); BUY/SELL/STRONG = strategy met
    base_type = _signal_base_type(signal)
    if signal in ("BUY", "SELL", "STRONG BUY", "STRONG SELL") and base_type and _should_send_signal_alert(base_type):
        try:
            spx_p = spx["Close"].iloc[-1]
            r_str = f"{rsi_val:.1f}" if rsi_val is not None else "—"
            v_str = f"{vix_value:.1f}" if vix_value is not None else "—"
            time_short = datetime.now(EST).strftime("%I:%M %p ET").lstrip("0")
            # Only list conditions that are both enabled (selected) and currently met
            conditions_met = [CONDITION_LABELS[k] for k in range(8) if (enabled_buy[k] or enabled_sell[k]) and (buy_bools[k] or sell_bools[k])]
            cond_str = ", ".join(conditions_met) if conditions_met else "—"
            mc, sc = _macd_signal_cols(spx)
            macd_status = "MACD > Signal" if (mc and sc and spx[mc].iloc[-1] > spx[sc].iloc[-1]) else "MACD < Signal"
            emoji = "🟢" if "BUY" in signal else "🔴"
            msg = (
                f"{emoji} *{signal} — SPX Day Trading*\n"
                f"💰 *SPX Price:* ${spx_p:,.2f}\n"
                f"📊 *Selected conditions met:* {display_score}/{display_score_denom}\n"
                f"⏰ *Time:* {time_short}\n"
                f"📈 *RSI:* {r_str} | *VIX:* {v_str}\n"
                f"📉 *MACD:* {macd_status}\n"
                f"✅ Conditions met: {cond_str}\n"
                f"⚠️ Not financial advice."
            )
            if send_slack_alert(msg):
                _record_signal_sent(base_type)
        except Exception:
            pass

    # Slack: daily summary at 4:05–4:20 PM ET (once per day)
    if _should_send_daily_summary() and not spx.empty and len(spx) >= 2:
        try:
            stats = _daily_signal_stats(spx, vix_value, data, enabled_buy, enabled_sell)
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

    # Simulated intraday: open position on BUY/STRONG BUY, close on SELL/STRONG SELL
    current_price = float(spx["Close"].iloc[-1])
    last_ts = spx.index[-1]
    if signal in ("BUY", "STRONG BUY") and st.session_state.sim_position is None:
        st.session_state.sim_position = {
            "entry_time": last_ts,
            "entry_price": current_price,
            "amount": SIM_POSITION_SIZE,
            "signal": signal,
        }
    elif signal in ("SELL", "STRONG SELL") and st.session_state.sim_position is not None:
        pos = st.session_state.sim_position
        exit_price = current_price
        pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
        pnl_dollars = pos["amount"] * (exit_price / pos["entry_price"] - 1)
        entry_ts = pos["entry_time"]
        try:
            delta = pd.Timestamp(last_ts) - pd.Timestamp(entry_ts)
            total_mins = int(delta.total_seconds() / 60)
            time_str = f"{total_mins // 60}h {total_mins % 60}m"
        except Exception:
            time_str = "—"
        buy_time_str = pd.Timestamp(entry_ts).strftime("%I:%M %p") if hasattr(pd.Timestamp(entry_ts), "strftime") else str(entry_ts)
        sell_time_str = pd.Timestamp(last_ts).strftime("%I:%M %p") if hasattr(pd.Timestamp(last_ts), "strftime") else str(last_ts)
        st.session_state.sim_trades.append({
            "Signal": f"{pos['signal']} → {signal}",
            "Buy time": buy_time_str,
            "Sell time": sell_time_str,
            "Entry price": pos["entry_price"],
            "Exit price": exit_price,
            "Time in trade": time_str,
            "P&L %": round(pnl_pct, 2),
            "P&L $": round(pnl_dollars, 2),
        })
        st.session_state.sim_position = None

    st.markdown("---")

    # When closed: summary card (SPX close, % change, BUY/SELL counts, best signal)
    if not is_market_open:
        stats = _daily_signal_stats(spx, vix_value, data, enabled_buy, enabled_sell)
        # Close any open sim position at session close
        if st.session_state.sim_position is not None and stats:
            pos = st.session_state.sim_position
            exit_price = float(stats["close"])
            pnl_pct = (exit_price / pos["entry_price"] - 1) * 100
            pnl_dollars = pos["amount"] * (exit_price / pos["entry_price"] - 1)
            entry_ts = pos["entry_time"]
            last_ts_close = spx.index[-1]
            try:
                delta = pd.Timestamp(last_ts_close) - pd.Timestamp(entry_ts)
                total_mins = int(delta.total_seconds() / 60)
                time_str = f"{total_mins // 60}h {total_mins % 60}m"
            except Exception:
                time_str = "—"
            buy_time_str = pd.Timestamp(entry_ts).strftime("%I:%M %p") if hasattr(pd.Timestamp(entry_ts), "strftime") else str(entry_ts)
            sell_time_str = pd.Timestamp(last_ts_close).strftime("%I:%M %p") if hasattr(pd.Timestamp(last_ts_close), "strftime") else str(last_ts_close)
            st.session_state.sim_trades.append({
                "Signal": f"{pos['signal']} → EOD",
                "Buy time": buy_time_str,
                "Sell time": sell_time_str,
                "Entry price": pos["entry_price"],
                "Exit price": exit_price,
                "Time in trade": time_str,
                "P&L %": round(pnl_pct, 2),
                "P&L $": round(pnl_dollars, 2),
            })
            st.session_state.sim_position = None
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

    # Simulated intraday trading (fake $50k; $10k per position; track P&L as if we traded on signals)
    st.subheader("Simulated intraday trading")
    st.caption(f"Starting capital ${SIM_STARTING_CAPITAL:,} · Position size ${SIM_POSITION_SIZE:,} (simulation only — for intraday buy/sell same day)")
    total_pnl = sum(t["P&L $"] for t in st.session_state.sim_trades)
    ending_balance = SIM_STARTING_CAPITAL + total_pnl
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        st.metric("Starting capital", f"${SIM_STARTING_CAPITAL:,}")
    with sim_c2:
        st.metric("Today's P&L", f"${total_pnl:+,.2f}")
    with sim_c3:
        st.metric("Ending balance", f"${ending_balance:,.2f}")
    if st.session_state.sim_position is not None:
        pos = st.session_state.sim_position
        entry_t = pd.Timestamp(pos["entry_time"]).strftime("%I:%M %p") if hasattr(pd.Timestamp(pos["entry_time"]), "strftime") else str(pos["entry_time"])
        st.info(f"📌 **Open position:** entered at ${pos['entry_price']:,.2f} at {entry_t} (${pos['amount']:,}) — waiting for SELL signal or EOD.")
    if st.session_state.sim_trades:
        df_sim = pd.DataFrame(st.session_state.sim_trades)
        df_sim["Entry price"] = df_sim["Entry price"].apply(lambda x: f"${x:,.2f}")
        df_sim["Exit price"] = df_sim["Exit price"].apply(lambda x: f"${x:,.2f}")
        df_sim["P&L %"] = df_sim["P&L %"].apply(lambda x: f"{x:+.2f}%")
        df_sim["P&L $"] = df_sim["P&L $"].apply(lambda x: f"${x:+,.2f}")
        st.dataframe(df_sim, use_container_width=True, hide_index=True)
    else:
        st.caption("No completed trades yet. BUY opens a $10k position; SELL closes it and records P&L. Positions are closed at session end if still open.")

    st.markdown("---")

    # Which indicator charts to show: only for conditions the user has selected (checked)
    show_vwap = enabled_buy[0] or enabled_sell[0]   # Price above/below VWAP
    show_rsi = enabled_buy[1] or enabled_sell[1]   # RSI zone
    show_ema = enabled_buy[2] or enabled_sell[2]   # EMA 9 crossed EMA 21
    show_macd = enabled_buy[3] or enabled_sell[3]  # MACD above/below signal

    # SPX intraday: price chart — always Close; VWAP and EMAs only if that indicator is selected
    chart_spx = spx[~spx.index.duplicated(keep="first")].sort_index()
    title_parts = [f"SPX {interval_label}"]
    if show_vwap:
        title_parts.append("VWAP")
    if show_ema:
        title_parts.append("EMAs")
    chart_title = f"{' & '.join(title_parts)} — {data_label} (9:30 AM–4:00 PM ET only)"
    st.subheader(f"SPX {interval_label} price" + (" & VWAP" if show_vwap else "") + (" & EMA 9/21" if show_ema else ""))
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=chart_spx.index, y=chart_spx["Close"], name="SPX",
        mode="lines", line=dict(color="#00d4aa", width=2),
    ))
    if show_vwap and "VWAP" in chart_spx.columns:
        fig_price.add_trace(go.Scatter(x=chart_spx.index, y=chart_spx["VWAP"], name="VWAP", mode="lines", line=dict(color="#f39c12", width=1.5, dash="dash")))
    if show_ema and "EMA9" in chart_spx.columns:
        fig_price.add_trace(go.Scatter(x=chart_spx.index, y=chart_spx["EMA9"], name="EMA 9", mode="lines", line=dict(color="#9b59b6", width=1.2)))
    if show_ema and "EMA21" in chart_spx.columns:
        fig_price.add_trace(go.Scatter(x=chart_spx.index, y=chart_spx["EMA21"], name="EMA 21", mode="lines", line=dict(color="#3498db", width=1.2)))
    fig_price.update_layout(**PLOTLY_LAYOUT, title=chart_title, height=380, xaxis=AXIS, yaxis=AXIS)
    for s in trading_window_shapes(chart_spx):
        fig_price.add_shape(s)
    st.plotly_chart(fig_price, use_container_width=True)

    # RSI chart — only if RSI condition is selected
    if show_rsi:
        st.subheader("RSI (14)")
        rsi_col = "RSI" if "RSI" in spx.columns else None
        if rsi_col:
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=spx.index, y=spx[rsi_col], name="RSI", line=dict(color="#00d4aa", width=2)))
            fig_rsi.add_hline(y=BUY_RSI_MAX, line_dash="dash", line_color="#2ecc71", annotation_text="45 (buy zone)")
            fig_rsi.add_hline(y=SELL_RSI_MIN, line_dash="dash", line_color="#e74c3c", annotation_text="55 (sell zone)")
            fig_rsi.update_layout(**PLOTLY_LAYOUT, title="RSI (14)", height=280, xaxis=AXIS, yaxis=dict(**AXIS, range=[0, 100]))
            st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD chart — only if MACD condition is selected
    if show_macd:
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

    # Signal History (last 10 signals fired today)
    if "signals_history" not in st.session_state:
        st.session_state.signals_history = []
    if signal in ("BUY", "SELL", "STRONG BUY", "STRONG SELL"):
        cond_triggered = [CONDITION_LABELS[k] for k in range(8) if (enabled_buy[k] or enabled_sell[k]) and (buy_bools[k] or sell_bools[k])]
        st.session_state.signals_history.append({
            "Time (ET)": now_est,
            "Type": signal,
            "SPX": f"${spx_price:,.2f}",
            "Conditions": ", ".join(cond_triggered[:3]) + ("..." if len(cond_triggered) > 3 else ""),
        })
        st.session_state.signals_history = st.session_state.signals_history[-10:]
    st.subheader("Signal History")
    if st.session_state.signals_history:
        st.dataframe(pd.DataFrame(st.session_state.signals_history).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No signals yet. BUY: 3 of 4 core (RSI<45, MACD>Signal, VIX<25, EMA9>EMA21) + 1 confirmation. SELL: 3 of 4 core (RSI>55, MACD<Signal, VIX≥22, EMA9<EMA21) + 1 confirmation.")

    # Auto-refresh note
    st.markdown("---")
    st.caption(f"Data at {interval_label} timeframe. Green zones on chart: 10:00–11:30 AM & 2:30–3:30 PM EST.")


# Auto-refresh every 5 minutes when Streamlit supports run_every
try:
    use_fragment = hasattr(st, "fragment") and callable(getattr(st.fragment, "__call__", None))
    run_every_available = use_fragment
except Exception:
    run_every_available = False

if run_every_available:
    _dashboard_header()
    try:
        @st.fragment(run_every=timedelta(seconds=300))
        def _auto_refresh():
            _run_dashboard_body()
        _auto_refresh()
    except Exception:
        _run_dashboard_body()
else:
    run_dashboard()
