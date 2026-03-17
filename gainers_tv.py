"""
Top Gainers — TradingView screener for leading gainers by % change.
S&P 500 gainers via yfinance (rank by % change from prior close).
Chart data (OHLCV, MACD, ADX, EMA) via yfinance for deep analysis.
"""
import contextlib
import csv
import io
import urllib.request
from typing import List, Dict, Any, Optional
import pandas as pd

try:
    from tradingview_screener import Query
    HAS_TV = True
except ImportError:
    HAS_TV = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

# Cache for S&P 500 tickers (list of symbols, yfinance-style e.g. BRK-B)
_SP500_TICKERS_CACHE: Optional[List[str]] = None
SP500_CSV_URL = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
# Fallback if CSV unavailable (subset of S&P 500)
SP500_FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "BRK-B", "UNH",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP", "KO", "COST",
    "WMT", "MCD", "CSCO", "ACN", "ABT", "TMO", "AVGO", "DHR", "NEE", "NKE", "BMY",
    "PM", "RTX", "HON", "INTC", "AMD", "AMGN", "ADBE", "CRM", "TXN", "QCOM", "LOW",
    "UPS", "IBM", "ORCL", "CAT", "GE", "BA", "XOM", "DIS", "NFLX", "PYPL", "SBUX",
    "MDT", "GILD", "INTU", "AMAT", "ISRG", "VRTX", "REGN", "LMT", "BKNG", "ADP",
    "DE", "C", "GS", "AXP", "PLD", "SYK", "MMC", "CI", "MO", "SO", "DUK", "BDX",
    "ELV", "CB", "CL", "PGR", "ZTS", "APD", "APTV", "AON", "CMCSA", "BSX", "EOG",
    "ITW", "SLB", "PSA", "KLAC", "SNPS", "CDNS", "LRCX", "ADI", "TGT",
]


def _get(r: pd.Series, *keys: str, default=None):
    """Get first existing key from row (case-insensitive); treat NaN/None as missing."""
    for k in keys:
        if k in r.index:
            v = r[k]
            if v is not None and not (isinstance(v, float) and pd.isna(v)):
                return v
        for idx in r.index:
            if str(idx).lower() == k.lower():
                v = r[idx]
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    return v
                break
    return default


def _num(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def get_sp500_tickers() -> List[str]:
    """Return list of S&P 500 tickers (yfinance-style, e.g. BRK-B). Cached after first fetch."""
    global _SP500_TICKERS_CACHE
    if _SP500_TICKERS_CACHE is not None:
        return _SP500_TICKERS_CACHE
    try:
        with urllib.request.urlopen(SP500_CSV_URL, timeout=10) as resp:
            text = resp.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        symbols = []
        for row in reader:
            sym = (row.get("Symbol") or "").strip()
            if not sym:
                continue
            # yfinance uses hyphen for class shares (e.g. BRK-B not BRK.B)
            sym = sym.replace(".", "-")
            symbols.append(sym)
        if symbols:
            _SP500_TICKERS_CACHE = symbols
            return symbols
    except Exception:
        pass
    _SP500_TICKERS_CACHE = SP500_FALLBACK_TICKERS
    return SP500_FALLBACK_TICKERS


def get_top_gainers_sp500(limit: int = 50, timeframe: str = "1d") -> List[Dict[str, Any]]:
    """
    Top gainers among S&P 500 stocks by % change from previous close. Uses yfinance.
    timeframe: '1d' (change from prior close); '5' ignored for SP500 (we use daily).
    Returns list of dicts: symbol, price, change_pct, volume, gap_pct, etc.
    """
    if not HAS_YF:
        return _demo_gainers(limit)
    tickers = get_sp500_tickers()
    chunk_size = 80
    all_rows: List[Dict[str, Any]] = []
    with contextlib.redirect_stderr(io.StringIO()):
        for i in range(0, len(tickers), chunk_size):
            chunk = tickers[i : i + chunk_size]
            try:
                df = yf.download(
                    chunk,
                    period="5d",
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    timeout=15,
                )
            except Exception:
                continue
            if df is None or df.empty or len(df) < 2:
                continue
            # MultiIndex columns: (Ticker, Open/High/Low/Close/Volume) when len(chunk) > 1
            if len(chunk) == 1:
                sym = chunk[0]
                try:
                    close_prev = df["Close"].iloc[-2]
                    close_curr = df["Close"].iloc[-1]
                    open_curr = df["Open"].iloc[-1]
                    vol = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else 0
                except (IndexError, KeyError):
                    continue
                if pd.isna(close_prev) or pd.isna(close_curr) or close_prev <= 0 or close_curr <= 0:
                    continue
                change_pct = (float(close_curr) - float(close_prev)) / float(close_prev) * 100
                gap_pct = (float(open_curr) - float(close_prev)) / float(close_prev) * 100 if not pd.isna(open_curr) else None
                all_rows.append({
                    "symbol": sym,
                    "price": round(float(close_curr), 4),
                    "change_pct": round(change_pct, 2),
                    "volume": vol,
                    "relative_volume": None,
                    "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
                    "float_shares": None,
                    "short_interest": None,
                })
            else:
                # MultiIndex: (ticker, ohlcv) or (ohlcv, ticker) depending on yfinance version
                for sym in chunk:
                    try:
                        def _col(c: str):
                            if (sym, c) in df.columns:
                                return df[(sym, c)]
                            if (c, sym) in df.columns:
                                return df[(c, sym)]
                            return None
                        close_ser = _col("Close")
                        open_ser = _col("Open")
                        vol_ser = _col("Volume")
                        if close_ser is None or len(close_ser) < 2:
                            continue
                        close_prev = close_ser.iloc[-2]
                        close_curr = close_ser.iloc[-1]
                        open_curr = open_ser.iloc[-1] if open_ser is not None and len(open_ser) else None
                        vol = int(vol_ser.iloc[-1]) if vol_ser is not None and len(vol_ser) else 0
                    except (IndexError, KeyError, TypeError):
                        continue
                    if pd.isna(close_prev) or pd.isna(close_curr) or close_prev <= 0 or close_curr <= 0:
                        continue
                    change_pct = (float(close_curr) - float(close_prev)) / float(close_prev) * 100
                    gap_pct = None
                    if open_curr is not None and not pd.isna(open_curr):
                        gap_pct = (float(open_curr) - float(close_prev)) / float(close_prev) * 100
                    all_rows.append({
                        "symbol": sym,
                        "price": round(float(close_curr), 4),
                        "change_pct": round(change_pct, 2),
                        "volume": vol,
                        "relative_volume": None,
                        "gap_pct": round(gap_pct, 2) if gap_pct is not None else None,
                        "float_shares": None,
                        "short_interest": None,
                    })
    all_rows.sort(key=lambda x: x["change_pct"], reverse=True)
    return all_rows[:limit]


def get_top_gainers(limit: int = 50, timeframe: str = "1d") -> List[Dict[str, Any]]:
    """
    Top gainers by % change from previous close. Uses TradingView screener.
    timeframe: '1d' (default, change from prior close) or '5' for 5-min (change|5).
    Returns list of dicts: symbol, price, change_pct, volume, rel_vol, gap_pct, float_shares (if available).
    """
    change_col = "change" if timeframe == "1d" else "change|5"
    if not HAS_TV:
        return _demo_gainers(limit)
    try:
        result = (
            Query()
            .select(
                "name",
                "close",
                "volume",
                change_col,
                "gap",
                "relative_volume_10d_calc",
            )
            .order_by(change_col, ascending=False)
            .limit(min(limit, 100))
            .get_scanner_data()
        )
        if isinstance(result, tuple):
            _, df = result
        else:
            df = result
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return _demo_gainers(limit)
        rows = []
        for _, r in df.iterrows():
            name = _get(r, "name", "Name") or _get(r, "ticker", "Ticker") or ""
            if pd.isna(name) or name is None or str(name).strip().lower() in ("", "nan", "none"):
                continue
            name = str(name).strip()
            if ":" in name:
                name = name.split(":")[-1]
            close = _num(_get(r, "close", "Close"))
            if close is None or close <= 0:
                continue
            ch = _num(_get(r, change_col, "change", "change|5"))
            gap_val = _num(_get(r, "gap", "Gap"))
            vol = _num(_get(r, "volume", "Volume"))
            rel_vol = _num(_get(r, "relative_volume_10d_calc"))
            rows.append({
                "symbol": name,
                "price": float(close),
                "change_pct": round(float(ch), 2) if ch is not None else 0.0,
                "volume": int(vol) if vol is not None else 0,
                "relative_volume": round(float(rel_vol), 2) if rel_vol is not None else None,
                "gap_pct": round(float(gap_val), 2) if gap_val is not None else None,
                "float_shares": None,
                "short_interest": None,
            })
        return rows[:limit]
    except Exception:
        return _demo_gainers(limit)


def _demo_gainers(limit: int) -> List[Dict[str, Any]]:
    """Demo top gainers when TradingView not available."""
    return [
        {"symbol": "AAPL", "price": 225.50, "change_pct": 2.5, "volume": 50_000_000, "relative_volume": 1.2, "gap_pct": 2.5, "float_shares": None, "short_interest": None},
        {"symbol": "NVDA", "price": 128.20, "change_pct": 4.1, "volume": 80_000_000, "relative_volume": 1.8, "gap_pct": 4.0, "float_shares": None, "short_interest": None},
        {"symbol": "TSLA", "price": 221.10, "change_pct": 3.2, "volume": 73_000_000, "relative_volume": 1.5, "gap_pct": 3.2, "float_shares": None, "short_interest": None},
        {"symbol": "AMD", "price": 156.40, "change_pct": 1.9, "volume": 76_000_000, "relative_volume": 1.1, "gap_pct": 1.8, "float_shares": None, "short_interest": None},
    ][:limit]


def _macd_series(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce").astype(float)
    e1 = close.ewm(span=fast, adjust=False).mean()
    e2 = close.ewm(span=slow, adjust=False).mean()
    macd_line = e1 - e2
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": macd_line, "MACD_signal": sig_line, "MACD_hist": macd_line - sig_line})


def _adx_series(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)
    close = pd.to_numeric(close, errors="coerce").astype(float)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1).astype(float)
    up = high - high.shift(1)
    down = low.shift(1) - low
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    # Avoid division by zero: use tiny epsilon instead of pd.NA so result stays float
    atr_safe = atr.replace(0.0, 1e-10)
    plus_di = (100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe)).astype(float)
    minus_di = (100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_safe)).astype(float)
    di_sum = (plus_di + minus_di).replace(0.0, 1e-10)
    dx = (100 * (plus_di - minus_di).abs() / di_sum).astype(float)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean().astype(float)
    return adx.fillna(0)


def fetch_stock_chart_data(
    ticker: str,
    period: str = "3mo",
    interval: Optional[str] = "1d",
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV for ticker and add MACD(12,26,9), ADX(14), EMA9, EMA21.
    Tries longer periods if needed so small/OTC tickers still get data. Min 5 bars required.
    """
    import time
    time.sleep(0.15)  # small delay to reduce yfinance rate limit when loading multiple tabs/symbols
    if not HAS_YF or not ticker or str(ticker).strip() in ("", "—"):
        return None
    sym = str(ticker).strip().upper()
    interval = interval or "1d"
    # For daily data, try longer periods if 3mo returns too little (e.g. OTC or new listings)
    periods_to_try = [period]
    if interval == "1d" and period in ("3mo", "1mo", "5d"):
        periods_to_try = [period, "6mo", "1y", "2y"]
    elif interval == "1d":
        periods_to_try = [period, "1y", "2y"]
    for p in periods_to_try:
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                # Use Ticker().history() for single ticker — returns single-level columns (no MultiIndex)
                ticker_obj = yf.Ticker(sym)
                df = ticker_obj.history(period=p, interval=interval, auto_adjust=True)
            if df is None or df.empty or len(df) < 5:
                continue
            # Flatten MultiIndex if present (e.g. some yfinance versions)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            req = ["Open", "High", "Low", "Close", "Volume"]
            if not all(c in df.columns for c in req):
                continue
            df = df.dropna(subset=["Close"], how="all").copy()
            if len(df) < 5:
                continue
            # Force numeric (yfinance can return object dtype)
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            close = df["Close"]
            if close.isna().all() or (close <= 0).all():
                continue
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            macd_df = _macd_series(close.astype(float), 12, 26, 9)
            for col in macd_df.columns:
                df[col] = macd_df[col]
            df["ADX"] = _adx_series(high, low, close.astype(float), 14)
            close_f = close.astype(float)
            df["EMA9"] = close_f.ewm(span=9, adjust=False).mean()
            df["EMA21"] = close_f.ewm(span=21, adjust=False).mean()
            return df
        except Exception:
            continue
    return None


# Universe of liquid tickers for backtest (scanner would have picked from these by % change)
GAINERS_BACKTEST_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "BRK-B", "UNH",
    "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "PEP",
    "KO", "COST", "WMT", "MCD", "CSCO", "ACN", "ABT", "TMO", "AVGO", "DHR",
    "NEE", "NKE", "BMY", "PM", "RTX", "HON", "INTC", "AMD", "AMGN", "ADBE",
    "CRM", "TXN", "QCOM", "LOW", "UPS", "IBM", "ORCL", "CAT", "GE", "BA",
    "SPY", "QQQ", "IWM", "XOM", "DIS", "NFLX", "PYPL", "SBUX", "MDT", "GILD",
    "INTU", "AMAT", "ISRG", "VRTX", "REGN", "LMT", "BKNG", "ADP", "DE", "C",
    "GS", "AXP", "PLD", "SYK", "MMC", "CI", "MO", "SO", "DUK", "BDX",
]


def run_gainers_backtest(
    from_date: str,
    to_date: str,
    top_n: int = 20,
    min_eod_pct: float = 2.0,
    universe: Optional[List[str]] = None,
) -> tuple:
    """
    Backtest: for each day, simulate "scanner would have alerted top N gainers by gap (open vs prev close)".
    Then check: did each alerted stock move up by at least min_eod_pct by end of day?
    Returns (summary_dict, details_list).
    """
    if not HAS_YF:
        return {"error": "yfinance required"}, []
    tickers = universe or GAINERS_BACKTEST_UNIVERSE[:50]  # limit for speed
    start = (pd.Timestamp(from_date) - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    end = pd.Timestamp(to_date).strftime("%Y-%m-%d")
    all_data = {}
    with contextlib.redirect_stderr(io.StringIO()):
        for sym in tickers:
            try:
                t = yf.Ticker(sym)
                df = t.history(start=start, end=end, auto_adjust=True)
                if df is not None and len(df) >= 2 and "Open" in df.columns and "Close" in df.columns:
                    all_data[sym] = df
            except Exception:
                continue
    if not all_data:
        return {"error": "No historical data downloaded"}, []

    details = []
    try:
        dates = pd.date_range(start=from_date, end=end, freq="B")
        for d in dates:
            d = d.date() if hasattr(d, "date") else d
            day_alerts = []
            for sym, hist in all_data.items():
                try:
                    hist = hist.copy()
                    idx = hist.index
                    idx_dates = [x.date() if hasattr(x, "date") else x for x in idx]
                    if d not in idx_dates:
                        continue
                    pos = list(idx_dates).index(d)
                    if pos < 1:
                        continue
                    prev_close = hist["Close"].iloc[pos - 1]
                    open_today = hist["Open"].iloc[pos]
                    close_today = hist["Close"].iloc[pos]
                    if pd.isna(prev_close) or pd.isna(open_today) or pd.isna(close_today) or prev_close <= 0 or open_today <= 0:
                        continue
                    gap_pct = (float(open_today) - float(prev_close)) / float(prev_close) * 100
                    eod_pct = (float(close_today) - float(open_today)) / float(open_today) * 100
                    day_alerts.append({"ticker": sym, "gap_pct": gap_pct, "eod_pct": eod_pct})
                except Exception:
                    continue
            day_alerts.sort(key=lambda x: x["gap_pct"], reverse=True)
            for a in day_alerts[:top_n]:
                win = a["eod_pct"] >= min_eod_pct
                details.append({
                    "date": str(d),
                    "ticker": a["ticker"],
                    "gap_pct": round(a["gap_pct"], 2),
                    "eod_pct": round(a["eod_pct"], 2),
                    "win": win,
                })
    except Exception as e:
        return {"error": str(e)}, []

    if not details:
        return {"error": "No backtest results (check date range)"}, []

    total = len(details)
    wins = sum(1 for x in details if x["win"])
    win_rate = (wins / total * 100) if total else 0
    avg_eod = sum(x["eod_pct"] for x in details) / total if total else 0
    days_run = len(set(x["date"] for x in details))
    summary = {
        "total_alerts": total,
        "wins": wins,
        "win_rate_pct": round(win_rate, 1),
        "avg_eod_pct": round(avg_eod, 2),
        "days_run": days_run,
        "min_eod_threshold": min_eod_pct,
        "top_n_per_day": top_n,
    }
    return summary, details
