"""
Technical indicators for SPX Day Trading Dashboard.
Uses pandas-ta when available, with pandas fallbacks.
"""
import pandas as pd
import numpy as np

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    if HAS_PANDAS_TA:
        return ta.rsi(close, length=length)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1 / length, adjust=False).mean()
    al = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = ag / al.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    if HAS_PANDAS_TA:
        m = ta.macd(close, fast=fast, slow=slow, signal=signal)
        if m is not None and not m.empty:
            return m
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return pd.DataFrame({"MACD": macd_line, "MACD_signal": sig_line, "MACD_hist": hist})


def vwap(df: pd.DataFrame) -> pd.Series:
    """Requires High, Low, Close, Volume."""
    if "High" not in df.columns or "Low" not in df.columns or "Volume" not in df.columns:
        return pd.Series(dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, pd.NA).ffill()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def bollinger_bands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    if HAS_PANDAS_TA:
        bb = ta.bbands(close, length=length, std=std)
        if bb is not None and not bb.empty:
            return bb
    mid = close.ewm(span=length, adjust=False).mean()
    sd = close.ewm(span=length, adjust=False).std()
    upper = mid + std * sd
    lower = mid - std * sd
    return pd.DataFrame({"BBL": lower, "BBM": mid, "BBU": upper})


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range. Requires High, Low, Close."""
    if "High" not in df.columns or "Low" not in df.columns:
        return pd.Series(dtype=float)
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(span=length, adjust=False).mean()


def add_spx_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, VWAP, EMA9, EMA21, BB, ATR to SPX OHLCV DataFrame."""
    if df.empty or "Close" not in df.columns:
        return df
    df = df.copy()
    close = df["Close"]
    df["RSI"] = rsi(close, 14)
    macd_df = macd(close, 12, 26, 9)
    for c in macd_df.columns:
        df[c] = macd_df[c]
    df["VWAP"] = vwap(df)
    df["EMA9"] = ema(close, 9)
    df["EMA21"] = ema(close, 21)
    bb = bollinger_bands(close, 20, 2.0)
    for c in bb.columns:
        df[c] = bb[c]
    df["ATR14"] = atr(df, 14)
    return df


def add_confirmation_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, % change today, and above/below VWAP for confirmation tickers."""
    if df.empty or "Close" not in df.columns:
        return df
    df = df.copy()
    close = df["Close"]
    df["RSI"] = rsi(close, 14)
    if "Open" in df.columns and len(df) > 0:
        first_open = df["Open"].iloc[0]
        df["PctChange"] = ((close - first_open) / first_open * 100) if first_open and first_open != 0 else 0
    else:
        df["PctChange"] = 0
    df["VWAP"] = vwap(df)
    df["AboveVWAP"] = close > df["VWAP"] if "VWAP" in df.columns else pd.Series(False, index=df.index)
    return df
