"""
Signal scoring system for SPX Day Trading Dashboard.
BUY and SELL scores 0-8 with condition breakdown.
"""
import pandas as pd
import numpy as np

# Condition names for display
BUY_CONDITIONS = [
    "SPX above VWAP",
    "RSI below 40",
    "EMA 9 above EMA 21",
    "MACD above signal",
    "VIX below 20",
    "AAPL & MSFT green today",
    "QQQ above VWAP",
    "10Y yield stable",
]
SELL_CONDITIONS = [
    "SPX below VWAP",
    "RSI above 60",
    "EMA 9 below EMA 21",
    "MACD below signal",
    "VIX above 20",
    "AAPL & MSFT red today",
    "QQQ below VWAP",
    "10Y yield rising fast",
]


def _ema_cross_above(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1 or "EMA9" not in df.columns or "EMA21" not in df.columns:
        return False
    return df["EMA9"].iloc[idx] > df["EMA21"].iloc[idx] and df["EMA9"].iloc[idx - 1] <= df["EMA21"].iloc[idx - 1]


def _ema_cross_below(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1 or "EMA9" not in df.columns or "EMA21" not in df.columns:
        return False
    return df["EMA9"].iloc[idx] < df["EMA21"].iloc[idx] and df["EMA9"].iloc[idx - 1] >= df["EMA21"].iloc[idx - 1]


def _macd_cols(df: pd.DataFrame):
    macd_c = [c for c in df.columns if c == "MACD" or (str(c).startswith("MACD") and "MACDh" not in str(c) and "MACDs" not in str(c))][:1]
    sig_c = [c for c in df.columns if "MACDs" in str(c) or ("MACD" in str(c) and "signal" in str(c).lower() and "hist" not in str(c).lower())][:1]
    if not macd_c:
        macd_c = [c for c in df.columns if "MACD" in str(c) and "MACDh" not in str(c) and "MACDs" not in str(c)][:1]
    return (macd_c[0] if macd_c else None, sig_c[0] if sig_c else None)


def _macd_cross_above(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    mc, sc = _macd_cols(df)
    if mc is None or sc is None:
        return False
    return df[mc].iloc[idx] > df[sc].iloc[idx] and df[mc].iloc[idx - 1] <= df[sc].iloc[idx - 1]


def _macd_cross_below(df: pd.DataFrame, idx: int) -> bool:
    if idx < 1:
        return False
    mc, sc = _macd_cols(df)
    if mc is None or sc is None:
        return False
    return df[mc].iloc[idx] < df[sc].iloc[idx] and df[mc].iloc[idx - 1] >= df[sc].iloc[idx - 1]


def compute_scores(
    spx: pd.DataFrame,
    vix_value: float,
    data: dict,
) -> tuple:
    """
    Compute BUY and SELL scores (0-8) and which conditions passed.
    data = dict from fetch_all_dashboard_data with keys: spx, vix_value, AAPL, MSFT, QQQ, ^TNX (or TNX key).
    Returns (buy_score, sell_score, buy_list, sell_list, signal_strength_str).
    """
    buy_list = []
    sell_list = []
    if spx.empty or len(spx) < 2:
        return 0, 0, buy_list, sell_list, "⚪ NO SIGNAL - WAIT"

    i = len(spx) - 1
    row = spx.iloc[i]
    close = row["Close"]
    vwap_val = row.get("VWAP")
    rsi_val = row.get("RSI")

    # 1. SPX vs VWAP
    if vwap_val is not None and not pd.isna(vwap_val):
        if close > vwap_val:
            buy_list.append(BUY_CONDITIONS[0])
        if close < vwap_val:
            sell_list.append(SELL_CONDITIONS[0])

    # 2. RSI
    if rsi_val is not None and not pd.isna(rsi_val):
        if rsi_val < 40:
            buy_list.append(BUY_CONDITIONS[1])
        if rsi_val > 60:
            sell_list.append(SELL_CONDITIONS[1])

    # 3. EMA 9 vs EMA 21
    if _ema_cross_above(spx, i):
        buy_list.append(BUY_CONDITIONS[2])
    if _ema_cross_below(spx, i):
        sell_list.append(SELL_CONDITIONS[2])

    # 4. MACD cross
    if _macd_cross_above(spx, i):
        buy_list.append(BUY_CONDITIONS[3])
    if _macd_cross_below(spx, i):
        sell_list.append(SELL_CONDITIONS[3])

    # 5. VIX
    if vix_value is not None:
        if vix_value < 20:
            buy_list.append(BUY_CONDITIONS[4])
        if vix_value > 20:
            sell_list.append(SELL_CONDITIONS[4])

    # 6. AAPL & MSFT green/red today
    aapl_df = data.get("AAPL")
    msft_df = data.get("MSFT")
    aapl_green = False
    msft_green = False
    if aapl_df is not None and not aapl_df.empty and "PctChange" in aapl_df.columns:
        aapl_green = aapl_df["PctChange"].iloc[-1] > 0
    if msft_df is not None and not msft_df.empty and "PctChange" in msft_df.columns:
        msft_green = msft_df["PctChange"].iloc[-1] > 0
    if aapl_green and msft_green:
        buy_list.append(BUY_CONDITIONS[5])
    if not aapl_green and not msft_green and (aapl_df is not None and msft_df is not None):
        sell_list.append(SELL_CONDITIONS[5])

    # 7. QQQ vs VWAP
    qqq_df = data.get("QQQ")
    if qqq_df is not None and not qqq_df.empty and "AboveVWAP" in qqq_df.columns:
        if qqq_df["AboveVWAP"].iloc[-1]:
            buy_list.append(BUY_CONDITIONS[6])
        else:
            sell_list.append(SELL_CONDITIONS[6])

    # 8. 10Y yield stable / rising fast (use ^TNX)
    tnx_key = "^TNX" if "^TNX" in data else "TNX"
    tnx_df = data.get(tnx_key)
    if tnx_df is not None and not tnx_df.empty and "Close" in tnx_df.columns:
        tnx_now = tnx_df["Close"].iloc[-1]
        tnx_open = tnx_df["Open"].iloc[0] if "Open" in tnx_df.columns and len(tnx_df) > 0 else tnx_now
        tnx_pct = ((tnx_now - tnx_open) / tnx_open * 100) if tnx_open and tnx_open != 0 else 0
        if abs(tnx_pct) < 0.5:  # stable
            buy_list.append(BUY_CONDITIONS[7])
        if tnx_pct > 1.0:  # rising fast
            sell_list.append(SELL_CONDITIONS[7])

    buy_score = len(buy_list)
    sell_score = len(sell_list)

    # Signal strength: use the higher score; if tie, WAIT
    if buy_score >= sell_score and buy_score >= 7:
        strength = "🔥 STRONG BUY"
    elif buy_score >= sell_score and buy_score >= 5:
        strength = "✅ MODERATE BUY"
    elif sell_score > buy_score and sell_score >= 7:
        strength = "🔥 STRONG SELL"
    elif sell_score > buy_score and sell_score >= 5:
        strength = "✅ MODERATE SELL"
    elif buy_score >= 3 or sell_score >= 3:
        strength = "⚠️ WEAK SIGNAL - WAIT"
    else:
        strength = "⚪ NO SIGNAL - WAIT"

    return buy_score, sell_score, buy_list, sell_list, strength
