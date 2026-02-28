"""
Backtesting engine for SPX Day Trading Dashboard.
Strategies: RSI+EMA50, MACD crossover, EMA 9/21 crossover.
"""
import pandas as pd
import numpy as np
from indicators import rsi, macd, ema


def _ensure_indicators_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14), MACD, EMA9, EMA21, EMA50 to daily OHLCV."""
    if df.empty or len(df) < 50:
        return df
    df = df.copy()
    close = df["Close"]
    df["RSI"] = rsi(close, 14)
    m = macd(close, 12, 26, 9)
    for c in m.columns:
        df[c] = m[c]
    df["EMA9"] = ema(close, 9)
    df["EMA21"] = ema(close, 21)
    df["EMA50"] = ema(close, 50)
    return df


def backtest_rsi_ema50(df: pd.DataFrame, stop_pct: float = 2.0) -> dict:
    """
    BUY: RSI < 30 and price > EMA50.
    SELL: RSI > 70 or price drops stop_pct% from entry.
    """
    df = _ensure_indicators_daily(df)
    if df.empty or len(df) < 50:
        return _empty_result("RSI + EMA50")

    trades = []
    position = None
    entry_price = None
    equity = 100000.0
    equity_curve = [equity]
    for i in range(50, len(df)):
        row = df.iloc[i]
        close = row["Close"]
        rsi_val = row["RSI"]
        ema50 = row["EMA50"]
        if position is None:
            if rsi_val < 30 and close > ema50:
                position = "long"
                entry_price = close
        else:
            exit_trade = False
            reason = ""
            if rsi_val > 70:
                exit_trade = True
                reason = "RSI>70"
            elif (entry_price - close) / entry_price >= stop_pct / 100:
                exit_trade = True
                reason = "2% stop"
            if exit_trade:
                pnl_pct = (close - entry_price) / entry_price * 100
                trades.append({"entry": entry_price, "exit": close, "pnl_pct": pnl_pct, "reason": reason})
                equity *= (1 + pnl_pct / 100)
                position = None
        equity_curve.append(equity)

    return _build_result(trades, equity_curve, "RSI + EMA50", df)


def backtest_macd_crossover(df: pd.DataFrame) -> dict:
    """BUY: MACD crosses above signal. SELL: MACD crosses below signal."""
    df = _ensure_indicators_daily(df)
    if df.empty or len(df) < 50:
        return _empty_result("MACD Crossover")

    mc, sc = "MACD", "MACD_signal"
    if mc not in df.columns:
        cols = [c for c in df.columns if "MACD" in str(c) and "MACDh" not in str(c) and "MACDs" not in str(c)]
        mc = cols[0] if cols else None
    if not mc:
        return _empty_result("MACD Crossover")
    sig_cols = [c for c in df.columns if "MACDs" in str(c)]
    sc = sig_cols[0] if sig_cols else None
    if not sc:
        return _empty_result("MACD Crossover")

    trades = []
    position = None
    entry_price = None
    equity = 100000.0
    equity_curve = [equity]
    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row["Close"]
        cross_up = df[mc].iloc[i] > df[sc].iloc[i] and df[mc].iloc[i - 1] <= df[sc].iloc[i - 1]
        cross_dn = df[mc].iloc[i] < df[sc].iloc[i] and df[mc].iloc[i - 1] >= df[sc].iloc[i - 1]
        if position is None and cross_up:
            position = "long"
            entry_price = close
        elif position and cross_dn:
            pnl_pct = (close - entry_price) / entry_price * 100
            trades.append({"entry": entry_price, "exit": close, "pnl_pct": pnl_pct, "reason": "MACD cross"})
            equity *= (1 + pnl_pct / 100)
            position = None
        equity_curve.append(equity)

    return _build_result(trades, equity_curve, "MACD Crossover", df)


def backtest_ema_crossover(df: pd.DataFrame) -> dict:
    """BUY: EMA9 crosses above EMA21. SELL: EMA9 crosses below EMA21."""
    df = _ensure_indicators_daily(df)
    if df.empty or len(df) < 21:
        return _empty_result("EMA 9/21 Crossover")

    trades = []
    position = None
    entry_price = None
    equity = 100000.0
    equity_curve = [equity]
    for i in range(1, len(df)):
        row = df.iloc[i]
        close = row["Close"]
        cross_up = df["EMA9"].iloc[i] > df["EMA21"].iloc[i] and df["EMA9"].iloc[i - 1] <= df["EMA21"].iloc[i - 1]
        cross_dn = df["EMA9"].iloc[i] < df["EMA21"].iloc[i] and df["EMA9"].iloc[i - 1] >= df["EMA21"].iloc[i - 1]
        if position is None and cross_up:
            position = "long"
            entry_price = close
        elif position and cross_dn:
            pnl_pct = (close - entry_price) / entry_price * 100
            trades.append({"entry": entry_price, "exit": close, "pnl_pct": pnl_pct, "reason": "EMA cross"})
            equity *= (1 + pnl_pct / 100)
            position = None
        equity_curve.append(equity)

    return _build_result(trades, equity_curve, "EMA 9/21 Crossover", df)


def _empty_result(name: str) -> dict:
    return {
        "name": name,
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate_pct": 0,
        "avg_win_pct": 0,
        "avg_loss_pct": 0,
        "total_return_pct": 0,
        "best_trade_pct": 0,
        "worst_trade_pct": 0,
        "max_drawdown_pct": 0,
        "equity_curve": pd.DataFrame(),
        "trades_df": pd.DataFrame(),
    }


def _build_result(trades: list, equity_curve: list, name: str, df: pd.DataFrame) -> dict:
    if not trades:
        return _empty_result(name)

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]
    total_return = (equity_curve[-1] / 100000 - 1) * 100
    n = min(len(equity_curve), len(df) + 1)
    eq_series = pd.Series(equity_curve[:n])
    if len(df.index) >= n:
        eq_series.index = df.index[:n]
    peak = eq_series.expanding().max()
    drawdown = (eq_series - peak) / peak * 100
    max_dd = drawdown.min()

    return {
        "name": name,
        "total_trades": len(trades),
        "winning_trades": len(wins),
        "losing_trades": len(losses),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0,
        "avg_win_pct": wins["pnl_pct"].mean() if len(wins) else 0,
        "avg_loss_pct": losses["pnl_pct"].mean() if len(losses) else 0,
        "total_return_pct": total_return,
        "best_trade_pct": trades_df["pnl_pct"].max(),
        "worst_trade_pct": trades_df["pnl_pct"].min(),
        "max_drawdown_pct": max_dd,
        "equity_curve": eq_series,
        "trades_df": trades_df,
    }
