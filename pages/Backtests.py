"""
Backtests — Upload SPX CSV, select interval and strategies, run backtests and view win rate + charts.
"""
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Backtests", page_icon="📈", layout="wide")

# ---------- Cost model (industry standard) ----------
SPREAD_PCT = 0.0001      # 1bp each side (bid-ask spread)
SLIPPAGE_PCT = 0.0001    # 1bp each side (execution slippage)
SEC_FEE_RATE = 0.0000278 # on sell side only (per $ of trade value)
COMMISSION = 0.0         # $0 (zero commission broker)


def _cost_per_round_trip(entry_price: float, exit_price: float, notional: float = None) -> float:
    """Total cost in dollars: spread + slippage both sides (×2) + SEC fee on exit value. If notional given, use it; else use entry_price as position value."""
    if notional is not None and notional > 0:
        exit_value = notional * (exit_price / entry_price) if entry_price else notional
        cost = notional * (SPREAD_PCT + SLIPPAGE_PCT) * 2 + exit_value * SEC_FEE_RATE
    else:
        cost = entry_price * (SPREAD_PCT + SLIPPAGE_PCT) * 2 + exit_price * SEC_FEE_RATE
    return cost


def _apply_costs_to_trade(entry_price: float, exit_price: float, gross_pnl_pct: float, notional: float = None) -> tuple:
    """Returns (net_pnl_pct, cost_usd). Uses notional for cost when provided (e.g. investment)."""
    cost_usd = _cost_per_round_trip(entry_price, exit_price, notional)
    basis = notional if (notional is not None and notional > 0) else entry_price
    gross_pnl_usd = basis * (gross_pnl_pct / 100)
    net_pnl_usd = gross_pnl_usd - cost_usd
    net_pnl_pct = (net_pnl_usd / basis) * 100 if basis else 0
    return net_pnl_pct, cost_usd


# ---------- Indicator helpers ----------
def _vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP = cumsum(typical_price * volume) / cumsum(volume). Typical price = (H+L+C)/3."""
    if "High" not in df.columns or "Low" not in df.columns or "Close" not in df.columns or "Volume" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    vol = df["Volume"].replace(0, pd.NA).ffill().fillna(1)
    return (tp * df["Volume"]).cumsum() / df["Volume"].cumsum().replace(0, pd.NA).ffill()

# User-selectable entry conditions (all selected must be true for long). Keys used in backtest.
INDICATOR_CONDITIONS = [
    ("Price above VWAP", "price_above_vwap"),
    ("RSI < 45 (oversold)", "rsi_oversold"),
    ("EMA 9 crossed above EMA 21", "ema9_above_ema21"),
    ("MACD above signal line", "macd_above_signal"),
    ("VIX < 25 (calm market)", "vix_under_25"),
]


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
    return pd.DataFrame({"MACD": macd_line, "MACD_signal": sig_line})


def _record_trade(d: pd.DataFrame, entry_idx: int, exit_idx: int, entry_price: float, exit_price: float, force_closed: bool, notional: float = None) -> dict:
    """Compute gross P&L, apply costs, return trade dict with net pnl_pct (and gross for display), cost_usd. notional = investment for correct cost scaling."""
    gross_pnl_pct = (exit_price - entry_price) / entry_price * 100
    net_pnl_pct, cost_usd = _apply_costs_to_trade(entry_price, exit_price, gross_pnl_pct, notional)
    return {
        "entry_idx": entry_idx, "exit_idx": exit_idx,
        "entry_price": entry_price, "exit_price": exit_price,
        "gross_pnl_pct": gross_pnl_pct,
        "pnl_pct": net_pnl_pct,
        "cost_usd": cost_usd,
        "force_closed": force_closed,
    }


def _backtest_rsi(df: pd.DataFrame, notional: float = None) -> tuple:
    """Entry: RSI < 30. Exit: RSI crosses above 50, or RSI > 70, or EOD (force close at last bar Close). Entry/exit prices: next bar Open; last bar uses Close. RSI EWM valid from bar 1."""
    d = df.copy()
    if "Open" not in d.columns:
        d["Open"] = d["Close"]
    d["RSI"] = _rsi(d["Close"], 14)
    trades = []
    pos = None
    entry_price = None
    entry_idx = None
    n = len(d)
    for i in range(1, n):
        r = d["RSI"].iloc[i]
        r_prev = d["RSI"].iloc[i - 1] if i > 0 else 50
        is_last_bar = i == n - 1
        # Exit conditions (whichever first): RSI cross above 50, RSI > 70, or EOD
        exit_cross_50 = pos is not None and r_prev < 50 and r >= 50
        exit_overbought = pos is not None and r > 70
        exit_eod = pos is not None and is_last_bar
        if exit_eod:
            exit_px = float(d["Close"].iloc[i])
            t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
            trades.append(t)
            pos = None
        elif exit_cross_50 or exit_overbought:
            j = i + 1
            if j < n:
                exit_px = float(d["Open"].iloc[j])
                t = _record_trade(d, entry_idx, j, entry_price, exit_px, force_closed=False, notional=notional)
                trades.append(t)
            else:
                exit_px = float(d["Close"].iloc[i])
                t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
                trades.append(t)
            pos = None
        # Entry: RSI < 30; enter at next bar's open
        if pos is None and r < 30:
            k = i + 1
            if k < n:
                pos = "long"
                entry_price = float(d["Open"].iloc[k])
                entry_idx = k
            # If k == n we don't enter (no next bar)
    metrics = _result_metrics(trades, "RSI (buy <30, exit at 50/70/EOD)")
    return metrics, trades, d


def _backtest_macd(df: pd.DataFrame, notional: float = None) -> tuple:
    """Signal on bar i → entry/exit at Open[i+1]. Force close at EOD at last bar Close."""
    d = df.copy()
    if "Open" not in d.columns:
        d["Open"] = d["Close"]
    m = _macd_df(d["Close"], 12, 26, 9)
    d["MACD"] = m["MACD"]
    d["MACD_signal"] = m["MACD_signal"]
    trades = []
    pos = None
    entry_price = None
    entry_idx = None
    n = len(d)
    for i in range(1, n):
        cross_up = d["MACD"].iloc[i] > d["MACD_signal"].iloc[i] and d["MACD"].iloc[i - 1] <= d["MACD_signal"].iloc[i - 1]
        cross_dn = d["MACD"].iloc[i] < d["MACD_signal"].iloc[i] and d["MACD"].iloc[i - 1] >= d["MACD_signal"].iloc[i - 1]
        is_last_bar = i == n - 1
        if pos is not None and is_last_bar:
            exit_px = float(d["Close"].iloc[i])
            t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
            trades.append(t)
            pos = None
        elif pos is None and cross_up:
            j = i + 1
            if j < n:
                pos = "long"
                entry_price = float(d["Open"].iloc[j])
                entry_idx = j
        elif pos and cross_dn:
            j = i + 1
            if j < n:
                exit_px = float(d["Open"].iloc[j])
                t = _record_trade(d, entry_idx, j, entry_price, exit_px, force_closed=False, notional=notional)
                trades.append(t)
                pos = None
    metrics = _result_metrics(trades, "MACD crossover")
    return metrics, trades, d


def _backtest_ema(df: pd.DataFrame, notional: float = None) -> tuple:
    """Signal on bar i → entry/exit at Open[i+1]. Force close at EOD at last bar Close."""
    d = df.copy()
    if "Open" not in d.columns:
        d["Open"] = d["Close"]
    d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    trades = []
    pos = None
    entry_price = None
    entry_idx = None
    n = len(d)
    for i in range(1, n):
        cross_up = d["EMA9"].iloc[i] > d["EMA21"].iloc[i] and d["EMA9"].iloc[i - 1] <= d["EMA21"].iloc[i - 1]
        cross_dn = d["EMA9"].iloc[i] < d["EMA21"].iloc[i] and d["EMA9"].iloc[i - 1] >= d["EMA21"].iloc[i - 1]
        is_last_bar = i == n - 1
        if pos is not None and is_last_bar:
            exit_px = float(d["Close"].iloc[i])
            t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
            trades.append(t)
            pos = None
        elif pos is None and cross_up:
            j = i + 1
            if j < n:
                pos = "long"
                entry_price = float(d["Open"].iloc[j])
                entry_idx = j
        elif pos and cross_dn:
            j = i + 1
            if j < n:
                exit_px = float(d["Open"].iloc[j])
                t = _record_trade(d, entry_idx, j, entry_price, exit_px, force_closed=False, notional=notional)
                trades.append(t)
                pos = None
    metrics = _result_metrics(trades, "EMA 9/21 crossover")
    return metrics, trades, d


COMBINED_OPTION_LABEL = "Combined (all selected must agree)"


def _run_combined_strategy(df: pd.DataFrame, selected_strategies: list, notional: float = None) -> tuple:
    """
    Combined signal: +1 only when ALL selected strategies agree LONG, -1 only when ALL agree SHORT, else 0.
    Entry at Open[i+1], exit at Open[i+1] when consensus breaks. Force close at last bar Close.
    Returns (metrics, trades, df_plot, signal_table_df).
    selected_strategies: list of display names e.g. ["MACD crossover", "EMA 9/21 crossover", "RSI (buy <30, exit at 50/70/EOD)"]
    """
    d = df.copy()
    if "Open" not in d.columns:
        d["Open"] = d["Close"]
    n = len(d)
    # Indicators
    d["RSI"] = _rsi(d["Close"], 14)
    m = _macd_df(d["Close"], 12, 26, 9)
    d["MACD"] = m["MACD"]
    d["MACD_signal"] = m["MACD_signal"]
    d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    # Bar-by-bar signals: +1 long, -1 short, 0 flat
    has_macd = "MACD crossover" in selected_strategies
    has_ema = "EMA 9/21 crossover" in selected_strategies
    has_rsi = "RSI (buy <30, exit at 50/70/EOD)" in selected_strategies
    sig_macd = []
    sig_ema = []
    sig_rsi = []
    combined = []
    pos = None  # "long" or None
    for i in range(n):
        macd_v = 0
        if has_macd and i > 0:
            if d["MACD"].iloc[i] > d["MACD_signal"].iloc[i]:
                macd_v = 1
            elif d["MACD"].iloc[i] < d["MACD_signal"].iloc[i]:
                macd_v = -1
        sig_macd.append(macd_v)
        ema_v = 0
        if has_ema and i > 0:
            if d["EMA9"].iloc[i] > d["EMA21"].iloc[i]:
                ema_v = 1
            elif d["EMA9"].iloc[i] < d["EMA21"].iloc[i]:
                ema_v = -1
        sig_ema.append(ema_v)
        rsi_v = 0
        if has_rsi and i >= 14:
            r = d["RSI"].iloc[i]
            if r < 30 or (pos and r < 50):
                rsi_v = 1
            elif r > 70:
                rsi_v = -1
        sig_rsi.append(rsi_v)
        # Combined: only +1 if all selected == +1, only -1 if all == -1
        signals = []
        if has_macd:
            signals.append(macd_v)
        if has_ema:
            signals.append(ema_v)
        if has_rsi:
            signals.append(rsi_v)
        if not signals:
            comb = 0
        elif all(s == 1 for s in signals):
            comb = 1
        elif all(s == -1 for s in signals):
            comb = -1
        else:
            comb = 0
        combined.append(comb)
        if comb == 1:
            pos = "long"
        elif comb != 1:
            pos = None
    d["sig_macd"] = sig_macd
    d["sig_ema"] = sig_ema
    d["sig_rsi"] = sig_rsi
    d["combined"] = combined
    # Trade: enter long when combined becomes +1 at bar i -> fill at Open[i+1]; exit when combined != +1 at bar i -> exit at Open[i+1]
    trades = []
    pos = None
    entry_price = None
    entry_idx = None
    for i in range(n):
        comb = combined[i]
        is_last = i == n - 1
        if pos is not None and is_last:
            exit_px = float(d["Close"].iloc[i])
            t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
            trades.append(t)
            pos = None
        elif pos is not None and comb != 1:
            j = i + 1
            if j < n:
                exit_px = float(d["Open"].iloc[j])
                t = _record_trade(d, entry_idx, j, entry_price, exit_px, force_closed=False, notional=notional)
                trades.append(t)
            else:
                exit_px = float(d["Close"].iloc[i])
                t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
                trades.append(t)
            pos = None
        elif pos is None and comb == 1:
            j = i + 1
            if j < n:
                pos = "long"
                entry_price = float(d["Open"].iloc[j])
                entry_idx = j
    short_names = []
    if has_macd:
        short_names.append("MACD")
    if has_ema:
        short_names.append("EMA")
    if has_rsi:
        short_names.append("RSI")
    combined_name = "🔗 Combined (" + " + ".join(short_names) + ")"
    metrics = _result_metrics(trades, combined_name)
    # Signal agreement table: only bars where at least 2 strategies agree (same direction)
    def _agree_count(i):
        s = []
        if has_macd:
            s.append(sig_macd[i])
        if has_ema:
            s.append(sig_ema[i])
        if has_rsi:
            s.append(sig_rsi[i])
        plus = sum(1 for x in s if x == 1)
        minus = sum(1 for x in s if x == -1)
        return plus >= 2 or minus >= 2
    rows = []
    for i in range(n):
        if not _agree_count(i):
            continue
        ts = d.index[i]
        time_str = str(ts)[:19] if hasattr(ts, "__str__") else str(ts)
        price = d["Close"].iloc[i]
        def sym(v):
            if v == 1:
                return "▲"
            if v == -1:
                return "▼"
            return "—"
        macd_s = sym(sig_macd[i]) if has_macd else "—"
        ema_s = sym(sig_ema[i]) if has_ema else "—"
        rsi_s = sym(sig_rsi[i]) if has_rsi else "—"
        comb_s = combined[i]
        if comb_s == 1:
            comb_str = "✅ LONG"
        elif comb_s == -1:
            comb_str = "❌ SHORT"
        else:
            comb_str = "❌ No trade"
        rows.append({
            "Time": time_str,
            "SPX Price": round(price, 2),
            "MACD": macd_s,
            "EMA": ema_s,
            "RSI": rsi_s,
            "Combined": comb_str,
            "_combined_val": comb_s,
        })
    signal_table_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return metrics, trades, d, signal_table_df


def _run_custom_conditions_backtest(df: pd.DataFrame, selected_condition_keys: list, notional: float = None) -> tuple:
    """
    Backtest where LONG entry only when ALL selected conditions are true at bar i; entry at Open[i+1], exit when any breaks at Open[i+1], force close EOD.
    Conditions: price_above_vwap, rsi_oversold (RSI<45), ema9_above_ema21, macd_above_signal, vix_under_25 (if VIX column exists).
    """
    if not selected_condition_keys:
        return _result_metrics([], "Custom (conditions)"), [], df.copy(), None
    d = df.copy()
    if "Open" not in d.columns:
        d["Open"] = d["Close"]
    n = len(d)
    d["VWAP"] = _vwap(d)
    d["RSI"] = _rsi(d["Close"], 14)
    m = _macd_df(d["Close"], 12, 26, 9)
    d["MACD"] = m["MACD"]
    d["MACD_signal"] = m["MACD_signal"]
    d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
    d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    has_vix = "VIX" in d.columns or any(c for c in d.columns if "vix" in str(c).lower())
    vix_col = "VIX" if "VIX" in d.columns else next((c for c in d.columns if "vix" in str(c).lower()), None)
    conditions_ok = []
    for i in range(n):
        ok = True
        for key in selected_condition_keys:
            if key == "price_above_vwap":
                vwap = d["VWAP"].iloc[i] if pd.notna(d["VWAP"].iloc[i]) else d["Close"].iloc[i]
                if d["Close"].iloc[i] <= vwap:
                    ok = False
                    break
            elif key == "rsi_oversold":
                if i < 1:
                    ok = False
                    break
                if d["RSI"].iloc[i] >= 45:
                    ok = False
                    break
            elif key == "ema9_above_ema21":
                # Crossover: previous bar EMA9 <= EMA21, current bar EMA9 > EMA21
                if i < 1:
                    ok = False
                    break
                if d["EMA9"].iloc[i - 1] > d["EMA21"].iloc[i - 1] or d["EMA9"].iloc[i] <= d["EMA21"].iloc[i]:
                    ok = False
                    break
            elif key == "macd_above_signal":
                if i < 1:
                    ok = False
                    break
                if d["MACD"].iloc[i] <= d["MACD_signal"].iloc[i]:
                    ok = False
                    break
            elif key == "vix_under_25":
                if not has_vix or vix_col is None:
                    continue
                if d[vix_col].iloc[i] >= 25:
                    ok = False
                    break
        conditions_ok.append(ok)
    trades = []
    pos = None
    entry_price = None
    entry_idx = None
    for i in range(n):
        is_last = i == n - 1
        all_ok = conditions_ok[i]
        if pos is not None and is_last:
            exit_px = float(d["Close"].iloc[i])
            t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
            trades.append(t)
            pos = None
        elif pos is not None and not all_ok:
            j = i + 1
            if j < n:
                exit_px = float(d["Open"].iloc[j])
                t = _record_trade(d, entry_idx, j, entry_price, exit_px, force_closed=False, notional=notional)
                trades.append(t)
            else:
                exit_px = float(d["Close"].iloc[i])
                t = _record_trade(d, entry_idx, i, entry_price, exit_px, force_closed=True, notional=notional)
                trades.append(t)
            pos = None
        elif pos is None and all_ok:
            j = i + 1
            if j < n:
                pos = "long"
                entry_price = float(d["Open"].iloc[j])
                entry_idx = j
    names = [name for name, k in INDICATOR_CONDITIONS if k in selected_condition_keys]
    combined_name = "📌 Custom (" + " + ".join(names) + ")"
    metrics = _result_metrics(trades, combined_name)
    return metrics, trades, d, None


def _result_metrics(trades: list, name: str) -> dict:
    """Win rate and all metrics use net P&L (after costs). Force-closed trades included. Adds avg_win_pct, avg_loss_pct, profit_factor, max_drawdown_pct, total_costs_usd."""
    if not trades:
        return {
            "name": name, "total_trades": 0, "win_rate_pct": 0, "total_return_pct": 0, "best_pct": 0, "worst_pct": 0,
            "avg_win_pct": 0, "avg_loss_pct": 0, "profit_factor": 0, "max_drawdown_pct": 0, "total_costs_usd": 0,
        }
    t = pd.DataFrame(trades)
    net_pnl = t["pnl_pct"]
    wins = t[net_pnl > 0]
    losses = t[net_pnl <= 0]
    win_rate_pct = len(wins) / len(trades) * 100
    total_return = (1 + net_pnl / 100).prod() - 1
    avg_win_pct = wins["pnl_pct"].mean() if len(wins) > 0 else 0
    avg_loss_pct = losses["pnl_pct"].mean() if len(losses) > 0 else 0
    sum_win = wins["pnl_pct"].sum()
    sum_loss_abs = abs(losses["pnl_pct"].sum()) if len(losses) > 0 else 1
    profit_factor = sum_win / sum_loss_abs if sum_loss_abs else (sum_win if sum_win > 0 else 0)
    total_costs_usd = t["cost_usd"].sum()
    # Max drawdown from equity curve (use net P&L)
    eq = [100]
    for p in net_pnl:
        eq.append(eq[-1] * (1 + p / 100))
    peak = eq[0]
    max_dd = 0
    for v in eq:
        if v > peak:
            peak = v
        dd = (peak - v) / peak * 100 if peak else 0
        if dd > max_dd:
            max_dd = dd
    return {
        "name": name,
        "total_trades": len(trades),
        "win_rate_pct": win_rate_pct,
        "total_return_pct": total_return * 100,
        "best_pct": net_pnl.max(),
        "worst_pct": net_pnl.min(),
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "profit_factor": profit_factor,
        "max_drawdown_pct": max_dd,
        "total_costs_usd": total_costs_usd,
    }


def _trades_by_day(df: pd.DataFrame, trades: list) -> pd.DataFrame:
    """Group trades by the calendar day of entry. Returns DataFrame: Date, Trades, Wins, Win rate %, Day return %."""
    if not trades or df.empty:
        return pd.DataFrame(columns=["Date", "Trades", "Wins", "Win rate %", "Day return %"])
    idx = df.index
    by_day = {}
    for t in trades:
        i = t["entry_idx"]
        if i >= len(idx):
            continue
        ts = idx[i]
        day = ts.date() if hasattr(ts, "date") else pd.Timestamp(ts).date()
        by_day.setdefault(day, []).append(t["pnl_pct"])
    rows = []
    for day in sorted(by_day.keys()):
        pnls = by_day[day]
        wins = sum(1 for p in pnls if p > 0)
        win_rate = (wins / len(pnls) * 100) if pnls else 0
        day_return = sum(pnls)  # simple sum of % gains/losses for that day
        rows.append({"Date": day, "Trades": len(pnls), "Wins": wins, "Win rate %": round(win_rate, 1), "Day return %": round(day_return, 2)})
    return pd.DataFrame(rows)


def _plot_strategy(df: pd.DataFrame, trades: list, metrics: dict) -> None:
    """Plot price with BUY/SELL markers and equity curve (single combined chart)."""
    if df.empty:
        return
    idx = df.index
    close = df["Close"].values
    buy_x, buy_y = [], []
    sell_x, sell_y = [], []
    for t in trades:
        i, j = t["entry_idx"], t["exit_idx"]
        if i < len(idx) and j < len(idx):
            buy_x.append(idx[i])
            buy_y.append(t["entry_price"])
            sell_x.append(idx[j])
            sell_y.append(t["exit_price"])
    eq = [100]
    for t in trades:
        eq.append(eq[-1] * (1 + t["pnl_pct"] / 100))
    eq_x = [idx[0]] + [idx[t["exit_idx"]] for t in trades] if trades else [idx[0]]
    eq_y = eq
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Price with BUY/SELL signals", "Equity curve (%)"),
        row_heights=[0.6, 0.4],
    )
    fig.add_trace(go.Scatter(x=idx, y=close, name="Close", line=dict(color="#00d4aa", width=2)), row=1, col=1)
    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=12, color="#2ecc71")), row=1, col=1)
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=12, color="#e74c3c")), row=1, col=1)
    fig.add_trace(go.Scatter(x=eq_x, y=eq_y, name="Equity", fill="tozeroy", line=dict(color="#3498db", width=2)), row=2, col=1)
    fig.update_layout(template="plotly_dark", height=400, showlegend=True)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,36,0.8)", font=dict(color="#e0e2e6"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    st.plotly_chart(fig, use_container_width=True)


def _plot_price_only(df: pd.DataFrame, trades: list, height: int = 400) -> go.Figure:
    """Price chart with BUY/SELL markers only."""
    if df.empty:
        return go.Figure()
    idx = df.index
    close = df["Close"].values
    buy_x, buy_y, sell_x, sell_y = [], [], [], []
    for t in trades:
        i, j = t["entry_idx"], t["exit_idx"]
        if i < len(idx) and j < len(idx):
            buy_x.append(idx[i])
            buy_y.append(t["entry_price"])
            sell_x.append(idx[j])
            sell_y.append(t["exit_price"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=close, name="Close", line=dict(color="#00d4aa", width=2)))
    if buy_x:
        fig.add_trace(go.Scatter(x=buy_x, y=buy_y, mode="markers", name="BUY", marker=dict(symbol="triangle-up", size=10, color="#00c853")))
    if sell_x:
        fig.add_trace(go.Scatter(x=sell_x, y=sell_y, mode="markers", name="SELL", marker=dict(symbol="triangle-down", size=10, color="#ff3d57")))
    fig.update_layout(template="plotly_dark", height=height, title="Price & signals", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,36,0.8)", font=dict(color="#e0e2e6"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    return fig


def _plot_equity_only(df: pd.DataFrame, trades: list, height: int = 400) -> go.Figure:
    """Equity curve only."""
    if df.empty or not trades:
        return go.Figure()
    idx = df.index
    eq = [100]
    for t in trades:
        eq.append(eq[-1] * (1 + t["pnl_pct"] / 100))
    eq_x = [idx[0]] + [idx[t["exit_idx"]] for t in trades]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq_x, y=eq, name="Equity", fill="tozeroy", line=dict(color="#4f8ef7", width=2)))
    fig.update_layout(template="plotly_dark", height=height, title="Equity curve (%)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,36,0.8)", font=dict(color="#e0e2e6"))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    return fig


def _plot_macd(df: pd.DataFrame, height: int = 380) -> go.Figure:
    """MACD (12, 26, 9) chart: MACD line, Signal line, and Histogram so you can see when MACD crosses the signal line."""
    if df.empty or "Close" not in df.columns:
        return go.Figure()
    d = df.copy()
    if "MACD" not in d.columns or "MACD_signal" not in d.columns:
        m = _macd_df(d["Close"], 12, 26, 9)
        d["MACD"] = m["MACD"]
        d["MACD_signal"] = m["MACD_signal"]
    idx = d.index
    macd_vals = d["MACD"].values
    sig_vals = d["MACD_signal"].values
    hist = macd_vals - sig_vals
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=macd_vals, name="MACD", line=dict(color="#00d4aa", width=2)))
    fig.add_trace(go.Scatter(x=idx, y=sig_vals, name="Signal", line=dict(color="#ff9800", width=1.5)))
    colors = ["#00c853" if h >= 0 else "#ff3d57" for h in hist]
    fig.add_trace(go.Bar(x=idx, y=hist, name="Histogram", marker_color=colors, opacity=0.7))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title="MACD (12, 26, 9)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,36,0.8)",
        font=dict(color="#e0e2e6"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)", title_text="MACD")
    return fig


def _plot_ema(df: pd.DataFrame, height: int = 380) -> go.Figure:
    """EMA 9/21 chart: Price (Close), EMA 9, and EMA 21 so you can see when EMA 9 crosses above/below EMA 21."""
    if df.empty or "Close" not in df.columns:
        return go.Figure()
    d = df.copy()
    if "EMA9" not in d.columns or "EMA21" not in d.columns:
        d["EMA9"] = d["Close"].ewm(span=9, adjust=False).mean()
        d["EMA21"] = d["Close"].ewm(span=21, adjust=False).mean()
    idx = d.index
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=idx, y=d["Close"].values, name="Close", line=dict(color="#8b8fa8", width=1.5)))
    fig.add_trace(go.Scatter(x=idx, y=d["EMA9"].values, name="EMA 9", line=dict(color="#00d4aa", width=2)))
    fig.add_trace(go.Scatter(x=idx, y=d["EMA21"].values, name="EMA 21", line=dict(color="#ff9800", width=2)))
    fig.update_layout(
        template="plotly_dark",
        height=height,
        title="EMA 9 / EMA 21",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,29,36,0.8)",
        font=dict(color="#e0e2e6"),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(80,85,95,0.5)", title_text="Price")
    return fig


def _trade_log_table(df: pd.DataFrame, trades: list) -> pd.DataFrame:
    """Build trade log with Entry Time, Exit Time, Entry Price, Exit Price, Gross %, Cost $, Net %, Result, force_closed flag."""
    if not trades or df.empty:
        return pd.DataFrame()
    idx = df.index
    rows = []
    for t in trades:
        i, j = t["entry_idx"], t["exit_idx"]
        entry_ts = idx[i] if i < len(idx) else None
        exit_ts = idx[j] if j < len(idx) else None
        ep, xp = t["entry_price"], t["exit_price"]
        gross_pct = t.get("gross_pnl_pct", (xp - ep) / ep * 100)
        rows.append({
            "Entry Time": str(entry_ts)[:19] if entry_ts is not None else "—",
            "Exit Time": str(exit_ts)[:19] if exit_ts is not None else "—",
            "Entry Price": f"{ep:,.2f}",
            "Exit Price": f"{xp:,.2f}",
            "Gross %": f"{gross_pct:+.2f}",
            "Cost $": f"${t['cost_usd']:.2f}",
            "Net %": f"{t['pnl_pct']:+.2f}",
            "Result": "✅ WIN" if t["pnl_pct"] > 0 else "❌ LOSS",
            "Note": "🔒 EOD" if t.get("force_closed") else "",
        })
    return pd.DataFrame(rows)


def _strategy_badge_color(name: str) -> str:
    if "MACD" in name:
        return "#4f8ef7"
    if "EMA" in name:
        return "#00c853"
    if "RSI" in name:
        return "#ff9800"
    return "#8b8fa8"


def _generate_insights(all_results: list, comparison: pd.DataFrame, combined_result: dict = None) -> list:
    """Generate dynamic insight bullets from actual result numbers."""
    insights = []
    if not all_results or comparison.empty:
        return insights
    best = comparison.iloc[0]
    best_name = best["Strategy"]
    best_pf = best.get("Profit factor", 0)
    best_wr = best.get("Win rate %", 0)
    best_ret = best.get("Total return %", 0)
    best_dd = best.get("Max drawdown %", 0)
    if best_pf and float(best_pf) > 1:
        insights.append(f"✅ **{best_name}** had the highest profit factor ({float(best_pf):.2f}) — wins were {float(best_pf):.1f}x the size of losses")
    worst_wr_row = comparison.loc[comparison["Win rate %"].idxmin()]
    if len(comparison) > 1 and worst_wr_row["Win rate %"] < 50:
        insights.append(f"⚠️ **{worst_wr_row['Strategy']}** had a {worst_wr_row['Win rate %']:.1f}% win rate — fewer than half of trades were profitable")
    costs = comparison.get("Total costs $", pd.Series([0] * len(comparison)))
    if len(costs) > 1 and costs.max() > 0:
        min_cost_idx = costs.idxmin()
        max_cost_idx = costs.idxmax()
        insights.append(f"💡 Fewer trades = lower costs. **{comparison.loc[min_cost_idx, 'Strategy']}** costs ${costs.min():.2f} vs **{comparison.loc[max_cost_idx, 'Strategy']}** ${costs.max():.2f}")
    for r in all_results:
        m = r["metrics"]
        for t in r.get("trades", []):
            if t.get("force_closed"):
                insights.append(f"🔒 **{m['name']}** had a trade force-closed at EOD — consider if holding overnight changes the thesis")
                break
    if best_dd is not None and best_dd < 1:
        insights.append(f"📉 Max drawdown was {best_dd:.2f}% for the best strategy — capital was not at significant risk on this period")
    elif best_dd is not None and best_dd >= 0.3:
        insights.append(f"📉 Max drawdown reached {best_dd:.2f}% — monitor risk when scaling position size")
    if best_ret and best_ret > 0:
        insights.append(f"📈 Best strategy **{best_name}** returned **{best_ret:.2f}%** net of costs")
    # Combined-strategy insights
    if combined_result:
        c_metrics = combined_result.get("metrics", {})
        c_name = c_metrics.get("name", "Combined")
        c_trades = c_metrics.get("total_trades", 0)
        c_ret = c_metrics.get("total_return_pct", 0)
        c_pf = c_metrics.get("profit_factor", 0)
        indiv_rows = comparison[~comparison["Strategy"].astype(str).str.startswith("🔗")]
        best_indiv_ret = indiv_rows["Total return %"].max() if not indiv_rows.empty else 0
        best_indiv_pf = indiv_rows["Profit factor"].max() if not indiv_rows.empty else 0
        if c_trades == 0:
            insights.append("⚠️ **Combined strategy** found no full agreement windows on this period — indicators were conflicting throughout")
        elif c_ret > best_indiv_ret and best_indiv_ret is not None:
            insights.append("🏆 **Combined strategy** outperformed all individual strategies — confluence filtering worked")
        elif c_trades > 0 and c_pf > best_indiv_pf and best_indiv_pf > 0:
            insights.append(f"✅ **Combined** had fewer trades but higher quality — profit factor {c_pf:.2f} vs best individual {best_indiv_pf:.2f}")
    return insights[:8]


# ---------- Design system ----------
ANALYTICS_CSS = """
<style>
:root {
  --bg: #0e1117;
  --card: #1a1d27;
  --border: #2d3142;
  --accent: #4f8ef7;
  --green: #00c853;
  --red: #ff3d57;
  --yellow: #ffd600;
  --text: #ffffff;
  --text2: #8b8fa8;
}
.analytics-header { font-size: 2rem; font-weight: 700; color: var(--text); margin-bottom: 0.25rem; background: linear-gradient(135deg, #4f8ef7 0%, #00c853 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.analytics-subtitle { color: var(--text2); font-size: 0.95rem; margin-bottom: 1rem; }
.analytics-info-bar { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.25rem; margin-bottom: 1.5rem; display: flex; flex-wrap: wrap; gap: 1.5rem; }
.analytics-info-item { color: var(--text2); font-size: 0.9rem; }
.analytics-info-item strong { color: var(--text); }
.analytics-step { background: var(--card); border: 1px solid var(--border); border-radius: 8px; padding: 0.5rem 0.75rem; font-size: 0.8rem; color: var(--text2); margin-bottom: 0.5rem; }
.analytics-pill { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 20px; font-size: 0.8rem; font-weight: 500; margin: 0.1rem; }
.analytics-badge-macd { background: rgba(79,142,247,0.25); color: #4f8ef7; }
.analytics-badge-ema { background: rgba(0,200,83,0.25); color: #00c853; }
.analytics-badge-rsi { background: rgba(255,152,0,0.25); color: #ff9800; }
.analytics-results-banner { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; margin-bottom: 1.5rem; }
.analytics-insight-box { background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem; margin: 1rem 0; }
.analytics-footer { color: var(--text2); font-size: 0.8rem; text-align: center; padding: 1.5rem 0; border-top: 1px solid var(--border); margin-top: 2rem; }
.analytics-return-positive { color: var(--green); font-weight: 700; }
.analytics-return-negative { color: var(--red); font-weight: 700; }
.analytics-spark-bar { height: 8px; border-radius: 4px; max-width: 80px; }
.analytics-table th, .analytics-table td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
.analytics-table tbody tr:hover { background: rgba(45,49,66,0.3); }
.analytics-table tr.combined-row { border-left: 4px solid #4f8ef7; background: rgba(79,142,247,0.08); }
section[data-testid="stSidebar"] { width: 320px !important; min-width: 320px !important; }
section[data-testid="stSidebar"] > div { width: 320px !important; }
</style>
"""

# ---------- Page UI ----------
st.markdown(ANALYTICS_CSS, unsafe_allow_html=True)

if st.button("← Back to Dashboard", type="secondary", key="analytics_back_dashboard"):
    st.switch_page("app_daytrade.py")

# Section 1 — Page header (title + subtitle; info bar after load)
st.markdown('<p class="analytics-header">Backtests</p>', unsafe_allow_html=True)
st.markdown('<p class="analytics-subtitle">Backtest your SPX intraday strategies with institutional-grade metrics</p>', unsafe_allow_html=True)

# File upload (Step 1)
uploaded = st.file_uploader("Select SPX data file (CSV)", type=["csv"], help="e.g. spx_5m_60d.csv with columns: Datetime, Open, High, Low, Close, Volume")
if not uploaded:
    st.info("👆 Upload a CSV file to begin. Use the file from `fetch_spx_1y.py` (e.g. spx_5m_60d.csv) or any OHLCV CSV with a datetime column.")
    st.stop()

# Parse CSV — defensive: read by column NAME only, never by position
try:
    raw = pd.read_csv(uploaded)
    # Normalize column names (strip spaces, fix case)
    raw.columns = [c.strip().title() for c in raw.columns]
    # Handle 'Adj Close' → 'Close'
    if "Adj Close" in raw.columns and "Close" not in raw.columns:
        raw.rename(columns={"Adj Close": "Close"}, inplace=True)
    # Verify required columns exist
    required = ["Open", "High", "Low", "Close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        st.error(f"❌ CSV missing required columns: {missing}. Found: {list(raw.columns)}")
        st.stop()
    # Volume required for backtest (VWAP, etc.)
    if "Volume" not in raw.columns:
        st.error(f"❌ CSV missing column: Volume. Found: {list(raw.columns)}")
        st.stop()
    # Find datetime column by name (never by position)
    date_col = None
    for c in ["Datetime", "DateTime", "Date", "Time"]:
        if c in raw.columns:
            date_col = c
            break
    if date_col is None:
        st.error(f"❌ CSV missing datetime column. Expected one of: Datetime, DateTime, Date, Time. Found: {list(raw.columns)}")
        st.stop()
    raw[date_col] = pd.to_datetime(raw[date_col], utc=True, errors="coerce")
    raw = raw.dropna(subset=[date_col])
    raw = raw.set_index(date_col)
    raw.index.name = "Datetime"
    # Always assign by name — NEVER use iloc for column selection
    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.sort_index()
    if len(df) < 30:
        st.warning("Not enough rows (need at least 30). Try a larger file.")
        st.stop()
    # Data Quality panel — show detected columns and confirm read-by-name
    detected = " | ".join([str(raw.index.name)] + list(df.columns))
    with st.expander("📋 Data Quality", expanded=True):
        st.markdown(f"**Detected columns:** {detected}")
        st.success("✅ All required columns found (reading by name, order doesn't matter).")
except Exception as e:
    st.error(f"Could not load CSV: {e}")
    st.stop()

# Section 2 — Config sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown('<span class="analytics-step">Step 1 — Upload CSV ✓</span>', unsafe_allow_html=True)
    st.markdown("**Step 2 — Interval**")
    interval = st.selectbox("Interval", ["5m", "15m", "30m", "1d"], index=0, label_visibility="collapsed")
    st.markdown("**Step 3 — Backtest mode**")
    run_scope = st.radio("Mode", ["Single day", "Entire period"], horizontal=True, label_visibility="collapsed")
    dates_in_file = sorted({(t.date() if hasattr(t, "date") else pd.Timestamp(t).date()) for t in df.index})
    st.markdown("**Step 4 — Select date**")
    if run_scope == "Single day":
        selected_date = st.selectbox("Date", options=dates_in_file, format_func=lambda d: str(d), label_visibility="collapsed")
        df_run = df[(df.index.map(lambda t: t.date() if hasattr(t, "date") else pd.Timestamp(t).date()) == selected_date)].copy()
        n_match = len(df_run)
        st.caption(f"**{n_match}** bars match")
    else:
        selected_date = None
        df_run = df
        n_match = len(df_run)
        st.caption("All bars")
    st.markdown("**Step 5 — Strategies**")
    strategy_options = ["RSI (buy <30, exit at 50/70/EOD)", "MACD crossover", "EMA 9/21 crossover", COMBINED_OPTION_LABEL]
    strategies = st.multiselect("Strategies", strategy_options, default=strategy_options, label_visibility="collapsed")
    st.markdown("**Step 5b — Custom entry conditions**")
    st.caption("Select which conditions must all be true for a LONG entry. Backtest runs as \"Custom\" strategy.")
    condition_options = [name for name, _ in INDICATOR_CONDITIONS]
    custom_conditions_display = st.multiselect("Conditions (all must agree)", condition_options, default=[], label_visibility="collapsed")
    custom_condition_keys = [k for name, k in INDICATOR_CONDITIONS if name in custom_conditions_display]
    st.markdown("**Step 6 — Investment amount**")
    if "analytics_inv" not in st.session_state:
        st.session_state.analytics_inv = 10000
    preset_col1, preset_col2 = st.columns(2)
    with preset_col1:
        if st.button("$1k", key="inv1k"):
            st.session_state.analytics_inv = 1000
        if st.button("$5k", key="inv5k"):
            st.session_state.analytics_inv = 5000
    with preset_col2:
        if st.button("$10k", key="inv10k"):
            st.session_state.analytics_inv = 10000
        if st.button("$25k", key="inv25k"):
            st.session_state.analytics_inv = 25000
    investment = st.number_input("Amount ($)", min_value=0, value=st.session_state.analytics_inv, step=500, key="inv_input")
    st.session_state.analytics_inv = int(investment) if investment else 0
    st.markdown("---")
    run = st.button("▶ Run Backtest", type="primary", use_container_width=True, key="analytics_run_backtest")

# Info bar (Section 1 continued — after sidebar so we have interval)
date_min = df.index.min()
date_max = df.index.max()
file_name = getattr(uploaded, "name", "uploaded.csv")
st.markdown(f'''
<div class="analytics-info-bar">
  <span class="analytics-info-item">📅 <strong>Date range</strong> {date_min.strftime("%Y-%m-%d") if hasattr(date_min, "strftime") else date_min} → {date_max.strftime("%Y-%m-%d") if hasattr(date_max, "strftime") else date_max}</span>
  <span class="analytics-info-item">📊 <strong>Total bars</strong> {len(df):,}</span>
  <span class="analytics-info-item">⏱️ <strong>Interval</strong> {interval}</span>
  <span class="analytics-info-item">📁 <strong>File</strong> {file_name}</span>
</div>
''', unsafe_allow_html=True)

if not strategies and not custom_condition_keys:
    st.warning("Select at least one strategy and/or custom entry conditions in the sidebar.")
    st.stop()
individual_strategies = [s for s in strategies if s != COMBINED_OPTION_LABEL]
if COMBINED_OPTION_LABEL in strategies and not individual_strategies:
    st.warning("Select at least one individual strategy (MACD, EMA, or RSI) along with Combined.")
    st.stop()

if not run:
    st.info("Configure options in the **sidebar** and click **▶ Run Backtest** to see results.")
    st.stop()

# Map display name -> function (individual only)
strategy_fns = {
    "RSI (buy <30, exit at 50/70/EOD)": _backtest_rsi,
    "MACD crossover": _backtest_macd,
    "EMA 9/21 crossover": _backtest_ema,
}
run_combined = COMBINED_OPTION_LABEL in strategies

# Run backtests
all_results = []
combined_result = None  # { metrics, trades, df_plot, signal_table_df } or None
notional = investment if (investment and investment > 0) else None
with st.spinner("Running backtests..."):
    for name in individual_strategies:
        fn = strategy_fns.get(name)
        if not fn:
            continue
        metrics, trades, df_plot = fn(df_run, notional=notional)
        all_results.append({"metrics": metrics, "trades": trades, "df_plot": df_plot, "is_combined": False})
    if run_combined and individual_strategies:
        try:
            c_metrics, c_trades, c_df_plot, c_signal_table = _run_combined_strategy(df_run, individual_strategies, notional=notional)
            combined_result = {"metrics": c_metrics, "trades": c_trades, "df_plot": c_df_plot, "signal_table_df": c_signal_table, "is_combined": True}
            all_results.append({"metrics": c_metrics, "trades": c_trades, "df_plot": c_df_plot, "is_combined": True, "signal_table_df": c_signal_table})
        except Exception as e:
            st.warning(f"Combined strategy failed: {e}")
    if custom_condition_keys:
        try:
            cst_metrics, cst_trades, cst_df_plot, _ = _run_custom_conditions_backtest(df_run, custom_condition_keys, notional=notional)
            all_results.append({"metrics": cst_metrics, "trades": cst_trades, "df_plot": cst_df_plot, "is_combined": False, "is_custom": True})
        except Exception as e:
            st.warning(f"Custom conditions backtest failed: {e}")

if not all_results:
    st.warning("No results to show.")
    st.stop()

# Build comparison (same logic as before)
comparison = pd.DataFrame([
    {
        "Strategy": r["metrics"]["name"],
        "Total trades": r["metrics"]["total_trades"],
        "Win rate %": round(r["metrics"]["win_rate_pct"], 1),
        "Total return %": round(r["metrics"]["total_return_pct"], 1),
        "Avg win %": round(r["metrics"].get("avg_win_pct", 0), 2),
        "Avg loss %": round(r["metrics"].get("avg_loss_pct", 0), 2),
        "Profit factor": round(r["metrics"].get("profit_factor", 0), 2),
        "Max drawdown %": round(r["metrics"].get("max_drawdown_pct", 0), 2),
        "Total costs $": round(r["metrics"].get("total_costs_usd", 0), 2),
        "Best trade %": round(r["metrics"]["best_pct"], 2),
        "Worst trade %": round(r["metrics"]["worst_pct"], 2),
    }
    for r in all_results
])
if investment > 0:
    comparison["Est. return ($)"] = [round(investment * r["metrics"]["total_return_pct"] / 100, 2) for r in all_results]
comparison = comparison.sort_values(by=["Total return %", "Win rate %"], ascending=[False, False]).reset_index(drop=True)
comparison["Rank"] = range(1, len(comparison) + 1)
cols = ["Rank", "Strategy", "Total trades", "Win rate %", "Total return %"]
if "Est. return ($)" in comparison.columns:
    cols.append("Est. return ($)")
cols += ["Avg win %", "Avg loss %", "Profit factor", "Max drawdown %", "Total costs $", "Best trade %", "Worst trade %"]
comparison = comparison[[c for c in cols if c in comparison.columns]]

# Section 3 — Results banner
best_row = comparison.iloc[0]
best_ret = best_row["Total return %"]
banner_badge = "🟢 Profitable" if best_ret > 0 else ("🔴 All strategies lost" if (comparison["Total return %"] <= 0).all() else "🟡 Mixed")
date_tested = str(selected_date) if run_scope == "Single day" and selected_date else f"{df_run.index.min()} → {df_run.index.max()}"
st.markdown(f'''
<div class="analytics-results-banner">
  <strong style="color: #ffffff;">Results</strong> — Date: {date_tested} &nbsp;|&nbsp; Best: <strong>{best_row["Strategy"]}</strong> ({best_ret:+.2f}%) &nbsp;|&nbsp; Strategies: {len(comparison)} &nbsp;|&nbsp; <span style="color: {"#00c853" if best_ret > 0 else "#ff3d57"};">{banner_badge}</span>
</div>
''', unsafe_allow_html=True)

# Section 4 — Comparison table (styled with HTML for colors + sparkline)
st.markdown("#### 🏆 Strategy comparison")
# Build HTML table with rank medals and color coding
def _cell_color_return(v):
    if v > 0: return "#00c853"
    if v < 0: return "#ff3d57"
    return "#8b8fa8"
def _cell_color_wr(v):
    if v >= 50: return "#00c853"
    if v >= 30: return "#ffd600"
    return "#ff3d57"
def _cell_color_pf(v):
    if v > 1.5: return "#00c853"
    if v >= 1: return "#ffd600"
    return "#ff3d57"
def _cell_color_dd(v):
    if v < 0.3: return "#00c853"
    if v <= 0.6: return "#ffd600"
    return "#ff3d57"

medals = ["🥇", "🥈", "🥉"]
table_rows = []
for idx, row in comparison.iterrows():
    rank = int(row["Rank"])
    medal = medals[rank - 1] if rank <= 3 else str(rank)
    ret = row["Total return %"]
    wr = row["Win rate %"]
    pf = row.get("Profit factor", 0)
    dd = row.get("Max drawdown %", 0)
    spark_color = "#00c853" if ret >= 0 else "#ff3d57"
    spark_w = min(100, max(0, abs(ret) * 2))  # scale for bar width
    est = row.get("Est. return ($)", "")
    est_str = f"<strong>${est:+,.2f}</strong>" if isinstance(est, (int, float)) else "—"
    tr_class = " class=\"combined-row\"" if str(row["Strategy"]).startswith("🔗") else ""
    table_rows.append(
        f"<tr{tr_class}><td>{medal}</td><td>{row['Strategy']}</td><td>{row['Total trades']}</td>"
        f"<td style=\"color:{_cell_color_wr(wr)};\">{wr:.1f}%</td>"
        f"<td style=\"color:{_cell_color_return(ret)};\">{ret:.1f}%</td>"
        f"<td>{est_str}</td>"
        f"<td style=\"color:{_cell_color_pf(pf)};\">{pf:.2f}</td>"
        f"<td style=\"color:{_cell_color_dd(dd)};\">{dd:.2f}%</td>"
        f"<td><div class=\"analytics-spark-bar\" style=\"width:{spark_w}%;background:{spark_color};\"></div></td></tr>"
    )
th_cells = "".join(f"<th style=\"text-align:left;\">{h}</th>" for h in ["Rank", "Strategy", "Trades", "Win rate %", "Return %", "Est. return $", "PF", "Max DD %", "Return"])
table_html = f"<div style=\"overflow-x:auto;\"><table class=\"analytics-table\" style=\"width:100%;border-collapse:collapse;color:#ffffff;font-size:0.9rem;\"><thead><tr>{th_cells}</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>"
st.markdown(table_html, unsafe_allow_html=True)
st.markdown("""
<div class="analytics-insight-box" style="margin-top: 0.5rem;">
  <small style="color: #8b8fa8;">Simulated as SPX index % return (equivalent to trading a CFD or index fund, not SPY shares directly). All returns net of spread, slippage, SEC fee. Entry/exit at next bar open; open positions force-closed at last bar close.</small>
</div>
""", unsafe_allow_html=True)

# Section 5 — Per-strategy cards
st.markdown("---")
st.markdown("#### 📋 Per-strategy details")

for card_idx, r in enumerate(all_results):
    m = r["metrics"]
    is_combined = r.get("is_combined", False)
    is_custom = r.get("is_custom", False)
    badge_color = "#9c27b0" if is_combined else ("#ff9800" if is_custom else _strategy_badge_color(m["name"]))
    summary = f"{m['total_trades']} trades | {m['win_rate_pct']:.1f}% win | {m['total_return_pct']:+.2f}% return"
    expander_label = f"**{m['name']}** — {summary}" if not is_combined else f"**{m['name']}** — {summary}"
    with st.expander(expander_label, expanded=True):
        header_note = " 🔗 Combined Strategy — trades only when ALL indicators agree" if is_combined else (" 📌 Custom — entry when your selected conditions all agree" if is_custom else "")
        st.markdown(f'<span style="display:inline-block;background:{badge_color}22;color:{badge_color};padding:4px 10px;border-radius:20px;font-size:0.85rem;font-weight:600;">{m["name"]}{header_note}</span>', unsafe_allow_html=True)
        st.markdown("")
        # "Why so few trades?" for combined
        if is_combined:
            indiv_names = [x for x in individual_strategies if x in strategy_fns]
            names_short = " + ".join(["MACD" if "MACD" in n else "EMA" if "EMA" in n else "RSI" for n in indiv_names])
            agreement_windows = len(r.get("signal_table_df", pd.DataFrame()))  # bars where ≥2 agreed
            completed = m["total_trades"]
            st.markdown(f"""
            <div class="analytics-insight-box" style="margin-bottom:1rem;">
            <strong>Why so few trades?</strong><br>
            <small>The combined strategy only enters when {names_short} all agree simultaneously. This filters out noise but also means fewer opportunities. On this period: <strong>{agreement_windows}</strong> bars had at least 2 strategies agreeing, <strong>{completed}</strong> resulted in completed trades.</small>
            </div>
            """, unsafe_allow_html=True)
            # Signal agreement visualizer (only for combined)
            sig_df = r.get("signal_table_df")
            if sig_df is not None and not sig_df.empty:
                st.markdown("**Signal agreement** *(bars where ≥2 strategies agree)*")
                # Build HTML table with ▲▼ colored (green/red), highlight LONG/SHORT rows
                sig_rows = []
                for _, sr in sig_df.iterrows():
                    comb_val = sr.get("_combined_val", 0)
                    row_style = "background:rgba(0,200,83,0.15);" if comb_val == 1 else ("background:rgba(255,61,87,0.15);" if comb_val == -1 else "")
                    macd_c = "#00c853" if sr.get("MACD") == "▲" else ("#ff3d57" if sr.get("MACD") == "▼" else "#8b8fa8")
                    ema_c = "#00c853" if sr.get("EMA") == "▲" else ("#ff3d57" if sr.get("EMA") == "▼" else "#8b8fa8")
                    rsi_c = "#00c853" if sr.get("RSI") == "▲" else ("#ff3d57" if sr.get("RSI") == "▼" else "#8b8fa8")
                    sig_rows.append(f"<tr style=\"{row_style}\"><td>{sr['Time']}</td><td>{sr['SPX Price']}</td><td style=\"color:{macd_c};\">{sr.get('MACD','—')}</td><td style=\"color:{ema_c};\">{sr.get('EMA','—')}</td><td style=\"color:{rsi_c};\">{sr.get('RSI','—')}</td><td>{sr['Combined']}</td></tr>")
                sig_html = f"<div style=\"overflow-x:auto;\"><table class=\"analytics-table\" style=\"width:100%;font-size:0.85rem;\"><thead><tr><th>Time</th><th>SPX Price</th><th>MACD</th><th>EMA</th><th>RSI</th><th>Combined</th></tr></thead><tbody>{''.join(sig_rows)}</tbody></table></div>"
                st.markdown(sig_html, unsafe_allow_html=True)
                st.markdown("")
        # Row 1 — Key metrics (5 tiles)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Trades", m["total_trades"])
        c2.metric("Win Rate %", f"{m['win_rate_pct']:.1f}%")
        c3.metric("Total Return %", f"{m['total_return_pct']:.1f}%")
        c4.metric("Best Trade %", f"{m['best_pct']:+.2f}%")
        c5.metric("Worst Trade %", f"{m['worst_pct']:+.2f}%")
        # Row 2 — Advanced metrics
        c6, c7, c8, c9, c10 = st.columns(5)
        c6.metric("Avg Win %", f"{m.get('avg_win_pct', 0):.2f}%")
        c7.metric("Avg Loss %", f"{m.get('avg_loss_pct', 0):.2f}%")
        c8.metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
        c9.metric("Max Drawdown %", f"{m.get('max_drawdown_pct', 0):.2f}%")
        c10.metric("Total Costs $", f"${m.get('total_costs_usd', 0):,.2f}")
        # Estimated return (large, green/red)
        if investment > 0:
            est_return = investment * m["total_return_pct"] / 100
            total_final = investment + est_return
            color_class = "analytics-return-positive" if est_return >= 0 else "analytics-return-negative"
            st.markdown(f'**If you invested ${investment:,.0f}** → **<span class="{color_class}">${est_return:+,.2f}</span>** → Total: **${total_final:,.2f}**', unsafe_allow_html=True)
            st.caption("Simulated as SPX index % return (equivalent to trading a CFD or index fund, not SPY shares directly).")
        # Trade log table
        st.markdown("**Trade log**")
        trade_df = _trade_log_table(r["df_plot"], r["trades"])
        if not trade_df.empty:
            st.dataframe(trade_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No trades.")
        # Charts side by side (price + equity, height 400)
        st.markdown("**Charts**")
        chart_left, chart_right = st.columns(2)
        with chart_left:
            fig_price = _plot_price_only(r["df_plot"], r["trades"], height=400)
            if fig_price and len(fig_price.data) > 0:
                st.plotly_chart(fig_price, use_container_width=True, key=f"chart_price_{card_idx}")
        with chart_right:
            fig_eq = _plot_equity_only(r["df_plot"], r["trades"], height=400)
            if fig_eq and len(fig_eq.data) > 0:
                st.plotly_chart(fig_eq, use_container_width=True, key=f"chart_equity_{card_idx}")
        # MACD (12, 26, 9) chart for MACD strategy — see when MACD line crosses the signal line
        if "MACD crossover" in m["name"]:
            st.markdown("**MACD (12, 26, 9)** — MACD vs Signal line; histogram shows difference (green = MACD above Signal, red = below).")
            fig_macd = _plot_macd(r["df_plot"], height=380)
            if fig_macd and len(fig_macd.data) > 0:
                st.plotly_chart(fig_macd, use_container_width=True, key=f"chart_macd_{card_idx}")
        # EMA 9/21 chart for EMA strategy — see when EMA 9 crosses above/below EMA 21
        if "EMA 9/21 crossover" in m["name"]:
            st.markdown("**EMA 9 / EMA 21** — Close price with EMAs; crossover up = EMA 9 crosses above EMA 21, crossover down = EMA 9 crosses below.")
            fig_ema = _plot_ema(r["df_plot"], height=380)
            if fig_ema and len(fig_ema.data) > 0:
                st.plotly_chart(fig_ema, use_container_width=True, key=f"chart_ema_{card_idx}")
        # Day-by-day table (Day return % colored green/red via HTML)
        by_day = _trades_by_day(r["df_plot"], r["trades"])
        if not by_day.empty:
            st.markdown("**Day-by-day**")
            day_rows = []
            for _, row in by_day.iterrows():
                dr = row["Day return %"]
                dr_color = "#00c853" if dr >= 0 else "#ff3d57"
                day_rows.append(f"<tr><td>{row['Date']}</td><td>{int(row['Trades'])}</td><td>{int(row['Wins'])}</td><td>{row['Win rate %']}%</td><td style='color:{dr_color};font-weight:600;'>{dr:+.2f}%</td></tr>")
            st.markdown(f"""
            <div style="overflow-x:auto;">
            <table class="analytics-table" style="width:100%; color: #fff; font-size: 0.9rem;">
            <thead><tr><th>Date</th><th>Trades</th><th>Wins</th><th>Win rate %</th><th>Day return %</th></tr></thead>
            <tbody>{"".join(day_rows)}</tbody>
            </table>
            </div>
            """, unsafe_allow_html=True)
            fig_day = go.Figure()
            fig_day.add_trace(go.Bar(x=by_day["Date"], y=by_day["Win rate %"], name="Win rate %", marker_color="#00c853", opacity=0.8))
            fig_day.add_trace(go.Scatter(x=by_day["Date"], y=by_day["Day return %"], name="Day return %", line=dict(color="#4f8ef7", width=2), yaxis="y2"))
            fig_day.update_layout(template="plotly_dark", height=280, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(26,29,36,0.8)", font=dict(color="#e0e2e6"))
            fig_day.update_layout(xaxis_title="Date", yaxis=dict(title="Win rate %"), yaxis2=dict(title="Day return %", side="right", overlaying="y"))
            st.plotly_chart(fig_day, use_container_width=True, key=f"chart_day_{card_idx}")

# Section 6 — Strategy insights
st.markdown("---")
st.markdown("#### 📋 Strategy insights")
insights = _generate_insights(all_results, comparison, combined_result)
if insights:
    for bullet in insights:
        st.markdown(f"- {bullet}")
else:
    st.caption("Run backtests to see auto-generated insights.")

# Section 7 — Footer
st.markdown("""
<div class="analytics-footer">
  All returns simulated as SPX index % return. Entry at next bar open. Costs include spread, slippage, SEC fee.<br>
  <em>Not financial advice. For educational and research purposes only.</em><br>
  <span style="color: #2d3142;">SPX Backtests v1.0</span>
</div>
""", unsafe_allow_html=True)
