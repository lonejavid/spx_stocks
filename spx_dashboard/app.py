"""
SPX Day Trading Dashboard - Main Streamlit application.
Professional signal system with scoring, backtesting, and alerts.
"""
from datetime import datetime, timedelta, time as dt_time
import pytz
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import indicators
import signals
import backtest
import notifications

EST = pytz.timezone("America/New_York")


def _macd_cols_spx(df):
    mc = [c for c in df.columns if c == "MACD" or (str(c).startswith("MACD") and "MACDh" not in str(c) and "MACDs" not in str(c))][:1]
    sc = [c for c in df.columns if "MACDs" in str(c)][:1]
    if not mc:
        mc = [c for c in df.columns if "MACD" in str(c) and "MACDh" not in str(c) and "MACDs" not in str(c)][:1]
    return (mc[0] if mc else None, sc[0] if sc else None)

# Trading windows (best times)
WINDOW1 = (9, 45), (11, 30)
WINDOW2 = (14, 0), (15, 30)

st.set_page_config(page_title="SPX Day Trading", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

DARK_CSS = """
<style>
/* Base */
.block-container { padding: 1rem; max-width: 1800px; margin-left: auto; margin-right: auto; width: 100%; box-sizing: border-box; }
h1, h2, h3 { color: #f0f2f6 !important; }
h1 { font-size: 1.75rem !important; border-bottom: 2px solid #00d4aa; padding-bottom: 0.4rem; }
h2 { font-size: 1.1rem !important; color: #b0b4bc !important; margin-top: 1rem; }
[data-testid="stMetricValue"] { font-size: 1.35rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #8892a0 !important; font-size: 0.75rem !important; text-transform: uppercase !important; }
#MainMenu, footer { visibility: hidden; }
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
  [data-testid="stSidebar"] { min-width: 100% !important; }
}
/* Small mobile */
@media (max-width: 480px) {
  .block-container { padding: 0.4rem 0.5rem; }
  h1 { font-size: 1.2rem !important; }
  h2 { font-size: 0.9rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
}
.market-banner { font-size: 1rem; padding: 12px 16px !important; }
.strong-signal-alert span { font-size: 1.8rem !important; }
@media (max-width: 768px) {
  .market-banner { font-size: 0.9rem !important; padding: 10px 12px !important; }
  .strong-signal-alert { padding: 14px 12px !important; }
  .strong-signal-alert span { font-size: 1.35rem !important; }
}
@media (max-width: 480px) {
  .market-banner { font-size: 0.85rem !important; padding: 8px 10px !important; }
  .strong-signal-alert { padding: 10px 8px !important; }
  .strong-signal-alert span { font-size: 1.15rem !important; }
}
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


def _next_open_et():
    now = datetime.now(EST)
    today = now.date()
    open_today = EST.localize(datetime.combine(today, dt_time(9, 30)))
    if now.weekday() < 5 and now < open_today:
        return open_today
    days = 3 if now.weekday() == 4 else (2 if now.weekday() == 5 else 1)
    next_day = today + timedelta(days=days)
    return EST.localize(datetime.combine(next_day, dt_time(9, 30)))


def countdown_next_open():
    now = datetime.now(EST)
    nxt = _next_open_et()
    delta = nxt - now
    if delta.total_seconds() <= 0:
        return "Opening soon"
    s = int(delta.total_seconds())
    d, r = divmod(s, 86400)
    h, r = divmod(r, 3600)
    m, _ = divmod(r, 60)
    return f"{d}d {h}h {m}m" if d else f"{h}h {m}m"


def trading_window_shapes(df):
    if df.empty:
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
    out = []
    for (h1, m1), (h2, m2) in [WINDOW1, WINDOW2]:
        x0 = EST.localize(datetime.combine(d, dt_time(h1, m1)))
        x1 = EST.localize(datetime.combine(d, dt_time(h2, m2)))
        out.append(dict(type="rect", x0=x0, x1=x1, y0=0, y1=1, yref="paper", fillcolor="rgba(0,212,170,0.12)", line=dict(width=0)))
    return out


def _in_optimal_window():
    now = datetime.now(EST).time()
    w1 = dt_time(9, 45) <= now <= dt_time(11, 30)
    w2 = dt_time(14, 0) <= now <= dt_time(15, 30)
    return w1 or w2


def run_dashboard():
    st.markdown(DARK_CSS, unsafe_allow_html=True)
    st.title("SPX Day Trading Dashboard")
    st.caption("Signal scoring 0-8 · RSI · MACD · VWAP · EMA · Confirmation stocks · Backtest")

    # Sidebar: Telegram config
    with st.sidebar:
        st.subheader("Telegram Alerts")
        telegram_token = st.text_input("Bot Token (optional)", type="password")
        telegram_chat = st.text_input("Chat ID (optional)")
        send_telegram = st.checkbox("Send on STRONG signal (6+)", value=False)

    is_open = data.is_market_open()
    show_date, data_label = data.get_display_date()
    show_date_str = show_date.isoformat() if hasattr(show_date, "isoformat") else str(show_date)

    if is_open:
        st.markdown(
            '<div class="market-banner" style="background:linear-gradient(90deg,rgba(46,204,113,0.25),rgba(46,204,113,0.1));border:1px solid #2ecc71;border-radius:8px;margin-bottom:1rem;">'
            '<span>🟢 <strong>MARKET OPEN</strong> — Live Data</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="market-banner" style="background:linear-gradient(90deg,rgba(231,76,60,0.25),rgba(231,76,60,0.1));border:1px solid #e74c3c;border-radius:8px;margin-bottom:0.5rem;">'
            '<span>🔴 <strong>MARKET CLOSED</strong></span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"**Next open in:** {countdown_next_open()}")

    with st.spinner("Loading data..."):
        try:
            payload = data.fetch_all_dashboard_data(show_date_str if not is_open else None)
        except Exception as e:
            st.error(f"Data error: {e}")
            return
    spx = payload.get("spx")
    if spx is None or spx.empty or len(spx) < 5:
        st.warning("Not enough SPX 5-min data. Try during market hours (9:30–16:00 ET) or view previous day when closed.")
        return

    spx = indicators.add_spx_indicators(spx)
    for t in data.CONFIRMATION_TICKERS:
        df_t = payload.get(t)
        if df_t is not None and not df_t.empty:
            payload[t] = indicators.add_confirmation_indicators(df_t)

    vix_value = payload.get("vix_value")
    buy_score, sell_score, buy_list, sell_list, strength = signals.compute_scores(spx, vix_value, payload)
    now_est = datetime.now(EST).strftime("%Y-%m-%d %H:%M ET")

    # Strong signal alert
    if (buy_score >= 6 or sell_score >= 6) and (buy_score >= sell_score or sell_score > buy_score):
        alert_type = "STRONG BUY" if buy_score >= sell_score else "STRONG SELL"
        score = max(buy_score, sell_score)
        st.markdown(
            f'<div class="strong-signal-alert" style="background:#1a1d24;border:3px solid #00d4aa;border-radius:12px;padding:20px;text-align:center;">'
            f'<span style="font-size:1.8rem;color:#00d4aa;">🔥 {alert_type} — Score {score}/8</span></div>',
            unsafe_allow_html=True,
        )
        if send_telegram and telegram_token and telegram_chat:
            triggers = buy_list if buy_score >= sell_score else sell_list
            notifications.send_telegram_alert(
                telegram_token, telegram_chat, alert_type, score,
                float(spx["Close"].iloc[-1]), now_est, triggers
            )

    # Top row: SPX, VIX, Signal, Time
    spx_close = spx["Close"].iloc[-1]
    spx_open = spx["Open"].iloc[0] if "Open" in spx.columns else spx_close
    spx_pct = ((spx_close - spx_open) / spx_open * 100) if spx_open and spx_open != 0 else 0
    vix_status = "Calm" if (vix_value is not None and vix_value < 20) else "Fearful" if (vix_value and vix_value < 30) else "Panic"
    display_score = f"{buy_score}/8" if buy_score >= sell_score else f"{sell_score}/8"
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("SPX", f"${spx_close:,.2f}", f"{spx_pct:+.2f}%")
    m2.metric("VIX", f"{vix_value:.1f}" if vix_value else "—", vix_status)
    m3.metric("Signal", strength, display_score)
    m4.metric("Time (ET)", now_est, "Open" if is_open else "Closed")
    st.markdown("---")

    # Second row: Score breakdown
    st.subheader("Signal score (0-8)")
    score_val = max(buy_score, sell_score)
    st.progress(score_val / 8.0)
    st.write(f"**{strength}** — {score_val}/8 conditions met")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**BUY conditions**")
        for i, name in enumerate(signals.BUY_CONDITIONS):
            st.write("✅" if name in buy_list else "❌", name)
    with col_b:
        st.write("**SELL conditions**")
        for i, name in enumerate(signals.SELL_CONDITIONS):
            st.write("✅" if name in sell_list else "❌", name)
    if not _in_optimal_window() and is_open:
        st.warning("⚠️ Outside optimal trading window (9:45–11:30 AM, 2:00–3:30 PM EST)")
    st.markdown("---")

    # SPX 5-min chart with VWAP, EMAs, BB, windows
    st.subheader(f"SPX 5-min — {data_label}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=spx.index, y=spx["Close"], name="SPX", line=dict(color="#00d4aa", width=2), fill="tozeroy", fillcolor="rgba(0,212,170,0.08)"))
    if "VWAP" in spx.columns:
        fig.add_trace(go.Scatter(x=spx.index, y=spx["VWAP"], name="VWAP", line=dict(color="#f1c40f", width=1.5, dash="dash")))
    if "EMA9" in spx.columns:
        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA9"], name="EMA 9", line=dict(color="#3498db", width=1.2)))
    if "EMA21" in spx.columns:
        fig.add_trace(go.Scatter(x=spx.index, y=spx["EMA21"], name="EMA 21", line=dict(color="#e67e22", width=1.2)))
    if "BBU" in spx.columns and "BBL" in spx.columns:
        fig.add_trace(go.Scatter(x=spx.index, y=spx["BBU"], name="BB Upper", line=dict(color="#95a5a6", width=1, dash="dot")))
        fig.add_trace(go.Scatter(x=spx.index, y=spx["BBL"], name="BB Lower", line=dict(color="#95a5a6", width=1, dash="dot")))
    for s in trading_window_shapes(spx):
        fig.add_shape(s)
    fig.update_layout(**PLOTLY_LAYOUT, height=400, xaxis=AXIS, yaxis=AXIS)
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    st.subheader("RSI (14)")
    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=spx.index, y=spx["RSI"], name="RSI", line=dict(color="#00d4aa", width=2)))
    fig_rsi.add_hline(y=40, line_dash="dash", line_color="#2ecc71")
    fig_rsi.add_hline(y=60, line_dash="dash", line_color="#e74c3c")
    fig_rsi.update_layout(**PLOTLY_LAYOUT, height=260, xaxis=AXIS, yaxis=dict(**AXIS, range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

    # MACD
    mc, sc = _macd_cols_spx(spx)
    if mc and sc:
        st.subheader("MACD (12, 26, 9)")
        fig_macd = go.Figure()
        fig_macd.add_trace(go.Scatter(x=spx.index, y=spx[mc], name="MACD", line=dict(color="#00d4aa", width=2)))
        fig_macd.add_trace(go.Scatter(x=spx.index, y=spx[sc], name="Signal", line=dict(color="#f39c12", width=1.5)))
        hist_c = [c for c in spx.columns if "MACDh" in str(c) or "hist" in c.lower()][:1]
        if hist_c:
            colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in spx[hist_c[0]]]
            fig_macd.add_trace(go.Bar(x=spx.index, y=spx[hist_c[0]], name="Hist", marker_color=colors, opacity=0.6))
        fig_macd.update_layout(**PLOTLY_LAYOUT, height=260, xaxis=AXIS, yaxis=AXIS)
        st.plotly_chart(fig_macd, use_container_width=True)

    # Volume
    if "Volume" in spx.columns:
        st.subheader("Volume")
        fig_vol = go.Figure(go.Bar(x=spx.index, y=spx["Volume"], marker_color="#7f8c8d", opacity=0.7))
        fig_vol.update_layout(**PLOTLY_LAYOUT, height=220, xaxis=AXIS, yaxis=AXIS)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Confirmation stocks table
    st.subheader("Confirmation stocks")
    rows = []
    for t in data.CONFIRMATION_TICKERS:
        df_t = payload.get(t)
        if df_t is None or df_t.empty or "Close" not in df_t.columns:
            rows.append({"Ticker": t, "Price": "—", "% Today": "—", "vs VWAP": "—", "RSI": "—", "Status": "—"})
            continue
        close_t = df_t["Close"].iloc[-1]
        pct = df_t["PctChange"].iloc[-1] if "PctChange" in df_t.columns else 0
        above = "Above" if df_t["AboveVWAP"].iloc[-1] else "Below" if "AboveVWAP" in df_t.columns else "—"
        rsi_t = df_t["RSI"].iloc[-1] if "RSI" in df_t.columns else None
        status = "Bullish" if pct > 0 and above == "Above" else "Bearish" if pct < 0 else "Neutral"
        rows.append({
            "Ticker": t, "Price": f"${close_t:,.2f}", "% Today": f"{pct:+.2f}%",
            "vs VWAP": above, "RSI": f"{rsi_t:.1f}" if rsi_t is not None else "—", "Status": status,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Backtesting
    st.markdown("---")
    st.subheader("Backtesting (1 year daily)")
    with st.spinner("Running backtests..."):
        try:
            daily = data.fetch_daily_for_backtest("^GSPC", 1)
            if daily is not None and len(daily) >= 50:
                r1 = backtest.backtest_rsi_ema50(daily)
                r2 = backtest.backtest_macd_crossover(daily)
                r3 = backtest.backtest_ema_crossover(daily)
                b1, b2, b3 = st.columns(3)
                for col, r in [(b1, r1), (b2, r2), (b3, r3)]:
                    with col:
                        st.metric(r["name"], f"{r['total_return_pct']:.1f}% return", f"{r['total_trades']} trades")
                        st.caption(f"Win rate {r['win_rate_pct']:.0f}% · Max DD {r['max_drawdown_pct']:.1f}%")
                if not r1["equity_curve"].empty:
                    fig_eq = go.Figure(go.Scatter(x=r1["equity_curve"].index, y=r1["equity_curve"].values, name="Equity", line=dict(color="#00d4aa", width=2)))
                    fig_eq.update_layout(**PLOTLY_LAYOUT, title="Equity curve (RSI+EMA50)", height=300, xaxis=AXIS, yaxis=AXIS)
                    st.plotly_chart(fig_eq, use_container_width=True)
            else:
                st.info("Not enough daily data for backtest.")
        except Exception as e:
            st.warning(f"Backtest error: {e}")

    # Signals log
    st.subheader("Trading signals log")
    if "signals_log" not in st.session_state:
        st.session_state.signals_log = []
    if buy_score >= 5 or sell_score >= 5:
        sig_type = "BUY" if buy_score >= sell_score else "SELL"
        sc = buy_score if sig_type == "BUY" else sell_score
        st.session_state.signals_log.append({
            "Time": now_est, "Signal": sig_type, "Score": f"{sc}/8",
            "SPX": f"${spx_close:,.2f}", "Conditions": ", ".join((buy_list if sig_type == "BUY" else sell_list)[:3]),
        })
        st.session_state.signals_log = st.session_state.signals_log[-100:]
    if st.session_state.signals_log:
        st.dataframe(pd.DataFrame(st.session_state.signals_log).iloc[::-1], use_container_width=True, hide_index=True)
    else:
        st.info("No signals logged yet. STRONG/MODERATE signals (5+) will appear here.")

    # Win/Loss tracker (simplified: count correct vs wrong from log)
    st.subheader("Win/Loss tracker")
    total_sigs = len(st.session_state.signals_log)
    st.metric("Signals today", total_sigs)
    st.caption("Outcome tracking (30 min later) and simulated P&L can be extended with session state.")

    st.markdown("---")
    st.caption("Data refreshes every 5 min when market open. Best windows: 9:45–11:30 AM & 2:00–3:30 PM EST.")


# Auto-refresh when market open
try:
    use_frag = hasattr(st, "fragment") and callable(getattr(st.fragment, "__call__", None))
except Exception:
    use_frag = False
if use_frag:
    try:
        @st.fragment(run_every=timedelta(seconds=300))
        def _refresh():
            run_dashboard()
        _refresh()
    except Exception:
        run_dashboard()
else:
    run_dashboard()
