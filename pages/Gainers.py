"""
Top Gainers — Leading gainers by % change from previous close.
Click a symbol for deep analysis: multi-timeframe candlestick, volume, MACD, ADX, EMA.
"""
import os
import sys
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gainers_tv import get_top_gainers_sp500, fetch_stock_chart_data, run_gainers_backtest

st.set_page_config(page_title="Top Gainers", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

if "gainers_selected_symbol" not in st.session_state:
    st.session_state.gainers_selected_symbol = None
if "gainers_bt_summary" not in st.session_state:
    st.session_state.gainers_bt_summary = None
if "gainers_bt_details" not in st.session_state:
    st.session_state.gainers_bt_details = None

def _fmt_vol(v):
    if v is None or (isinstance(v, (int, float)) and (v != v or v < 0)):
        return "—"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.2f}K"
    return str(int(v))

def _fmt_pct(x):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    return f"{float(x):.2f}%"

st.title("S&P 500 Top Gainers")
st.caption("Gainers among S&P 500 stocks by % change from prior close • Data: yfinance • Backtest: did alerted stocks move up by EOD?")

tab_live, tab_backtest = st.tabs(["Live Gainers", "Backtest"])

# ---------- TAB 1: LIVE GAINERS ----------
with tab_live:
    now = datetime.now()
    window_end = now.strftime("%H:%M:%S")
    st.markdown(f"**S&P 500 gainers: {window_end} (daily data)**")

    limit = st.slider("Number of rows", 10, 100, 50, key="gainers_limit")

    if st.button("↻ Refresh gainers", type="primary", key="bt_refresh"):
        st.rerun()

    with st.spinner("Loading S&P 500 gainers..."):
        gainers = get_top_gainers_sp500(limit=limit, timeframe="1d")

    if not gainers:
        st.info("No S&P 500 gainers data. Check connection and try again.")
    else:
        # Build display table: Change From Close(%), Symbol, Price, Volume, Float, Rel Vol Daily, Rel Vol 5m, Gap(%), Short Interest
        rows = []
        for g in gainers:
            change_pct = g.get("change_pct")
            symbol = (g.get("symbol") or "—").strip()
            if not symbol or str(symbol).lower() in ("none", "nan"):
                continue
            price_val = g.get("price")
            # Skip rows with no price or zero (avoids misleading $0.00 from data glitches)
            if price_val is None or (isinstance(price_val, (int, float)) and (price_val != price_val or price_val <= 0)):
                continue
            rows.append({
                "Change From Close(%)": change_pct if change_pct is not None else 0.0,
                "Symbol": symbol,
                "Price": price_val,
                "Volume": _fmt_vol(g.get("volume")),
                "Float": "—",
                "Relative Volume (Daily)": g.get("relative_volume") if g.get("relative_volume") is not None else "—",
                "Gap(%)": g.get("gap_pct") if g.get("gap_pct") is not None else (change_pct if change_pct is not None else "—"),
                "Short Interest": "—",
            })

        df = pd.DataFrame(rows)

        # Style: green for change %, sort by Change From Close descending (already from API)
        df = df.sort_values("Change From Close(%)", ascending=False).reset_index(drop=True)

        # Color change column: green shades for positive
        def _color_change(v):
            if v is None or (isinstance(v, float) and (pd.isna(v) or v < 0)):
                return ""
            try:
                x = float(v)
                if x >= 20:
                    return "background-color: rgba(34, 197, 94, 0.35); color: #22c55e; font-weight: 600;"
                if x >= 10:
                    return "background-color: rgba(34, 197, 94, 0.2); color: #4ade80;"
                return "color: #86efac;"
            except Exception:
                return ""

        def _color_gap(v):
            if v is None or v == "—" or (isinstance(v, float) and pd.isna(v)):
                return ""
            try:
                x = float(v)
                if x > 0:
                    return "color: #22c55e;"
            except Exception:
                pass
            return ""

        # Format for display (keep df numeric for styling)
        def _fmt_price(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "—"
            v = float(x)
            return f"${v:,.4f}" if 0 < v < 1 else f"${v:,.2f}" if v >= 1 else "—"
        if "Gap(%)" in df.columns:
            display_df = df.copy()
            display_df["Change From Close(%)"] = display_df["Change From Close(%)"].apply(lambda x: f"{x:.2f}%" if x is not None and not (isinstance(x, float) and pd.isna(x)) else "—")
            display_df["Price"] = display_df["Price"].apply(_fmt_price)
            gap_vals = display_df["Gap(%)"]
            display_df["Gap(%)"] = gap_vals.apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and not (isinstance(x, float) and pd.isna(x)) else str(x) if x != "—" else "—")
        else:
            display_df = df.copy()
            display_df["Change From Close(%)"] = display_df["Change From Close(%)"].apply(lambda x: f"{x:.2f}%" if x is not None and not (isinstance(x, float) and pd.isna(x)) else "—")
            display_df["Price"] = display_df["Price"].apply(_fmt_price)

        # Table with clickable symbols (each symbol is a button — click to open charts)
        cols = ["Change From Close(%)", "Symbol", "Price", "Volume", "Float", "Relative Volume (Daily)", "Gap(%)", "Short Interest"]
        header_cols = st.columns(8)
        for i, c in enumerate(cols):
            header_cols[i].markdown(f"**{c}**")
        for row_idx, row in display_df.iterrows():
            r = display_df.loc[row_idx]
            c0, c1, c2, c3, c4, c5, c6, c7 = st.columns(8)
            c0.write(r["Change From Close(%)"])
            symbol = str(r["Symbol"]).strip()
            if symbol and symbol.lower() not in ("none", "nan", "—"):
                # Key by symbol only so clicking any symbol updates selected regardless of row order
                if c1.button(symbol, key=f"gainers_btn_{symbol}", use_container_width=True):
                    st.session_state.gainers_selected_symbol = symbol
                    st.rerun()
            else:
                c1.write(symbol or "—")
            c2.write(r["Price"])
            c3.write(r["Volume"])
            c4.write(r["Float"])
            c5.write(r["Relative Volume (Daily)"])
            c6.write(r["Gap(%)"])
            c7.write(r["Short Interest"])

        st.caption(f"Showing top {len(display_df)} S&P 500 gainers • Click any symbol for charts (price, volume, MACD, ADX)")

        # When a symbol was clicked, show charts below (for any selected symbol)
        if st.session_state.gainers_selected_symbol:
            st.markdown("---")
            ticker = st.session_state.gainers_selected_symbol
            if st.button("✕ Close deep analysis", key=f"close_deep_{ticker}"):
                st.session_state.gainers_selected_symbol = None
                st.rerun()
            st.markdown(f"### {ticker} — Price, Volume, MACD & ADX")
            timeframes = [
                ("1D (3 months)", "3mo", "1d"),
                ("5m (5 days)", "5d", "5m"),
                ("1h (1 month)", "1mo", "1h"),
            ]
            tabs = st.tabs([t[0] for t in timeframes])
            any_chart_shown = False
            for tab, (label, period, interval) in zip(tabs, timeframes):
                with tab:
                    with st.spinner(f"Loading {ticker} {label}…"):
                        chart_df = fetch_stock_chart_data(ticker, period=period, interval=interval)
                    fallback_note = None
                    if chart_df is None or chart_df.empty or len(chart_df) < 5:
                        # Fallback: show daily chart so user always sees something when daily data exists
                        if interval != "1d":
                            chart_df = fetch_stock_chart_data(ticker, period="3mo", interval="1d")
                            if chart_df is not None and not chart_df.empty and len(chart_df) >= 5:
                                fallback_note = f"No {label} data for this symbol. Showing daily chart instead."
                        if chart_df is None or chart_df.empty or len(chart_df) < 5:
                            st.warning(f"No data for {ticker}. This symbol may be OTC, delisted, or have no history.")
                            continue
                    any_chart_shown = True
                    if fallback_note:
                        st.info(fallback_note)
                    elif len(chart_df) < 15:
                        st.caption(f"Limited history for {ticker} ({len(chart_df)} bars).")
                    idx = chart_df.index.tolist()
                    fig = make_subplots(
                        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=(f"{ticker} Price (OHLC) + EMA 9/21", "Volume", "MACD (12, 26, 9)", "ADX (14)"),
                        row_heights=[0.45, 0.15, 0.2, 0.2],
                    )
                    # Candlestick
                    fig.add_trace(
                        go.Candlestick(
                            x=idx, open=chart_df["Open"], high=chart_df["High"], low=chart_df["Low"], close=chart_df["Close"],
                            name="OHLC", increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
                        ),
                        row=1, col=1,
                    )
                    if "EMA9" in chart_df.columns:
                        fig.add_trace(go.Scatter(x=idx, y=chart_df["EMA9"], name="EMA 9", line=dict(color="#a855f7", width=1.5)), row=1, col=1)
                    if "EMA21" in chart_df.columns:
                        fig.add_trace(go.Scatter(x=idx, y=chart_df["EMA21"], name="EMA 21", line=dict(color="#3b82f6", width=1.5)), row=1, col=1)
                    # Volume (green/red by close vs open)
                    vol_colors = ["#22c55e" if chart_df["Close"].iloc[i] >= chart_df["Open"].iloc[i] else "#ef4444" for i in range(len(chart_df))]
                    fig.add_trace(go.Bar(x=idx, y=chart_df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)
                    # MACD
                    if "MACD" in chart_df.columns and "MACD_signal" in chart_df.columns:
                        fig.add_trace(go.Scatter(x=idx, y=chart_df["MACD"], name="MACD", line=dict(color="#00d4aa")), row=3, col=1)
                        fig.add_trace(go.Scatter(x=idx, y=chart_df["MACD_signal"], name="Signal", line=dict(color="#f59e0b")), row=3, col=1)
                    if "MACD_hist" in chart_df.columns:
                        hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in chart_df["MACD_hist"]]
                        fig.add_trace(go.Bar(x=idx, y=chart_df["MACD_hist"], name="Histogram", marker_color=hist_colors, opacity=0.6), row=3, col=1)
                    # ADX
                    if "ADX" in chart_df.columns:
                        fig.add_trace(go.Scatter(x=idx, y=chart_df["ADX"], name="ADX", line=dict(color="#a855f7", width=2)), row=4, col=1)
                    fig.update_layout(template="plotly_dark", height=720, showlegend=True, margin=dict(t=50), xaxis_rangeslider_visible=False)
                    fig.update_yaxes(title_text="Price", row=1, col=1)
                    fig.update_yaxes(title_text="Volume", row=2, col=1)
                    fig.update_yaxes(title_text="MACD", row=3, col=1)
                    fig.update_yaxes(title_text="ADX", row=4, col=1)
                    st.plotly_chart(fig, use_container_width=True, key=f"gainers_chart_{ticker}_{period}_{interval}")
            if not any_chart_shown:
                st.info(f"No chart data available for **{ticker}** right now. Try again in a minute (rate limit) or check the symbol.")

# ---------- TAB 2: BACKTEST ----------
with tab_backtest:
    st.markdown("Simulate the scanner: each day we take the **top N stocks by gap %** (open vs prior close), then check how many **moved up by EOD** (close vs open ≥ threshold).")
    default_end = datetime.now().date()
    default_start = default_end - timedelta(days=60)
    col_d1, col_d2, col_n, col_pct = st.columns(4)
    with col_d1:
        from_date = st.date_input("From date", value=default_start, key="bt_from")
    with col_d2:
        to_date = st.date_input("To date", value=default_end, key="bt_to")
    with col_n:
        top_n = st.number_input("Top N per day", min_value=5, max_value=50, value=20, key="bt_top_n")
    with col_pct:
        min_eod_pct = st.number_input("Min EOD move % (win)", min_value=0.5, max_value=10.0, value=2.0, step=0.5, key="bt_min_eod")
    if st.button("Run Backtest", type="primary", key="bt_run"):
        with st.spinner("Running backtest (fetching daily data per ticker)…"):
            summary, details = run_gainers_backtest(
                from_date=str(from_date), to_date=str(to_date), top_n=int(top_n), min_eod_pct=float(min_eod_pct)
            )
        st.session_state.gainers_bt_summary = summary
        st.session_state.gainers_bt_details = details
    if st.session_state.gainers_bt_summary is not None:
        s = st.session_state.gainers_bt_summary
        if s.get("error"):
            st.error(s["error"])
        else:
            st.metric("Total alerts", s.get("total_alerts", 0))
            st.metric("Wins (EOD ≥ threshold)", s.get("wins", 0))
            st.metric("Win rate %", f"{s.get('win_rate_pct', 0):.1f}%")
            st.metric("Avg EOD %", f"{s.get('avg_eod_pct', 0):.2f}%")
            st.caption(f"Days: {s.get('days_run', 0)} • Top N/day: {s.get('top_n_per_day')} • Threshold: {s.get('min_eod_threshold')}%")
            if st.session_state.gainers_bt_details:
                det_df = pd.DataFrame(st.session_state.gainers_bt_details)
                st.dataframe(det_df, use_container_width=True, hide_index=True)
