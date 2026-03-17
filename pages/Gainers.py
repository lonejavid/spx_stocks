"""
Top Gainers — Leading gainers by % change from previous close.
Click a symbol for deep analysis: multi-timeframe candlestick, volume, MACD, ADX, EMA.
Live Mode: auto-refresh every 60s with audio/visual alerts for new high-score stocks.
"""
import os
import sys
import time
from datetime import datetime, timedelta

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gainers_tv import get_top_gainers_sp500, fetch_stock_chart_data, run_gainers_backtest

st.set_page_config(page_title="Top Gainers", page_icon="📈", layout="wide", initial_sidebar_state="expanded")


@st.cache_data(ttl=300)
def _cached_gainers(limit: int, timeframe: str):
    return get_top_gainers_sp500(limit=limit, timeframe=timeframe)

if "gainers_selected_symbol" not in st.session_state:
    st.session_state.gainers_selected_symbol = None
if "gainers_bt_summary" not in st.session_state:
    st.session_state.gainers_bt_summary = None
if "gainers_bt_details" not in st.session_state:
    st.session_state.gainers_bt_details = None
# Live alert system
if "gainers_live_mode" not in st.session_state:
    st.session_state.gainers_live_mode = False
if "previous_scanner_symbols" not in st.session_state:
    st.session_state.previous_scanner_symbols = None  # set of symbols that met alert criteria last scan
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []  # last 20 alert strings

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

    live_col, refresh_col, _ = st.columns([1, 1, 2])
    with live_col:
        live_mode = st.toggle("Live Mode ON/OFF", value=st.session_state.gainers_live_mode, key="gainers_live_toggle")
        st.session_state.gainers_live_mode = live_mode
    with refresh_col:
        if st.button("↻ Refresh", type="primary", key="bt_refresh"):
            _cached_gainers.clear()
            st.rerun()

    with st.spinner("Loading S&P 500 gainers..."):
        gainers = _cached_gainers(limit, "1d")

    if not gainers:
        st.info("No S&P 500 gainers data.. Check connection and try again.")
    else:
        # Filter controls above table
        st.markdown("**Filters**")
        f1, f2, f3 = st.columns(3)
        with f1:
            min_gap_filter = st.slider("Min Gap %", 0.0, 10.0, 0.0, 0.5, key="live_min_gap")
        with f2:
            min_vol_ratio_filter = st.slider("Min Volume Ratio", 0.0, 5.0, 0.0, 0.1, key="live_min_vol")
        with f3:
            min_score_filter = st.slider("Min Score", 0.0, 20.0, 0.0, 0.5, key="live_min_score")

        # Build rows with new fields; apply filters
        rows = []
        for g in gainers:
            change_pct = g.get("change_pct")
            symbol = (g.get("symbol") or "—").strip()
            if not symbol or str(symbol).lower() in ("none", "nan"):
                continue
            price_val = g.get("price")
            if price_val is None or (isinstance(price_val, (int, float)) and (price_val != price_val or price_val <= 0)):
                continue
            gap_pct = g.get("gap_pct")
            gap_val = float(gap_pct) if gap_pct is not None and not (isinstance(gap_pct, float) and pd.isna(gap_pct)) else 0.0
            vol_ratio = g.get("volume_ratio")
            vol_ratio_val = float(vol_ratio) if vol_ratio is not None and not (isinstance(vol_ratio, float) and pd.isna(vol_ratio)) else None
            score_val = g.get("scanner_score")
            score_val = float(score_val) if score_val is not None and not (isinstance(score_val, float) and pd.isna(score_val)) else None
            # Apply filters: treat missing volume_ratio/score as "pass" so old or incomplete data still shows rows
            if gap_val < min_gap_filter:
                continue
            if vol_ratio_val is not None and vol_ratio_val < min_vol_ratio_filter:
                continue
            if score_val is not None and score_val < min_score_filter:
                continue
            intraday = g.get("intraday_move_pct")
            intraday_val = intraday if intraday is not None else 0.0
            rows.append({
                "Change From Close(%)": change_pct if change_pct is not None else 0.0,
                "Symbol": symbol,
                "Price": price_val,
                "Volume": _fmt_vol(g.get("volume")),
                "Volume Ratio": vol_ratio_val if vol_ratio_val is not None else 0.0,
                "Intraday Move %": intraday_val,
                "Score": score_val if score_val is not None else 0.0,
                "Float": "—",
                "Relative Volume (Daily)": g.get("relative_volume") if g.get("relative_volume") is not None else "—",
                "Gap(%)": gap_pct if gap_pct is not None else (change_pct if change_pct is not None else "—"),
                "Short Interest": "—",
            })

        df = pd.DataFrame(rows)
        # When all rows are filtered out, DataFrame is empty and has no columns — sort would raise KeyError
        if not df.empty and "Change From Close(%)" in df.columns:
            df = df.sort_values("Change From Close(%)", ascending=False).reset_index(drop=True)

        # Alert detection: scanner_score > 5 and gap_pct > 2 (from raw gainers)
        alert_symbols_this_scan = set()
        for g in gainers:
            sym = (g.get("symbol") or "").strip()
            if not sym:
                continue
            score_val = g.get("scanner_score")
            score_val = float(score_val) if score_val is not None and not (isinstance(score_val, float) and pd.isna(score_val)) else 0.0
            gap_val = g.get("gap_pct")
            gap_val = float(gap_val) if gap_val is not None and not (isinstance(gap_val, float) and pd.isna(gap_val)) else 0.0
            if score_val > 5 and gap_val > 2:
                alert_symbols_this_scan.add(sym)
        prev = st.session_state.previous_scanner_symbols
        if prev is None:
            st.session_state.previous_scanner_symbols = set(alert_symbols_this_scan)
            new_alert_symbols = set()
        else:
            new_alert_symbols = alert_symbols_this_scan - prev
            st.session_state.previous_scanner_symbols = set(alert_symbols_this_scan)
        for sym in new_alert_symbols:
            g = next((x for x in gainers if (x.get("symbol") or "").strip() == sym), None)
            if g:
                ts = now.strftime("%H:%M")
                gap = g.get("gap_pct") or 0
                vr = g.get("volume_ratio") or 0
                sc = g.get("scanner_score") or 0
                line = f"{ts} — {sym} gapped {gap:.2f}% with volume ratio {vr:.2f} — Score: {sc:.2f}"
                st.session_state.alert_log.append(line)
        st.session_state.alert_log = st.session_state.alert_log[-20:]

        # New-alerts badge and beep
        if new_alert_symbols:
            st.markdown(
                f'<span style="background-color: #ef4444; color: white; padding: 6px 12px; border-radius: 6px; font-weight: 700;">🔔 {len(new_alert_symbols)} NEW ALERTS</span>',
                unsafe_allow_html=True,
            )
            beep_js = """
            <script>
            (function() {
                try {
                    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const oscillator = audioCtx.createOscillator();
                    const gainNode = audioCtx.createGain();
                    oscillator.connect(gainNode);
                    gainNode.connect(audioCtx.destination);
                    oscillator.type = 'sine';
                    oscillator.frequency.setValueAtTime(880, audioCtx.currentTime);
                    gainNode.gain.setValueAtTime(0.3, audioCtx.currentTime);
                    gainNode.gain.exponentialRampToValueAtTime(0.0001, audioCtx.currentTime + 0.8);
                    oscillator.start(audioCtx.currentTime);
                    oscillator.stop(audioCtx.currentTime + 0.8);
                } catch (e) {}
            })();
            </script>
            """
            components.html(beep_js, height=0)

        # Summary bar (safe when df is empty)
        n_found = len(df)
        avg_gap = df["Gap(%)"].mean() if n_found and not df.empty and "Gap(%)" in df.columns else 0.0
        try:
            avg_gap = float(avg_gap) if avg_gap is not None and not (isinstance(avg_gap, float) and pd.isna(avg_gap)) else 0.0
        except Exception:
            avg_gap = 0.0
        avg_vol_ratio = df["Volume Ratio"].mean() if n_found and not df.empty and "Volume Ratio" in df.columns else 0.0
        st.markdown(
            f"**{n_found} stocks found** | **Avg Gap:** {avg_gap:.2f}% | **Avg Volume Ratio:** {avg_vol_ratio:.2f} | **Last updated:** {now.strftime('%H:%M')}"
        )

        def _fmt_price(x):
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return "—"
            v = float(x)
            return f"${v:,.4f}" if 0 < v < 1 else f"${v:,.2f}" if v >= 1 else "—"

        # Table only when we have rows and expected columns (avoid KeyError on empty filtered result)
        if not df.empty and "Change From Close(%)" in df.columns:
            display_df = df.copy()
            display_df["Change From Close(%)"] = display_df["Change From Close(%)"].apply(lambda x: f"{x:.2f}%" if x is not None and not (isinstance(x, float) and pd.isna(x)) else "—")
            display_df["Price"] = display_df["Price"].apply(_fmt_price)
            gap_vals = display_df["Gap(%)"]
            display_df["Gap(%)"] = gap_vals.apply(lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) and not (isinstance(x, float) and pd.isna(x)) else str(x) if x != "—" else "—")
            display_df["Volume Ratio"] = display_df["Volume Ratio"].apply(lambda x: f"{x:.2f}")
            display_df["Intraday Move %"] = display_df["Intraday Move %"].apply(lambda x: f"{x:.2f}%")
            display_df["Score"] = display_df["Score"].apply(lambda x: f"{x:.2f}")

            # Table with indicator column + new-alert row highlight (yellow/green glow)
            cols = ["", "Change From Close(%)", "Symbol", "Price", "Volume", "Volume Ratio", "Intraday Move %", "Score", "Gap(%)", "Float", "Relative Volume (Daily)", "Short Interest"]
            header_cols = st.columns(12)
            for i, c in enumerate(cols):
                header_cols[i].markdown(f"**{c}**" if c else " ")
            for row_idx, row in display_df.iterrows():
                r = display_df.loc[row_idx]
                symbol = str(r["Symbol"]).strip()
                is_new_alert = symbol in new_alert_symbols
                row_style = ""
                if is_new_alert:
                    row_style = "border: 2px solid #22c55e; border-radius: 6px; padding: 2px 4px; margin: 2px 0; box-shadow: 0 0 12px rgba(34, 197, 94, 0.5);"
                c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11 = st.columns(12)
                if is_new_alert:
                    c0.markdown('<span style="background: #22c55e; color: white; padding: 2px 6px; border-radius: 4px; font-size: 0.75em; font-weight: 700;">NEW</span>', unsafe_allow_html=True)
                    c1.markdown(f'<div style="{row_style}">{r["Change From Close(%)"]}</div>', unsafe_allow_html=True)
                else:
                    c0.write("")
                    c1.write(r["Change From Close(%)"])
                if symbol and symbol.lower() not in ("none", "nan", "—"):
                    if c2.button(symbol, key=f"gainers_btn_{symbol}", use_container_width=True):
                        st.session_state.gainers_selected_symbol = symbol
                        st.rerun()
                else:
                    c2.write(symbol or "—")
                c3.write(r["Price"])
                c4.write(r["Volume"])
                vr = df.loc[row_idx, "Volume Ratio"]
                vr_style = "color: #22c55e;" if vr > 2 else "color: #eab308;" if vr >= 1.5 else "color: #9ca3af;"
                c5.markdown(f"<span style='{vr_style}'>{r['Volume Ratio']}</span>", unsafe_allow_html=True)
                im = df.loc[row_idx, "Intraday Move %"]
                im_style = "color: #22c55e;" if im >= 0 else "color: #ef4444;"
                c6.markdown(f"<span style='{im_style}'>{r['Intraday Move %']}</span>", unsafe_allow_html=True)
                sc = df.loc[row_idx, "Score"]
                sc_style = "color: #22c55e;" if sc > 5 else "color: #eab308;" if sc >= 2 else "color: #ef4444;"
                c7.markdown(f"<span style='{sc_style}'>{r['Score']}</span>", unsafe_allow_html=True)
                c8.write(r["Gap(%)"])
                c9.write(r["Float"])
                c10.write(r["Relative Volume (Daily)"])
                c11.write(r["Short Interest"])

            st.caption(f"Showing {n_found} S&P 500 gainers • Click any symbol for charts (price, volume, MACD, ADX)")
        else:
            st.caption("No rows match the current filters. Try lowering Min Gap %, Min Volume Ratio, or Min Score.")

        # Alert Log (last 20 alerts, scrollable)
        st.markdown("**Alert Log**")
        log_lines = st.session_state.alert_log[-20:]
        log_text = "\n".join(reversed(log_lines)) if log_lines else "No alerts yet. Alerts trigger when scanner_score > 5 and gap_pct > 2 and the stock is new this scan."
        st.text_area("Alert log content", value=log_text, height=120, disabled=True, key="alert_log_display", label_visibility="collapsed")

        # When a symbol was clicked, show charts in a single placeholder (avoids duplicate section on rerun/Live Mode)
        _charts_placeholder = st.empty()
        if st.session_state.gainers_selected_symbol:
            ticker = st.session_state.gainers_selected_symbol
            with _charts_placeholder.container():
                st.markdown("---")
                if st.button("✕ Close deep analysis", key=f"close_deep_{ticker}"):
                    st.session_state.gainers_selected_symbol = None
                    st.rerun()
                st.markdown(f"### {ticker} — Price, Volume, MACD & ADX")
                timeframes = [
                    ("5min Today", "1d", "5m"),
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
                        vol_colors = ["#22c55e" if chart_df["Close"].iloc[i] >= chart_df["Open"].iloc[i] else "#ef4444" for i in range(len(chart_df))]
                        fig.add_trace(go.Bar(x=idx, y=chart_df["Volume"], name="Volume", marker_color=vol_colors, opacity=0.7), row=2, col=1)
                        if "MACD" in chart_df.columns and "MACD_signal" in chart_df.columns:
                            fig.add_trace(go.Scatter(x=idx, y=chart_df["MACD"], name="MACD", line=dict(color="#00d4aa")), row=3, col=1)
                            fig.add_trace(go.Scatter(x=idx, y=chart_df["MACD_signal"], name="Signal", line=dict(color="#f59e0b")), row=3, col=1)
                        if "MACD_hist" in chart_df.columns:
                            hist_colors = ["#22c55e" if v >= 0 else "#ef4444" for v in chart_df["MACD_hist"]]
                            fig.add_trace(go.Bar(x=idx, y=chart_df["MACD_hist"], name="Histogram", marker_color=hist_colors, opacity=0.6), row=3, col=1)
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
        else:
            _charts_placeholder.empty()

        # Live Mode: countdown and auto-refresh every 60s (after charts so symbol click always shows graphs)
        if live_mode:
            countdown_placeholder = st.empty()
            for secs_left in range(60, 0, -1):
                countdown_placeholder.markdown(f"**Next refresh in: {secs_left}s**")
                time.sleep(1)
            _cached_gainers.clear()
            st.rerun()

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
    st.markdown("**Filters (applied before taking top N per day)**")
    col_gap, col_vol = st.columns(2)
    with col_gap:
        min_gap_pct = st.slider("Min Gap %", 0.5, 10.0, 2.0, 0.5, key="bt_min_gap")
    with col_vol:
        min_volume_ratio = st.slider("Min Volume Ratio", 1.0, 5.0, 1.5, 0.1, key="bt_min_vol")
    if st.button("Run Backtest", type="primary", key="bt_run"):
        with st.spinner("Running backtest (fetching daily data per ticker)…"):
            summary, details = run_gainers_backtest(
                from_date=str(from_date),
                to_date=str(to_date),
                top_n=int(top_n),
                min_eod_pct=float(min_eod_pct),
                min_gap_pct=float(min_gap_pct),
                min_volume_ratio=float(min_volume_ratio),
            )
        st.session_state.gainers_bt_summary = summary
        st.session_state.gainers_bt_details = details
    if st.session_state.gainers_bt_summary is not None:
        s = st.session_state.gainers_bt_summary
        if s.get("error"):
            st.error(s["error"])
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Alerts", s.get("total_alerts", 0))
            m2.metric("Total Wins", s.get("wins", 0))
            m3.metric("Win Rate %", f"{s.get('win_rate_pct', 0):.1f}%")
            m4.metric("Avg EOD Move %", f"{s.get('avg_eod_move_pct', s.get('avg_eod_pct', 0)):.2f}%")
            st.caption(f"Days: {s.get('days_run', 0)} • Top N/day: {s.get('top_n_per_day')} • Threshold: {s.get('min_eod_threshold')}%")
            # Win Rate by Month bar chart
            by_month = s.get("by_month") or []
            if by_month:
                months = [x["month"] for x in by_month]
                win_rates = [x["win_rate_pct"] for x in by_month]
                fig_bt = go.Figure(data=[go.Bar(x=months, y=win_rates, marker_color="#22c55e", opacity=0.8)])
                fig_bt.update_layout(
                    title="Win Rate by Month",
                    xaxis_title="Month",
                    yaxis_title="Win Rate %",
                    template="plotly_dark",
                    height=360,
                    margin=dict(t=50),
                )
                st.plotly_chart(fig_bt, use_container_width=True, key="bt_month_chart")
            if st.session_state.gainers_bt_details:
                det_df = pd.DataFrame(st.session_state.gainers_bt_details)
                # Style Win column: green "✓ Win" / red "✗ Loss"
                def _style_win(val):
                    if val is True:
                        return "background-color: rgba(34, 197, 94, 0.35); color: #166534; font-weight: 600;"
                    return "background-color: rgba(239, 68, 68, 0.35); color: #991b1b; font-weight: 600;"
                display_det = det_df.copy()
                display_det["Win"] = display_det["win"].map(lambda w: "✓ Win" if w else "✗ Loss")
                display_det = display_det.drop(columns=["win"])
                def _win_cell_style(v):
                    if v == "✓ Win":
                        return "background-color: rgba(34, 197, 94, 0.35); color: #166534; font-weight: 600;"
                    if v == "✗ Loss":
                        return "background-color: rgba(239, 68, 68, 0.35); color: #991b1b; font-weight: 600;"
                    return ""
                st.dataframe(
                    display_det.style.map(_win_cell_style, subset=["Win"]),
                    use_container_width=True,
                    hide_index=True,
                )
