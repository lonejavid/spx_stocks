"""
S&P 500 (^GSPC) Dashboard with RSI, MACD, and trading signals.
"""
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False


def _rsi_pandas(close: pd.Series, length: int = 14) -> pd.Series:
    """RSI using plain pandas (fallback when pandas-ta not installed)."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return (100 - (100 / (1 + rs))).fillna(50)


def _macd_pandas(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD using plain pandas (fallback when pandas-ta not installed)."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({
        f"MACD_{fast}_{slow}_{signal}": macd_line,
        f"MACDs_{fast}_{slow}_{signal}": signal_line,
        f"MACDh_{fast}_{slow}_{signal}": histogram,
    })

st.set_page_config(page_title="S&P 500 Dashboard", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# Dark dashboard theme — custom CSS
DASHBOARD_CSS = """
<style>
/* Main container */
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }
/* Header strip */
div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stMarkdown"] > p) { }
/* Section titles */
h1, h2, h3 { color: #f0f2f6 !important; font-weight: 600 !important; }
h1 { font-size: 1.85rem !important; border-bottom: 2px solid #00d4aa; padding-bottom: 0.5rem; margin-bottom: 0.5rem !important; }
h2 { font-size: 1.15rem !important; color: #b0b4bc !important; margin-top: 1.5rem !important; }
/* Metric cards */
[data-testid="stMetricValue"] { font-size: 1.5rem !important; font-weight: 700 !important; color: #00d4aa !important; }
[data-testid="stMetricLabel"] { color: #8892a0 !important; font-size: 0.8rem !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
/* Cards around metrics */
div[data-testid="metric-container"] { background: linear-gradient(135deg, #1a1d24 0%, #252830 100%); border: 1px solid #2d3139; border-radius: 10px; padding: 1rem 1.25rem !important; box-shadow: 0 4px 14px rgba(0,0,0,0.25); }
/* Dataframe */
div[data-testid="stDataFrame"] { border: 1px solid #2d3139; border-radius: 8px; overflow: hidden; }
/* Dividers */
hr { border-color: #2d3139 !important; margin: 1.5rem 0 !important; }
/* Caption / subtitle */
p { color: #8892a0 !important; }
/* Spinner */
.stSpinner > div { border-top-color: #00d4aa !important; }
/* Expander / info boxes */
.stAlert { border-radius: 8px; border: 1px solid #2d3139; }
/* Remove Streamlit branding for cleaner look */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { background: linear-gradient(90deg, #0e1117 0%, #1a1d24 100%); padding: 0.5rem 0 !important; }
</style>
"""

# Plotly dark layout (no title/xaxis/yaxis — pass those per chart to avoid duplicate keyword errors)
PLOTLY_DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(26,29,36,0.8)",
    font=dict(color="#e0e2e6", size=12),
    margin=dict(t=50, b=50, l=50, r=30),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b0b4bc")),
)
AXIS_STYLE = dict(showgrid=True, gridcolor="rgba(80,85,95,0.5)", zeroline=False)

TICKER = "^GSPC"
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70


@st.cache_data(ttl=3600)
def fetch_sp500_data():
    """Fetch S&P 500 data for the last 1 year."""
    end = datetime.now()
    start = end - timedelta(days=365)
    df = yf.download(TICKER, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if isinstance(df.index, pd.MultiIndex):
        df.index = df.index.get_level_values(0)
    df = df.dropna(how="all")
    return df


def add_indicators(df):
    """Add RSI and MACD (pandas-ta if available, else pandas fallback)."""
    df = df.copy()
    close = df["Close"]
    if HAS_PANDAS_TA:
        df["RSI"] = ta.rsi(close, length=14)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None and not macd.empty:
            for col in macd.columns:
                df[col] = macd[col]
    else:
        df["RSI"] = _rsi_pandas(close, length=14)
        macd = _macd_pandas(close, fast=12, slow=26, signal=9)
        for col in macd.columns:
            df[col] = macd[col]
    return df


def main():
    st.markdown(DASHBOARD_CSS, unsafe_allow_html=True)

    st.title("S&P 500 (^GSPC) Dashboard")
    st.caption("1-year price · RSI · MACD · BUY when RSI < 30 · SELL when RSI > 70")

    with st.spinner("Loading S&P 500 data..."):
        df = fetch_sp500_data()

    if df.empty or len(df) < 30:
        st.error("Could not load enough data. Please try again later.")
        return

    df = add_indicators(df)
    df = df.dropna(subset=["RSI"])

    # Detect MACD column names (pandas-ta naming can vary)
    macd_cols = [c for c in df.columns if "MACD" in c and "MACDh" not in c and "MACDs" not in c]
    signal_cols = [c for c in df.columns if "MACDs" in c]
    hist_cols = [c for c in df.columns if "MACDh" in c]
    macd_line = macd_cols[0] if macd_cols else None
    macd_signal = signal_cols[0] if signal_cols else None
    macd_hist = hist_cols[0] if hist_cols else None

    last = df.iloc[-1]
    rsi_val = last["RSI"]
    current_signal = "BUY" if rsi_val < RSI_OVERSOLD else "SELL" if rsi_val > RSI_OVERBOUGHT else "—"
    last_date = df.index[-1].strftime("%Y-%m-%d") if hasattr(df.index[-1], "strftime") else str(df.index[-1])

    # --- KPI row at top ---
    st.markdown("---")
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Close", f"${last['Close']:,.2f}")
    with k2:
        st.metric("RSI (14)", f"{rsi_val:.1f}")
    with k3:
        st.metric("Signal", current_signal)
    with k4:
        st.metric("As of", last_date)
    st.markdown("---")

    # --- Price chart ---
    st.subheader("Price")
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            name="Close",
            line=dict(color="#00d4aa", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(0,212,170,0.12)",
        )
    )
    fig_price.update_layout(**PLOTLY_DARK_LAYOUT, title="S&P 500 — 1 Year", height=380, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
    fig_price.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig_price, use_container_width=True)

    # --- RSI ---
    st.subheader("RSI")
    rsi_colors = [
        "#2ecc71" if v < RSI_OVERSOLD else "#e74c3c" if v > RSI_OVERBOUGHT else "#00d4aa"
        for v in df["RSI"]
    ]
    fig_rsi = go.Figure()
    fig_rsi.add_trace(
        go.Scatter(
            x=df.index,
            y=df["RSI"],
            name="RSI (14)",
            line=dict(color="#00d4aa", width=1.8),
            marker=dict(color=rsi_colors, size=5),
        )
    )
    fig_rsi.add_hline(y=RSI_OVERSOLD, line_dash="dash", line_color="#2ecc71", annotation_text="Oversold 30")
    fig_rsi.add_hline(y=RSI_OVERBOUGHT, line_dash="dash", line_color="#e74c3c", annotation_text="Overbought 70")
    fig_rsi.update_layout(**PLOTLY_DARK_LAYOUT, title="RSI (14)", height=320, xaxis=AXIS_STYLE, yaxis=dict(**AXIS_STYLE, range=[0, 100]))
    st.plotly_chart(fig_rsi, use_container_width=True)

    # --- MACD ---
    st.subheader("MACD")
    if macd_line and macd_signal:
        fig_macd = go.Figure()
        fig_macd.add_trace(
            go.Scatter(x=df.index, y=df[macd_line], name="MACD", line=dict(color="#00d4aa", width=2))
        )
        fig_macd.add_trace(
            go.Scatter(x=df.index, y=df[macd_signal], name="Signal", line=dict(color="#f39c12", width=1.5))
        )
        if macd_hist and macd_hist in df.columns:
            colors_hist = ["#2ecc71" if v >= 0 else "#e74c3c" for v in df[macd_hist]]
            fig_macd.add_trace(
                go.Bar(x=df.index, y=df[macd_hist], name="Histogram", marker_color=colors_hist, opacity=0.7)
            )
        fig_macd.update_layout(**PLOTLY_DARK_LAYOUT, title="MACD (12, 26, 9)", height=320, xaxis=AXIS_STYLE, yaxis=AXIS_STYLE)
        st.plotly_chart(fig_macd, use_container_width=True)

    # --- Signals ---
    st.subheader("Trading signals")
    df["Signal"] = "—"
    df.loc[df["RSI"] < RSI_OVERSOLD, "Signal"] = "BUY"
    df.loc[df["RSI"] > RSI_OVERBOUGHT, "Signal"] = "SELL"

    signals = df[df["Signal"].isin(["BUY", "SELL"])].copy()
    if signals.empty:
        st.info("No BUY or SELL signals in the last year (no RSI < 30 or RSI > 70).")
    else:
        signals_display = signals[["Close", "RSI", "Signal"]].sort_index(ascending=False)
        signals_display.index = signals_display.index.strftime("%Y-%m-%d")
        st.dataframe(
            signals_display.head(50),
            use_container_width=True,
            column_config={
                "Close": st.column_config.NumberColumn("Close", format="$%.2f"),
                "RSI": st.column_config.NumberColumn("RSI", format="%.1f"),
                "Signal": st.column_config.TextColumn("Signal"),
            },
        )

if __name__ == "__main__":
    main()
