# S&P 500 Dashboard — What This Is and How It Works

## What You Have (In One Sentence)

A dark-themed web dashboard that loads the last year of S&P 500 (^GSPC) prices from the internet, draws a price chart, computes RSI and MACD, and shows BUY/SELL signals when RSI is below 30 or above 70.

---

## Where the Data Comes From

- **Source:** Yahoo Finance, via the **yfinance** Python library.
- **What we ask for:** 1 year of daily data for the ticker **^GSPC** (S&P 500 index).
- **What we get:** For each day: Open, High, Low, **Close**, Volume. The app mainly uses **Close** for all calculations and charts.
- **Where it’s fetched:** In `app.py`, the function `fetch_sp500_data()` calls `yf.download(TICKER, ...)`. Data is cached for 1 hour so we don’t hit Yahoo on every refresh.

So: **data = last 1 year of S&P 500 daily closing prices from Yahoo Finance.**

---

## What the Dashboard Shows

1. **Top row (KPIs)**  
   - **Close** — Latest closing price (e.g. $6,946.13).  
   - **RSI (14)** — Current 14‑period RSI (e.g. 54.5).  
   - **Signal** — Either **BUY**, **SELL**, or **—** (no signal).  
   - **As of** — Date of the latest data (e.g. 2026-02-25).

2. **Price chart**  
   - One line: S&P 500 **closing price** for each day over the last year (e.g. Mar 2025 → Jan 2026).

3. **RSI chart**  
   - One line: **RSI(14)** for each day (0–100).  
   - Horizontal lines at **30** (oversold, green) and **70** (overbought, red).  
   - Green when RSI &lt; 30, red when RSI &gt; 70, teal otherwise.

4. **MACD chart**  
   - **Green line:** MACD (12, 26).  
   - **Orange line:** Signal (9).  
   - **Bars:** Histogram = MACD − Signal (green if positive, red if negative).

5. **Trading signals table**  
   - Rows where RSI gave a **BUY** (RSI &lt; 30) or **SELL** (RSI &gt; 70): **Date**, **Close**, **RSI**, **Signal**.  
   - So you see *when* and *at what price/RSI* those signals happened.

---

## How the Calculations Are Done

### RSI (Relative Strength Index), 14 periods

- **Input:** Daily **Close** prices.
- **Idea:** Compare average “up” moves vs “down” moves over the last 14 days.
- **Steps (simplified):**  
  1. For each day, compute price change from the previous day.  
  2. Separate gains (positive changes) and losses (negative, as positive numbers).  
  3. Smooth them with an exponential average (like a 14‑period EMA).  
  4. RSI = 100 − (100 / (1 + (avg gain / avg loss))).
- **Result:** A number between 0 and 100.  
  - **&lt; 30** → oversold (price dropped a lot) → **BUY** signal.  
  - **&gt; 70** → overbought (price rose a lot) → **SELL** signal.

In the code: if **pandas-ta** is installed we use `ta.rsi(close, length=14)`; otherwise we use the built-in `_rsi_pandas()` that does the same math with pandas.

---

### MACD (12, 26, 9)

- **Input:** Daily **Close** prices.
- **Three parts:**  
  1. **MACD line** = 12‑day EMA of Close − 26‑day EMA of Close.  
  2. **Signal line** = 9‑day EMA of the MACD line.  
  3. **Histogram** = MACD line − Signal line (the bars: green when positive, red when negative).
- **Use:** Shows short‑term vs medium‑term trend and momentum; crossovers and histogram sign changes are often used for timing (we don’t auto-signal from MACD in this app; we only use RSI for BUY/SELL).

In the code: same as RSI — **pandas-ta** if available, otherwise built-in `_macd_pandas()` with the same formulas (EMAs and differences).

---

### BUY / SELL Signals (What “Signal” Means)

- **Rule in this app:**  
  - **BUY** when **RSI &lt; 30** (oversold).  
  - **SELL** when **RSI &gt; 70** (overbought).  
  - Otherwise **—** (no signal).
- **Where it’s used:**  
  - The **Signal** in the top row = current day’s signal.  
  - The **Trading signals** table = every day in the last year that had RSI &lt; 30 or RSI &gt; 70, with date, close, RSI, and BUY/SELL.

So: **data from Yahoo → Close prices → we compute RSI and MACD → we show charts and a table of RSI-based BUY/SELL dates.**

---

## File Roles (Very Short)

- **`app.py`** — The whole app: fetch data, compute RSI/MACD, draw Plotly charts, show Streamlit layout and dark theme.
- **`.streamlit/config.toml`** — Dark theme (black background, teal accent).
- **`requirements.txt`** — List of Python packages: streamlit, yfinance, plotly, pandas (no pandas-ta required; we have a fallback).

That’s exactly what was built and how the data and calculations work.
