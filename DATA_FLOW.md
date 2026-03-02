# Where Data Is Received and How It’s Stored for Calculations

## 1. Where data is received (exact locations)

### Yahoo Finance 5‑minute data (main source)

| Location | What happens |
|----------|----------------|
| **`fetch_intraday_5m()`** (lines ~318–330) | Calls **`yf.download(ticker, start=..., end=..., interval="5m", ...)`** or **`yf.download(ticker, period="1d", interval="5m", ...)`**. This is where **raw OHLCV data is received** from the internet (Yahoo Finance). |
| **`fetch_all_intraday()`** (lines ~352–375) | Calls `fetch_intraday_5m()` for **each ticker**: SPX (^GSPC), VIX (^VIX), then AAPL, MSFT, NVDA, AMZN, QQQ, SPY. So **all 5m data is received inside these two functions** via `yf.download()`. |

So: **data is “received” only in `fetch_intraday_5m()` → `yf.download()`**. Nothing is received in `add_indicators()` or `get_signal()`; they only read data that was already fetched.

---

## 2. How data is “saved” for calculations (no disk, only memory)

Nothing is written to a file or database. Everything stays **in memory** (and in Streamlit’s cache).

### Step-by-step flow

1. **`run_dashboard()`** (around line 717)  
   - Calls **`data = fetch_all_intraday(show_date_str)`** or **`fetch_all_intraday(None)`**.

2. **`fetch_all_intraday()`** (lines 352–375)  
   - Builds a dict **`out`** in memory:
     - **`out["spx"]`** = DataFrame of SPX 5m bars (after `_filter_session`).
     - **`out["vix"]`** = DataFrame of VIX 5m bars.
     - **`out["vix_value"]`** = last VIX close (float).
     - **`out["AAPL"]`**, **`out["MSFT"]`, …** = DataFrames for each stock.
   - **Returns** this dict. So “saved for calculations” here = **this dict in memory**.

3. **Caching (in-memory only)**  
   - **`@st.cache_data(ttl=120)`** on `fetch_intraday_5m` and `fetch_all_intraday` (lines 317, 350) means Streamlit **caches the return value in memory** for **120 seconds**. So for 2 minutes, repeated calls get the same `out` without calling `yf.download()` again. Still **no disk**; cache is in process memory.

4. **In `run_dashboard()` after the fetch** (lines 727–733):
   - **`spx = data["spx"]`** → `spx` is the SPX DataFrame (in memory).
   - **`vix_df = data["vix"]`**, **`vix_value = data.get("vix_value")`**.
   - **`spx = add_indicators(spx)`** → RSI, MACD, VWAP, EMA9, EMA21 are **added as new columns** on the same DataFrame (still in memory). No separate “saved” copy; the same `spx` is updated.

5. **Where calculations use this “saved” data**
   - **`get_signal(spx, vix_value, data)`** (line 734)  
     Uses:
     - **`spx`** (with Close, RSI, EMA9, EMA21, and MACD columns from `add_indicators`).
     - **`vix_value`** (single number from `data["vix_value"]`).
     - **`data`** (for AAPL/MSFT/QQQ DataFrames for confirmations).
   - **RSI/MACD/VWAP/EMA**  
     Calculated inside **`add_indicators(spx)`** (lines 379–400) from **`spx["Close"]`** (and High/Low/Volume for VWAP). The result is stored only in the **`spx`** DataFrame columns (e.g. `spx["RSI"]`), not in a separate table or file.

---

## 3. Summary diagram

```
yf.download()  (inside fetch_intraday_5m)
       ↓
   Raw OHLCV DataFrames per ticker
       ↓
fetch_all_intraday()  builds  out = { "spx": df_spx, "vix": df_vix, "vix_value": float, "AAPL": df_aapl, ... }
       ↓
   Cached in memory (ttl=120 sec) — NOT saved to disk
       ↓
run_dashboard()  gets  data = fetch_all_intraday(...)
       ↓
spx = data["spx"]   →  spx = add_indicators(spx)   →  spx now has Close, RSI, MACD, VWAP, EMA9, EMA21 (all in memory)
       ↓
get_signal(spx, vix_value, data)  uses spx and data for BUY/SELL logic
```

---

## 4. Exact line references

| What | Where in code |
|------|----------------|
| Data **received** from the internet | `yf.download(...)` in **`fetch_intraday_5m()`** (lines 322, 324). |
| Data **stored** for the session | Dict **`out`** in **`fetch_all_intraday()`** (lines 354–375); keys: `"spx"`, `"vix"`, `"vix_value"`, `"AAPL"`, etc. |
| Where that stored data is **used** | **`run_dashboard()`**: `data = fetch_all_intraday(...)` then `spx = data["spx"]`, `vix_value = data.get("vix_value")`, and `data` passed to `get_signal(spx, vix_value, data)`. |
| Where **calculations** (RSI, MACD, etc.) are done and “saved” | **`add_indicators(spx)`** (lines 379–400): adds columns to **`spx`** (RSI, MACD, VWAP, EMA9, EMA21). No separate storage; only the **`spx`** DataFrame in memory. |

So: **data is received only in `fetch_intraday_5m()` via `yf.download()`**, and **“saved” for calculations means: held in the `data` dict and in the `spx` DataFrame (and Streamlit cache) in memory only**.
