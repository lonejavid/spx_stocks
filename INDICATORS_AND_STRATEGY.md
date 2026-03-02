# SPX Day Trading Dashboard — Indicators & BUY/SELL Strategy

## Data

- **Symbol:** SPX (S&P 500), with VIX and optional stocks (AAPL, MSFT, NVDA, AMZN, QQQ, SPY) for context.
- **Timeframe:** 5-minute bars (intraday).
- **Session:** US regular hours only (9:30 AM–4:00 PM ET); bars outside this are filtered out.
- **Preferred windows (for context):** 10:00–11:30 AM ET and 2:30–3:30 PM ET.

---

## Indicators Used (All on SPX 5m)

### 1. RSI (Relative Strength Index)

- **Period:** 14.
- **Formula:** Smoothed average gain / average loss over 14 bars; RSI = 100 − (100 / (1 + RS)).
- **Role in BUY:** Price must be **oversold** → **RSI < 40** (BUY_RSI_MAX = 40).

### 2. VWAP (Volume-Weighted Average Price)

- **Formula:** Cumulative (Typical Price × Volume) / Cumulative Volume, where Typical Price = (High + Low + Close) / 3.
- **Role in BUY:** Price must be **above** VWAP (buyers in control on the session).

### 3. MACD (Moving Average Convergence Divergence)

- **Parameters:** Fast = 12, Slow = 26, Signal = 9 (all exponential).
- **Components:**
  - MACD line = EMA(12) − EMA(26) of Close.
  - Signal line = EMA(9) of MACD line.
- **Role in BUY:** **MACD must cross above the signal line** on the current bar (bullish momentum).

### 4. VIX (Volatility Index)

- **Source:** Separate ticker (^VIX); we use the **latest close**.
- **Role in BUY:** **VIX < 20** (VIX_CALM). Avoids BUY when fear/volatility is high.

### 5. EMA 9 & EMA 21

- **Formula:** Exponential moving average of Close, span 9 and 21.
- **Used for:** Signal strength display (e.g. “EMA 9 crossed EMA 21”) and context; **not** required for the core BUY rule.

---

## BUY Signal — Exact Rule (All 4 Required)

A **BUY** is triggered only when **all** of the following are true on the **latest 5m bar**:

| # | Condition        | Code / threshold |
|---|------------------|------------------|
| 1 | **RSI < 40**     | Oversold (BUY_RSI_MAX = 40). |
| 2 | **SPX Close > VWAP** | Price above session VWAP. |
| 3 | **MACD cross up**   | MACD line crosses above signal line on current bar. |
| 4 | **VIX < 20**     | Low volatility (VIX_CALM = 20). |

- If **any** of these fails → signal stays **WAIT** (no BUY).
- No minimum “signal strength” score is required for BUY; the 8-condition score is for display/context only.

---

## SELL Signal — Exact Rule (All 4 Required)

A **SELL** is triggered only when **all** of the following are true on the **latest 5m bar**:

| # | Condition        | Code / threshold |
|---|------------------|------------------|
| 1 | **RSI > 60**     | (SELL_RSI_MIN = 60). |
| 2 | **SPX Close < VWAP** | Price below session VWAP. |
| 3 | **MACD cross down**  | MACD line crosses below signal line on current bar. |
| 4 | **VIX ≥ 20**     | (same VIX_CALM; 20 or above for SELL). |

---

## Signal Strength (0–8) — Display Only

The dashboard shows “Signal strength: X/8” from **8 conditions**. These do **not** change whether BUY/SELL fires; they only add context:

1. Price above VWAP (BUY) / below VWAP (SELL).
2. RSI &lt; 40 (BUY) / RSI &gt; 60 (SELL).
3. EMA 9 crossed above EMA 21 (BUY) / below (SELL).
4. MACD crossed signal line (up for BUY, down for SELL).
5. VIX &lt; 20 (BUY) / VIX ≥ 20 (SELL).
6. AAPL green today (BUY) / red (SELL).
7. MSFT green today (BUY) / red (SELL).
8. QQQ above its VWAP (BUY) / below (SELL).

---

## Summary Table

| Item        | BUY requirement | SELL requirement |
|------------|------------------|-------------------|
| RSI        | &lt; 40          | &gt; 60           |
| Price vs VWAP | Above VWAP   | Below VWAP        |
| MACD       | Cross **above** signal | Cross **below** signal |
| VIX        | &lt; 20          | ≥ 20              |

All four must be true for the respective signal. Slack alerts are sent automatically when BUY or SELL triggers (with a 30-minute cooldown per signal type).
