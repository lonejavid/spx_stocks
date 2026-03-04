"""
Download 5-minute intraday data for the last 2 months (60d) for 10 stocks.
Saves each as a CSV in spx_11/ with columns: Datetime, Close, High, Low, Open, Volume, SPX_Weight_Pct.
Uses only yfinance and pandas.
"""
import os
import pandas as pd
import yfinance as yf

stocks = {
    "NVDA":  7.10,
    "AAPL":  6.24,
    "MSFT":  4.74,
    "AMZN":  3.58,
    "GOOGL": 3.07,
    "INTC":  0.36,
    "T":     0.31,
    "VZ":    0.34,
    "PFE":   0.25,
    "QCOM":  0.24
}

# Folder in same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "spx_11")

# Required column order (before adding weight)
COL_ORDER = ["Datetime", "Close", "High", "Low", "Open", "Volume"]


def download_one(ticker: str, weight: float) -> tuple:
    """
    Download 5m data for one ticker. Returns (success: bool, df or None, error_msg or None).
    """
    try:
        df = yf.download(
            ticker,
            period="60d",
            interval="5m",
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return False, None, "No data returned"
        # Flatten MultiIndex columns if present (e.g. from multi-ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Ensure we have required columns (yfinance may use different casing)
        df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
        required = ["Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in df.columns:
                return False, None, f"Missing column: {col}"
        # Drop Adj Close if present (we use auto_adjust so Close is adjusted)
        df = df[[c for c in df.columns if c in required]].copy()
        # Reset index so Datetime becomes a column
        df = df.reset_index()
        # Standardize datetime column name
        date_col = df.columns[0]
        df = df.rename(columns={date_col: "Datetime"})
        # Order: Datetime, Close, High, Low, Open, Volume
        df = df[COL_ORDER].copy()
        df["SPX_Weight_Pct"] = weight
        return True, df, None
    except Exception as e:
        return False, None, str(e)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    success_count = 0
    failed = []

    for ticker, weight in stocks.items():
        ok, df, err = download_one(ticker, weight)
        if not ok:
            print(f"{ticker}: Error — {err}")
            failed.append(ticker)
            continue
        out_path = os.path.join(OUT_DIR, f"{ticker}_5min_2mo.csv")
        df.to_csv(out_path, index=False)
        n = len(df)
        dt_min = df["Datetime"].min()
        dt_max = df["Datetime"].max()
        if hasattr(dt_min, "strftime"):
            range_str = f"{dt_min.strftime('%Y-%m-%d')} to {dt_max.strftime('%Y-%m-%d')}"
        else:
            range_str = f"{dt_min} to {dt_max}"
        print(f"{ticker}: {n} rows saved to spx_11/{ticker}_5min_2mo.csv | Date range: {range_str}")
        success_count += 1

    print()
    print(f"✅ Done. {success_count}/10 stocks saved successfully in spx_11/")
    if failed:
        print(f"❌ Failed: {failed}")
    else:
        print("❌ Failed: []")


if __name__ == "__main__":
    main()
