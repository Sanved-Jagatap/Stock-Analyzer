import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import streamlit as st
from sklearn.linear_model import LinearRegression
import os
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================
DB_PATH = "stock_data.db"
EXPORT_DIR = "data"
os.makedirs(EXPORT_DIR, exist_ok=True)

st.set_page_config(page_title="Stock Analysis App", layout="centered")

# ======================================================
# DATA FETCH WITH CACHE (CRITICAL FIX)
# ======================================================
@st.cache_data(ttl=3600)
def fetch_stock(symbol, start_date, end_date):
    """
    Cached stock fetch to avoid Yahoo Finance rate limits
    """
    return yf.download(
        symbol,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=True,
        threads=False
    )

# ======================================================
# INDICATOR FUNCTIONS
# ======================================================
def calculate_rsi(df, window=14):
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(df, window=20):
    sma = df["Close"].rolling(window).mean()
    std = df["Close"].rolling(window).std()
    df["Bollinger_Upper"] = sma + 2 * std
    df["Bollinger_Lower"] = sma - 2 * std
    return df


# ======================================================
# CORE ANALYSIS FUNCTION
# ======================================================
def analyze_stocks(stock_symbols, start_date, end_date):
    symbols = [s.strip().upper() for s in stock_symbols.split(",")]
    conn = sqlite3.connect(DB_PATH)

    valid_stocks = {}

    for symbol in symbols:
        st.subheader(f"📊 {symbol}")

        try:
            df = fetch_stock(symbol, start_date, end_date)
        except Exception as e:
            st.error(f"Download failed for {symbol}: {e}")
            continue

        # ------------------ DATA VALIDATION ------------------
        if df.empty or len(df) < 30:
            st.warning(f"No sufficient data available for {symbol}. Skipping.")
            continue

        df.reset_index(inplace=True)
        valid_stocks[symbol] = df.copy()

        st.dataframe(df.head(10))

        # ------------------ INDICATORS ------------------
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()
        df["RSI"] = calculate_rsi(df)
        df = calculate_bollinger_bands(df)
        df["Returns"] = df["Close"].pct_change()
        df.dropna(inplace=True)

        # ------------------ PRICE TREND ------------------
        plt.figure(figsize=(10, 4))
        plt.plot(df["Date"], df["Close"], label="Close Price")
        plt.title(f"{symbol} Price Trend")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        st.pyplot(plt)
        plt.close()

        # ------------------ MA + BOLLINGER ------------------
        plt.figure(figsize=(10, 4))
        plt.plot(df["Date"], df["Close"], label="Close")
        plt.plot(df["Date"], df["SMA_20"], label="SMA 20")
        plt.plot(df["Date"], df["EMA_20"], label="EMA 20")
        plt.plot(df["Date"], df["Bollinger_Upper"], "--", label="Upper Band")
        plt.plot(df["Date"], df["Bollinger_Lower"], "--", label="Lower Band")
        plt.legend()
        st.pyplot(plt)
        plt.close()

        # ------------------ RISK TO REWARD ------------------
        volatility = df["Returns"].std()
        avg_return = df["Returns"].mean()
        rr = round(abs(avg_return / volatility), 4) if volatility != 0 else np.nan
        st.write(f"📈 Risk-to-Reward Ratio: **{rr if not np.isnan(rr) else 'N/A'}**")

        # ------------------ LINEAR REGRESSION (SAFE) ------------------
        if len(df) >= 2:
            df["Day_Index"] = np.arange(len(df))
            X = df[["Day_Index"]]
            y = df["Close"]

            lr = LinearRegression()
            lr.fit(X, y)

            df["Trendline"] = lr.predict(X)
            slope = float(lr.coef_[0])
            trend_status = "Uptrend 📈" if slope > 0 else "Downtrend 📉"

            plt.figure(figsize=(10, 4))
            plt.plot(df["Date"], df["Close"], label="Actual")
            plt.plot(df["Date"], df["Trendline"], "--", label="Trendline")
            plt.legend()
            plt.title(f"{symbol} Trend ({trend_status})")
            st.pyplot(plt)
            plt.close()

            st.success(f"{symbol} is in **{trend_status}** (Slope: {slope:.4f})")
        else:
            st.warning("Not enough data for trend prediction.")

        # ------------------ EXPORT ------------------
        export_cols = [
            "Date", "Open", "High", "Low", "Close", "Volume",
            "SMA_20", "EMA_20", "RSI",
            "Bollinger_Upper", "Bollinger_Lower",
            "Returns", "Trendline"
        ]
        export_cols = [c for c in export_cols if c in df.columns]

        export_df = df[export_cols]
        export_df.to_csv(f"{EXPORT_DIR}/{symbol}_analysis.csv", index=False)

        st.download_button(
            f"⬇️ Download {symbol} CSV",
            export_df.to_csv(index=False),
            file_name=f"{symbol}_analysis.csv",
            mime="text/csv"
        )

        # ------------------ DATABASE ------------------
        df.to_sql(symbol.replace(".", "_"), conn, if_exists="replace", index=False)

    conn.close()

    # ------------------ COMPARISON PLOT ------------------
    if len(valid_stocks) >= 2:
        plt.figure(figsize=(10, 4))
        for sym, df in valid_stocks.items():
            plt.plot(df["Date"], df["Close"], label=sym)
        plt.legend()
        plt.title("Stock Comparison")
        st.pyplot(plt)
        plt.close()


# ======================================================
# STREAMLIT UI
# ======================================================
st.title("📈 Stock Analysis App")

stock_input = st.text_input(
    "Enter stock symbols (comma-separated)",
    placeholder="TCS.NS, RELIANCE.NS"
)

start_date = st.date_input("Start Date", pd.to_datetime("2024-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))

if st.button("Analyze Stocks"):
    if not stock_input.strip():
        st.warning("Please enter at least one stock symbol.")
    else:
        analyze_stocks(stock_input, start_date, end_date)
