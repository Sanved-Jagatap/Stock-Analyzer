import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
import streamlit as st
from sklearn.linear_model import LinearRegression

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate Bollinger Bands
def calculate_bollinger_bands(data, window=20):
    rolling_mean = data['Close'].rolling(window=window).mean()
    rolling_std = data['Close'].rolling(window=window).std()
    data['Bollinger_Upper'] = rolling_mean + (rolling_std * 2)
    data['Bollinger_Lower'] = rolling_mean - (rolling_std * 2)
    return data

# Function to fetch and analyze stock data
def analyze_stocks(stock_symbols, start_date, end_date):
    stocks = [symbol.strip() for symbol in stock_symbols.split(',')]
    hds = {}

    # Fetch historical data
    for stock in stocks:
        hds[stock] = yf.download(stock, start=start_date, end=end_date)

    # Connect to SQLite database
    conn = sqlite3.connect('stock_data.db')

    for stock, hd in hds.items():
        st.subheader(f"Stock Data for {stock}")
        #st.dataframe(hd)
        # Flatten column names if MultiIndex (like TCS.NS, TCS.NS...)
        if isinstance(hd.columns, pd.MultiIndex):
           hd.columns = hd.columns.get_level_values(0)

# Optionally reset index for export
        hd.reset_index(inplace=True)

    # Show full data in Streamlit
        st.dataframe(hd)

        # Download CSV option
        #csv = hd.to_csv(index=False).encode('utf-8')
        #st.download_button(
        #    label=f"Download {stock} Data as CSV",
        #    data=csv,
        #    file_name=f'{stock}_data.csv',
        #    mime='text/csv'
        #)
        # Add trendline to DataFrame for export
        

        

        
       



        # Trend Analysis
        plt.figure(figsize=(12, 6))
        plt.plot(hd['Date'], hd['Close'], label=f'{stock} Close Price')
        plt.title(f'{stock} Stock Trend')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        #plt.xticks(rotation=45)
        st.pyplot(plt)

        # Moving Averages, RSI, Bollinger Bands
        hd['SMA_20'] = hd['Close'].rolling(window=20).mean()
        hd['EMA_20'] = hd['Close'].ewm(span=20, adjust=False).mean()
        hd['RSI'] = calculate_rsi(hd)
        hd = calculate_bollinger_bands(hd)
        hd.dropna(inplace=True)


        plt.figure(figsize=(12, 6))
        #plt.plot(hd.index, hd['Close'], label=f'{stock} Close Price')
        #plt.plot(hd.index, hd['SMA_20'], label=f'{stock} SMA 20')
        #plt.plot(hd.index, hd['EMA_20'], label=f'{stock} EMA 20')
        #plt.plot(hd.index, hd['Bollinger_Upper'], label='Bollinger Upper Band', linestyle='--', color='orange')
        #plt.plot(hd.index, hd['Bollinger_Lower'], label='Bollinger Lower Band', linestyle='--', color='orange')
        plt.plot(hd['Date'], hd['Close'], label=f'{stock} Close Price')
        plt.plot(hd['Date'], hd['SMA_20'], label=f'{stock} SMA 20')
        plt.plot(hd['Date'], hd['EMA_20'], label=f'{stock} EMA 20')
        plt.plot(hd['Date'], hd['Bollinger_Upper'], label='Bollinger Upper Band', linestyle='--', color='orange')
        plt.plot(hd['Date'], hd['Bollinger_Lower'], label='Bollinger Lower Band', linestyle='--', color='orange')
        plt.title('Moving Averages and Bollinger Bands')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)
        #plt.xticks(rotation=45)


        # Risk-to-Reward Ratio
        hd['Returns'] = hd['Close'].pct_change()
        volatility = hd['Returns'].std()
        avg_return = hd['Returns'].mean()
        rrw = abs(avg_return / volatility) if volatility != 0 else 0
        st.write(f'{stock} Risk-to-Reward Ratio: {rrw:.4f}')

        # Linear Regression Trendline
        #hd['Days'] = np.arange(len(hd))
        #X = hd[['Days']]
        #y = hd['Close']
        #lr = LinearRegression()
        #lr.fit(X, y)
        #trendline = lr.predict(X)
        #trend_slope = float(lr.coef_[0])  # Fixing TypeError issue

        #plt.figure(figsize=(12, 6))
        #plt.plot(hd['Days'], hd['Close'], label=f'{stock} Actual Price', color='blue', alpha=0.6)
        #plt.plot(hd['Days'], trendline, label=f'{stock} Trendline', color='red', linestyle='--')

        # Title for uptrend or downtrend
        #trend_status = "Uptrend 📈" if trend_slope > 0 else "Downtrend 📉"
        #plt.title(f'{stock}: {trend_status} (Slope: {trend_slope:.4f})')
        #plt.xlabel('Days')
        #plt.ylabel('Price')
        #plt.legend()
        #st.pyplot(plt)
        # Use numerical index for regression, but keep Date for plotting
        hd['Days_Index'] = np.arange(len(hd))
        X = hd[['Days_Index']]
        y = hd['Close']
        lr = LinearRegression()
        lr.fit(X, y)
        trendline = lr.predict(X)
        trend_slope = float(lr.coef_[0])
        hd['Trendline'] = trendline
        # Select key columns to include in CSV
        # Define desired columns
        desired_cols = [
            'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
            'SMA_20', 'EMA_20', 'RSI', 'Bollinger_Upper', 'Bollinger_Lower',
            'Returns', 'Days_Index', 'Trendline'
        ]

        # Keep only columns that actually exist in hd
        existing_cols = [col for col in desired_cols if col in hd.columns]
        csv_export = hd[existing_cols]


        # Generate and offer download
        csv = csv_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label=f"Download {stock} Full Data as CSV",
            data=csv,
            file_name=f'{stock}_analysis.csv',
            mime='text/csv'
        )

        plt.figure(figsize=(12, 6))
        plt.plot(hd['Date'], hd['Close'], label=f'{stock} Actual Price', color='blue', alpha=0.6)
        plt.plot(hd['Date'], trendline, label=f'{stock} Trendline', color='red', linestyle='--')
        #plt.xticks(rotation=45)

        # Title for uptrend or downtrend
        trend_status = "Uptrend 📈" if trend_slope > 0 else "Downtrend 📉"
        plt.title(f'{stock}: {trend_status} (Slope: {trend_slope:.4f})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        st.pyplot(plt)


        # Print uptrend or downtrend
        st.write(f'{stock} is in a **{trend_status}** (Slope: {trend_slope:.4f}).')

        # Save stock data to SQLite database
        hd.to_sql(stock, conn, if_exists='replace')

    # Close database connection
    conn.close()

    # Stock Comparison Plot
    plt.figure(figsize=(12, 6))
    for stock, hd in hds.items():
        # Ensure 'Date' is datetime and sorted
        if 'Date' in hd.columns:
            plt.plot(hd['Date'], hd['Close'], label=f'{stock} Close Price')

    plt.title('Stock Comparison')
    plt.xlabel('Date')
    plt.ylabel('Price')
    #plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)
    

# Streamlit UI
st.title("Stock Analysis App")


# User input for stock symbols
stock_input = st.text_input("Enter stock symbols (comma-separated, e.g., INFY.NS, TCS.NS):")
start_date = st.date_input("Select start date", value=pd.to_datetime('2023-01-01'))
end_date = st.date_input("Select end date", value=pd.to_datetime('2024-10-22'))

if st.button("Analyze Stocks"):
    if stock_input:
        analyze_stocks(stock_input, start_date, end_date)
    else:
        st.warning("Please enter at least one stock symbol.")