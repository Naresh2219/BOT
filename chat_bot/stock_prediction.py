import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import requests

# Fetch USD to INR conversion rate
def get_usd_to_inr():
    try:
        url = "https://api.exchangerate-api.com/v4/latest/USD"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return data["rates"]["INR"]
    except Exception as e:
        st.error(f"Error fetching exchange rate: {e}")
    return 83.0  # Default rate if API fails

usd_to_inr = get_usd_to_inr()

# Fetch Stock Data
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    if stock_data.empty:
        st.warning(f"No data found for {ticker} in the given date range.")
        return None
    return stock_data[['Close']]

# Preprocess Data
def preprocess_data(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler

# Prepare Data for Model
def create_sequences(data, time_step=60):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

# Build Model
def build_rf_model():
    return RandomForestRegressor(n_estimators=100, random_state=42)

# Train and Predict
def train_and_predict(ticker, start_date, end_date, days_ahead):
    data = fetch_stock_data(ticker, start_date, end_date)
    if data is None:
        return None, None, None
    
    scaled_data, scaler = preprocess_data(data)
    if len(scaled_data) < 61:
        st.warning("Not enough data to create sequences.")
        return None, None, None
    
    X, Y = create_sequences(scaled_data)
    model = build_rf_model()
    model.fit(X, Y)
    
    # Predict Future Prices
    future_predictions = []
    last_60_days = scaled_data[-60:].reshape(1, -1)
    
    for _ in range(days_ahead):
        predicted_price_scaled = model.predict(last_60_days)
        predicted_price = scaler.inverse_transform([[predicted_price_scaled[0]]])[0][0]
        future_predictions.append(predicted_price)
        last_60_days = np.append(last_60_days[:, 1:], predicted_price_scaled).reshape(1, -1)
    
    return data, future_predictions, scaler

# Streamlit Dashboard
st.title("📈 Investment & Prediction Dashboard (INR ₹)")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, IDEA.NS, AAPL):", "RELIANCE.NS")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
days_ahead = st.number_input("Days Ahead to Predict", min_value=1, max_value=30, value=7)
investment_amount = st.number_input("Enter Investment Amount (INR ₹)", min_value=100, value=1000, step=100)

if st.button("Fetch & Predict"):
    data, future_predictions, scaler = train_and_predict(ticker, start_date, end_date, days_ahead)
    
    if data is not None:
        st.subheader(f"📊 Stock Data for {ticker}")
        start_price = data.iloc[0]['Close'] * usd_to_inr
        latest_price = data.iloc[-1]['Close'] * usd_to_inr
        future_prices_inr = [price * usd_to_inr for price in future_predictions]
        
        col1, col2 = st.columns(2)
        col1.metric(label="Start Price (5 years ago)", value=f"₹{start_price.iloc[0]:.2f}")
        col2.metric(label="Latest Price (Today)", value=f"₹{latest_price.iloc[0]:.2f}")
        
        predicted_price = future_prices_inr[-1]
        future_value = (investment_amount / latest_price) * predicted_price
        st.metric(label="Predicted Investment Value", value=f"₹{future_value.iloc[0]:.2f}")
        
        future_dates = pd.date_range(end_date, periods=days_ahead + 1)[1:]
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price (INR ₹)": future_prices_inr})
        st.write("### 📅 Future Predictions")
        st.dataframe(future_df)
        
        st.subheader("📉 Stock Price Trend")
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'] * usd_to_inr, label="Actual Prices", color="blue")
        plt.plot(future_dates, future_prices_inr, label="Predicted Prices", linestyle="dashed", color="red")
        plt.xlabel("Date")
        plt.ylabel("Stock Price (INR ₹)")
        plt.title(f"Stock Price Prediction for {ticker}")
        plt.legend()
        st.pyplot(plt)

# Cryptocurrency Prices
st.subheader("💰 Live Cryptocurrency Prices (INR ₹)")
crypto_tickers = ['BTC-USD', 'ETH-USD', 'DOGE-USD']
crypto_prices = {crypto: yf.Ticker(crypto).history(period='1d')['Close'][-1] * usd_to_inr for crypto in crypto_tickers}
st.write(pd.DataFrame(crypto_prices.items(), columns=["Cryptocurrency", "Price (INR ₹)"]))

# Upcoming IPO Listings
st.subheader("🚀 Upcoming IPOs in India")
ipo_data = pd.DataFrame({
    "Company": ["XYZ Ltd.", "ABC Tech", "NextGen Pharma"],
    "Issue Price (INR ₹)": [500, 320, 740],
    "Open Date": ["2025-02-15", "2025-03-01", "2025-03-10"],
    "Close Date": ["2025-02-20", "2025-03-05", "2025-03-15"]
})
st.dataframe(ipo_data)
