import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

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
        
        # Update input sequence for next prediction
        last_60_days = np.append(last_60_days[:, 1:], predicted_price_scaled).reshape(1, -1)
    
    return data, future_predictions, scaler

# Streamlit Dashboard
st.title("📈 Investment & Stock Prediction Dashboard")

# User Inputs
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, IDEA.NS):", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("today"))
days_ahead = st.number_input("Days Ahead to Predict", min_value=1, max_value=30, value=7)
investment_amount = st.number_input("Enter Investment Amount (₹):", min_value=1, value=100)

if st.button("Fetch & Predict"):
    data, future_predictions, scaler = train_and_predict(ticker, start_date, end_date, days_ahead)

    if data is not None:
        st.subheader(f"📊 Stock Data for {ticker}")
        start_price = data.iloc[0]['Close']
        latest_price = data.iloc[-1]['Close']
        
        # Calculate investment potential
        stocks_can_buy = investment_amount / latest_price
        future_value = stocks_can_buy * future_predictions[-1]
        
        # Display Key Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Start Price (5 years ago)", value=f"₹{start_price.iloc[0]:.2f}")
        col2.metric(label="Latest Price (Today)", value=f"₹{latest_price.iloc[0]:.2f}")
        col3.metric(label=f"Predicted Price ({days_ahead} days later)", value=f"₹{future_predictions[-1]:.2f}")
        
        st.success(f"📢 If you invest ₹{investment_amount}, you can buy {stocks_can_buy.iloc[0]:.2f} shares.\nYour estimated investment value in {days_ahead} days: ₹{future_value.iloc[0]:.2f}")
        
        # Future Predictions
        future_dates = pd.date_range(end_date, periods=days_ahead + 1)[1:]
        future_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_predictions})
        st.write("### 📅 Future Predictions")
        st.dataframe(future_df)
        
        # Plot Data
        st.subheader("📉 Stock Price Trend")
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'], label="Actual Prices", color="blue")
        plt.plot(future_dates, future_predictions, label="Predicted Prices", linestyle="dashed", color="red")
        plt.xlabel("Date")
        plt.ylabel("Stock Price (₹)")
        plt.title(f"Stock Price Prediction for {ticker}")
        plt.legend()
        st.pyplot(plt)
        
        # Other Investment Options
        st.subheader("💰 Other Investment Opportunities")
        st.write("**Upcoming IPOs:** Reliance Retail, Ola Electric, Swiggy IPOs launching soon!")
        st.write("**Cryptocurrencies:** Bitcoin (BTC), Ethereum (ETH), Solana (SOL)")
        st.write("**Digital Gold:** Safe investment with steady returns!")
