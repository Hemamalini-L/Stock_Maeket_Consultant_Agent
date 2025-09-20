import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# -------------------------------
# App Configuration
# -------------------------------
st.set_page_config(page_title="Stock Market Consultant Agent", layout="wide")
st.title("📊 Stock Market Consultant Agent")
st.write("Beginner-friendly guidance for your investments — live data, portfolio analysis, and clear advice.")

# -------------------------------
# Helper Functions
# -------------------------------
def fetch_stock_data(ticker, period="6mo"):
    """Fetch stock data safely"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {e}")
        return None

def analyze_portfolio(portfolio):
    """Provide beginner-friendly advice based on portfolio"""
    advice = []
    sectors = set()
    
    for stock in portfolio:
        data = fetch_stock_data(stock)
        if data is not None and len(data) >= 2:
            latest_price = data['Close'].dropna().iloc[-1]
            prev_price = data['Close'].dropna().iloc[-2]
            pct_change = ((latest_price - prev_price) / prev_price) * 100

            if pct_change > 2:
                advice.append(f"📈 {stock}: Price rising fast — consider booking partial profit.")
            elif pct_change < -2:
                advice.append(f"📉 {stock}: Price falling — review your risk tolerance.")
            else:
                advice.append(f"🤝 {stock}: Stable — holding is reasonable.")

            # Dummy sector logic (in real system fetch via API/DB)
            if "INFY" in stock or "TCS" in stock:
                sectors.add("IT")
            elif "HDFC" in stock or "ICICI" in stock:
                sectors.add("Banking")
        else:
            advice.append(f"⚠️ {stock}: Data unavailable.")

    if len(sectors) < 2:
        advice.append("⚠️ Your portfolio lacks diversification. Try adding stocks from different sectors.")

    return advice

def plot_stock(ticker):
    """Plot stock price chart"""
    data = fetch_stock_data(ticker)
    if data is not None and not data['Close'].isna().all():
        st.subheader(f"📉 Price Trend for {ticker}")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(data.index, data['Close'], label='Closing Price')
        ax.set_xlabel("Date")
        ax.set_ylabel("Price (₹)")
        ax.legend()
        st.pyplot(fig)

        # ✅ Safe metric display
        if len(data['Close'].dropna()) >= 2:
            latest_price = data['Close'].dropna().iloc[-1]
            prev_price = data['Close'].dropna().iloc[-2]
            change = latest_price - prev_price
            pct_change = (change / prev_price) * 100
            st.metric("Latest Price", f"₹{latest_price:.2f}", f"{pct_change:.2f}%")
        else:
            st.warning("Not enough valid data to display latest price.")
    else:
        st.warning(f"No chart data for {ticker}")

# -------------------------------
# User Inputs
# -------------------------------
st.sidebar.header("📌 Portfolio Input")
portfolio_input = st.sidebar.text_area("Enter your stock symbols (comma separated, e.g., RELIANCE.NS, TCS.NS, INFY.NS):")
analyze_btn = st.sidebar.button("Analyze Portfolio")

# -------------------------------
# Portfolio Analysis
# -------------------------------
if analyze_btn and portfolio_input:
    portfolio = [s.strip() for s in portfolio_input.split(",")]
    st.subheader("🧾 Portfolio Advice")
    advice = analyze_portfolio(portfolio)
    for line in advice:
        st.write(line)

    # Plot for each stock
    for stock in portfolio:
        plot_stock(stock)

# -------------------------------
# Usage Counter
# -------------------------------
if "usage_count" not in st.session_state:
    st.session_state.usage_count = 0

if analyze_btn:
    st.session_state.usage_count += 1

st.sidebar.markdown(f"🔢 **Advice Generated:** {st.session_state.usage_count}")
