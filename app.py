import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime, timedelta

# -------------------------
# Helper Functions
# -------------------------
def get_stock_data(ticker, period="6mo"):
    try:
        data = yf.download(ticker, period=period, interval="1d")
        if data.empty:
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def plot_candlestick(data, ticker):
    fig = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name="Candlestick"
    )])
    # Add moving averages
    for ma in [20, 50, 200]:
        data[f"MA{ma}"] = data['Close'].rolling(ma).mean()
        fig.add_trace(go.Scatter(x=data.index, y=data[f"MA{ma}"], mode='lines', name=f"MA{ma}"))
    fig.update_layout(title=f"{ticker} Stock Price", xaxis_rangeslider_visible=False)
    return fig

def get_recommendations():
    # Example beginner-friendly suggestions
    return {
        "Safe Choices": ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        "High Risk - High Reward": ["ZOMATO.NS", "PAYTM.NS"]
    }

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Stock Market Consultant", layout="wide")
st.title("📈 Stock Market Consultant for Beginners")

st.sidebar.header("Choose a Stock")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., RELIANCE.NS, AAPL)", "RELIANCE.NS")

period = st.sidebar.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y"])
data = get_stock_data(ticker, period)

if data is not None:
    st.subheader(f"Real-Time Data for {ticker}")
    latest_price = data['Close'].iloc[-1]
    prev_price = data['Close'].iloc[-2]
    change = latest_price - prev_price
    pct_change = (change / prev_price) * 100

    st.metric("Latest Price", f"₹{latest_price:.2f}", f"{pct_change:.2f}%")

    st.plotly_chart(plot_candlestick(data, ticker), use_container_width=True)
else:
    st.warning("No data found for this ticker.")

# -------------------------
# Portfolio Tracker
# -------------------------
st.header("💼 Your Portfolio Tracker")

if "portfolio" not in st.session_state:
    st.session_state.portfolio = []

with st.form("Add Stock"):
    stock_ticker = st.text_input("Stock Ticker (e.g., TCS.NS)")
    qty = st.number_input("Quantity", min_value=1, value=1)
    buy_price = st.number_input("Buy Price (₹)", min_value=1.0, value=100.0)
    submitted = st.form_submit_button("Add to Portfolio")

    if submitted:
        st.session_state.portfolio.append({"Ticker": stock_ticker, "Qty": qty, "Buy Price": buy_price})
        st.success(f"Added {stock_ticker} to portfolio!")

if st.session_state.portfolio:
    portfolio_data = []
    total_value, total_cost = 0, 0
    for stock in st.session_state.portfolio:
        stock_data = get_stock_data(stock["Ticker"], "1mo")
        if stock_data is not None:
            current_price = stock_data['Close'].iloc[-1]
            value = stock["Qty"] * current_price
            cost = stock["Qty"] * stock["Buy Price"]
            pl = value - cost
            portfolio_data.append([stock["Ticker"], stock["Qty"], stock["Buy Price"], current_price, value, pl])
            total_value += value
            total_cost += cost
    df = pd.DataFrame(portfolio_data, columns=["Ticker", "Qty", "Buy Price", "Current Price", "Value", "P/L"])
    st.dataframe(df)
    st.subheader(f"📊 Total Portfolio Value: ₹{total_value:.2f} | P/L: ₹{total_value-total_cost:.2f}")

# -------------------------
# Beginner Guide
# -------------------------
st.header("🧑‍🏫 Beginner Investment Advice")
st.info("""
- **Start Small:** Invest only what you can afford to lose.  
- **Diversify:** Don’t put all money in one stock.  
- **Focus on Blue-Chip:** Safer for beginners (Reliance, Infosys, TCS).  
- **Avoid Day Trading:** Start with long-term investments.  
- **Learn Indicators:** Moving averages, P/E ratio, market cap.  
""")

# -------------------------
# Suggested Stocks
# -------------------------
st.header("🔍 Stock Suggestions for Beginners")
recommendations = get_recommendations()
st.success(f"Safe Choices: {', '.join(recommendations['Safe Choices'])}")
st.error(f"Risky Picks: {', '.join(recommendations['High Risk - High Reward'])}")
