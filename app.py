import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# --------------------------
# Mock billing & usage system
# --------------------------
if "usage_count" not in st.session_state:
    st.session_state.usage_count = 0
if "credit_balance" not in st.session_state:
    st.session_state.credit_balance = 10  # give 10 free credits

# --------------------------
# App Title
# --------------------------
st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
st.title("📊 Stock Market Consultant Agent")
st.markdown("Beginner-friendly stock advice in **plain English** with real-time market data.")

# --------------------------
# Portfolio Input
# --------------------------
st.sidebar.header("Enter Your Portfolio")
tickers_input = st.sidebar.text_area(
    "Enter stock tickers (comma-separated, e.g., TCS.NS, INFY.NS, RELIANCE.NS):"
)
quantities_input = st.sidebar.text_area(
    "Enter quantities (comma-separated, same order as tickers, e.g., 10, 5, 3):"
)

# Convert inputs
tickers = [t.strip() for t in tickers_input.split(",")] if tickers_input else []
quantities = [int(q.strip()) for q in quantities_input.split(",")] if quantities_input else []

# --------------------------
# Fetch real-time stock data
# --------------------------
def fetch_stock_data(tickers):
    try:
        data = yf.download(tickers, period="5d")["Adj Close"]
        return data
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# --------------------------
# Advice Rules
# --------------------------
def generate_advice(stock_data, tickers, quantities):
    advice_list = []
    diversification = {}
    
    for i, ticker in enumerate(tickers):
        try:
            price_now = stock_data[ticker].iloc[-1]
            price_week_ago = stock_data[ticker].iloc[0]
            change = ((price_now - price_week_ago) / price_week_ago) * 100

            # Rule-based advice
            if change > 5:
                advice = f"Price up {change:.2f}% this week → Consider **Selling** some {ticker}."
            elif change < -5:
                advice = f"Price down {change:.2f}% this week → Good time to **Buy/Hold** {ticker} if you trust fundamentals."
            else:
                advice = f"{ticker} is stable (change {change:.2f}%). → **Hold**."

            # Add diversification category
            if "INFY" in ticker or "TCS" in ticker or "WIPRO" in ticker:
                sector = "IT"
            elif "RELIANCE" in ticker or "ONGC" in ticker:
                sector = "Energy"
            elif "HDFCBANK" in ticker or "ICICIBANK" in ticker:
                sector = "Banking"
            else:
                sector = "Other"

            diversification[sector] = diversification.get(sector, 0) + quantities[i] * price_now

            advice_list.append({"Stock": ticker, "Advice": advice, "Current Price": price_now})
        except Exception:
            advice_list.append({"Stock": ticker, "Advice": "⚠️ Data unavailable", "Current Price": None})
    
    return advice_list, diversification

# --------------------------
# Run Analysis Button
# --------------------------
if st.sidebar.button("Analyze Portfolio"):
    if not tickers or not quantities or len(tickers) != len(quantities):
        st.error("Please enter tickers and quantities correctly.")
    elif st.session_state.credit_balance <= 0:
        st.error("❌ No credits left! Please upgrade your plan.")
    else:
        st.session_state.usage_count += 1
        st.session_state.credit_balance -= 1

        stock_data = fetch_stock_data(tickers)
        if stock_data is not None:
            advice_list, diversification = generate_advice(stock_data, tickers, quantities)

            st.subheader("📌 Personalized Advice")
            df_advice = pd.DataFrame(advice_list)
            st.table(df_advice)

            # Download Report
            csv = df_advice.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Advice Report", data=csv, file_name="portfolio_advice.csv", mime="text/csv")

            # Charts
            st.subheader("📊 Diversification Overview")
            fig, ax = plt.subplots()
            ax.pie(diversification.values(), labels=diversification.keys(), autopct='%1.1f%%')
            ax.set_title("Portfolio Diversification by Sector")
            st.pyplot(fig)

            st.subheader("📈 Stock Price Trends (Last 5 Days)")
            st.line_chart(stock_data)

# --------------------------
# Usage & Billing Info
# --------------------------
st.sidebar.markdown("## 💰 Usage & Billing")
st.sidebar.write(f"Portfolios analyzed: {st.session_state.usage_count}")
st.sidebar.write(f"Remaining Credits: {st.session_state.credit_balance}")
