import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import random
import io

# ---------------------------
# Mock sector mapping for diversification
# ---------------------------
SECTOR_MAP = {
    "RELIANCE.NS": "Energy",
    "TCS.NS": "IT",
    "INFY.NS": "IT",
    "HDFCBANK.NS": "Banking",
    "ICICIBANK.NS": "Banking",
    "SBIN.NS": "Banking",
    "HINDUNILVR.NS": "FMCG",
    "ITC.NS": "FMCG",
    "LT.NS": "Infrastructure",
}

# ---------------------------
# State initialization
# ---------------------------
if "usage" not in st.session_state:
    st.session_state.usage = {"portfolios": 0, "advices": 0, "chat_queries": 0}

# ---------------------------
# Helper functions
# ---------------------------
def fetch_stock_data(ticker, period="6mo"):
    """Fetch stock data from Yahoo Finance"""
    try:
        data = yf.download(ticker, period=period, interval="1d")
        if data.empty:
            return None
        return data
    except Exception as e:
        st.warning(f"Live data fetch failed for {ticker}, using Pathway mock data.")
        # Mock Pathway fallback with CSV
        try:
            return pd.read_csv("mock_stock_data.csv", index_col=0, parse_dates=True)
        except:
            return None


def generate_advice(data, ticker):
    """Generate beginner-friendly stock advice"""
    latest_price = data["Close"][-1]
    week_ago_price = data["Close"][-5] if len(data) >= 5 else data["Close"][0]
    pct_change = ((latest_price - week_ago_price) / week_ago_price) * 100

    if pct_change > 5:
        action = "Consider Selling"
        explanation = f"{ticker} rose {pct_change:.2f}% this week. It may be a good time to book profits."
    elif pct_change < -5:
        action = "Consider Buying"
        explanation = f"{ticker} dropped {pct_change:.2f}%. If you believe in long-term value, it may be a chance to accumulate."
    else:
        action = "Hold"
        explanation = f"{ticker} has been stable ({pct_change:.2f}%). No major action needed."
    
    return action, explanation, latest_price, pct_change


def diversification_check(portfolio):
    """Check if portfolio is diversified across sectors"""
    sectors = [SECTOR_MAP.get(stock, "Other") for stock in portfolio]
    unique_sectors = set(sectors)
    if len(unique_sectors) < 2:
        return "⚠️ Your portfolio is not well-diversified. Consider adding stocks from different sectors."
    return "✅ Your portfolio looks diversified across multiple sectors."


def export_report(advice_list):
    """Export portfolio analysis as CSV"""
    df = pd.DataFrame(advice_list, columns=["Stock", "Action", "Explanation", "Latest Price", "Weekly % Change"])
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📈 Stock Market Consultant Agent")
st.write("Your beginner-friendly stock advisor. Get live advice, portfolio insights, and diversification tips!")

portfolio_input = st.text_area("Enter your stock portfolio (comma separated)", "RELIANCE.NS, TCS.NS, INFY.NS")
portfolio = [s.strip() for s in portfolio_input.split(",") if s.strip()]

if st.button("Analyze Portfolio"):
    st.session_state.usage["portfolios"] += 1
    advice_list = []
    
    for stock in portfolio:
        data = fetch_stock_data(stock)
        if data is not None:
            action, explanation, latest_price, pct_change = generate_advice(data, stock)
            st.metric(stock, f"₹{latest_price:.2f}", f"{pct_change:.2f}%")
            st.write(f"**Advice:** {action}")
            st.write(f"**Reason:** {explanation}")
            advice_list.append([stock, action, explanation, latest_price, pct_change])
            st.session_state.usage["advices"] += 1

            # Chart
            fig, ax = plt.subplots()
            ax.plot(data.index, data["Close"], label=stock)
            ax.set_title(f"{stock} Performance")
            ax.legend()
            st.pyplot(fig)
        else:
            st.error(f"No data available for {stock}")

    # Diversification
    st.subheader("📊 Diversification Check")
    st.info(diversification_check(portfolio))

    # Report export
    csv_data = export_report(advice_list)
    st.download_button("Download Portfolio Report (CSV)", data=csv_data, file_name="portfolio_report.csv")

# ---------------------------
# Beginner Chat System
# ---------------------------
st.subheader("💬 Beginner Q&A Chat")
user_question = st.text_input("Ask about your portfolio (e.g., 'Should I hold TCS?')")

if st.button("Get Answer"):
    if user_question:
        st.session_state.usage["chat_queries"] += 1
        random_response = random.choice([
            "Holding looks safe for now.",
            "You may consider reducing exposure if risk feels high.",
            "Diversify into other sectors to reduce risk.",
            "Check long-term trends before deciding."
        ])
        st.success(f"Answer: {random_response}")

# ---------------------------
# Usage & Billing Mock
# ---------------------------
st.sidebar.header("📌 Usage & Billing (Mock Flexprice)")
st.sidebar.write(f"Portfolios analyzed: {st.session_state.usage['portfolios']}")
st.sidebar.write(f"Advice generated: {st.session_state.usage['advices']}")
st.sidebar.write(f"Chat queries asked: {st.session_state.usage['chat_queries']}")
st.sidebar.info("Billing: ₹5 per advice, ₹10 per portfolio analysis")

