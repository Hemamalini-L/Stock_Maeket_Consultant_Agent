import streamlit as st
import yfinance as yf
import sqlite3
import datetime
import pandas as pd
from openai import OpenAI

# =========================
#  API SETUP
# =========================
# ✅ OPTION 1: Hardcode API key (for testing only!)
client = OpenAI(api_key="sk-YOUR_REAL_KEY_HERE")

# ✅ OPTION 2 (Recommended for deployment):
# client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# =========================
#  DATABASE SETUP
# =========================
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_user(name, email):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (name, email) VALUES (?, ?)", (name, email))
    conn.commit()
    conn.close()

# =========================
#  STOCK DATA FETCH
# =========================
def get_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1mo")
        if data.empty:
            st.error("No data found for this ticker.")
            return None
        return data
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

# =========================
#  AI ADVICE
# =========================
def generate_advice(data, ticker):
    latest_price = round(data["Close"].iloc[-1], 2)
    pct_change = round(((data["Close"].iloc[-1] - data["Close"].iloc[0]) / data["Close"].iloc[0]) * 100, 2)

    prompt = f"""
    You are a financial advisor. A beginner investor is asking about {ticker}.
    The current price is {latest_price}, and the stock has changed {pct_change}% in the last month.
    Give a clear and simple recommendation: BUY, SELL, or HOLD.
    Also explain risks, beginner-friendly tips, and current market value insights.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a financial stock advisor."},
                      {"role": "user", "content": prompt}]
        )
        advice = response.choices[0].message.content.strip()
    except Exception as e:
        advice = f"Error generating AI advice: {e}"

    return latest_price, pct_change, advice

# =========================
#  STREAMLIT UI
# =========================
def main():
    st.set_page_config(page_title="AI Stock Advisor", layout="wide")
    st.title("📈 AI-Powered Stock Investment Advisor")

    # User Registration
    st.sidebar.header("👤 New User Registration")
    name = st.sidebar.text_input("Name")
    email = st.sidebar.text_input("Email")
    if st.sidebar.button("Register"):
        if name and email:
            add_user(name, email)
            st.sidebar.success("User registered successfully!")
        else:
            st.sidebar.error("Please enter both name and email.")

    # Stock input
    ticker = st.text_input("Enter Stock Ticker (e.g., TCS.NS, INFY.NS, AAPL)", "AAPL")

    if st.button("Get Stock Insights"):
        data = get_stock_data(ticker)
        if data is not None:
            st.subheader(f"📊 Stock Data for {ticker}")
            st.line_chart(data["Close"])

            latest_price, pct_change, advice = generate_advice(data, ticker)

            st.metric(label="Latest Price", value=f"${latest_price}")
            st.metric(label="1-Month Change", value=f"{pct_change}%")

            st.subheader("🤖 AI Investment Advice")
            st.write(advice)

            st.warning("⚠️ Investment Disclaimer: This is AI-generated advice. Please consult a financial advisor before making decisions.")

# =========================
#  RUN
# =========================
if __name__ == "__main__":
    init_db()
    main()
