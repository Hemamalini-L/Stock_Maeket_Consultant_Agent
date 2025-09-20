import os
import sqlite3
import yfinance as yf
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from openai import OpenAI

# --------------------------
# API KEY SETUP
# --------------------------
api_key = None
try:
    api_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

if not api_key:
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    api_key = "sk-your-test-key-here"  # Replace only for local testing

client = OpenAI(api_key=api_key)

# --------------------------
# SQLITE DATABASE SETUP
# --------------------------
DB_FILE = "stock_consultant.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            portfolio TEXT,
            usage_count INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    conn.close()

def save_user(name, portfolio):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO users (name, portfolio, usage_count) VALUES (?, ?, ?)", 
              (name, portfolio, 0))
    conn.commit()
    conn.close()

def update_usage(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE users SET usage_count = usage_count + 1 WHERE name=?", (name,))
    conn.commit()
    conn.close()

def get_usage(name):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT usage_count FROM users WHERE name=?", (name,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 0

init_db()

# --------------------------
# STOCK DATA FUNCTIONS
# --------------------------
def get_stock_data(symbol, period="6mo"):
    stock = yf.Ticker(symbol)
    data = stock.history(period=period)
    return data

def plot_stock(data, symbol):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(data.index, data["Close"], label="Close Price", color="blue")
    ax.set_title(f"{symbol} Price Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (₹)")
    ax.legend()
    st.pyplot(fig)

# --------------------------
# AI ADVICE GENERATOR
# --------------------------
def generate_advice(portfolio, stock_data):
    prompt = f"""
    You are a financial consultant for beginner investors.
    The user's portfolio is: {portfolio}.
    Based on the latest stock data: {stock_data.tail(5).to_dict()},
    provide simple advice:
    - Whether to Buy, Hold, or Sell.
    - Risk warnings (if any).
    - Diversification tips.
    Answer in plain English, short and clear.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful stock consultant."},
                  {"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(page_title="📊 Stock Consultant Agent", layout="wide")

st.title("📊 Stock Market Consultant Agent")
st.write("Beginner-friendly stock advice using **real-time market data**.")

# Sidebar User Info
st.sidebar.header("👤 User Info")
name = st.sidebar.text_input("Enter your name")
portfolio = st.sidebar.text_area("Enter your portfolio (comma-separated stock symbols, e.g., RELIANCE.NS, TCS.NS)")

if st.sidebar.button("Save User"):
    if name and portfolio:
        save_user(name, portfolio)
        st.sidebar.success("✅ User saved successfully!")

# Main App
if name and portfolio:
    stocks = [s.strip() for s in portfolio.split(",") if s.strip()]
    st.subheader(f"📌 Portfolio Analysis for {name}")

    for stock in stocks:
        st.markdown(f"### 📈 {stock}")
        try:
            data = get_stock_data(stock)
            if not data.empty:
                st.metric("Latest Price", f"₹{data['Close'].iloc[-1]:.2f}")
                plot_stock(data, stock)

                # Generate AI advice
                with st.spinner("Analyzing..."):
                    advice = generate_advice(portfolio, data)
                st.success(advice)

                # Update usage
                update_usage(name)
            else:
                st.warning(f"No data found for {stock}")
        except Exception as e:
            st.error(f"Error fetching data for {stock}: {e}")

    usage_count = get_usage(name)
    st.info(f"📊 You have asked for advice **{usage_count} times**.")

else:
    st.warning("👉 Please enter your name and portfolio in the sidebar to get started.")
