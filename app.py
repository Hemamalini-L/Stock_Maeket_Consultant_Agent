import streamlit as st
import yfinance as yf
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
import openai
import os

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="📊 Stock Consultant Agent", layout="wide")
DB_FILE = "portfolio.db"

# -------------------------------
# DATABASE FUNCTIONS
# -------------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS portfolio (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user TEXT,
            symbol TEXT,
            quantity INTEGER,
            buy_price REAL,
            created TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def add_stock(user, symbol, quantity, buy_price):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO portfolio (user, symbol, quantity, buy_price) VALUES (?, ?, ?, ?)",
              (user, symbol.upper(), quantity, buy_price))
    conn.commit()
    conn.close()

def get_portfolio(user):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM portfolio WHERE user=?", conn, params=(user,))
    conn.close()
    return df

def delete_stock(stock_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM portfolio WHERE id=?", (stock_id,))
    conn.commit()
    conn.close()

# -------------------------------
# STOCK FUNCTIONS
# -------------------------------
def get_stock_data(symbol, period="6mo"):
    stock = yf.Ticker(symbol)
    hist = stock.history(period=period)
    return hist

def generate_advice(symbol, quantity, buy_price):
    try:
        hist = get_stock_data(symbol, "1mo")
        if hist.empty:
            return "❌ No data available", 0, 0, 0
        
        latest_price = hist["Close"].iloc[-1]
        change_pct = ((latest_price - buy_price) / buy_price) * 100

        # Simple advice rules
        if change_pct > 15:
            advice = f"📈 Your {symbol} is up {change_pct:.2f}%. Consider partial selling to book profits."
        elif change_pct < -10:
            advice = f"📉 Your {symbol} is down {change_pct:.2f}%. Risk is high. Consider stop-loss or long-term hold."
        else:
            advice = f"⚖️ Hold {symbol}. It's stable with {change_pct:.2f}% change."

        return advice, latest_price, change_pct, quantity * latest_price

    except Exception as e:
        return f"❌ Error: {str(e)}", 0, 0, 0

# -------------------------------
# DASHBOARD
# -------------------------------
st.title("📊 Stock Market Consultant Agent")
st.caption("Beginner-friendly advisor for stock investing with real-time data, portfolio tracking, and AI-powered advice.")

# Sidebar
st.sidebar.header("User Login")
username = st.sidebar.text_input("Enter your name:", "guest")

if username:
    init_db()

    st.sidebar.subheader("Add Stock to Portfolio")
    symbol = st.sidebar.text_input("Stock Symbol (e.g., RELIANCE.NS, TCS.NS, INFY.NS)")
    qty = st.sidebar.number_input("Quantity", min_value=1, value=10)
    buy_price = st.sidebar.number_input("Buy Price", min_value=1.0, value=1000.0)
    if st.sidebar.button("Add to Portfolio"):
        add_stock(username, symbol, qty, buy_price)
        st.sidebar.success(f"✅ Added {symbol} to {username}'s portfolio")

    # Show portfolio
    st.header(f"📂 {username}'s Portfolio")
    portfolio = get_portfolio(username)

    if not portfolio.empty:
        portfolio_data = []
        total_value = 0
        total_investment = 0

        for _, row in portfolio.iterrows():
            advice, latest_price, pct, value = generate_advice(row['symbol'], row['quantity'], row['buy_price'])
            portfolio_data.append([row['id'], row['symbol'], row['quantity'], row['buy_price'], latest_price, pct, value, advice])
            total_value += value
            total_investment += row['quantity'] * row['buy_price']

        df = pd.DataFrame(portfolio_data, columns=["ID", "Symbol", "Qty", "Buy Price", "Latest Price", "Change %", "Value", "Advice"])
        st.dataframe(df, hide_index=True)

        # Portfolio summary
        st.subheader("💰 Portfolio Summary")
        st.metric("Total Investment", f"₹{total_investment:,.2f}")
        st.metric("Current Value", f"₹{total_value:,.2f}")
        st.metric("Net Gain/Loss", f"₹{total_value - total_investment:,.2f}")

        # Diversification chart
        fig = go.Figure(data=[go.Pie(labels=df["Symbol"], values=df["Value"], hole=0.3)])
        fig.update_layout(title_text="Portfolio Diversification")
        st.plotly_chart(fig)

        # Graph for single stock
        st.subheader("📉 Stock Charts")
        stock_choice = st.selectbox("Select stock for chart:", df["Symbol"].tolist())
        if stock_choice:
            hist = get_stock_data(stock_choice, "6mo")
            fig = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close']
            )])
            fig.update_layout(title=f"{stock_choice} - Last 6 Months", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig)

    else:
        st.info("ℹ️ Your portfolio is empty. Add stocks from the sidebar to begin.")

    # Delete stock option
    st.sidebar.subheader("Manage Portfolio")
    del_id = st.sidebar.number_input("Delete stock by ID", min_value=0)
    if st.sidebar.button("Delete Stock"):
        delete_stock(del_id)
        st.sidebar.warning(f"🗑️ Deleted stock with ID {del_id}")

    # AI Chatbot (Beginner guide)
    st.subheader("🤖 Stock Consultant Chatbot")
    api_key = st.text_input("Enter OpenAI API Key:", type="password")
    query = st.text_input("Ask a question (e.g., Should I sell Infosys?)")

    if st.button("Get AI Advice") and api_key:
        try:
            openai.api_key = api_key
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are a beginner-friendly stock advisor. Explain in simple terms."},
                          {"role": "user", "content": query}]
            )
            st.success(response["choices"][0]["message"]["content"])
        except Exception as e:
            st.error(f"Error: {str(e)}")

