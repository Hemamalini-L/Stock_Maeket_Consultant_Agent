import streamlit as st
import yfinance as yf
import sqlite3
from datetime import datetime
from openai import OpenAI
import plotly.graph_objects as go
import pandas as pd

# Initialize OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Database setup
conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()
c.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    portfolio TEXT,
    created_at TIMESTAMP
)
""")
c.execute("""
CREATE TABLE IF NOT EXISTS usage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    queries INTEGER,
    portfolios INTEGER,
    last_used TIMESTAMP
)
""")
conn.commit()

# --- Functions ---
def fetch_stock_data(symbol):
    """Fetch last 3 months stock history"""
    stock = yf.Ticker(symbol)
    hist = stock.history(period="3mo")
    return hist

def generate_advice(stock_name, latest_price, pct_change):
    """Generate plain English stock advice"""
    prompt = f"""
    You are a stock consultant. A beginner investor is holding {stock_name}.
    Latest price: {latest_price:.2f} INR. Daily change: {pct_change:.2f}%.
    
    Give simple advice in one line: Should they BUY, SELL, HOLD, or DIVERSIFY?
    Also explain the risk in plain English.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful stock consultant."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def update_usage(user_id, query=False, portfolio=False):
    """Update usage counters"""
    c.execute("SELECT * FROM usage WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if row:
        queries = row[2] + (1 if query else 0)
        portfolios = row[3] + (1 if portfolio else 0)
        c.execute("UPDATE usage SET queries=?, portfolios=?, last_used=? WHERE user_id=?",
                  (queries, portfolios, datetime.now(), user_id))
    else:
        c.execute("INSERT INTO usage (user_id, queries, portfolios, last_used) VALUES (?, ?, ?, ?)",
                  (user_id, 1 if query else 0, 1 if portfolio else 0, datetime.now()))
    conn.commit()

def plot_stock_graph(df, stock_name):
    """Show candlestick + moving averages"""
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'],
                                         name="Candlestick")])
    # Moving Averages
    df["MA20"] = df["Close"].rolling(window=20).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()

    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], mode="lines", name="20-day MA"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], mode="lines", name="50-day MA"))

    fig.update_layout(title=f"{stock_name} Stock Price",
                      xaxis_title="Date", yaxis_title="Price (INR)",
                      template="plotly_dark", height=600)
    return fig

# --- Streamlit App ---
st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
st.title("📈 Stock Market Consultant Agent")

menu = ["Register", "Portfolio", "Ask Advice", "Usage"]
choice = st.sidebar.radio("Menu", menu)

if choice == "Register":
    st.subheader("Register User")
    name = st.text_input("Enter your name")
    if st.button("Register"):
        c.execute("INSERT INTO users (name, portfolio, created_at) VALUES (?, ?, ?)",
                  (name, "", datetime.now()))
        conn.commit()
        st.success(f"✅ User {name} registered! Note your User ID from database.")

elif choice == "Portfolio":
    st.subheader("Enter Your Portfolio")
    user_id = st.number_input("Enter your User ID", min_value=1, step=1)
    portfolio = st.text_area("Enter stock symbols (comma separated, e.g., TCS.NS, INFY.NS, RELIANCE.NS)")
    if st.button("Save Portfolio"):
        c.execute("UPDATE users SET portfolio=? WHERE id=?", (portfolio, user_id))
        conn.commit()
        update_usage(user_id, portfolio=True)
        st.success("📌 Portfolio saved successfully!")

elif choice == "Ask Advice":
    st.subheader("Get Stock Advice")
    user_id = st.number_input("Enter your User ID", min_value=1, step=1)

    c.execute("SELECT portfolio FROM users WHERE id=?", (user_id,))
    row = c.fetchone()

    if row and row[0]:
        stocks = row[0].split(",")
        for stock in stocks:
            stock = stock.strip()
            try:
                df = fetch_stock_data(stock)
                if not df.empty:
                    latest_price = df["Close"][-1]
                    pct_change = ((df["Close"][-1] - df["Close"][-2]) / df["Close"][-2]) * 100
                    advice = generate_advice(stock, latest_price, pct_change)

                    st.markdown(f"### {stock}")
                    st.write(f"💰 Price: {latest_price:.2f} INR | 📊 Change: {pct_change:.2f}%")
                    st.info(advice)

                    # Plot graph
                    fig = plot_stock_graph(df, stock)
                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.warning(f"No data found for {stock}")
            except Exception as e:
                st.error(f"❌ Error fetching {stock}: {e}")

        update_usage(user_id, query=True)
    else:
        st.error("⚠️ No portfolio found for this user.")

elif choice == "Usage":
    st.subheader("Usage Statistics")
    user_id = st.number_input("Enter your User ID", min_value=1, step=1)
    c.execute("SELECT queries, portfolios, last_used FROM usage WHERE user_id=?", (user_id,))
    row = c.fetchone()
    if row:
        st.write(f"📌 Queries Asked: {row[0]}")
        st.write(f"📌 Portfolios Saved: {row[1]}")
        st.write(f"📌 Last Used: {row[2]}")
    else:
        st.warning("No usage found for this user yet.")

