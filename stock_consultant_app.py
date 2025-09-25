import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go
from fpdf import FPDF
import sqlite3
import datetime
from io import BytesIO

st.set_page_config(page_title="Beginner-Friendly Stock Consultant", layout="wide")

# ---------- Helper Functions ----------

def fetch_price(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="1d")
        if not data.empty:
            return float(data['Close'].iloc[-1])
    except:
        return None
    return None

def portfolio_analysis(portfolio, goal):
    results = []
    for _, row in portfolio.iterrows():
        ticker = row['name']
        qty = row['quantity']
        buy_price = row['buyPrice']

        current_price = fetch_price(ticker)
        if current_price:
            pnl = ((current_price - buy_price) / buy_price) * 100
            if goal == "Short-term":
                if pnl > 5:
                    advice = "Consider booking profits."
                elif pnl < -5:
                    advice = "Consider stop-loss to avoid further loss."
                else:
                    advice = "Hold for now."
            else:
                if pnl > 20:
                    advice = "Partial profit booking recommended."
                elif pnl < -20:
                    advice = "Review fundamentals. Consider diversifying."
                else:
                    advice = "Hold for long-term stability."
        else:
            current_price, pnl, advice = None, None, "No data available."

        results.append({
            "Symbol": ticker,
            "Quantity": qty,
            "Buy Price": buy_price,
            "Current Price": round(current_price, 2) if current_price else None,
            "P&L %": round(pnl, 2) if pnl else None,
            "Advice": advice
        })
    return pd.DataFrame(results)

def generate_pdf(report_df, user_id):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, f"Stock Consultant Report - {user_id}", ln=True, align="C")
    pdf.ln(10)
    for i, row in report_df.iterrows():
        line = f"{row['Symbol']} | Qty: {row['Quantity']} | Buy: {row['Buy Price']} | Current: {row['Current Price']} | P&L: {row['P&L %']}% | Advice: {row['Advice']}"
        pdf.multi_cell(0, 10, line)
    buf = BytesIO()
    pdf.output(buf)
    return buf

def save_log(user_id, goal, portfolio_df):
    conn = sqlite3.connect("consultant_usage.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS logs 
                 (user TEXT, goal TEXT, timestamp TEXT)""")
    c.execute("INSERT INTO logs VALUES (?,?,?)", 
              (user_id, goal, str(datetime.datetime.now())))
    conn.commit()
    conn.close()

def stock_chart(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="6mo")
    if hist.empty:
        return None
    fig = go.Figure(data=[go.Candlestick(
        x=hist.index,
        open=hist['Open'],
        high=hist['High'],
        low=hist['Low'],
        close=hist['Close']
    )])
    fig.update_layout(title=f"{ticker} - 6 Months Candlestick", xaxis_rangeslider_visible=False)
    return fig

# ---------- Website Layout ----------

st.title("📈 Beginner-Friendly Stock Consultant Agent")
st.write("Welcome! This app is designed to help **beginners** understand the stock market, analyze their portfolio, and get simple advice in plain English. 🚀")

# ---- Section 1: Watchlist ----
st.header("1️⃣ Quick Market Watchlist")
tickers = st.text_input("Enter tickers (comma separated)", "AAPL, MSFT, INFY.NS")
ticker_list = [t.strip() for t in tickers.split(",")]

data = []
for t in ticker_list:
    price = fetch_price(t)
    if price:
        data.append({"Ticker": t, "Price": price})
df_watch = pd.DataFrame(data)
st.dataframe(df_watch)

# ---- Section 2: Portfolio ----
st.header("2️⃣ Your Portfolio")
user_id = st.text_input("Enter your name / user ID", "student123")
uploaded = st.file_uploader("Upload a CSV (name,quantity,buyPrice)", type=['csv'])
if uploaded:
    portfolio = pd.read_csv(uploaded)
else:
    st.info("No file uploaded. Using sample portfolio below.")
    portfolio = pd.DataFrame([
        {"name": "AAPL", "quantity": 5, "buyPrice": 150},
        {"name": "INFY.NS", "quantity": 10, "buyPrice": 1200}
    ])
st.dataframe(portfolio)

# ---- Section 3: Goal ----
st.header("3️⃣ Investment Goal")
goal = st.radio("Select your goal:", ["Short-term", "Long-term"])

# ---- Section 4: Analysis ----
st.header("4️⃣ Portfolio Analysis & Advice")
report_df = portfolio_analysis(portfolio, goal)
st.dataframe(report_df)

# ---- Section 5: Charts ----
st.header("5️⃣ Stock Charts & Trends")
selected_ticker = st.selectbox("Pick a ticker to view chart:", ticker_list)
chart = stock_chart(selected_ticker)
if chart:
    st.plotly_chart(chart, use_container_width=True)

# ---- Section 6: Report & Billing ----
st.header("6️⃣ Download Report / Invoice")
if st.button("Generate PDF Report"):
    buf = generate_pdf(report_df, user_id)
    st.download_button("📥 Download Report", buf, file_name="stock_report.pdf")

# ---- Section 7: Beginner's Guide ----
st.header("7️⃣ Beginner's Guide to Investing")
st.write("""
- Start small with well-known companies.  
- Understand **short-term** vs **long-term**:  
  - Short-term → quick trades, more risk.  
  - Long-term → steady growth, less stress.  
- Diversify (don’t put all money in one stock).  
- Always set a budget & risk limit.  
- Don’t panic with market fluctuations.  
- Learn step by step. 📚
""")

# ---- Section 8: Logging ----
save_log(user_id, goal, portfolio)
st.success("Your session has been saved. ✅")
