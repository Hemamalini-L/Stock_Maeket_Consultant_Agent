import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# -------------------------
# Mock Stock Data (can replace with API)
# -------------------------
stock_data = pd.DataFrame({
    "Symbol": ["TCS", "INFY", "RELIANCE", "HDFC", "ICICI", "WIPRO"],
    "Price": [3300, 1800, 2500, 1500, 900, 450],
    "Risk": [75, 40, 60, 25, 35, 50],
    "Sector": ["IT", "IT", "Energy", "Banking", "Banking", "IT"]
})

# -------------------------
# Session state for usage tracking
# -------------------------
if 'portfolios_analyzed' not in st.session_state:
    st.session_state.portfolios_analyzed = 0
if 'advice_generated' not in st.session_state:
    st.session_state.advice_generated = 0
if 'billing_amount' not in st.session_state:
    st.session_state.billing_amount = 0

# -------------------------
# App Design: Zerodha-style Dashboard
# -------------------------
st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
st.title("📊 Stock Consultant Agent - Zerodha Clone")

# Sidebar for portfolio input
st.sidebar.header("Portfolio Input")
num_stocks = st.sidebar.number_input("Number of stocks", min_value=1, max_value=20, value=1)

portfolio = []
for i in range(num_stocks):
    symbol = st.sidebar.text_input(f"Stock {i+1} Symbol", key=f"symbol_{i}")
    quantity = st.sidebar.number_input(f"Stock {i+1} Quantity", min_value=1, value=1, key=f"qty_{i}")
    portfolio.append({"symbol": symbol.upper(), "quantity": quantity})

# -------------------------
# Functions
# -------------------------
def generate_advice(portfolio):
    advice_list = []
    total_value = 0
    stock_summary = []

    for stock in portfolio:
        symbol = stock["symbol"]
        qty = stock["quantity"]
        data = stock_data[stock_data["Symbol"] == symbol]

        if data.empty:
            advice_list.append(f"{symbol}: No data available")
            continue

        price = int(data["Price"].values[0])
        risk = int(data["Risk"].values[0])
        sector = data["Sector"].values[0]
        value = qty * price
        total_value += value
        stock_summary.append({"Symbol": symbol, "Quantity": qty, "Price": price, "Value": value, "Risk": risk, "Sector": sector})

        if risk > 70:
            advice_list.append(f"You already hold {symbol}. Risk is high – consider reducing.")
        elif risk < 30:
            advice_list.append(f"Consider adding {symbol} to your portfolio – low risk.")
        else:
            advice_list.append(f"Hold your {symbol}.")

    # Diversification advice
    current_symbols = [p['symbol'] for p in portfolio]
    missing_sectors = [s for s in stock_data['Symbol'] if s not in current_symbols]
    if missing_sectors:
        advice_list.append(f"Consider adding these stocks for diversification: {', '.join(missing_sectors[:2])}")

    return advice_list, stock_summary, total_value

def plot_portfolio(stock_summary):
    df = pd.DataFrame(stock_summary)
    fig, ax = plt.subplots()
    sns.barplot(x="Symbol", y="Value", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Portfolio Value Distribution")
    ax.set_ylabel("Value (₹)")
    return fig

def generate_invoice(advice_count, portfolio_count, total_value):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Stock Consultant Invoice", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Portfolios Analyzed: {portfolio_count}", ln=True)
    pdf.cell(0, 10, f"Advice Generated: {advice_count}", ln=True)
    pdf.cell(0, 10, f"Total Portfolio Value: ₹{total_value}", ln=True)
    pdf.cell(0, 10, f"Billing Amount: ₹{portfolio_count*10 + advice_count*5}", ln=True)
    file_name = f"Invoice_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(file_name)
    return file_name

# -------------------------
# Main App
# -------------------------
if st.button("Analyze Portfolio"):
    advice, summary, total_value = generate_advice(portfolio)

    # Advice Section
    st.subheader("💡 Advice")
    for a in advice:
        st.write("- " + a)

    # Portfolio Summary Table
    st.subheader("📈 Portfolio Summary")
    st.dataframe(pd.DataFrame(summary))

    # Portfolio Distribution Chart
    fig = plot_portfolio(summary)
    st.pyplot(fig)

    # Update usage stats
    st.session_state.portfolios_analyzed += 1
    st.session_state.advice_generated += len(advice)
    st.session_state.billing_amount += st.session_state.portfolios_analyzed*10 + st.session_state.advice_generated*5

    # Download Invoice
    invoice_file = generate_invoice(st.session_state.advice_generated, st.session_state.portfolios_analyzed, total_value)
    st.download_button("📄 Download Invoice", invoice_file, file_name=invoice_file, mime="application/pdf")

# -------------------------
# Usage Dashboard (like Zerodha dashboard)
# -------------------------
st.sidebar.subheader("📝 Usage Dashboard")
st.sidebar.write(f"Portfolios Analyzed: {st.session_state.portfolios_analyzed}")
st.sidebar.write(f"Advice Generated: {st.session_state.advice_generated}")
st.sidebar.write(f"Estimated Billing Amount: ₹{st.session_state.billing_amount}")
