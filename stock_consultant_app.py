import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime
import random

# -------------------------
# Mock real-time stock data
# -------------------------
companies = ["TCS", "INFY", "RELIANCE", "HDFC", "ICICI", "WIPRO", "AMZN", "AAPL", "MSFT", "GOOG"]
stock_data = pd.DataFrame({
    "Symbol": companies,
    "Price": [random.randint(300, 5000) for _ in companies],
    "Risk": [random.randint(10, 90) for _ in companies],
    "Sector": ["IT", "IT", "Energy", "Banking", "Banking", "IT", "Tech", "Tech", "Tech", "Tech"]
})

# -------------------------
# Session state
# -------------------------
if "portfolios_analyzed" not in st.session_state:
    st.session_state.portfolios_analyzed = 0
if "advice_generated" not in st.session_state:
    st.session_state.advice_generated = 0
if "billing_amount" not in st.session_state:
    st.session_state.billing_amount = 0

# -------------------------
# App Layout
# -------------------------
st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
st.title("📊 Stock Consultant Agent - Zerodha Clone")

# Sidebar: Portfolio input
st.sidebar.header("Portfolio Input")
num_stocks = st.sidebar.number_input("Number of stocks", 1, 10, 1)
portfolio = []
for i in range(num_stocks):
    symbol = st.sidebar.text_input(f"Stock {i+1} Symbol", key=f"symbol_{i}")
    qty = st.sidebar.number_input(f"Stock {i+1} Quantity", min_value=1, value=1, key=f"qty_{i}")
    portfolio.append({"symbol": symbol.upper(), "quantity": qty})

# -------------------------
# Functions
# -------------------------
def generate_advice(portfolio):
    advice_list = []
    stock_summary = []
    total_value = 0

    for stock in portfolio:
        symbol = stock["symbol"]
        qty = stock["quantity"]
        data = stock_data[stock_data["Symbol"] == symbol]

        if data.empty:
            advice_list.append(f"{symbol}: Data not available")
            continue

        price = int(data["Price"].values[0])
        risk = int(data["Risk"].values[0])
        sector = data["Sector"].values[0]
        value = qty * price
        total_value += value

        stock_summary.append({
            "Symbol": symbol,
            "Quantity": qty,
            "Price": price,
            "Value": value,
            "Risk": risk,
            "Sector": sector
        })

        # Simple advice rules
        if risk > 70:
            advice_list.append(f"{symbol}: High risk – consider selling.")
        elif risk < 30:
            advice_list.append(f"{symbol}: Low risk – consider buying more.")
        else:
            advice_list.append(f"{symbol}: Hold your position.")

    # Diversification suggestion
    missing_symbols = [s for s in companies if s not in [p["symbol"] for p in portfolio]]
    if missing_symbols:
        advice_list.append(f"Diversification: Consider adding {', '.join(missing_symbols[:2])}")

    return advice_list, stock_summary, total_value

def plot_portfolio(stock_summary):
    df = pd.DataFrame(stock_summary)
    fig, ax = plt.subplots()
    sns.barplot(x="Symbol", y="Value", data=df, palette="coolwarm", ax=ax)
    ax.set_title("Portfolio Value Distribution")
    return fig

def generate_pdf(advice, stock_summary, total_value):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stock Portfolio Analysis Report", ln=True, align='C')
    pdf.ln(10)

    for line in advice:
        pdf.multi_cell(0, 10, line)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Total Portfolio Value: ₹{total_value}", ln=True)
    pdf.ln(10)
    pdf.cell(200, 10, txt="Stock Summary:", ln=True)
    for s in stock_summary:
        pdf.cell(200, 10, txt=f"{s['Symbol']} - {s['Quantity']} shares @ ₹{s['Price']} each", ln=True)

    filename = f"Portfolio_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# -------------------------
# Main App
# -------------------------
if st.button("Analyze Portfolio"):
    advice, summary, total_value = generate_advice(portfolio)

    st.subheader("💡 Investment Advice")
    for line in advice:
        st.write("- " + line)

    st.subheader("📈 Portfolio Summary")
    st.dataframe(pd.DataFrame(summary))

    fig = plot_portfolio(summary)
    st.pyplot(fig)

    st.session_state.portfolios_analyzed += 1
    st.session_state.advice_generated += len(advice)
    st.session_state.billing_amount += st.session_state.portfolios_analyzed*10 + st.session_state.advice_generated*5

    pdf_file = generate_pdf(advice, summary, total_value)
    st.download_button("Download PDF Report", pdf_file, file_name=pdf_file, mime="application/pdf")

# -------------------------
# Usage
# -------------------------
st.sidebar.subheader("Usage Dashboard")
st.sidebar.write(f"Portfolios Analyzed: {st.session_state.portfolios_analyzed}")
st.sidebar.write(f"Advice Generated: {st.session_state.advice_generated}")
st.sidebar.write(f"Estimated Billing Amount: ₹{st.session_state.billing_amount}")
