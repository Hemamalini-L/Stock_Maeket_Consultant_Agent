import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# -------------------------
# Mock Stock Data
# -------------------------
stock_data = pd.DataFrame({
    "Symbol": ["TCS", "INFY", "RELIANCE", "HDFC", "ICICI", "WIPRO"],
    "Price": [3300, 1800, 2500, 1500, 900, 450],
    "Risk": [75, 40, 60, 25, 35, 50]
})

# -------------------------
# Session State for Usage Tracking
# -------------------------
if 'portfolios_analyzed' not in st.session_state:
    st.session_state.portfolios_analyzed = 0
if 'advice_generated' not in st.session_state:
    st.session_state.advice_generated = 0
if 'billing_amount' not in st.session_state:
    st.session_state.billing_amount = 0

# -------------------------
# Title
# -------------------------
st.title("📈 Stock Market Consultant Agent (Advanced Version)")
st.write("Enter your portfolio and get simple advice like a real stock consultant.")

# -------------------------
# Portfolio Input
# -------------------------
num_stocks = st.number_input("How many stocks in your portfolio?", min_value=1, max_value=20, value=1)

portfolio = []
for i in range(num_stocks):
    col1, col2 = st.columns(2)
    with col1:
        symbol = st.text_input(f"Stock {i+1} Symbol", key=f"symbol_{i}")
    with col2:
        quantity = st.number_input(f"Stock {i+1} Quantity", min_value=1, value=1, key=f"qty_{i}")
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
        value = qty * price
        total_value += value
        stock_summary.append({"Symbol": symbol, "Quantity": qty, "Price": price, "Value": value, "Risk": risk})

        if risk > 70:
            advice_list.append(f"You already hold {symbol}. Risk is high – consider reducing.")
        elif risk < 30:
            advice_list.append(f"Consider adding {symbol} to your portfolio – low risk.")
        else:
            advice_list.append(f"Hold your {symbol}.")

    # Portfolio diversification suggestion
    sectors_needed = [s for s in stock_data['Symbol'] if s not in [p['symbol'] for p in portfolio]]
    if sectors_needed:
        advice_list.append(f"Your portfolio lacks these stocks for better diversification: {', '.join(sectors_needed[:2])}")

    return advice_list, stock_summary, total_value

# Function to plot portfolio distribution
def plot_portfolio(stock_summary):
    df = pd.DataFrame(stock_summary)
    fig, ax = plt.subplots()
    sns.barplot(x="Symbol", y="Value", data=df, palette="viridis", ax=ax)
    ax.set_title("Portfolio Value Distribution")
    ax.set_ylabel("Value (₹)")
    return fig

# Function to generate invoice
def generate_invoice(advice_count, portfolio_count, total_value):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Stock Consultant Agent Invoice", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}", ln=True)
    pdf.cell(0, 10, f"Portfolios Analyzed: {portfolio_count}", ln=True)
    pdf.cell(0, 10, f"Advice Generated: {advice_count}", ln=True)
    pdf.cell(0, 10, f"Total Portfolio Value: ₹{total_value}", ln=True)
    pdf.cell(0, 10, f"Billing Amount: ₹{portfolio_count*10 + advice_count*5}", ln=True)  # Example billing formula
    file_name = f"Invoice_{datetime.now().strftime('%H%M%S')}.pdf"
    pdf.output(file_name)
    return file_name

# -------------------------
# Button to Analyze Portfolio
# -------------------------
if st.button("Analyze Portfolio"):
    advice, summary, total_value = generate_advice(portfolio)
    st.subheader("📊 Advice:")
    for a in advice:
        st.write("- " + a)

    st.subheader("💰 Portfolio Summary:")
    st.table(pd.DataFrame(summary))

    fig = plot_portfolio(summary)
    st.pyplot(fig)

    # Update usage
    st.session_state.portfolios_analyzed += 1
    st.session_state.advice_generated += len(advice)
    st.session_state.billing_amount += st.session_state.portfolios_analyzed*10 + st.session_state.advice_generated*5

    # Generate invoice button
    invoice_file = generate_invoice(st.session_state.advice_generated, st.session_state.portfolios_analyzed, total_value)
    st.download_button("📄 Download Invoice", invoice_file, file_name=invoice_file, mime="application/pdf")

# -------------------------
# Display Usage
# -------------------------
st.subheader("📝 Usage:")
st.write(f"Portfolios analyzed: {st.session_state.portfolios_analyzed}")
st.write(f"Advice generated: {st.session_state.advice_generated}")
st.write(f"Estimated Billing Amount: ₹{st.session_state.billing_amount}")
