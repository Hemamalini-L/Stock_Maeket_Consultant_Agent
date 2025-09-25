import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
from datetime import datetime

# Initialize session state
if 'portfolios_analyzed' not in st.session_state:
    st.session_state.portfolios_analyzed = 0
if 'advice_generated' not in st.session_state:
    st.session_state.advice_generated = 0
if 'billing_amount' not in st.session_state:
    st.session_state.billing_amount = 0

# Streamlit app layout
st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
st.title("📊 Stock Consultant Agent")

# Sidebar for portfolio input
st.sidebar.header("Enter Your Portfolio")
num_stocks = st.sidebar.number_input("Number of stocks", min_value=1, max_value=20, value=1)

portfolio = []
for i in range(num_stocks):
    symbol = st.sidebar.text_input(f"Stock {i+1} Symbol (e.g., AAPL)", key=f"symbol_{i}")
    quantity = st.sidebar.number_input(f"Stock {i+1} Quantity", min_value=1, value=1, key=f"qty_{i}")
    portfolio.append({"symbol": symbol.upper(), "quantity": quantity})

# Function to fetch real-time stock data
def fetch_stock_data(symbol):
    data = yf.download(symbol, period="1d", interval="1m")
    return data['Close'].iloc[-1] if not data.empty else None

# Function to generate advice
def generate_advice(portfolio):
    advice_list = []
    stock_summary = []
    total_value = 0

    for stock in portfolio:
        symbol = stock["symbol"]
        qty = stock["quantity"]
        price = fetch_stock_data(symbol)

        if price is None:
            advice_list.append(f"{symbol}: Data not available")
            continue

        value = price * qty
        total_value += value
        stock_summary.append({"Symbol": symbol, "Quantity": qty, "Price": price, "Value": value})

        # Simple advice logic
        if price < 1000:
            advice_list.append(f"{symbol}: Consider buying more – price is low.")
        elif price > 5000:
            advice_list.append(f"{symbol}: Consider selling – price is high.")
        else:
            advice_list.append(f"{symbol}: Hold your position.")

    return advice_list, stock_summary, total_value

# Function to plot portfolio distribution
def plot_portfolio(stock_summary):
    df = pd.DataFrame(stock_summary)
    fig, ax = plt.subplots()
    sns.barplot(x="Symbol", y="Value", data=df, ax=ax)
    ax.set_title("Portfolio Value Distribution")
    return fig

# Function to generate PDF report
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
    for stock in stock_summary:
        pdf.cell(200, 10, txt=f"{stock['Symbol']} - {stock['Quantity']} shares @ ₹{stock['Price']} each", ln=True)

    filename = f"Portfolio_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf.output(filename)
    return filename

# Main app logic
if st.button("Analyze Portfolio"):
    advice, summary, total_value = generate_advice(portfolio)

    # Display advice
    st.subheader("💡 Investment Advice")
    for line in advice:
        st.write(line)

    # Display portfolio summary
    st.subheader("📈 Portfolio Summary")
    st.dataframe(pd.DataFrame(summary))

    # Display portfolio distribution chart
    fig = plot_portfolio(summary)
    st.pyplot(fig)

    # Update session state
    st.session_state.portfolios_analyzed += 1
    st.session_state.advice_generated += len(advice)
    st.session_state.billing_amount += st.session_state.portfolios_analyzed * 10 + st.session_state.advice_generated * 5

    # Generate and provide download link for PDF report
    pdf_filename = generate_pdf(advice, summary, total_value)
    st.download_button("Download PDF Report", pdf_filename, file_name=pdf_filename, mime="application/pdf")

# Display usage statistics
st.sidebar.subheader("Usage Statistics")
st.sidebar.write(f"Portfolios Analyzed: {st.session_state.portfolios_analyzed}")
st.sidebar.write(f"Advice Generated: {st.session_state.advice_generated}")
st.sidebar.write(f"Billing Amount: ₹{st.session_state.billing_amount}")
