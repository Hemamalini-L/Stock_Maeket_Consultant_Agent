import streamlit as st
import yfinance as yf
import pandas as pd
import random
import datetime

# ---------------------------
# Helper Functions
# ---------------------------

def fetch_stock_data(ticker):
    """Fetch latest stock price using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5d")
        if hist.empty:
            return None
        return hist['Close'].iloc[-1]  # latest closing price
    except Exception:
        return None

def generate_advice(portfolio_df):
    """Generate simple advice for each stock"""
    advice_list = []
    for _, row in portfolio_df.iterrows():
        decision = random.choice(["Buy", "Sell", "Hold", "Diversify"])
        advice_list.append({
            "Stock": row['Stock'],
            "Current Price": row['Price'],
            "Quantity": row['Quantity'],
            "Advice": decision
        })
    return pd.DataFrame(advice_list)

# ---------------------------
# Streamlit App
# ---------------------------

st.set_page_config(page_title="Stock Market Consultant Agent", layout="wide")

st.title("📊 Stock Market Consultant Agent")
st.markdown("Get simple, beginner-friendly portfolio advice.")

# Session state for usage counter
if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

# Input Portfolio
st.sidebar.header("Enter Your Portfolio")
stocks_input = st.sidebar.text_area(
    "Enter stock tickers separated by commas (e.g., TCS.NS, INFY.NS, RELIANCE.NS)"
)
quantities_input = st.sidebar.text_area(
    "Enter quantities separated by commas (e.g., 10, 5, 2)"
)

if st.sidebar.button("Analyze Portfolio"):
    if stocks_input and quantities_input:
        tickers = [s.strip() for s in stocks_input.split(",")]
        quantities = [int(q.strip()) for q in quantities_input.split(",")]

        if len(tickers) == len(quantities):
            data = []
            for i in range(len(tickers)):
                price = fetch_stock_data(tickers[i])
                if price:
                    data.append({"Stock": tickers[i], "Quantity": quantities[i], "Price": price})
                else:
                    data.append({"Stock": tickers[i], "Quantity": quantities[i], "Price": "N/A"})

            portfolio_df = pd.DataFrame(data)
            st.write("### Your Portfolio")
            st.dataframe(portfolio_df)

            # Generate advice
            advice_df = generate_advice(portfolio_df)
            st.write("### Consultant Advice")
            st.dataframe(advice_df)

            # Increment counter
            st.session_state.analysis_count += 1
            st.success(f"✅ Analysis done! Total analyses so far: {st.session_state.analysis_count}")

            # Report download
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            report_csv = advice_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Advice Report",
                data=report_csv,
                file_name=f"portfolio_advice_{timestamp}.csv",
                mime="text/csv"
            )
        else:
            st.error("❌ Number of tickers and quantities must match!")
    else:
        st.error("❌ Please enter both stock tickers and quantities.")

# Usage Counter
st.sidebar.markdown("---")
st.sidebar.subheader("📌 Usage")
st.sidebar.write(f"Portfolios analyzed: {st.session_state.analysis_count}")

