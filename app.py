import streamlit as st
import pandas as pd

# -------------------------
# Mock Stock Data (replace with Pathway API later)
# -------------------------
stock_data = pd.DataFrame({
    "Symbol": ["TCS", "INFY", "RELIANCE", "HDFC"],
    "Price": [3300, 1800, 2500, 1500],
    "Risk": [75, 40, 60, 25]  # 0-100 scale
})

# -------------------------
# Initialize session state for usage tracking
# -------------------------
if 'portfolios_analyzed' not in st.session_state:
    st.session_state.portfolios_analyzed = 0
if 'advice_generated' not in st.session_state:
    st.session_state.advice_generated = 0

# -------------------------
# Title
# -------------------------
st.title("📈 Stock Market Consultant Agent")
st.write("Enter your portfolio and get simple buy/sell/hold/diversify advice.")

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
# Generate Advice Function
# -------------------------
def generate_advice(portfolio):
    advice_list = []
    for stock in portfolio:
        symbol = stock["symbol"]
        data = stock_data[stock_data["Symbol"] == symbol]
        if data.empty:
            advice_list.append(f"{symbol}: No data available")
            continue
        
        risk = int(data["Risk"].values[0])
        
        if risk > 70:
            advice_list.append(f"You already hold {symbol}. Risk is high – consider reducing.")
        elif risk < 30:
            advice_list.append(f"Consider adding {symbol} to your portfolio – low risk.")
        else:
            advice_list.append(f"Hold your {symbol}.")
    return advice_list

# -------------------------
# Button to analyze
# -------------------------
if st.button("Get Advice"):
    advice = generate_advice(portfolio)
    st.subheader("📊 Advice:")
    for a in advice:
        st.write("- " + a)

    # Update usage
    st.session_state.portfolios_analyzed += 1
    st.session_state.advice_generated += len(advice)

# -------------------------
# Display usage
# -------------------------
st.subheader("📝 Usage:")
st.write(f"Portfolios analyzed: {st.session_state.portfolios_analyzed}")
st.write(f"Advice generated: {st.session_state.advice_generated}")
