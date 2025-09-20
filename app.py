import os
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from agno.agent import Agent
from agno.models.google import Gemini

# ==============================
# Helper Functions
# ==============================

def compare_stocks(symbols):
    """Compare stock % change over last 6 months"""
    data = {}
    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")

            if hist.empty:
                continue
            data[symbol] = round(hist['Close'].pct_change().sum() * 100, 2)
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    return data


def get_company_info(symbol):
    stock = yf.Ticker(symbol)
    info = stock.info
    return {
        "name": info.get("longName", "N/A"),
        "sector": info.get("sector", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "summary": info.get("longBusinessSummary", "N/A"),
    }


def get_company_news(symbol):
    stock = yf.Ticker(symbol)
    try:
        return stock.news[:5]
    except Exception:
        return []


# ==============================
# Define Agents
# ==============================

market_analyst = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    description="Analyzes stock performance",
    instructions=[
        "Compare stock performance over 6 months",
        "Rank stocks based on % change",
    ],
    markdown=True,
)

company_researcher = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    description="Analyzes company fundamentals and news",
    instructions=[
        "Summarize company profile and sector",
        "Add key news updates",
    ],
    markdown=True,
)

stock_strategist = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    description="Recommends best stocks",
    instructions=[
        "Use performance + fundamentals",
        "Provide Buy/Hold/Sell advice",
    ],
    markdown=True,
)

team_lead = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    description="Final investment report",
    instructions=[
        "Aggregate insights from analysts",
        "Make a final ranked recommendation",
    ],
    markdown=True,
)


# ==============================
# Pipeline Functions
# ==============================

def get_market_analysis(symbols):
    perf = compare_stocks(symbols)
    if not perf:
        return "No valid stock data found."
    return market_analyst.run(f"Compare stock performance: {perf}").content


def get_company_analysis(symbol):
    info = get_company_info(symbol)
    news = get_company_news(symbol)
    return company_researcher.run(
        f"Company: {info['name']} ({symbol})\n"
        f"Sector: {info['sector']}\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Summary: {info['summary']}\n"
        f"News: {news}\n"
    ).content


def get_final_report(symbols):
    market = get_market_analysis(symbols)
    companies = {s: get_company_analysis(s) for s in symbols}
    strategy = stock_strategist.run(
        f"Market: {market}\nCompanies: {companies}\n"
    ).content

    return team_lead.run(
        f"Market Analysis: {market}\n"
        f"Company Analyses: {companies}\n"
        f"Strategy: {strategy}\n"
        f"Final ranked Buy/Hold/Sell recommendations please."
    ).content


# ==============================
# Streamlit App
# ==============================

st.set_page_config(page_title="AI Stock Insights", page_icon="📈", layout="wide")
st.title("📈 AI-Powered Stock Market Insights")
st.caption("Backed by Gemini + Yahoo Finance")

with st.sidebar:
    api_key = st.text_input("🔑 Enter your Gemini API key", type="password")
    st.markdown("---")
    st.markdown("Built with ❤️ using [Build Fast with AI](https://buildfastwithai.com)")

if not api_key:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

os.environ["GOOGLE_API_KEY"] = api_key

# Stock input
symbols = st.text_input("Enter stock symbols (comma separated)", "AAPL, TSLA, GOOG")
symbols = [s.strip().upper() for s in symbols.split(",") if s.strip()]

if st.button("🚀 Generate Investment Report"):
    with st.spinner("Analyzing..."):
        report = get_final_report(symbols)

    st.subheader("📊 Investment Report")
    st.markdown(report)

    # Plot stock performance
    st.subheader("📈 Stock Performance (6 months)")
    data = yf.download(symbols, period="6mo")["Close"]

    fig = go.Figure()
    for sym in symbols:
        if sym in data:
            fig.add_trace(go.Scatter(x=data.index, y=data[sym], mode="lines", name=sym))

    fig.update_layout(title="6-Month Stock Trend", xaxis_title="Date", yaxis_title="Price (USD)")
    st.plotly_chart(fig)
