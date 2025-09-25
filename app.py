# app.py
"""
Stock Market Consultant Agent - Full-featured Streamlit app
Features:
 - Portfolio input (manual + CSV upload)
 - Live market data via yfinance (quotes + history)
 - Technical indicators: MA, RSI, MACD (simple)
 - Buy / Sell / Hold rules with plain-English reasoning
 - Portfolio analytics: current value, P/L, allocation, volatility, Sharpe
 - Graphs: price line, candlestick, RSI, allocation pie
 - Usage tracking and billing (SQLite)
 - Export JSON + PDF invoice/report
 - Supports any ticker yfinance recognizes; optionally load tickers.csv for a master list
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sqlite3, json, os
from datetime import datetime, timedelta
from io import BytesIO
from fpdf import FPDF  # for simple PDF generation
import math

# ----------------------------
# CONFIG
# ----------------------------
DB_FILE = "usage.db"
PER_ADVICE_CHARGE = 0.1      # currency units per stock advice
PER_ANALYSIS_CHARGE = 1.0    # currency units per portfolio analysis
DEFAULT_TICKERS_FILE = "tickers.csv"  # optional master list
# ----------------------------

# ----------------------------
# DB helpers
# ----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS usage (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      action TEXT, detail TEXT, charge REAL, timestamp TEXT
    )''')
    c.execute('''
    CREATE TABLE IF NOT EXISTS analyses (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT, portfolio_json TEXT, advice_json TEXT, charge REAL, timestamp TEXT
    )''')
    conn.commit()
    conn.close()

def log_usage(action, detail, charge=0.0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO usage (action, detail, charge, timestamp) VALUES (?,?,?,?)',
              (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_analysis(user_id, portfolio, advice, charge):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO analyses (user_id, portfolio_json, advice_json, charge, timestamp) VALUES (?,?,?,?,?)',
              (user_id, json.dumps(portfolio), json.dumps(advice), charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def get_usage_summary():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT action, COUNT(*), SUM(charge) FROM usage GROUP BY action')
    rows = c.fetchall(); conn.close()
    return rows

def get_recent_analyses(limit=10):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT id, user_id, portfolio_json, advice_json, charge, timestamp FROM analyses ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall(); conn.close()
    return rows

# ----------------------------
# Market data helpers (yfinance)
# ----------------------------
@st.cache_data(ttl=60)  # cache for 60 seconds for "real-time-ish" performance
def fetch_ticker_info(ticker):
    t = yf.Ticker(ticker)
    info = t.info
    return info

@st.cache_data(ttl=60)
def fetch_history(ticker, period="6mo", interval="1d"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# quick quote
@st.cache_data(ttl=20)
def get_quote(ticker):
    t = yf.Ticker(ticker)
    q = t.history(period="2d", interval="1d")  # fallback
    # use fast method
    try:
        last = t.info.get('regularMarketPrice', None)
    except Exception:
        last = None
    # fallback to history close
    if last is None and not q.empty:
        last = float(q['Close'].iloc[-1])
    return last

# ----------------------------
# Technical indicators
# ----------------------------
def add_moving_averages(df, windows=[20,50,200]):
    for w in windows:
        df[f"MA_{w}"] = df['Close'].rolling(window=w).mean()
    return df

def compute_RSI(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    df['RSI'] = rsi
    return df

# simple MACD
def compute_MACD(df, a=12, b=26, c=9):
    df['EMA_fast'] = df['Close'].ewm(span=a, adjust=False).mean()
    df['EMA_slow'] = df['Close'].ewm(span=b, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=c, adjust=False).mean()
    return df

# ----------------------------
# Advice engine (rules + beginner friendly explanation)
# ----------------------------
def compute_signals(df):
    signals = {}
    if df.empty:
        return signals
    latest = df.iloc[-1]
    # MA cross (using MA_50 and MA_200)
    try:
        ma50 = latest.get('MA_50', None)
        ma200 = latest.get('MA_200', None)
    except:
        ma50 = ma200 = None
    # RSI
    rsi = latest.get('RSI', None)
    # MACD
    macd = latest.get('MACD', None)
    macd_signal = latest.get('MACD_signal', None)

    # simple rules
    if ma50 and ma200:
        if ma50 > ma200:
            signals['ma'] = ('bullish', "50-day MA is above 200-day MA (uptrend).")
        else:
            signals['ma'] = ('bearish', "50-day MA is below 200-day MA (downtrend).")

    if rsi is not None:
        if rsi < 30:
            signals['rsi'] = ('oversold', f"RSI={rsi:.1f} (oversold). Potential buying opportunity for long-term investors.")
        elif rsi > 70:
            signals['rsi'] = ('overbought', f"RSI={rsi:.1f} (overbought). Consider taking profits or wait.")
        else:
            signals['rsi'] = ('neutral', f"RSI={rsi:.1f} (neutral).")

    if macd is not None and macd_signal is not None:
        if macd > macd_signal:
            signals['macd'] = ('bullish', "MACD line above signal line (momentum improving).")
        else:
            signals['macd'] = ('bearish', "MACD line below signal line (momentum weakening).")

    return signals

def stock_advice(ticker, qty, buy_price, timeframe='short'):
    """
    Returns dict with currentPrice, pnl_pct, recommendation (Buy/Sell/Hold/Strong Buy/Strong Sell),
    and a beginner-friendly explanation.
    timeframe: 'short' or 'long' affects recommendation thresholds
    """
    df = fetch_history(ticker, period="1y", interval="1d")
    if df.empty:
        return {"error": "No data for ticker"}
    df = add_moving_averages(df, windows=[20,50,200])
    df = compute_RSI(df)
    df = compute_MACD(df)
    current = get_quote(ticker)
    buy_price = float(buy_price or 0)
    qty = float(qty or 0)
    value = current * qty
    pnl_pct = ((current - buy_price) / buy_price * 100) if buy_price>0 else None

    signals = compute_signals(df)

    # rule-based recommendation (combining signals)
    score = 0
    reasons = []
    # MA signal
    ma = signals.get('ma', (None,''))[0]
    if ma == 'bullish':
        score += 2
        reasons.append("Long-term trend looks positive (MA crossover).")
    elif ma == 'bearish':
        score -= 2
        reasons.append("Long-term trend looks negative (MA crossover).")
    # MACD
    macd = signals.get('macd', (None,''))[0]
    if macd == 'bullish':
        score += 1
    elif macd == 'bearish':
        score -= 1
    # RSI
    rsi_tag = signals.get('rsi', (None,''))[0]
    if rsi_tag == 'oversold':
        score += 1
        reasons.append("RSI indicates oversold conditions (possible buy).")
    elif rsi_tag == 'overbought':
        score -= 1
        reasons.append("RSI indicates overbought conditions (consider taking profits).")

    # PnL threshold adjustments depending on timeframe
    if pnl_pct is not None:
        if timeframe == 'short':
            if pnl_pct <= -5:
                score -= 1; reasons.append("Short-term loss >5% — consider stop-loss.")
            if pnl_pct >= 8:
                score += 1; reasons.append("Short-term gain >8% — consider booking partial profits.")
        else:  # long
            if pnl_pct <= -20:
                score -= 2; reasons.append("Long-term loss >20% — review position.")
            if pnl_pct >= 50:
                score += 2; reasons.append("Long-term gain >50% — consider rebalancing/taking profits.")

    # Final recommendation mapping
    if score >= 3:
        rec = "Strong Buy"
    elif score == 2:
        rec = "Buy"
    elif score == 1 or score == 0:
        rec = "Hold"
    elif score == -1 or score == -2:
        rec = "Reduce / Take Profit"
    else:
        rec = "Sell / Cut Loss"

    # Beginner friendly explanation
    explanation = " ".join(reasons) if reasons else "No strong technical signals; consider your investment horizon and risk tolerance."

    return {
        "ticker": ticker.upper(),
        "currentPrice": round(current,2) if current else None,
        "quantity": qty,
        "buyPrice": buy_price,
        "value": round(value,2),
        "pnl_pct": round(pnl_pct,2) if pnl_pct is not None else None,
        "score": score,
        "recommendation": rec,
        "explanation": explanation,
        "signals": signals
    }

# ----------------------------
# Portfolio analytics
# ----------------------------
def analyze_portfolio(portfolio: dict, timeframe='long'):
    """
    portfolio: {'userId':..., 'stocks': [{'name','quantity','buyPrice'},...]}
    Returns: detailed advice for each stock, portfolio summary
    """
    stocks = portfolio.get('stocks', [])
    advices = []
    total_value = 0.0
    values = []
    # per-stock advice
    for s in stocks:
        name = s.get('name')
        qty = s.get('quantity', 0)
        buy = s.get('buyPrice', 0)
        adv = stock_advice(name, qty, buy, timeframe=timeframe)
        if 'error' in adv:
            continue
        advices.append(adv)
        total_value += adv['value']
        values.append({'ticker': adv['ticker'], 'value': adv['value'], 'pnl_pct': adv['pnl_pct']})
    # portfolio metrics
    # weights
    for v in values:
        v['weight_pct'] = round((v['value'] / total_value * 100) if total_value>0 else 0,2)
    # simple volatility & Sharpe approx using historical daily returns
    returns = []
    for s in stocks:
        df = fetch_history(s['name'], period="6mo", interval="1d")
        if df.empty: continue
        df['ret'] = df['Close'].pct_change()
        returns.append(df['ret'].dropna())
    if returns:
        allrets = pd.concat(returns)
        vol = float(allrets.std() * math.sqrt(252))  # annualized volatility
        sharpe = float((allrets.mean() * 252) / (allrets.std() * math.sqrt(252))) if allrets.std()!=0 else None
    else:
        vol = None; sharpe = None

    summary = {
        "total_value": round(total_value,2),
        "positions": values,
        "annual_volatility": round(vol,4) if vol else None,
        "sharpe": round(sharpe,3) if sharpe else None
    }

    return advices, summary

# ----------------------------
# UI Helpers & plots
# ----------------------------
def plot_price_chart(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'))
    if 'MA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_20'], name='MA 20'))
    if 'MA_50' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_50'], name='MA 50'))
    if 'MA_200' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA_200'], name='MA 200'))
    fig.update_layout(title=f"{ticker} price chart", xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_candlestick(df, ticker):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(title=f"{ticker} Candlestick", xaxis_rangeslider_visible=False)
    return fig

def plot_rsi(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
    fig.update_layout(title="RSI", yaxis=dict(range=[0,100]))
    fig.add_hline(y=70, line_dash="dash", line_color="red")
    fig.add_hline(y=30, line_dash="dash", line_color="green")
    return fig

# PDF invoice generation
def generate_pdf_invoice(user_id, analysis_id, portfolio, advices, charge):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10, txt="Stock Consultant Agent - Analysis Invoice", ln=True, align='C')
    pdf.ln(5)
    pdf.cell(200,10, txt=f"Analysis ID: {analysis_id}  User: {user_id}", ln=True)
    pdf.cell(200,10, txt=f"Timestamp: {datetime.utcnow().isoformat()}", ln=True)
    pdf.ln(5)
    pdf.cell(200,10, txt=f"Charge: {charge} (mock currency)", ln=True)
    pdf.ln(5)
    pdf.cell(200,10, txt="Portfolio Snapshot:", ln=True)
    for s in portfolio.get('stocks', []):
        pdf.cell(200,8, txt=f"- {s['name']} | qty: {s['quantity']} | buyPrice: {s['buyPrice']}", ln=True)
    pdf.ln(5)
    pdf.cell(200,10, txt="Advice:", ln=True)
    for a in advices:
        pdf.multi_cell(0, 6, txt=f"{a['ticker']}: {a['recommendation']} — {a['explanation']}")
    # return bytes
    b = BytesIO()
    pdf.output(b)
    b.seek(0)
    return b

# ----------------------------
# Page layout
# ----------------------------
def main():
    st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
    init_db()

    st.title("Stock Market Consultant Agent — Complete")
    st.markdown("**Disclaimer:** This app provides educational guidance and is not financial advice. Always verify before trading.")

    # left: portfolio input, right: live market / search
    col1, col2 = st.columns([2,1])

    # LOAD master tickers list if provided
    tickers_list = []
    if os.path.exists(DEFAULT_TICKERS_FILE):
        try:
            df_t = pd.read_csv(DEFAULT_TICKERS_FILE)
            if 'symbol' in df_t.columns:
                tickers_list = df_t['symbol'].astype(str).tolist()
        except Exception:
            tickers_list = []

    with col2:
        st.header("Market Lookup")
        search_ticker = st.text_input("Enter ticker (e.g. AAPL, TCS.NS)", value="AAPL")
        if st.button("Lookup"):
            try:
                q = get_quote(search_ticker)
                info = fetch_ticker_info(search_ticker)
                st.metric(label=f"{search_ticker} price", value=q)
                st.json({k: info.get(k) for k in ['sector','industry','marketCap','regularMarketPrice'] if k in info})
                # show recent chart
                df = fetch_history(search_ticker, period="3mo")
                if not df.empty:
                    df = add_moving_averages(df, [20,50,200])
                    df = compute_RSI(df)
                    st.plotly_chart(plot_price_chart(df, search_ticker), use_container_width=True)
                    st.plotly_chart(plot_candlestick(df, search_ticker), use_container_width=True)
                    st.plotly_chart(plot_rsi(df), use_container_width=True)
                    # signals summary
                    signals = compute_signals(df)
                    st.write("Signals:")
                    st.write(signals)
            except Exception as e:
                st.error("Lookup failed: " + str(e))

    with col1:
        st.header("Portfolio Input & Analysis")
        st.write("Add holdings manually below, or upload a CSV with columns: name,quantity,buyPrice")
        with st.form("portfolio_form"):
            user_id = st.text_input("User ID", value="hema_user")
            default = st.checkbox("Load example portfolio", value=True)
            upload = st.file_uploader("Upload CSV (optional)", type=['csv'])
            if upload:
                try:
                    df = pd.read_csv(upload)
                    df = df[['name','quantity','buyPrice']]
                except Exception as e:
                    st.error("CSV error: "+str(e)); df = pd.DataFrame(columns=['name','quantity','buyPrice'])
            else:
                if default:
                    df = pd.DataFrame([{"name":"AAPL","quantity":10,"buyPrice":150},
                                       {"name":"MSFT","quantity":5,"buyPrice":300}])
                else:
                    df = pd.DataFrame(columns=['name','quantity','buyPrice'])
            st.dataframe(df)
            timeframe = st.selectbox("Advice timeframe", ["short", "long"], index=1)
            analyze = st.form_submit_button("Analyze Portfolio")
        if analyze:
            # build portfolio
            portfolio = {"userId": user_id, "stocks": df.to_dict(orient='records')}
            # get advices
            advs, summary = analyze_portfolio(portfolio, timeframe=timeframe)
            # show summary
            st.subheader("Portfolio Summary")
            st.metric("Total value (mock real-time)", f"{summary['total_value']}")
            if summary.get('annual_volatility') is not None:
                st.write(f"Annual vol: {summary['annual_volatility']}, Sharpe (approx): {summary['sharpe']}")
            # allocation pie
            if summary['positions']:
                dfpos = pd.DataFrame(summary['positions'])
                figpie = go.Figure(data=[go.Pie(labels=dfpos['ticker'], values=dfpos['value'], hole=.3)])
                st.plotly_chart(figpie, use_container_width=True)
            # show advices table
            if advs:
                st.subheader("Per-stock Advice")
                dft = pd.DataFrame(advs)
                st.dataframe(dft[['ticker','quantity','buyPrice','currentPrice','pnl_pct','recommendation']])
                # detailed expanders
                for a in advs:
                    with st.expander(f"{a['ticker']} — {a['recommendation']}"):
                        st.write("Current price:", a['currentPrice'])
                        st.write("P/L %:", a['pnl_pct'])
                        st.write("Reasoning:", a['explanation'])
                        st.json(a['signals'])
                        # charts for each
                        dfh = fetch_history(a['ticker'], period="6mo")
                        if not dfh.empty:
                            dfh = add_moving_averages(dfh, [20,50,200])
                            dfh = compute_RSI(dfh)
                            st.plotly_chart(plot_price_chart(dfh, a['ticker']), use_container_width=True)
            else:
                st.info("No positions or no data found.")
            # Billing & saving
            n = len(advs)
            charge = PER_ANALYSIS_CHARGE + PER_ADVICE_CHARGE * n
            st.write(f"Charge for this analysis (mock): {charge}")
            if st.button("Save analysis & generate invoice"):
                save_analysis(user_id, portfolio, advs, charge)
                log_usage("analysis_saved", f"user={user_id},count={n}", charge)
                st.success("Saved. Invoice available in Usage & Billing.")
                # retrieve latest analysis id
                recent = get_recent_analyses(1)
                if recent:
                    aid = recent[0][0]
                    pdfb = generate_pdf_invoice(user_id, aid, portfolio, advs, charge)
                    st.download_button("Download Invoice PDF", data=pdfb, file_name=f"invoice_{aid}.pdf", mime="application/pdf")
            # immediate json download
            st.download_button("Download analysis JSON", data=json.dumps({"portfolio":portfolio,"advice":advs,"summary":summary}, indent=2), file_name="analysis.json")

    st.markdown("---")
    # Usage & billing panel
    st.header("Usage & Billing")
    rows = get_usage_summary()
    if rows:
        dfu = pd.DataFrame(rows, columns=['action','count','total_charge'])
        st.table(dfu)
        st.write("Total charges (mock):", dfu['total_charge'].sum())
    else:
        st.info("No usage recorded yet.")

    st.markdown("---")
    # Education section
    st.header("Beginner's Guide: How to Invest")
    st.markdown("""
    **Steps for a beginner**
    1. Start with goal: Define time horizon (short <1yr, medium 1-3yr, long >3yr).  
    2. Build emergency fund before investing.  
    3. Diversify across sectors and avoid concentration > 30-40%.  
    4. For long-term investing: focus on fundamentals (earnings, revenue growth, ROE).  
    5. For short-term trading: combine technical signals (MAs, RSI) with strict stop-loss rules.  
    6. Always size positions relative to portfolio risk (e.g., 1-5% of portfolio per trade).
    """)
    st.markdown("**When to buy / sell / hold (plain English):**")
    st.markdown("""
    - **Buy** when the long-term trend is positive (MA crossover), the stock is not overbought (RSI < 70), and you have conviction in fundamentals.  
    - **Sell / Reduce** when the stock is overconcentrated, fundamentals deteriorate, or technicals show reversal and you hit stop-loss.  
    - **Hold** when there are mixed signals or you're long-term focused and short-term volatility appears.
    """)
    st.markdown("---")
    st.caption("App note: Real-time data depends on yfinance. For production-grade real-time, integrate a market-data API with websocket support or paid data provider.")

if __name__ == "__main__":
    main()
