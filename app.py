# app_real_time.py
"""
Real-time-ish Stock Market Consultant Agent (Streamlit)
- Use TWELVEDATA_API_KEY for better near-real-time quotes (recommended)
- Falls back to yfinance when TwelveData key is not present
- Portfolio upload/manual, watchlist, interactive charts, buy/sell/hold signals,
  simulated orders, alerts, billing, persistence (Mongo optional)
"""

import os, time, json, math
from datetime import datetime, timedelta
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
import sqlite3
from fpdf import FPDF  # or fpdf2

# ---------- CONFIG ----------
TWELVE_KEY = os.environ.get("TWELVEDATA_API_KEY")  # optional
MONGO_URI = os.environ.get("MONGO_URI")            # optional: use for persistence if you want
DB_FILE = "usage_real_time.db"
DEFAULT_REFRESH = 15   # seconds (watchlist / pricing poll)
PER_ADVICE = 0.1
PER_ANALYSIS = 1.0
# ----------------------------

# ---------- DB helpers (SQLite fallback) ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS usage (id INTEGER PRIMARY KEY, action TEXT, detail TEXT, charge REAL, ts TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS analyses (id INTEGER PRIMARY KEY, user TEXT, portfolio TEXT, advice TEXT, charge REAL, ts TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS orders (id INTEGER PRIMARY KEY, user TEXT, ticker TEXT, qty REAL, price REAL, side TEXT, ts TEXT)')
    conn.commit()
    conn.close()

def log_usage(action, detail="", charge=0.0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO usage (action, detail, charge, ts) VALUES (?,?,?,?)', (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_analysis(user, portfolio, advice, charge):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO analyses (user, portfolio, advice, charge, ts) VALUES (?,?,?,?,?)',
              (user, json.dumps(portfolio), json.dumps(advice), charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_order(user, ticker, qty, price, side):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO orders (user,ticker,qty,price,side,ts) VALUES (?,?,?,?,?,?)',
              (user,ticker,qty,price,side,datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

# ---------- Market data adapters ----------
def quote_twelvedata(ticker):
    """Return current quote using TwelveData (fast)."""
    if not TWELVE_KEY:
        return None
    url = "https://api.twelvedata.com/quote"
    params = {"symbol": ticker, "apikey": TWELVE_KEY}
    try:
        r = requests.get(url, params=params, timeout=6)
        data = r.json()
        if "status" in data and data["status"] == "error":
            return None
        # most important fields: price, percent_change
        return {
            "price": float(data.get("price")) if data.get("price") else None,
            "percent_change": float(data.get("percent_change")) if data.get("percent_change") else None,
            "timestamp": data.get("timestamp")
        }
    except Exception as e:
        return None

@st.cache_data(ttl=60)
def history_yf(ticker, period="6mo", interval="1d"):
    t = yf.Ticker(ticker)
    df = t.history(period=period, interval=interval)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_data(ttl=20)
def fast_yf_quote(ticker):
    t = yf.Ticker(ticker)
    try:
        info = t.fast_info
        price = info.get("last_price") or info.get("lastTradePriceOnly") or info.get("regularMarketPrice")
    except Exception:
        price = None
    # fallback history
    if price is None:
        df = history_yf(ticker, period="5d", interval="1d")
        if not df.empty:
            price = float(df['Close'].iloc[-1])
    return {"price": float(price) if price else None}

def get_realtime_quote(ticker):
    """Unified function: prefer TwelveData, fallback to yfinance."""
    if TWELVE_KEY:
        q = quote_twelvedata(ticker)
        if q and q.get("price") is not None:
            return q
    # fallback
    q = fast_yf_quote(ticker)
    return {"price": q.get("price"), "percent_change": None, "timestamp": None}

# ---------- Technical indicators ----------
def add_indicators(df):
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    df['MA200'] = df['Close'].rolling(200).mean()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = -delta.clip(upper=0).rolling(14).mean()
    rs = up / down
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def compute_signals(df):
    """Return dict of signals and friendly text."""
    if df.empty:
        return {}
    last = df.iloc[-1]
    signals = {}
    # MA crossover
    if not np.isnan(last.get('MA50', np.nan)) and not np.isnan(last.get('MA200', np.nan)):
        signals['MA'] = "bullish" if last['MA50'] > last['MA200'] else "bearish"
    # RSI
    rsi = last.get('RSI', None)
    if rsi is not None:
        if rsi < 30: signals['RSI'] = ("oversold", f"RSI {rsi:.1f} — oversold")
        elif rsi > 70: signals['RSI'] = ("overbought", f"RSI {rsi:.1f} — overbought")
        else: signals['RSI'] = ("neutral", f"RSI {rsi:.1f}")
    # MACD
    macd, sig = last.get('MACD', None), last.get('MACD_SIGNAL', None)
    if macd is not None and sig is not None:
        signals['MACD'] = "bullish" if macd > sig else "bearish"
    return signals

# ---------- Advice engine ----------
def stock_recommendation(ticker, qty, buy_price, timeframe='long'):
    df = history_yf(ticker, period="1y", interval="1d")
    if df.empty:
        return {"error": "no data"}
    df = add_indicators(df)
    signals = compute_signals(df)
    quote = get_realtime_quote(ticker)
    current = quote.get("price")
    pnl_pct = ((current - buy_price) / buy_price * 100) if buy_price and current else None

    score = 0
    reasons = []
    # MA
    ma = signals.get('MA', None)
    if ma == "bullish":
        score += 2; reasons.append("MA trend positive")
    elif ma == "bearish":
        score -= 2; reasons.append("MA trend negative")
    # RSI
    rsi_tag = signals.get('RSI', (None, None))[0]
    if rsi_tag == "oversold": score += 1; reasons.append("RSI oversold")
    elif rsi_tag == "overbought": score -= 1; reasons.append("RSI overbought")
    # MACD
    if signals.get('MACD') == "bullish": score += 1
    elif signals.get('MACD') == "bearish": score -= 1

    # PnL adjustments
    if pnl_pct is not None:
        if timeframe == 'short':
            if pnl_pct <= -5: score -= 1; reasons.append("short-term loss >5%")
            if pnl_pct >= 8: score += 1; reasons.append("short-term gain >8%")
        else:
            if pnl_pct <= -20: score -= 2; reasons.append("long-term loss >20%")
            if pnl_pct >= 50: score += 2; reasons.append("long-term gain >50%")

    # final mapping
    if score >= 3: rec = "Strong Buy"
    elif score == 2: rec = "Buy"
    elif score >= 0: rec = "Hold"
    elif score >= -2: rec = "Reduce"
    else: rec = "Sell"

    expl = "; ".join(reasons) if reasons else "No strong technical signal."
    return {
        "ticker": ticker,
        "current": current,
        "quantity": qty,
        "buy_price": buy_price,
        "pnl_pct": round(pnl_pct,2) if pnl_pct is not None else None,
        "score": score,
        "recommendation": rec,
        "explanation": expl,
        "signals": signals
    }

# ---------- Charts ----------
def price_chart(df, ticker, indicators=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
    if indicators:
        for col in ['MA20','MA50','MA200']:
            if col in df.columns:
                fig.add_trace(go.Scatter(x=df['Date'], y=df[col], name=col))
    fig.update_layout(title=f"{ticker} Close with MAs", xaxis_title="Date", yaxis_title="Price")
    return fig

def candlestick(df, ticker):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(title=f"{ticker} Candlestick", xaxis_rangeslider_visible=False)
    return fig

# ---------- Utilities ----------
def human_read_recommendation(rec):
    mapping = {
        "Strong Buy": "Good buying opportunity — consider adding if it fits your plan (long-term).",
        "Buy": "Positive signals; consider buying gradually.",
        "Hold": "Mixed or neutral signals — hold for now.",
        "Reduce": "Consider reducing your position or booking partial profits.",
        "Sell": "Strong sell signal — consider exiting to cut losses."
    }
    return mapping.get(rec, "")

def export_pdf_invoice(user, analysis_id, portfolio, advices, charge):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stock Consultant - Invoice", ln=True, align="C")
    pdf.ln(5)
    pdf.cell(200,10, txt=f"Analysis ID: {analysis_id}  User: {user}", ln=True)
    pdf.cell(200,10, txt=f"Charge: {charge}", ln=True)
    pdf.ln(5)
    pdf.cell(200,10, txt="Portfolio:", ln=True)
    for p in portfolio.get("stocks", []):
        pdf.cell(200,8, txt=f"- {p['name']} qty:{p['quantity']} buy:{p['buyPrice']}", ln=True)
    pdf.ln(5)
    pdf.cell(200,10, txt="Advice Summary:", ln=True)
    for a in advices:
        pdf.multi_cell(0,6, txt=f"{a['ticker']}: {a['recommendation']} -- {a['explanation']}")
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

# ---------- Streamlit UI ----------
st.set_page_config(layout="wide", page_title="Stock Consultant Agent - Real Time")
init_db()

st.title("Stock Consultant Agent — Real-time Dashboard (polling)")

col_left, col_right = st.columns([3,1])

with col_right:
    st.header("Controls")
    refresh = st.slider("Refresh frequency (sec)", min_value=5, max_value=60, value=DEFAULT_REFRESH, step=5)
    watchlist_tickers = st.text_area("Watchlist (comma-separated tickers)", value="AAPL,MSFT,GOOGL", height=100)
    watchlist = [t.strip().upper() for t in watchlist_tickers.split(",") if t.strip()]
    st.write("TwelveData API:", "Present" if TWELVE_KEY else "Not configured (using yfinance fallback)")
    st.write("Mongo persistence:", "Present" if MONGO_URI else "Not configured (using SQLite)")
    if st.button("Refresh now"):
        st.experimental_rerun()

with col_left:
    st.header("Watchlist (live-ish)")
    # live watchlist table
    data = []
    for t in watchlist:
        q = get_realtime_quote(t)
        price = q.get("price")
        pct = q.get("percent_change")
        data.append({"ticker": t, "price": price, "pct_change": pct})
    df_watch = pd.DataFrame(data)
    st.dataframe(df_watch)

    # allow selecting a ticker to inspect
    selected = st.selectbox("Select a ticker to inspect", options=watchlist)
    if selected:
        # show metric
        quote = get_realtime_quote(selected)
        st.metric(label=f"{selected} price", value=str(quote.get("price")), delta=str(quote.get("percent_change")))
        # show historical chart & indicators
        hist = history_yf(selected, period="6mo")
        if not hist.empty:
            hist = add_indicators(hist)
            st.plotly_chart(price_chart(hist, selected), use_container_width=True)
            st.plotly_chart(candlestick(hist, selected), use_container_width=True)
            # show RSI plot if present
            if 'RSI' in hist.columns:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist['RSI'], name='RSI'))
                fig.add_hline(y=70, line_dash="dash", line_color="red")
                fig.add_hline(y=30, line_dash="dash", line_color="green")
                st.plotly_chart(fig, use_container_width=True)
            # signals
            sigs = compute_signals(hist)
            st.write("Technical signals:", sigs)
            # recommendation for 1 share example
            rec = stock_recommendation(selected, qty=1, buy_price=quote.get("price") or 0, timeframe='long')
            st.subheader("Auto Recommendation")
            st.write(rec['recommendation'])
            st.write(rec['explanation'])
            st.info(human_read_recommendation(rec['recommendation']))

# ------ Portfolio section ------
st.markdown("---")
st.header("Your Portfolio")
col1, col2 = st.columns([2,1])
with col1:
    with st.form("portfolio_form"):
        user = st.text_input("User ID", value="hema_user")
        csvfile = st.file_uploader("Upload CSV (name,quantity,buyPrice)", type=["csv"])
        default = st.checkbox("Use example portfolio", value=True)
        if csvfile:
            df_port = pd.read_csv(csvfile)[['name','quantity','buyPrice']]
        else:
            if default:
                df_port = pd.DataFrame([{"name":"AAPL","quantity":10,"buyPrice":150},{"name":"MSFT","quantity":5,"buyPrice":300}])
            else:
                df_port = pd.DataFrame(columns=['name','quantity','buyPrice'])
        st.dataframe(df_port)
        timeframe = st.selectbox("Advice timeframe", ["short", "long"], index=1)
        submit = st.form_submit_button("Analyze Portfolio")
    if submit:
        port = {"user": user, "stocks": df_port.to_dict(orient='records')}
        advs = []
        for s in port['stocks']:
            adv = stock_recommendation(s['name'], s['quantity'], s['buyPrice'], timeframe)
            advs.append(adv)
        # portfolio summary
        total = sum([a.get('value') or (a.get('current',0) * a.get('quantity',0)) for a in advs])
        st.metric("Portfolio total (approx)", f"{round(total,2)}")
        df_ad = pd.DataFrame(advs)
        if not df_ad.empty:
            st.dataframe(df_ad[['ticker','quantity','buy_price','current','pnl_pct','recommendation']])
        # billing & save
        charge = PER_ANALYSIS + PER_ADVICE * len(advs)
        st.write(f"Estimated charge (mock): {charge}")
        if st.button("Save analysis & invoice"):
            save_analysis(user, port, advs, charge)
            log_usage("analysis_saved", f"user={user}", charge)
            st.success("Analysis saved.")
            recent = get_recent_analyses(limit=1)
            # create PDF
            if recent:
                aid = recent[0][0]
                pdfb = export_pdf_invoice(user, aid, port, advs, charge)
                st.download_button("Download Invoice PDF", data=pdfb, file_name=f"invoice_{aid}.pdf", mime="application/pdf")
            st.download_button("Download analysis JSON", data=json.dumps({"portfolio":port,"advice":advs}, indent=2), file_name="analysis.json")

with col2:
    st.subheader("Simulated Orders (paper trades)")
    with st.form("order_form"):
        o_user = st.text_input("User ID (for order)", value="hema_user")
        o_ticker = st.text_input("Ticker", value="AAPL")
        o_side = st.selectbox("Side", ["BUY","SELL"])
        o_qty = st.number_input("Quantity", min_value=1, value=1)
        o_price = st.number_input("Price (execute at)", min_value=0.0, value=float(get_realtime_quote(o_ticker).get("price") or 0.0))
        place = st.form_submit_button("Place paper order")
    if place:
        save_order(o_user, o_ticker, o_qty, o_price, o_side)
        st.success("Order saved (paper trade).")
        log_usage("paper_order", f"{o_user} {o_side} {o_ticker} {o_qty}@{o_price}", 0.0)

# ---------- Usage & Alerts ----------
st.markdown("---")
st.header("Usage & Alerts")
rows = get_usage_summary()
if rows:
    dfu = pd.DataFrame(rows, columns=['action','count','total_charge'])
    st.table(dfu)
else:
    st.info("No usage recorded yet.")

# ---------- Auto-refresh mechanism ----------
# Use st_autorefresh to poll at 'refresh' seconds. Keep this at the end so rest renders first.
count = st.experimental_get_query_params().get("ref", [0])[0]
# Pressing the Refresh button triggered st.experimental_rerun earlier. Here we set auto-refresh:
st.experimental_set_query_params(ref=int(time.time()))
st.experimental_rerun() if False else None

# Note: to implement live polling without full rerun, Streamlit's native options are limited.
# We emulate "live" by encouraging the user to set a low refresh and pressing Refresh.
# For production with real streaming, integrate a websocket client and update state continuously.
