# app_final.py
"""
Beginner-friendly Stock Consultant Agent (Streamlit)
Simple, interactive, lively — includes portfolio input, charts, advice,
usage & billing (SQLite), save/export analysis.
Uses yfinance for market data (near-real-time polling).
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import sqlite3, json, os, math
from datetime import datetime
from io import BytesIO
from fpdf import FPDF

# --------- Configuration ----------
DB_FILE = "app_usage.db"
PER_ADVICE_CHARGE = 0.05    # mock charge per stock advice
PER_ANALYSIS_CHARGE = 0.5   # mock charge per portfolio analysis
DEFAULT_REFRESH = 30        # seconds (suggested polling in UI)
# ---------------------------------

# --------- Database helpers (SQLite) ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS usage (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         action TEXT,
         detail TEXT,
         charge REAL,
         ts TEXT
      )
    ''')
    c.execute('''
      CREATE TABLE IF NOT EXISTS analyses (
         id INTEGER PRIMARY KEY AUTOINCREMENT,
         user_id TEXT,
         portfolio_json TEXT,
         advice_json TEXT,
         charge REAL,
         ts TEXT
      )
    ''')
    conn.commit()
    conn.close()

def log_usage(action, detail="", charge=0.0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO usage (action, detail, charge, ts) VALUES (?,?,?,?)',
              (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_analysis(user_id, portfolio, advice, charge):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO analyses (user_id, portfolio_json, advice_json, charge, ts) VALUES (?,?,?,?,?)',
              (user_id, json.dumps(portfolio), json.dumps(advice), charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def get_usage_summary():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT action, COUNT(*), SUM(charge) FROM usage GROUP BY action')
    rows = c.fetchall()
    conn.close()
    return rows

def get_recent_analyses(limit=5):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT id, user_id, portfolio_json, advice_json, charge, ts FROM analyses ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
# -----------------------------------------------

# --------- Market helpers (yfinance) -----------
@st.cache_data(ttl=30)
def fetch_history(ticker, period="6mo", interval="1d"):
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=20)
def get_latest_price(ticker):
    try:
        t = yf.Ticker(ticker)
        # try info then fallback to last close
        info = t.fast_info if hasattr(t, "fast_info") else {}
        price = info.get("last_price") or info.get("lastTradePriceOnly") or info.get("regularMarketPrice")
        if price is None:
            df = fetch_history(ticker, period="5d")
            if not df.empty:
                price = float(df['Close'].iloc[-1])
        return float(price) if price is not None else None
    except Exception:
        return None
# -----------------------------------------------

# --------- Technical indicators & advice -------
def add_indicators(df):
    df = df.copy()
    if 'Close' not in df.columns:
        return df
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def simple_stock_advice(ticker, qty, buy_price, timeframe='long'):
    """
    Simple, beginner-friendly rules:
      - MA20 > MA50 -> uptrend (positive)
      - RSI < 30 -> oversold (buy signal)
      - RSI > 70 -> overbought (sell signal)
    Returns dict with recommendation + explanation.
    """
    result = {"ticker": ticker.upper(), "quantity": qty, "buyPrice": buy_price}
    df = fetch_history(ticker, period="1y")
    if df.empty:
        result.update({"currentPrice": None, "pnl_pct": None, "recommendation": "No data", "explanation": "No historical data found."})
        return result
    df = add_indicators(df)
    current = get_latest_price(ticker)
    result["currentPrice"] = round(current, 2) if current else None
    pnl_pct = ((current - buy_price) / buy_price * 100) if (current and buy_price and buy_price>0) else None
    result["pnl_pct"] = round(pnl_pct,2) if pnl_pct is not None else None

    # derive signals
    last = df.iloc[-1]
    ma20 = last.get('MA20', None)
    ma50 = last.get('MA50', None)
    rsi = last.get('RSI', None)

    reasons = []
    score = 0
    if ma20 is not None and ma50 is not None:
        if ma20 > ma50:
            score += 1
            reasons.append("Short-term trend (MA20 > MA50) looks up.")
        else:
            score -= 1
            reasons.append("Short-term trend (MA20 <= MA50) looks weak.")
    if rsi is not None:
        if rsi < 30:
            score += 1
            reasons.append(f"RSI={rsi:.1f} (oversold) — potential buying opportunity.")
        elif rsi > 70:
            score -= 1
            reasons.append(f"RSI={rsi:.1f} (overbought) — consider taking profit or waiting.")

    # PnL considerations (simple)
    if pnl_pct is not None:
        if timeframe == 'short':
            if pnl_pct <= -5:
                score -= 1; reasons.append("Short-term loss >5% — consider stop-loss.")
            if pnl_pct >= 8:
                score += 1; reasons.append("Short-term gain >8% — consider booking partial profits.")
        else:
            if pnl_pct <= -20:
                score -= 2; reasons.append("Long-term loss >20% — review position.")
            if pnl_pct >= 50:
                score += 2; reasons.append("Long-term gain >50% — consider rebalancing.")

    # map score to recommendation
    if score >= 2:
        rec = "Buy / Strong Buy"
    elif score == 1:
        rec = "Buy"
    elif score == 0:
        rec = "Hold"
    elif score == -1:
        rec = "Reduce / Take Profit"
    else:
        rec = "Sell / Cut Loss"

    explanation = " ".join(reasons) if reasons else "No strong technical signal; consider your horizon & risk tolerance."
    result.update({"recommendation": rec, "explanation": explanation})
    return result
# -----------------------------------------------

# --------- Charts ----------
def plot_price_with_ma(df, ticker):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'))
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20'))
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50'))
    fig.update_layout(title=f"{ticker} Price & Moving Averages", xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_rsi(df):
    fig = go.Figure()
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
        fig.update_layout(title="RSI (14)", yaxis=dict(range=[0,100]))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
    return fig
# --------------------------------

# --------- PDF invoice helper ----------
def create_pdf_invoice(user_id, analysis_id, portfolio, advices, charge):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stock Consultant Agent - Invoice", ln=True, align='C')
    pdf.ln(6)
    pdf.cell(200, 8, txt=f"Analysis ID: {analysis_id}  User: {user_id}", ln=True)
    pdf.cell(200, 8, txt=f"Timestamp: {datetime.utcnow().isoformat()}", ln=True)
    pdf.cell(200, 8, txt=f"Charge (mock): {charge}", ln=True)
    pdf.ln(6)
    pdf.cell(200, 8, txt="Portfolio:", ln=True)
    for s in portfolio.get('stocks', []):
        pdf.cell(200, 6, txt=f"- {s['name']} | qty: {s['quantity']} | buyPrice: {s['buyPrice']}", ln=True)
    pdf.ln(4)
    pdf.cell(200, 8, txt="Advice summary:", ln=True)
    for a in advices:
        pdf.multi_cell(0, 6, txt=f"{a['ticker']}: {a['recommendation']} — {a['explanation']}")
    b = BytesIO()
    pdf.output(b)
    b.seek(0)
    return b
# --------------------------------------

# --------- Streamlit UI (main) ----------
def main():
    st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
    init_db()

    st.title("Stock Consultant Agent — Beginner Friendly")
    st.markdown("**Disclaimer:** This app provides educational guidance. Not financial advice. Always verify before trading.")

    # Top row: controls + watchlist
    col1, col2 = st.columns([3,1])
    with col2:
        st.header("Controls")
        refresh_sec = st.slider("Polling (seconds)", min_value=10, max_value=120, value=DEFAULT_REFRESH, step=5)
        st.write("Charges (mock):", f"Per analysis = {PER_ANALYSIS_CHARGE}, Per stock advice = {PER_ADVICE_CHARGE}")
        if st.button("Show Usage & Recent Analyses"):
            rows = get_usage_summary()
            if rows:
                st.table(pd.DataFrame(rows, columns=['action','count','total_charge']))
            else:
                st.info("No usage yet.")
            recent = get_recent_analyses(5)
            if recent:
                st.subheader("Recent Analyses")
                for r in recent:
                    aid, uid, pjson, ajson, ch, ts = r
                    st.markdown(f"**#{aid}** user={uid} charge={ch} ts={ts}")
    with col1:
        st.subheader("Watchlist (quick prices)")
        wl_text = st.text_input("Tickers (comma separated)", value="AAPL, MSFT, GOOGL")
        tickers = [t.strip().upper() for t in wl_text.split(",") if t.strip()]
        if tickers:
            data = []
            for t in tickers:
                price = get_latest_price(t)
                data.append({"Ticker": t, "Price": price})
            st.table(pd.DataFrame(data))

    st.markdown("---")
    # Portfolio input & analysis
    left, right = st.columns([2,1])
    with left:
        st.header("Your Portfolio")
        st.write("You can upload CSV with columns: name,quantity,buyPrice OR use the example and edit.")
        with st.form("portfolio_form", clear_on_submit=False):
            user_id = st.text_input("User ID", value="you123")
            uploaded = st.file_uploader("Upload CSV (optional)", type=['csv'])
            use_example = st.checkbox("Load example portfolio", value=True)
            if uploaded:
                try:
                    df = pd.read_csv(uploaded)[['name','quantity','buyPrice']]
                except Exception:
                    st.error("CSV must include columns: name,quantity,buyPrice")
                    df = pd.DataFrame(columns=['name','quantity','buyPrice'])
            else:
                if use_example:
                    df = pd.DataFrame([{"name":"AAPL","quantity":10,"buyPrice":150},
                                       {"name":"INFY.NS","quantity":5,"buyPrice":1200}])
                else:
                    df = pd.DataFrame(columns=['name','quantity','buyPrice'])
            st.dataframe(df)
            timeframe = st.selectbox("Advice timeframe", ["short","long"], index=1)
            analyze = st.form_submit_button("Analyze Portfolio")
        if analyze:
            # build portfolio and run advice
            portfolio = {"user_id": user_id, "stocks": df.to_dict(orient='records')}
            advices = []
            total_value = 0.0
            for s in portfolio['stocks']:
                name = s.get('name')
                qty = float(s.get('quantity') or 0)
                buy = float(s.get('buyPrice') or 0)
                adv = simple_stock_advice(name, qty, buy, timeframe)
                advices.append(adv)
                # compute approximate value
                cur = adv.get('currentPrice') or get_latest_price(name) or 0
                total_value += cur * qty
            total_value = round(total_value,2)
            st.subheader("Portfolio Summary")
            st.metric("Total (approx)", f"{total_value}")
            # positions table
            df_adv = pd.DataFrame(advices)
            if not df_adv.empty:
                st.dataframe(df_adv[['ticker','quantity','buyPrice','currentPrice','pnl_pct','recommendation']])
                # per-stock expanders with charts
                for a in advices:
                    with st.expander(f"{a['ticker']} — {a['recommendation']}"):
                        st.write("Current price:", a.get('currentPrice'))
                        st.write("P/L %:", a.get('pnl_pct'))
                        st.write("Explanation:", a.get('explanation'))
                        # show charts
                        hist = fetch_history(a['ticker'], period="6mo")
                        if not hist.empty:
                            hist = add_indicators(hist)
                            st.plotly_chart(plot_price_with_ma(hist, a['ticker']), use_container_width=True)
                            st.plotly_chart(plot_rsi(hist), use_container_width=True)
            else:
                st.info("No valid positions found.")

            # billing & save
            n = len(advices)
            charge = PER_ANALYSIS_CHARGE + PER_ADVICE_CHARGE * n
            st.write(f"Estimated charge (mock): {charge}")
            if st.button("Save analysis & download invoice"):
                save_analysis(user_id, portfolio, advices, charge)
                log_usage("analysis_saved", f"user={user_id},count={n}", charge)
                st.success("Saved to database (SQLite).")
                # get latest analysis id
                recent = get_recent_analyses(1)
                aid = recent[0][0] if recent else None
                if aid:
                    pdfb = create_pdf_invoice(user_id, aid, portfolio, advices, charge)
                    st.download_button("Download PDF Invoice", data=pdfb, file_name=f"invoice_{aid}.pdf", mime="application/pdf")
                st.download_button("Download analysis JSON", data=json.dumps({"portfolio":portfolio,"advice":advices,"total":total_value}, indent=2), file_name="analysis.json")

    with right:
        st.header("Search & Learn (single stock)")
        search = st.text_input("Enter ticker (e.g. AAPL or INFY.NS)", value="AAPL")
        if st.button("Lookup"):
            hist = fetch_history(search, period="6mo")
            if hist.empty:
                st.error("No data found for ticker.")
            else:
                hist = add_indicators(hist)
                st.plotly_chart(plot_price_with_ma(hist, search), use_container_width=True)
                st.plotly_chart(plot_rsi(hist), use_container_width=True)
                # show quick advice for 1 share
                quick = simple_stock_advice(search, qty=1, buy_price=(hist['Close'].iloc[-1]), timeframe='long')
                st.subheader("Quick Advice")
                st.write("Recommendation:", quick.get('recommendation'))
                st.write("Why:", quick.get('explanation'))
                st.info("Beginner tip: " + beginner_tip(quick.get('recommendation')))

    st.markdown("---")
    st.header("Beginner Guide: How to invest (simple)")
    st.markdown("""
    - Decide your **goal** and **time horizon** (short <1yr, medium 1-3yr, long >3yr).  
    - Start small. Use the app to learn signals (MA, RSI) but always check fundamentals.  
    - Diversify: avoid more than 30-40% in one stock.  
    - Use stop-loss for short-term trades; review position sizing for risk control.
    """)
    st.markdown("**Helpful steps:** 1) Learn basics 2) Paper-trade 3) Start small 4) Rebalance yearly")

# small helper for plain-language tips
def beginner_tip(recommendation):
    tips = {
        "Buy / Strong Buy": "This means technical signals are favorable. If you're new, consider buying small amounts over time (dollar-cost averaging).",
        "Buy": "Consider buying gradually—avoid investing all at once.",
        "Hold": "Mixed signals. If long-term investor, you can hold and re-evaluate later.",
        "Reduce / Take Profit": "Consider booking profits or lowering position size.",
        "Sell / Cut Loss": "Technical signals are negative — consider closing to protect capital."
    }
    return tips.get(recommendation, "Use your judgement; consult trusted sources.")

if __name__ == "__main__":
    main()


  
   

   
  

           
       
          
    
    



       
             
           
