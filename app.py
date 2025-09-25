# app_realtime_full.py
"""
Stock Market Consultant Agent — Real-time demo (beginner focused)
Features:
 - Watchlist (near-real-time quotes via TwelveData or yfinance fallback)
 - Portfolio input (manual + CSV) and portfolio analysis
 - Per-stock advice (MA crossover, RSI, MACD heuristics) with plain-English reasoning
 - Interactive charts (Plotly) and candlestick view
 - Billing simulation (per analysis + per advice), save analyses to MongoDB (optional) or SQLite
 - PDF invoice generation and JSON export
 - Voice guidance via gTTS (optional)
 - Beginner-friendly copy, tooltips, and guided UI
"""

import os
import time
import json
import math
import sqlite3
from datetime import datetime
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import requests
import yfinance as yf
import plotly.graph_objects as go
from fpdf import FPDF
from gtts import gTTS

# ----------------- CONFIG -----------------
TWELVE_KEY = os.environ.get("TWELVEDATA_API_KEY")  # optional; recommended
MONGO_URI = os.environ.get("MONGO_URI")            # optional; if provided, app persists to MongoDB
USE_MONGO = bool(MONGO_URI)
DB_SQLITE = "app_realtime.db"
PER_ADVICE = 0.1
PER_ANALYSIS = 1.0
REFRESH_DEFAULT = 15  # seconds
# ------------------------------------------

# ---------- optional pymongo use ----------
try:
    if USE_MONGO:
        from pymongo import MongoClient
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # test connection
        mongo_client.admin.command('ping')
        db_mongo = mongo_client.get_default_database()
        coll_analyses = db_mongo.get_collection("analyses")
        coll_usage = db_mongo.get_collection("usage")
    else:
        coll_analyses = coll_usage = None
except Exception as e:
    # fallback to sqlite if Mongo connection fails
    st = None  # avoid linter confusion
    USE_MONGO = False
    coll_analyses = coll_usage = None

# ------------- DB helpers (SQLite fallback) -------------
def init_sqlite():
    conn = sqlite3.connect(DB_SQLITE, check_same_thread=False)
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

def log_usage_sql(action, detail="", charge=0.0):
    conn = sqlite3.connect(DB_SQLITE, check_same_thread=False)
    c = conn.cursor()
    c.execute('INSERT INTO usage (action, detail, charge, ts) VALUES (?,?,?,?)',
              (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def save_analysis_sql(user_id, portfolio, advice, charge):
    conn = sqlite3.connect(DB_SQLITE, check_same_thread=False)
    c = conn.cursor()
    c.execute('INSERT INTO analyses (user_id, portfolio_json, advice_json, charge, ts) VALUES (?,?,?,?,?)',
              (user_id, json.dumps(portfolio), json.dumps(advice), charge, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_usage_sql():
    conn = sqlite3.connect(DB_SQLITE, check_same_thread=False)
    c = conn.cursor()
    c.execute('SELECT action, COUNT(*), SUM(charge) FROM usage GROUP BY action')
    rows = c.fetchall()
    conn.close()
    return rows

def get_recent_analyses_sql(limit=5):
    conn = sqlite3.connect(DB_SQLITE, check_same_thread=False)
    c = conn.cursor()
    c.execute('SELECT id, user_id, portfolio_json, advice_json, charge, ts FROM analyses ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows
# -----------------------------------------------------

# --------------- Unified persistence functions ---------------
def init_db():
    if USE_MONGO:
        # no-op: Mongo collections auto-create
        pass
    else:
        init_sqlite()

def log_usage(action, detail="", charge=0.0):
    if USE_MONGO:
        try:
            coll_usage.insert_one({"action": action, "detail": detail, "charge": charge, "ts": datetime.utcnow()})
        except Exception:
            log_usage_sql(action, detail, charge)
    else:
        log_usage_sql(action, detail, charge)

def save_analysis(user_id, portfolio, advice, charge):
    if USE_MONGO:
        try:
            coll_analyses.insert_one({"user_id": user_id, "portfolio": portfolio, "advice": advice, "charge": charge, "ts": datetime.utcnow()})
        except Exception:
            save_analysis_sql(user_id, portfolio, advice, charge)
    else:
        save_analysis_sql(user_id, portfolio, advice, charge)

def get_usage_summary():
    if USE_MONGO:
        try:
            rows = list(coll_usage.aggregate([{"$group":{"_id":"$action","count":{"$sum":1},"total":{"$sum":"$charge"}}}]))
            return [(r['_id'], r['count'], r['total']) for r in rows]
        except Exception:
            return get_usage_sql()
    else:
        return get_usage_sql()

def get_recent_analyses(limit=5):
    if USE_MONGO:
        try:
            rows = list(coll_analyses.find().sort([("_id", -1)]).limit(limit))
            out = []
            for r in rows:
                out.append((str(r.get("_id")), r.get("user_id"), r.get("portfolio"), r.get("advice"), r.get("charge"), r.get("ts")))
            return out
        except Exception:
            return get_recent_analyses_sql(limit)
    else:
        return get_recent_analyses_sql(limit)
# ---------------------------------------------------------------

# ----------------- Market data: TwelveData or yfinance -----------------
def quote_twelvedata(symbol):
    if not TWELVE_KEY:
        return None
    try:
        url = "https://api.twelvedata.com/quote"
        r = requests.get(url, params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=6)
        data = r.json()
        if data.get("status") == "error":
            return None
        return {
            "price": float(data.get("price")) if data.get("price") else None,
            "percent_change": float(data.get("percent_change")) if data.get("percent_change") else None,
            "timestamp": data.get("timestamp")
        }
    except Exception:
        return None

@st.cache_data(ttl=30)
def fetch_history(symbol, period="6mo", interval="1d"):
    try:
        t = yf.Ticker(symbol)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=15)
def quick_price_yf(symbol):
    try:
        t = yf.Ticker(symbol)
        info = t.fast_info if hasattr(t, "fast_info") else {}
        price = info.get("last_price") or info.get("regularMarketPrice")
        if price is None:
            df = fetch_history(symbol, period="5d")
            if not df.empty:
                price = float(df['Close'].iloc[-1])
        return {"price": float(price) if price else None, "pct": None}
    except Exception:
        return {"price": None, "pct": None}

def get_realtime(symbol):
    # prefer TwelveData
    if TWELVE_KEY:
        q = quote_twelvedata(symbol)
        if q and q.get("price") is not None:
            return q
    # fallback
    q = quick_price_yf(symbol)
    return {"price": q.get("price"), "percent_change": q.get("pct"), "timestamp": None}
# -----------------------------------------------------------------------

# ----------------- Indicators & signals -----------------
def add_indicators(df):
    df = df.copy()
    if 'Close' not in df.columns:
        return df
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    df['RSI'] = 100 - (100 / (1 + rs))
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_SIGNAL'] = df['MACD'].ewm(span=9, adjust=False).mean()
    return df

def compute_signals(df):
    if df.empty:
        return {}
    last = df.iloc[-1]
    signals = {}
    # MA cross
    if not np.isnan(last.get('MA20', np.nan)) and not np.isnan(last.get('MA50', np.nan)):
        signals['ma'] = 'bullish' if last['MA20'] > last['MA50'] else 'bearish'
    # RSI
    rsi = last.get('RSI')
    if rsi is not None:
        if rsi < 30:
            signals['rsi'] = ('oversold', rsi)
        elif rsi > 70:
            signals['rsi'] = ('overbought', rsi)
        else:
            signals['rsi'] = ('neutral', rsi)
    # MACD
    macd = last.get('MACD'); macdsig = last.get('MACD_SIGNAL')
    if macd is not None and macdsig is not None:
        signals['macd'] = 'bullish' if macd > macdsig else 'bearish'
    return signals

def stock_advice(symbol, qty, buy_price, timeframe='long'):
    df = fetch_history(symbol, period="1y")
    if df.empty:
        return {"symbol": symbol, "error": "No data"}
    df = add_indicators(df)
    signals = compute_signals(df)
    q = get_realtime(symbol)
    current = q.get("price")
    pnl_pct = ((current - buy_price)/buy_price*100) if (current and buy_price and buy_price>0) else None

    score = 0; reasons = []
    # MA
    ma = signals.get('ma')
    if ma == 'bullish':
        score += 2; reasons.append("Short-term MA trend positive.")
    elif ma == 'bearish':
        score -= 2; reasons.append("Short-term MA trend negative.")
    # RSI
    rsi_info = signals.get('rsi')
    if rsi_info:
        tag, r = rsi_info
        if tag == 'oversold':
            score += 1; reasons.append(f"RSI {r:.1f} (oversold).")
        elif tag == 'overbought':
            score -= 1; reasons.append(f"RSI {r:.1f} (overbought).")
    # MACD
    macd = signals.get('macd')
    if macd == 'bullish': score += 1
    elif macd == 'bearish': score -= 1

    # PnL thresholds
    if pnl_pct is not None:
        if timeframe == 'short':
            if pnl_pct <= -5: score -= 1; reasons.append("Short-term loss >5%")
            if pnl_pct >= 8: score += 1; reasons.append("Short-term gain >8%")
        else:
            if pnl_pct <= -20: score -= 2; reasons.append("Long-term loss >20%")
            if pnl_pct >= 50: score += 2; reasons.append("Long-term gain >50%")

    # map to advice
    if score >= 3: rec = "Strong Buy"
    elif score == 2: rec = "Buy"
    elif score >= 0: rec = "Hold"
    elif score >= -2: rec = "Reduce"
    else: rec = "Sell"

    explanation = "; ".join(reasons) if reasons else "No strong technical signals."
    return {
        "symbol": symbol,
        "current": round(current,2) if current else None,
        "qty": qty,
        "buy_price": buy_price,
        "pnl_pct": round(pnl_pct,2) if pnl_pct is not None else None,
        "signals": signals,
        "score": score,
        "recommendation": rec,
        "explanation": explanation
    }
# ---------------------------------------------------------

# --------------- UI helpers: charts, pdf, voice --------------
def plot_price(df, symbol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
    if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20'))
    if 'MA50' in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50'))
    fig.update_layout(title=f"{symbol} Price", xaxis_title="Date", yaxis_title="Price")
    return fig

def plot_candle(df, symbol):
    fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(title=f"{symbol} Candlestick", xaxis_rangeslider_visible=False)
    return fig

def create_invoice_pdf(user, aid, portfolio, advices, charge):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200,10,txt="Stock Consultant Agent - Invoice",ln=True,align='C')
    pdf.ln(6)
    pdf.cell(200,8,txt=f"Analysis ID: {aid}   User: {user}",ln=True)
    pdf.cell(200,8,txt=f"Timestamp: {datetime.utcnow().isoformat()}",ln=True)
    pdf.cell(200,8,txt=f"Charge (mock): {charge}",ln=True)
    pdf.ln(6)
    pdf.cell(200,8,txt="Portfolio:",ln=True)
    for s in portfolio.get("stocks", []):
        pdf.cell(200,6,txt=f"- {s['name']} | qty:{s['quantity']} | buy:{s['buyPrice']}",ln=True)
    pdf.ln(4)
    pdf.cell(200,8,txt="Advice:",ln=True)
    for a in advices:
        pdf.multi_cell(0,6,txt=f"{a['symbol']}: {a['recommendation']} -- {a['explanation']}")
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

def speak_text(text, lang="en"):
    # generate speech mp3 bytes via gTTS
    try:
        tts = gTTS(text=text, lang=lang)
        b = BytesIO()
        tts.write_to_fp(b)
        b.seek(0)
        return b
    except Exception:
        return None
# ---------------------------------------------------------------

# ------------------- Main Streamlit App -------------------------
def main():
    st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
    init_db()
    st.title("Stock Consultant Agent — Beginner-friendly Real-time Demo")
    st.markdown("**Disclaimer:** Educational only. Not financial advice.")

    # top controls
    col1, col2 = st.columns([3,1])
    with col2:
        st.header("Settings")
        refresh = st.slider("Refresh (seconds)", min_value=5, max_value=60, value=REFRESH_DEFAULT, step=5)
        st.write("Provider:", "TwelveData" if TWELVE_KEY else "yfinance (fallback)")
        st.write("Persistence:", "MongoDB" if USE_MONGO else "SQLite (local)")
        if st.button("Clear cache (dev)"):
            st.cache_data.clear()
            st.success("Cache cleared.")

    with col1:
        st.header("Quick Watchlist")
        wl = st.text_input("Comma separated tickers (e.g. AAPL, INFY.NS)", value="AAPL, MSFT, INFY.NS")
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        if tickers:
            rows = []
            for tk in tickers:
                q = get_realtime(tk)
                rows.append({"Ticker": tk, "Price": q.get("price"), "Change%": q.get("percent_change")})
            st.table(pd.DataFrame(rows))

    st.markdown("---")

    # Portfolio area
    left, right = st.columns([2,1])
    with left:
        st.header("Your Portfolio — Simple & Clear")
        st.write("Upload a CSV with columns: name,quantity,buyPrice, or use the example and edit.")
        with st.form("portfolio_form"):
            user = st.text_input("Your name / user id", value="student123")
            up = st.file_uploader("Upload CSV (optional)", type=["csv"])
            sample = st.checkbox("Load example portfolio", value=True)
            if up:
                try:
                    df_port = pd.read_csv(up)[['name','quantity','buyPrice']]
                except Exception:
                    st.error("CSV must include columns name,quantity,buyPrice")
                    df_port = pd.DataFrame(columns=['name','quantity','buyPrice'])
            else:
                if sample:
                    df_port = pd.DataFrame([{"name":"AAPL","quantity":5,"buyPrice":150},{"name":"INFY.NS","quantity":10,"buyPrice":1200}])
                else:
                    df_port = pd.DataFrame(columns=['name','quantity','buyPrice'])
            st.dataframe(df_port)
            timeframe = st.radio("Advice timeframe (how you think)", ("short","long"), index=1, help="Short = days/weeks, Long = months/years")
            analyze = st.form_submit_button("Analyze Portfolio")
        if analyze:
            portfolio = {"user": user, "stocks": df_port.to_dict(orient='records')}
            advs = []
            total_value = 0.0
            for p in portfolio['stocks']:
                sym = p['name']
                qty = float(p.get('quantity') or 0)
                bp = float(p.get('buyPrice') or 0)
                a = stock_advice(sym, qty, bp, timeframe)
                advs.append(a)
                cur = a.get('current') or 0
                total_value += cur * qty
            total_value = round(total_value,2)
            st.subheader("Portfolio Summary")
            st.metric("Total (approx)", f"{total_value}")
            if advs:
                df_adv = pd.DataFrame(advs)
                st.dataframe(df_adv[['symbol','qty','buy_price','current','pnl_pct','recommendation']])
                for a in advs:
                    with st.expander(f"{a['symbol']} — {a['recommendation']}"):
                        st.write("Current price:", a.get('current'))
                        st.write("P/L %:", a.get('pnl_pct'))
                        st.write("Signals:", a.get('signals'))
                        st.write("Why:", a.get('explanation'))
                        # chart
                        h = fetch_history(a['symbol'], period="6mo")
                        if not h.empty:
                            h = add_indicators(h)
                            st.plotly_chart(plot_price(h, a['symbol']), use_container_width=True)
                            st.plotly_chart(plot_candle(h, a['symbol']), use_container_width=True)
                        # voice
                        if st.button(f"Play voice tip for {a['symbol']}", key=f"voice_{a['symbol']}"):
                            txt = f"For {a['symbol']}, recommended action: {a['recommendation']}. {a['explanation']}"
                            mp3 = speak_text(txt)
                            if mp3:
                                st.audio(mp3.read(), format="audio/mp3")
                            else:
                                st.warning("Voice service unavailable.")

            # Billing & save
            n = len(advs)
            charge = PER_ANALYSIS + PER_ADVICE * n
            st.write(f"Estimated charge (mock): {charge:.2f}")
            if st.button("Save analysis & generate invoice"):
                save_analysis(user, portfolio, advs, charge)
                log_usage("analysis_saved", f"user={user},count={n}", charge)
                st.success("Saved.")
                recent = get_recent_analyses(limit=1)
                if recent:
                    aid = recent[0][0]
                    pdfb = create_invoice_pdf(user, aid, portfolio, advs, charge)
                    st.download_button("Download Invoice PDF", data=pdfb, file_name=f"invoice_{aid}.pdf", mime="application/pdf")
                st.download_button("Download analysis JSON", data=json.dumps({"portfolio":portfolio,"advice":advs}, indent=2), file_name="analysis.json")
    with right:
        st.header("Lookup & Learn — Single Ticker")
        query = st.text_input("Ticker (ex: AAPL or INFY.NS)", value="AAPL")
        if st.button("Lookup"):
            hist = fetch_history(query, period="1y")
            if hist.empty:
                st.error("No data found.")
            else:
                st.subheader(f"{query} Overview")
                hist = add_indicators(hist)
                st.plotly_chart(plot_price(hist, query), use_container_width=True)
                st.plotly_chart(plot_candle(hist, query), use_container_width=True)
                st.plotly_chart(pd.DataFrame({"Date":hist['Date'], "RSI":hist['RSI']}).set_index('Date').reset_index(), use_container_width=True)  # simple
                # quick advice for 1 share
                quick = stock_advice(query, 1, hist['Close'].iloc[-1], timeframe='long')
                st.metric("Live price", quick.get('current'))
                st.write("Recommendation:", quick.get('recommendation'))
                st.write("Why:", quick.get('explanation'))
                if st.button("Play voice guidance for this ticker"):
                    mp3 = speak_text(f"{query}. Recommendation: {quick.get('recommendation')}. {quick.get('explanation')}")
                    if mp3:
                        st.audio(mp3.read(), format="audio/mp3")
                    else:
                        st.warning("Voice unavailable.")
    st.markdown("---")
    st.header("Usage & Billing (mock)")
    rows = get_usage_summary()
    if rows:
        st.table(pd.DataFrame(rows, columns=['action','count','total_charge']))
    else:
        st.info("No usage yet.")

    st.markdown("---")
    st.header("Beginner's Guide (short & friendly)")
    st.markdown("""
    **Start here if you're new**
    1. Define a goal and time horizon (short/long).  
    2. Start with small amounts; paper-trade first.  
    3. Diversify — don't put >30-40% in a single stock.  
    4. Use Stop-loss for short-term trades.  
    5. For long-term, focus on fundamentals (earnings, growth).  
    6. Revisit your portfolio every month or quarter.
    """)
    st.caption("For production: integrate a paid streaming provider and a brokerage API for execution. This demo focuses on guidance & learning.")

if __name__ == "__main__":
    main()
