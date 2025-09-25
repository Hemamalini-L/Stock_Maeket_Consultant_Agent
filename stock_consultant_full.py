# stock_consultant_full.py
"""
Stock Consultant Agent — Full feature app (single file)
- Watchlist (yfinance fallback; TwelveData optional)
- Portfolio upload/manual + Pathway mock feed override
- Indicators: MA20, MA50, RSI
- Advice engine (explainable)
- Interactive Plotly charts (price+MA, candlestick, RSI)
- Wizard-style voice guidance (gTTS) + audio playback
- Billing (mock): PER_ANALYSIS + PER_ADVICE, save usage
- Save analyses to SQLite (default) or MongoDB (optional MONGO_URI)
- Download JSON analysis + PDF invoice
- Beginner-friendly UI & demo script
"""

import os
import json
import math
import sqlite3
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Optional

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from fpdf import FPDF

# Optional packages
try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

# Optional MongoDB persistence (set MONGO_URI env var)
MONGO_URI = os.environ.get("MONGO_URI")
USE_MONGO = False
if MONGO_URI:
    try:
        from pymongo import MongoClient
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command("ping")
        db_mongo = client.get_default_database()
        coll_analyses = db_mongo.get_collection("analyses")
        coll_usage = db_mongo.get_collection("usage")
        USE_MONGO = True
    except Exception:
        USE_MONGO = False

# Optional TwelveData key for near-real-time quotes
TWELVE_KEY = os.environ.get("TWELVEDATA_API_KEY")

# Pricing defaults (overridable via env)
PER_ADVICE = float(os.environ.get("PER_ADVICE", 0.10))
PER_ANALYSIS = float(os.environ.get("PER_ANALYSIS", 1.00))

# SQLite file
SQLITE_FILE = "stock_consultant_data.db"

# Streamlit page config
st.set_page_config(page_title="Stock Consultant Agent", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Persistence helpers
# ---------------------------
def init_sqlite():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
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

def log_usage_sql(action: str, detail: str = "", charge: float = 0.0):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO usage (action, detail, charge, ts) VALUES (?,?,?,?)",
              (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def save_analysis_sql(user_id: str, portfolio: dict, advice: list, charge: float):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO analyses (user_id, portfolio_json, advice_json, charge, ts) VALUES (?,?,?,?,?)",
              (user_id, json.dumps(portfolio), json.dumps(advice), charge, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_usage_sql():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT action, COUNT(*), SUM(charge) FROM usage GROUP BY action")
    rows = c.fetchall()
    conn.close()
    return rows

def get_recent_analyses_sql(limit=5):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, user_id, portfolio_json, advice_json, charge, ts FROM analyses ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

# Unified persistence functions (Mongo optional)
def init_db():
    if USE_MONGO:
        # nothing to create — collections auto-create
        return
    else:
        init_sqlite()

def log_usage(action: str, detail: str = "", charge: float = 0.0):
    if USE_MONGO:
        try:
            coll_usage.insert_one({"action": action, "detail": detail, "charge": charge, "ts": datetime.utcnow()})
            return
        except Exception:
            pass
    log_usage_sql(action, detail, charge)

def save_analysis(user_id: str, portfolio: dict, advice: list, charge: float):
    if USE_MONGO:
        try:
            coll_analyses.insert_one({"user_id": user_id, "portfolio": portfolio, "advice": advice, "charge": charge, "ts": datetime.utcnow()})
            return
        except Exception:
            pass
    save_analysis_sql(user_id, portfolio, advice, charge)

def get_usage_summary():
    if USE_MONGO:
        try:
            rows = list(coll_usage.aggregate([{"$group":{"_id":"$action","count":{"$sum":1},"total":{"$sum":"$charge"}}}]))
            return [(r["_id"], r["count"], r["total"]) for r in rows]
        except Exception:
            return get_usage_sql()
    else:
        return get_usage_sql()

def get_recent_analyses(limit=5):
    if USE_MONGO:
        try:
            rows = list(coll_analyses.find().sort([("_id",-1)]).limit(limit))
            out=[]
            for r in rows:
                out.append((str(r.get("_id")), r.get("user_id"), r.get("portfolio"), r.get("advice"), r.get("charge"), r.get("ts")))
            return out
        except Exception:
            return get_recent_analyses_sql(limit)
    else:
        return get_recent_analyses_sql(limit)

# Initialize persistence
init_db()

# ---------------------------
# Market data functions
# ---------------------------
@st.cache_data(ttl=20)
def fetch_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, interval=interval)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    except Exception:
        return pd.DataFrame()

def quote_twelvedata(symbol: str) -> Optional[dict]:
    if not TWELVE_KEY:
        return None
    try:
        r = requests.get("https://api.twelvedata.com/quote", params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=6)
        j = r.json()
        if j.get("status") == "error":
            return None
        return {"price": float(j.get("price")) if j.get("price") else None, "pct": float(j.get("percent_change")) if j.get("percent_change") else None}
    except Exception:
        return None

@st.cache_data(ttl=10)
def quick_yf_price(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", {})
        price = info.get("last_price") or info.get("regularMarketPrice")
        if price is None:
            df = fetch_history(symbol, period="5d")
            if not df.empty:
                price = float(df["Close"].iloc[-1])
        return {"price": float(price) if price else None, "pct": None}
    except Exception:
        return {"price": None, "pct": None}

def get_quote(symbol: str) -> dict:
    if TWELVE_KEY:
        q = quote_twelvedata(symbol)
        if q and q.get("price") is not None:
            return q
    return quick_yf_price(symbol)

# ---------------------------
# Pathway/mock CSV loader
# ---------------------------
def load_pathway_csv(uploaded_file) -> Dict[str,float]:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return {}
    df.columns = [c.lower() for c in df.columns]
    if "symbol" in df.columns and "price" in df.columns:
        df2 = df[["symbol","price"]].dropna()
        df2["symbol"] = df2["symbol"].astype(str).str.upper()
        mapping = dict(zip(df2["symbol"], df2["price"].astype(float)))
        return mapping
    return {}

# ---------------------------
# Indicators & Advice Engine
# ---------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "Close" not in df.columns:
        return df
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    df["RSI"] = 100 - (100 / (1 + rs))
    return df

def compute_signals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]
    signals = {}
    if not np.isnan(last.get("MA20", np.nan)) and not np.isnan(last.get("MA50", np.nan)):
        signals["ma"] = "bullish" if last["MA20"] > last["MA50"] else "bearish"
    rsi = last.get("RSI")
    if rsi is not None:
        if rsi < 30:
            signals["rsi"] = ("oversold", round(rsi,2))
        elif rsi > 70:
            signals["rsi"] = ("overbought", round(rsi,2))
        else:
            signals["rsi"] = ("neutral", round(rsi,2))
    return signals

def analyze_stock(symbol: str, qty: float, buy_price: float, timeframe: str = "long", override_price: Optional[float] = None) -> dict:
    df = fetch_history(symbol, period="1y")
    if df.empty:
        return {"symbol": symbol, "error": "no historical data"}
    df_i = add_indicators(df)
    signals = compute_signals(df_i)
    q = get_quote(symbol)
    current = override_price if override_price is not None else q.get("price")
    pnl_pct = ((current - buy_price)/buy_price*100) if (current and buy_price and buy_price>0) else None

    score = 0
    reasons = []
    ma = signals.get("ma")
    if ma == "bullish":
        score += 2; reasons.append("Short-term trend positive (MA20 > MA50).")
    elif ma == "bearish":
        score -= 2; reasons.append("Short-term trend negative (MA20 ≤ MA50).")
    rsi_info = signals.get("rsi")
    if rsi_info:
        tag,val = rsi_info
        if tag == "oversold":
            score += 1; reasons.append(f"RSI {val} (oversold).")
        elif tag == "overbought":
            score -= 1; reasons.append(f"RSI {val} (overbought).")
    # PnL adjustments
    if pnl_pct is not None:
        if timeframe == "short":
            if pnl_pct <= -5:
                score -= 1; reasons.append("Short-term loss >5% — consider stop-loss.")
            if pnl_pct >= 8:
                score += 1; reasons.append("Short-term gain >8% — consider booking partial profits.")
        else:
            if pnl_pct <= -20:
                score -= 2; reasons.append("Long-term loss >20% — review position.")
            if pnl_pct >= 50:
                score += 2; reasons.append("Long-term gain >50% — consider rebalancing.")
    if score >= 3:
        rec = "Strong Buy"
    elif score == 2:
        rec = "Buy"
    elif score >= 0:
        rec = "Hold"
    elif score >= -2:
        rec = "Reduce / Take Profit"
    else:
        rec = "Sell / Cut Loss"
    explanation = " ".join(reasons) if reasons else "No clear technical signal; consider fundamentals & your horizon."
    return {"symbol": symbol, "quantity": qty, "buyPrice": buy_price, "current": round(current,2) if current else None,
            "pnl_pct": round(pnl_pct,2) if pnl_pct is not None else None, "signals": signals,
            "recommendation": rec, "explanation": explanation}

# ---------------------------
# Plotly figure creators
# ---------------------------
def fig_price_with_ma(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], name="Close", mode="lines"))
    if "MA20" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA20"], name="MA20"))
    if "MA50" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["MA50"], name="MA50"))
    fig.update_layout(title=f"{symbol} Price & MAs", xaxis_title="Date", yaxis_title="Price", height=440)
    return fig

def fig_candlestick(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"])])
    fig.update_layout(title=f"{symbol} Candlestick", xaxis_rangeslider_visible=False, height=440)
    return fig

def fig_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if "RSI" in df.columns:
        fig.add_trace(go.Scatter(x=df["Date"], y=df["RSI"], name="RSI"))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.update_layout(title="RSI (14)", height=240)
    return fig

# ---------------------------
# Invoice + voice helpers
# ---------------------------
def make_invoice_pdf(user_id: str, analysis_id: int, portfolio: dict, advices: List[dict], charge: float) -> BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stock Consultant Agent - Invoice", ln=True, align="C")
    pdf.ln(4)
    pdf.cell(200, 8, txt=f"Analysis ID: {analysis_id}  User: {user_id}", ln=True)
    pdf.cell(200, 8, txt=f"Timestamp: {datetime.utcnow().isoformat()}", ln=True)
    pdf.cell(200, 8, txt=f"Charge (mock): {charge:.2f}", ln=True)
    pdf.ln(6)
    pdf.cell(200, 8, txt="Portfolio:", ln=True)
    for s in portfolio.get("stocks", []):
        pdf.cell(200, 6, txt=f"- {s['name']} | qty: {s['quantity']} | buyPrice: {s['buyPrice']}", ln=True)
    pdf.ln(4)
    pdf.cell(200, 8, txt="Advice summary:", ln=True)
    for a in advices:
        pdf.multi_cell(0, 6, txt=f"{a['symbol']}: {a['recommendation']} — {a['explanation']}")
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

def speak_text(text: str) -> Optional[BytesIO]:
    if not HAVE_GTTS:
        return None
    try:
        tts = gTTS(text=text, lang="en")
        b = BytesIO()
        tts.write_to_fp(b)
        b.seek(0)
        return b
    except Exception:
        return None

def play_voice(text: str):
    audio = speak_text(text)
    if audio:
        st.audio(audio.read(), format="audio/mp3")
    else:
        # fallback: show text bubble so user can read if voice not available
        st.info(text)

# ---------------------------
# Beginner-friendly helpers
# ---------------------------
def beginner_guide_md() -> str:
    return """
**Beginner's quick guide**
1. Decide your goal & horizon (Short-term vs Long-term).  
2. Try paper-trading first.  
3. Start small; diversify across sectors (avoid >30-40% concentration).  
4. Use stop-loss on short-term trades; hold and review fundamentals for long-term.  
5. Rebalance periodically and keep learning.
"""

def demo_script_md() -> str:
    return """
Demo script for jury:
1. Start guided session (play voice instructions).  
2. Upload example portfolio -> select timeframe -> analyze.  
3. Expand a few stocks to show charts and voice tips.  
4. Save analysis -> download PDF invoice (shows billing flow).  
5. Lookup single ticker to show interactive chart & advice.
"""

# ---------------------------
# Wizard-style guided flow + dashboard
# ---------------------------
def set_step(step: str):
    st.session_state.step = step

def wizard_ui():
    st.title("Stock Consultant Agent — Guided (Voice + Interactive)")
    st.caption("Beginner-friendly. Not financial advice. This demo uses yfinance (or TwelveData when configured).")

    # initialize session_state
    if "step" not in st.session_state:
        st.session_state.step = "welcome"
        st.session_state.portfolio = None
        st.session_state.pathway_map = {}
        st.session_state.advices = []
        st.session_state.timeframe = "long"
        st.session_state.total_value = 0.0

    # --- Welcome ---
    if st.session_state.step == "welcome":
        st.header("Welcome")
        st.write("Hello! I will guide you step-by-step. Click start to begin.")
        if st.button("Start guided session"):
            play_voice("Welcome to the Stock Consultant Agent. Please upload your portfolio or use the example.")
            set_step("portfolio")
            st.experimental_rerun()
        st.markdown("---")
        st.subheader("Or open the full dashboard (non-guided)")
        if st.button("Open dashboard"):
            set_step("dashboard")
            st.experimental_rerun()
        return

    # --- Portfolio step ---
    if st.session_state.step == "portfolio":
        st.header("Step 1 — Enter your portfolio")
        st.write("Upload a CSV (columns: name,quantity,buyPrice) or use example and edit.")
        with st.form("portfolio_form"):
            user_id = st.text_input("Your name / user id", value="demo_user")
            uploaded = st.file_uploader("Upload portfolio CSV (optional)", type=["csv"])
            use_example = st.checkbox("Load example portfolio", value=True)
            if uploaded:
                try:
                    df_port = pd.read_csv(uploaded)[["name","quantity","buyPrice"]]
                except Exception:
                    st.error("CSV must include columns: name,quantity,buyPrice")
                    df_port = pd.DataFrame(columns=["name","quantity","buyPrice"])
            else:
                if use_example:
                    df_port = pd.DataFrame([{"name":"AAPL","quantity":5,"buyPrice":150},{"name":"INFY.NS","quantity":10,"buyPrice":1200}])
                else:
                    df_port = pd.DataFrame(columns=["name","quantity","buyPrice"])
            st.dataframe(df_port)
            proceed = st.form_submit_button("Proceed to Goals")
        if proceed:
            st.session_state.portfolio = {"user_id": user_id, "stocks": df_port.to_dict(orient="records")}
            play_voice("Great. Next: choose your timeframe. Short-term or long-term?")
            set_step("goals")
            st.experimental_rerun()
        if st.button("Back to Welcome"):
            set_step("welcome")
            st.experimental_rerun()
        return

    # --- Goals step ---
    if st.session_state.step == "goals":
        st.header("Step 2 — Investment horizon & demo feed")
        st.write("Choose timeframe and optionally upload a Pathway/mock CSV that provides demo live prices.")
        timeframe = st.radio("Choose timeframe", options=["short","long"], index=1)
        pathway_file = st.file_uploader("Optional: Pathway/mock CSV (symbol,price) — used as live override", type=["csv"])
        if pathway_file is not None:
            pm = load_pathway_csv(pathway_file)
            if pm:
                st.session_state.pathway_map = pm
                st.success("Pathway demo feed loaded and will override live quotes during analysis.")
                play_voice("Demo feed loaded. I will use these prices for the analysis.")
            else:
                st.warning("Pathway CSV invalid — needs columns 'symbol' and 'price'.")
        if st.button("Proceed to Market Overview"):
            st.session_state.timeframe = timeframe
            play_voice(f"You chose {timeframe} timeframe. Now I'll show a quick market snapshot.")
            set_step("market")
            st.experimental_rerun()
        if st.button("Back to Portfolio"):
            set_step("portfolio")
            st.experimental_rerun()
        return

    # --- Market step ---
    if st.session_state.step == "market":
        st.header("Step 3 — Market overview (watchlist)")
        st.write("Quick watchlist. You can edit tickers and then analyze portfolio.")
        wl = st.text_input("Watchlist tickers (comma separated)", value="AAPL, MSFT, INFY.NS")
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        rows=[]
        pm = st.session_state.get("pathway_map", {})
        for tk in tickers:
            ov = pm.get(tk) if pm else None
            if ov is not None:
                rows.append({"Ticker": tk, "Price": ov, "Source": "Pathway override"})
            else:
                q = get_quote(tk)
                rows.append({"Ticker": tk, "Price": round(q.get("price"),2) if q.get("price") else None, "Source": "yfinance/TwelveData"})
        st.table(pd.DataFrame(rows))
        play_voice("This is your quick watchlist snapshot. Click analyze when ready.")
        if st.button("Analyze my portfolio now"):
            set_step("analyze")
            st.experimental_rerun()
        if st.button("Back to Goals"):
            set_step("goals")
            st.experimental_rerun()
        return

    # --- Analyze step ---
    if st.session_state.step == "analyze":
        st.header("Step 4 — Analyzing portfolio")
        portfolio = st.session_state.portfolio or {"user_id":"guest","stocks":[]}
        timeframe = st.session_state.timeframe or "long"
        pm = st.session_state.pathway_map or {}
        play_voice("Analyzing portfolio now. This may take a few seconds.")
        advices=[]
        total_value=0.0
        for pos in portfolio["stocks"]:
            sym = pos.get("name")
            qty = float(pos.get("quantity") or 0)
            bp = float(pos.get("buyPrice") or 0)
            override = pm.get(sym.upper()) if pm else None
            a = analyze_stock(sym, qty, bp, timeframe, override_price=override)
            advices.append(a)
            cur = a.get("current") or 0.0
            total_value += cur * qty
        st.session_state.advices = advices
        st.session_state.total_value = round(total_value,2)
        # quick spoken summary
        buy_count = sum(1 for a in advices if a.get("recommendation","").lower().startswith("buy"))
        sell_count = sum(1 for a in advices if a.get("recommendation","").lower().startswith("sell"))
        summary = f"Analysis done. Portfolio approx value {st.session_state.total_value}."
        if buy_count:
            summary += f" {buy_count} positions show buy signals."
        if sell_count:
            summary += f" {sell_count} positions show sell signals."
        play_voice(summary)
        st.success("Analysis complete — view results next.")
        if st.button("View Results"):
            set_step("results")
            st.experimental_rerun()
        if st.button("Back to Market Overview"):
            set_step("market")
            st.experimental_rerun()
        return

    # --- Results step ---
    if st.session_state.step == "results":
        st.header("Step 5 — Results & Advice (plain English)")
        advices = st.session_state.get("advices", [])
        st.metric("Portfolio value (approx)", st.session_state.get("total_value", 0.0))
        if advices:
            df_adv = pd.DataFrame(advices)
            cols = [c for c in ["symbol","quantity","buyPrice","current","pnl_pct","recommendation"] if c in df_adv.columns]
            st.dataframe(df_adv[cols])
            for a in advices:
                with st.expander(f"{a.get('symbol')} — {a.get('recommendation')}"):
                    st.write("Current:", a.get("current"))
                    st.write("P/L %:", a.get("pnl_pct"))
                    st.write("Signals:", a.get("signals"))
                    st.write("Why:", a.get("explanation"))
                    hist = fetch_history(a.get("symbol"), period="6mo")
                    if not hist.empty:
                        hist_i = add_indicators(hist)
                        st.plotly_chart(fig_price_with_ma(hist_i, a.get("symbol")), use_container_width=True)
                        st.plotly_chart(fig_candlestick(hist_i, a.get("symbol")), use_container_width=True)
                        st.plotly_chart(fig_rsi(hist_i), use_container_width=True)
                    # voice play per stock
                    if HAVE_GTTS and st.button(f"Play voice for {a.get('symbol')}", key=f"voice_result_{a.get('symbol')}"):
                        txt = f"For {a.get('symbol')}, recommended action: {a.get('recommendation')}. {a.get('explanation')}"
                        audio = speak_text(txt)
                        if audio:
                            st.audio(audio.read(), format="audio/mp3")
                        else:
                            st.warning("Voice not available.")
        else:
            st.info("No advice available.")
        # billing & save
        n = sum(1 for a in advices if not a.get("error"))
        charge = round(PER_ANALYSIS + PER_ADVICE * n, 2)
        st.write(f"Session (mock) charge: {charge}")
        if st.button("Save analysis & generate invoice"):
            user = st.session_state.portfolio.get("user_id","guest")
            save_analysis(user, st.session_state.portfolio, advices, charge)
            log_usage("analysis_saved", f"user={user},count={n}", charge)
            st.success("Saved analysis.")
            recent = get_recent_analyses(1)
            if recent:
                aid = recent[0][0]
                pdfb = make_invoice_pdf(user, aid, st.session_state.portfolio, advices, charge)
                st.download_button("Download invoice (PDF)", data=pdfb, file_name=f"invoice_{aid}.pdf", mime="application/pdf")
            st.download_button("Download analysis (JSON)", data=json.dumps({"portfolio":st.session_state.portfolio,"advice":advices,"total":st.session_state.total_value}, indent=2),
                                file_name="analysis.json")
            play_voice("Saved analysis and generated invoice. Thank you.")
        if st.button("Analyze again"):
            set_step("analyze")
            st.experimental_rerun()
        if st.button("Finish session (go to welcome)"):
            set_step("welcome")
            st.experimental_rerun()
        return

    # --- Dashboard (non-wizard) ---
    if st.session_state.step == "dashboard":
        st.header("Full Dashboard — All features")
        # top row: watchlist & controls
        c1, c2 = st.columns([3,1])
        with c2:
            st.write("Provider:", "TwelveData (if configured)" if TWELVE_KEY else "yfinance (fallback)")
            st.write("Persistence:", "MongoDB" if USE_MONGO else "SQLite (local)")
            if st.button("Refresh (clear cache)"):
                st.cache_data.clear()
                play_voice("Cache cleared. UI refreshed.")
                st.experimental_rerun()
        with c1:
            st.subheader("Watchlist")
            wl = st.text_input("Tickers (comma separated)", value="AAPL, MSFT, INFY.NS")
            tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
            rows=[]
            pm = st.session_state.get("pathway_map", {})
            for tk in tickers:
                ov = pm.get(tk) if pm else None
                if ov is not None:
                    rows.append({"Ticker": tk, "Price": ov, "Source": "Pathway override"})
                else:
                    q = get_quote(tk)
                    rows.append({"Ticker": tk, "Price": round(q.get("price"),2) if q.get("price") else None, "Source": "yfinance/TwelveData"})
            st.table(pd.DataFrame(rows))
        st.markdown("---")
        # Portfolio quick
        st.subheader("Quick Portfolio (upload & analyze)")
        with st.form("dash_form"):
            user = st.text_input("User id", value="demo_user")
            upload = st.file_uploader("Upload portfolio CSV (optional)", type=["csv"])
            example = st.checkbox("Load example", value=True)
            if upload:
                try:
                    dfp = pd.read_csv(upload)[["name","quantity","buyPrice"]]
                except Exception:
                    st.error("CSV must include columns: name,quantity,buyPrice")
                    dfp = pd.DataFrame(columns=["name","quantity","buyPrice"])
            else:
                dfp = pd.DataFrame([{"name":"AAPL","quantity":5,"buyPrice":150},{"name":"INFY.NS","quantity":10,"buyPrice":1200}]) if example else pd.DataFrame(columns=["name","quantity","buyPrice"])
            st.dataframe(dfp)
            tf = st.selectbox("Timeframe", options=["short","long"], index=1)
            run = st.form_submit_button("Analyze")
        if run:
            advs=[]
            total=0.0
            pm = st.session_state.get("pathway_map", {})
            for r in dfp.to_dict(orient="records"):
                override = pm.get(r["name"].upper()) if pm else None
                a = analyze_stock(r["name"], float(r["quantity"]), float(r["buyPrice"]), tf, override_price=override)
                advs.append(a)
                cur = a.get("current") or 0.0
                total += cur * a.get("quantity", 0)
            st.success(f"Analysis done. approx value {round(total,2)}")
            st.dataframe(pd.DataFrame(advs)[["symbol","quantity","buyPrice","current","pnl_pct","recommendation"]])
            if st.button("Play short summary voice"):
                play_voice(f"Analyzed {len(advs)} stocks. Portfolio approx value {round(total,2)}.")
        st.markdown("---")
        # usage & beginner guide
        st.subheader("Usage & Billing")
        rows = get_usage_summary()
        if rows:
            dfu = pd.DataFrame(rows, columns=["action","count","total_charge"])
            st.table(dfu)
            try:
                st.write("Total mock charges:", round(dfu["total_charge"].sum(),2))
            except Exception:
                pass
        else:
            st.info("No usage recorded yet.")
        st.subheader("Beginner's Guide & Demo Script")
        st.markdown(beginner_guide_md())
        st.code(demo_script_md())
        if st.button("Back to wizard"):
            set_step("welcome")
            st.experimental_rerun()
        return

# Run app
def main():
    wizard_ui()

if __name__ == "__main__":
    main()
