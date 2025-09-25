# app_voice.py
"""
Voice-guided Stock Consultant Agent (Streamlit)
- Wizard-style voice guidance with gTTS (text -> mp3 -> st.audio).
- Portfolio upload/manual, Pathway/mock CSV override, watchlist, charts, advice, billing, invoices.
- Uses yfinance for history & quotes; TwelveData optional via TWELVEDATA_API_KEY env var.
- Persistence: SQLite local (simple).
"""

import os
import json
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

# Voice (gTTS)
try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

# Optional TwelveData key (set in env if available)
TWELVE_KEY = os.environ.get("TWELVEDATA_API_KEY")

# App configuration
SQLITE_FILE = "app_voice_data.db"
PER_ADVICE = float(os.environ.get("PER_ADVICE", 0.1))
PER_ANALYSIS = float(os.environ.get("PER_ANALYSIS", 1.0))

st.set_page_config(page_title="Voice Stock Consultant Agent", layout="wide")

# -----------------------------
# Persistence helpers (SQLite)
# -----------------------------
def init_db():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
      CREATE TABLE IF NOT EXISTS usage (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        action TEXT, detail TEXT, charge REAL, ts TEXT
      )
    ''')
    c.execute('''
      CREATE TABLE IF NOT EXISTS analyses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, portfolio_json TEXT, advice_json TEXT, charge REAL, ts TEXT
      )
    ''')
    conn.commit(); conn.close()

def log_usage(action: str, detail: str = "", charge: float = 0.0):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO usage (action, detail, charge, ts) VALUES (?,?,?,?)",
              (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_analysis(user_id: str, portfolio: dict, advice: List[dict], charge: float):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO analyses (user_id, portfolio_json, advice_json, charge, ts) VALUES (?,?,?,?,?)",
              (user_id, json.dumps(portfolio), json.dumps(advice), charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def fetch_recent_analysis(limit=5):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id,user_id,portfolio_json,advice_json,charge,ts FROM analyses ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

def fetch_usage_summary():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT action, COUNT(*), SUM(charge) FROM usage GROUP BY action")
    rows = c.fetchall()
    conn.close()
    return rows

# initialize DB at startup
init_db()

# -----------------------------
# Market data helpers
# -----------------------------
@st.cache_data(ttl=20)
def fetch_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
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

def quote_twelvedata(symbol: str) -> Optional[dict]:
    if not TWELVE_KEY:
        return None
    try:
        r = requests.get("https://api.twelvedata.com/quote", params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=5)
        j = r.json()
        if j.get("status") == "error":
            return None
        return {"price": float(j.get("price")) if j.get("price") else None, "pct": float(j.get("percent_change")) if j.get("percent_change") else None}
    except Exception:
        return None

@st.cache_data(ttl=10)
def quick_yf_quote(symbol: str) -> dict:
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", {})
        price = info.get("last_price") or info.get("regularMarketPrice")
        if price is None:
            df = fetch_history(symbol, period="5d")
            if not df.empty:
                price = float(df['Close'].iloc[-1])
        return {"price": float(price) if price else None, "pct": None}
    except Exception:
        return {"price": None, "pct": None}

def get_quote(symbol: str) -> dict:
    if TWELVE_KEY:
        q = quote_twelvedata(symbol)
        if q and q.get("price") is not None:
            return q
    return quick_yf_quote(symbol)

# -----------------------------
# Pathway/mock CSV loader
# -----------------------------
def load_pathway_csv(uploaded_file) -> Dict[str, float]:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return {}
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if 'symbol' in df.columns and 'price' in df.columns:
        df2 = df[['symbol','price']].dropna()
        df2['symbol'] = df2['symbol'].astype(str).str.upper()
        mapping = dict(zip(df2['symbol'], df2['price'].astype(float)))
        return mapping
    return {}

# -----------------------------
# Indicators & advice
# -----------------------------
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or 'Close' not in df.columns:
        return df
    df = df.copy()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()
    delta = df['Close'].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def compute_signals(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    last = df.iloc[-1]
    signals = {}
    if not np.isnan(last.get('MA20', np.nan)) and not np.isnan(last.get('MA50', np.nan)):
        signals['ma'] = 'bullish' if last['MA20'] > last['MA50'] else 'bearish'
    rsi = last.get('RSI')
    if rsi is not None:
        if rsi < 30: signals['rsi'] = ('oversold', round(rsi,2))
        elif rsi > 70: signals['rsi'] = ('overbought', round(rsi,2))
        else: signals['rsi'] = ('neutral', round(rsi,2))
    return signals

def advice_for_stock(symbol: str, qty: float, buy_price: float, timeframe: str = 'long', override_price: float = None) -> dict:
    df = fetch_history(symbol, period="1y")
    if df.empty:
        return {"symbol": symbol, "error": "no data"}
    df_i = add_indicators(df)
    signals = compute_signals(df_i)
    q = get_quote(symbol)
    current = override_price if override_price is not None else q.get("price")
    pnl_pct = ((current - buy_price) / buy_price * 100) if (current and buy_price and buy_price>0) else None

    score = 0; reasons=[]
    if signals.get('ma') == 'bullish':
        score += 2; reasons.append("Short-term MA trend positive (MA20 > MA50).")
    elif signals.get('ma') == 'bearish':
        score -= 2; reasons.append("Short-term MA trend negative (MA20 ≤ MA50).")
    rsi_info = signals.get('rsi')
    if rsi_info:
        tag, val = rsi_info
        if tag == 'oversold':
            score += 1; reasons.append(f"RSI {val} (oversold).")
        elif tag == 'overbought':
            score -= 1; reasons.append(f"RSI {val} (overbought).")
    # PnL adjustments by timeframe
    if pnl_pct is not None:
        if timeframe == 'short':
            if pnl_pct <= -5:
                score -= 1; reasons.append("Short-term loss >5% — consider stop-loss.")
            if pnl_pct >= 8:
                score += 1; reasons.append("Short-term gain >8% — consider booking profits.")
        else:
            if pnl_pct <= -20:
                score -= 2; reasons.append("Long-term loss >20% — review position.")
            if pnl_pct >= 50:
                score += 2; reasons.append("Long-term gain >50% — consider rebalancing.")
    if score >= 3: rec = "Strong Buy"
    elif score == 2: rec = "Buy"
    elif score >= 0: rec = "Hold"
    elif score >= -2: rec = "Reduce / Take Profit"
    else: rec = "Sell / Cut Loss"
    explanation = " ".join(reasons) if reasons else "No strong technical signal; check fundamentals & horizon."
    return {"symbol": symbol, "quantity": qty, "buyPrice": buy_price, "current": round(current,2) if current else None,
            "pnl_pct": round(pnl_pct,2) if pnl_pct is not None else None, "signals": signals,
            "recommendation": rec, "explanation": explanation}

# -----------------------------
# Plotly figure helpers
# -----------------------------
def fig_price_ma(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'))
    if 'MA20' in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20'))
    if 'MA50' in df.columns: fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50'))
    fig.update_layout(title=f"{symbol} Price & MA", xaxis_title="Date", yaxis_title="Price", height=420)
    return fig

def fig_candlestick(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(title=f"{symbol} Candlestick", xaxis_rangeslider_visible=False, height=420)
    return fig

def fig_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.update_layout(title="RSI (14)", height=240)
    return fig

# -----------------------------
# PDF invoice and voice helpers
# -----------------------------
def make_invoice_pdf(user_id: str, analysis_id: int, portfolio: dict, advices: List[dict], charge: float) -> BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stock Consultant Agent - Invoice", ln=True, align='C')
    pdf.ln(4)
    pdf.cell(200, 8, txt=f"Analysis ID: {analysis_id}  User: {user_id}", ln=True)
    pdf.cell(200, 8, txt=f"Timestamp: {datetime.utcnow().isoformat()}", ln=True)
    pdf.cell(200, 8, txt=f"Charge (mock): {charge:.2f}", ln=True)
    pdf.ln(6)
    pdf.cell(200, 8, txt="Portfolio:", ln=True)
    for s in portfolio.get('stocks', []):
        pdf.cell(200, 6, txt=f"- {s['name']} | qty: {s['quantity']} | buyPrice: {s['buyPrice']}", ln=True)
    pdf.ln(4)
    pdf.cell(200, 8, txt="Advice:", ln=True)
    for a in advices:
        pdf.multi_cell(0, 6, txt=f"{a['symbol']}: {a['recommendation']} - {a['explanation']}")
    buf = BytesIO()
    pdf.output(buf)
    buf.seek(0)
    return buf

def speak_text(text: str) -> Optional[BytesIO]:
    if not HAVE_GTTS:
        return None
    try:
        tts = gTTS(text=text, lang='en')
        b = BytesIO(); tts.write_to_fp(b); b.seek(0)
        return b
    except Exception:
        return None

# -----------------------------
# Voice-guided wizard UI
# -----------------------------
def play_voice(text: str):
    """Generate TTS and display an audio player. Returns True if played."""
    audio = speak_text(text)
    if audio:
        st.audio(audio.read(), format='audio/mp3')
        return True
    else:
        # fallback: show text if voice not available
        st.info(text)
        return False

def wizard():
    st.title("Voice-guided Stock Consultant Agent — Beginner friendly")
    st.write("This is a guided workflow. The app will speak instructions and suggestions. Use headphones for best demo experience.")
    st.caption("Tip: set TWELVEDATA_API_KEY in environment for better near-real-time quotes (optional).")

    # session state for wizard step
    if 'step' not in st.session_state:
        st.session_state.step = 'welcome'
        st.session_state.portfolio = None
        st.session_state.pathway_map = {}
        st.session_state.advices = None
        st.session_state.charge = 0.0

    # ---------- step: welcome ----------
    if st.session_state.step == 'welcome':
        st.header("Welcome")
        st.write("Hello! I'm your Stock Consultant Agent. I will guide you step-by-step.")
        if st.button("Start guided session"):
            play_voice("Welcome. Please enter your portfolio. You can upload a CSV or use the example.")
            st.session_state.step = 'portfolio'
            st.experimental_rerun()
        st.markdown("---")
        st.subheader("Or use dashboard (skip wizard)")
        if st.button("Open full dashboard"):
            st.session_state.step = 'dashboard'
            st.experimental_rerun()
        return

    # ---------- step: portfolio ----------
    if st.session_state.step == 'portfolio':
        st.header("Step 1 — Enter your portfolio")
        st.write("Upload a CSV with columns: name,quantity,buyPrice or use the example and edit the table.")
        with st.form("portfolio_form"):
            user_id = st.text_input("Your name / user id", value="demo_user")
            uploaded = st.file_uploader("Portfolio CSV (optional)", type=['csv'])
            example = st.checkbox("Load example portfolio", value=True)
            if uploaded:
                try:
                    df_port = pd.read_csv(uploaded)[['name','quantity','buyPrice']]
                except Exception:
                    st.error("CSV must have columns: name,quantity,buyPrice")
                    df_port = pd.DataFrame(columns=['name','quantity','buyPrice'])
            else:
                if example:
                    df_port = pd.DataFrame([{"name":"AAPL","quantity":5,"buyPrice":150},{"name":"INFY.NS","quantity":10,"buyPrice":1200}])
                else:
                    df_port = pd.DataFrame(columns=['name','quantity','buyPrice'])
            st.dataframe(df_port)
            proceed = st.form_submit_button("Proceed to goals")
        if proceed:
            st.session_state.portfolio = {"user_id": user_id, "stocks": df_port.to_dict(orient='records')}
            play_voice("Got it. Next question: are you looking for short-term or long-term investments?")
            st.session_state.step = 'goals'
            st.experimental_rerun()
        st.button("Back to welcome", on_click=lambda: set_step('welcome'))
        return

    # ---------- step: goals ----------
    if st.session_state.step == 'goals':
        st.header("Step 2 — Investment horizon & demo feed")
        st.write("Choose your investment timeframe (this affects advice).")
        timeframe = st.radio("Choose timeframe:", options=['short','long'], index=1, help="Short: days/weeks. Long: months/years.")
        st.markdown("Optional: Upload a Pathway/mock CSV (symbol,price) to use as live demo prices.")
        pathway = st.file_uploader("Pathway/mock CSV (optional)", type=['csv'])
        if pathway is not None:
            mapping = load_pathway_csv(pathway)
            if mapping:
                st.success("Pathway/mock demo feed loaded.")
                st.session_state.pathway_map = mapping
                play_voice("Demo feed loaded. I will use these prices during analysis.")
            else:
                st.warning("Could not read Pathway CSV — ensure columns 'symbol' and 'price' exist.")
        if st.button("Proceed to market overview"):
            st.session_state.timeframe = timeframe
            play_voice(f"You selected {timeframe} timeframe. Now I will show market overview and your watchlist.")
            st.session_state.step = 'market'
            st.experimental_rerun()
        st.button("Back to portfolio", on_click=lambda: set_step('portfolio'))
        return

    # ---------- step: market overview ----------
    if st.session_state.step == 'market':
        st.header("Step 3 — Market overview & watchlist")
        st.write("Here's a quick watchlist and market snapshot.")
        wl = st.text_input("Watchlist tickers (comma separated)", value="AAPL, MSFT, INFY.NS")
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        rows=[]
        for tk in tickers:
            # if pathway override exists, show that price
            override = st.session_state.pathway_map.get(tk.upper()) if st.session_state.pathway_map else None
            if override is not None:
                rows.append({"Ticker": tk, "Price": override, "Source": "Pathway override"})
            else:
                q = get_quote(tk)
                rows.append({"Ticker": tk, "Price": round(q.get('price'),2) if q.get('price') else None, "Source":"yfinance/TwelveData"})
        st.table(pd.DataFrame(rows))
        play_voice("This is the current market snapshot for your watchlist. When you are ready, I will analyze your portfolio.")
        if st.button("Analyze my portfolio now"):
            st.session_state.step = 'analyze'
            st.experimental_rerun()
        st.button("Back to goals", on_click=lambda: set_step('goals'))
        return

    # ---------- step: analyze ----------
    if st.session_state.step == 'analyze':
        st.header("Step 4 — Analyzing portfolio")
        portfolio = st.session_state.portfolio
        timeframe = st.session_state.get('timeframe','long')
        pathway_map = st.session_state.get('pathway_map', {})
        advices=[]
        total_value = 0.0
        play_voice("Analyzing your portfolio now. This may take a few seconds.")
        for pos in portfolio['stocks']:
            sym = pos['name']
            qty = float(pos['quantity'] or 0)
            bp = float(pos['buyPrice'] or 0)
            override = pathway_map.get(sym.upper()) if pathway_map else None
            a = advice_for_stock(sym, qty, bp, timeframe, override_price=override)
            advices.append(a)
            cur = a.get('current') or 0.0
            total_value += cur * qty
        st.session_state.advices = advices
        st.session_state.total_value = round(total_value,2)
        # Summarize in voice
        speak_lines = [f"I analyzed {len(advices)} holdings. Your approximate portfolio value is {st.session_state.total_value}."]
        # mention top recommendations in short english
        buy_count = sum(1 for a in advices if a.get('recommendation','').lower().startswith('buy'))
        sell_count = sum(1 for a in advices if a.get('recommendation','').lower().startswith('sell'))
        if buy_count>0:
            speak_lines.append(f"{buy_count} positions show buy signals.")
        if sell_count>0:
            speak_lines.append(f"{sell_count} positions show sell signals.")
        play_voice(" ".join(speak_lines))
        st.success("Analysis complete.")
        st.button("View results", on_click=lambda: set_step('results'))
        st.button("Back to market overview", on_click=lambda: set_step('market'))
        return

    # ---------- step: results ----------
    if st.session_state.step == 'results':
        st.header("Step 5 — Results & Advice (plain English)")
        advices = st.session_state.get('advices', [])
        st.metric("Estimated portfolio value", st.session_state.get('total_value', 0.0))
        if advices:
            df_adv = pd.DataFrame(advices)
            show_cols = [c for c in ['symbol','quantity','buyPrice','current','pnl_pct','recommendation'] if c in df_adv.columns]
            st.dataframe(df_adv[show_cols])
            for a in advices:
                with st.expander(f"{a.get('symbol')} — {a.get('recommendation')}"):
                    st.write("Current:", a.get('current'))
                    st.write("P/L %:", a.get('pnl_pct'))
                    st.write("Signals:", a.get('signals'))
                    st.write("Why:", a.get('explanation'))
                    hist = fetch_history(a.get('symbol'), period="6mo")
                    if not hist.empty:
                        hist_i = add_indicators(hist)
                        st.plotly_chart(fig_price_ma(hist_i, a.get('symbol')), use_container_width=True)
                        st.plotly_chart(fig_candlestick(hist_i, a.get('symbol')), use_container_width=True)
                        st.plotly_chart(fig_rsi(hist_i), use_container_width=True)
                    # voice for this stock
                    if HAVE_GTTS and st.button(f"Play voice for {a.get('symbol')}", key=f"voice_res_{a.get('symbol')}"):
                        txt = f"For {a.get('symbol')}, my recommendation is {a.get('recommendation')}. {a.get('explanation')}"
                        audio = speak_text(txt)
                        if audio:
                            st.audio(audio.read(), format="audio/mp3")
                        else:
                            st.warning("Voice not available.")
        else:
            st.info("No advice items found.")
        # Ask to bill/save
        n = sum(1 for a in advices if not a.get('error'))
        charge = round(PER_ANALYSIS + PER_ADVICE * n,2)
        st.write(f"Session charge (mock): {charge}")
        if st.button("Save session & generate invoice"):
            save_analysis(st.session_state.portfolio.get('user_id','guest'), st.session_state.portfolio, advices, charge)
            log_usage("analysis_saved", f"user={st.session_state.portfolio.get('user_id','guest')},count={n}", charge)
            st.success("Saved analysis. Invoice ready.")
            recent = fetch_recent_analysis(1)
            if recent:
                aid = recent[0][0]
                pdfb = make_invoice_pdf(st.session_state.portfolio.get('user_id','guest'), aid, st.session_state.portfolio, advices, charge)
                st.download_button("Download invoice (PDF)", data=pdfb, file_name=f"invoice_{aid}.pdf", mime="application/pdf")
            st.download_button("Download analysis (JSON)", data=json.dumps({"portfolio":st.session_state.portfolio,"advice":advices,"total":st.session_state.total_value}, indent=2),
                                file_name="analysis.json")
            play_voice("Saved the analysis and generated an invoice. Thank you for using the Stock Consultant Agent.")
        if st.button("Back to analyze again"):
            st.session_state.step = 'analyze'
            st.experimental_rerun()
        if st.button("Finish session and return to welcome"):
            st.session_state.step = 'welcome'
            st.experimental_rerun()
        return

    # ---------- dashboard (non-wizard full view) ----------
    if st.session_state.step == 'dashboard':
        st.header("Full Dashboard (non-guided)")
        st.write("This view contains all controls and features on one page.")
        # Watchlist
        st.subheader("Watchlist")
        wl = st.text_input("Tickers", value="AAPL, MSFT, INFY.NS")
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        rows=[]
        for tk in tickers:
            ov = st.session_state.get('pathway_map',{}).get(tk)
            if ov is not None:
                rows.append({"Ticker": tk, "Price": ov, "Source":"Pathway override"})
            else:
                q = get_quote(tk)
                rows.append({"Ticker": tk, "Price": round(q.get('price'),2) if q.get('price') else None, "Source":"yfinance/TwelveData"})
        st.table(pd.DataFrame(rows))
        # Portfolio upload + analyze
        st.subheader("Portfolio (full dashboard)")
        with st.form("dash_pf"):
            user = st.text_input("User ID", value="demo_user")
            uploaded = st.file_uploader("Upload CSV", type=['csv'])
            example = st.checkbox("Load example", value=True)
            if uploaded:
                try:
                    dfp = pd.read_csv(uploaded)[['name','quantity','buyPrice']]
                except Exception:
                    st.error("Invalid CSV format")
                    dfp = pd.DataFrame(columns=['name','quantity','buyPrice'])
            else:
                dfp = pd.DataFrame([{"name":"AAPL","quantity":5,"buyPrice":150},{"name":"INFY.NS","quantity":10,"buyPrice":1200}]) if example else pd.DataFrame(columns=['name','quantity','buyPrice'])
            st.dataframe(dfp)
            tf = st.selectbox("Timeframe", options=['short','long'], index=1)
            go_btn = st.form_submit_button("Analyze")
        if go_btn:
            advs=[]
            total=0.0
            pm = st.session_state.get('pathway_map',{})
            for r in dfp.to_dict(orient='records'):
                o = pm.get(r['name'].upper()) if pm else None
                a = advice_for_stock(r['name'], float(r['quantity']), float(r['buyPrice']), tf, override_price=o)
                advs.append(a)
                cur = a.get('current') or 0
                total += cur * a.get('quantity',0)
            st.success(f"Analysis produced for {len(advs)} stocks. Total approx value: {round(total,2)}")
            st.dataframe(pd.DataFrame(advs)[['symbol','quantity','buyPrice','current','pnl_pct','recommendation']])
            if st.button("Play summary voice"):
                summary = f"I analyzed {len(advs)} positions. Portfolio approx value {round(total,2)} rupees."
                play_voice(summary)
        # Usage & tips
        st.subheader("Usage & Billing")
        rows = fetch_usage_summary()
        if rows:
            st.table(pd.DataFrame(rows, columns=['action','count','total_charge']))
        else:
            st.info("No usage yet.")
        st.subheader("Beginner's quick guide")
        st.markdown("1. Define goal & horizon. 2. Start small. 3. Diversify. 4. Use stop-loss for short trades.")
        if st.button("Back to wizard"):
            st.session_state.step = 'welcome'
            st.experimental_rerun()
        return

# small helper to set step externally
def set_step(step_name: str):
    st.session_state.step = step_name

# -----------------------------
# Plot helper wrappers (used in wizard)
# -----------------------------
def fig_price_ma(df, s): return fig_price_ma  # placeholders not used

# -----------------------------
# Run the app
# -----------------------------
if __name__ == "__main__":
    wizard()
