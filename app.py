# app_full_fixed.py
"""
Stock Consultant Agent — Clean & working single-file Streamlit app.

Features:
- Watchlist (near-real-time via yfinance; TwelveData optional)
- Portfolio input (manual / CSV) + Pathway/mock CSV override
- Per-stock & portfolio advice (Buy/Sell/Hold/Diversify) in plain English
- Interactive charts (Plotly): price, candlestick, MA20/MA50, RSI
- Usage tracking & mock billing (SQLite)
- Save analyses, download JSON & PDF invoice
- Optional voice guidance via gTTS (if installed)
- Beginner-friendly tips and demo script
"""

import os
import json
import sqlite3
from io import BytesIO
from datetime import datetime
from typing import Dict, List

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import requests
from fpdf import FPDF

# optional voice
try:
    from gtts import gTTS
    HAVE_GTTS = True
except Exception:
    HAVE_GTTS = False

# Optional TwelveData API key (set in env if you want)
TWELVE_KEY = os.environ.get("TWELVEDATA_API_KEY")

# App config
SQLITE_FILE = "app_data_fixed.db"
PER_ADVICE = float(os.environ.get("PER_ADVICE", 0.10))
PER_ANALYSIS = float(os.environ.get("PER_ANALYSIS", 1.00))

st.set_page_config(page_title="Stock Consultant Agent (Fixed)", layout="wide")

# ---------------------------
# Persistence (SQLite)
# ---------------------------
def init_sqlite():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS usage (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          action TEXT, detail TEXT, charge REAL, ts TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT, portfolio_json TEXT, advice_json TEXT, charge REAL, ts TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_usage(action: str, detail: str = "", charge: float = 0.0):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO usage (action, detail, charge, ts) VALUES (?,?,?,?)",
              (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def save_analysis_sql(user_id: str, portfolio: dict, advice: list, charge: float):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("INSERT INTO analyses (user_id, portfolio_json, advice_json, charge, ts) VALUES (?,?,?,?,?)",
              (user_id, json.dumps(portfolio), json.dumps(advice), charge, datetime.utcnow().isoformat()))
    conn.commit(); conn.close()

def get_usage_summary_sql():
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT action, COUNT(*), SUM(charge) FROM usage GROUP BY action")
    rows = c.fetchall(); conn.close()
    return rows

def get_recent_analyses_sql(limit=5):
    conn = sqlite3.connect(SQLITE_FILE, check_same_thread=False)
    c = conn.cursor()
    c.execute("SELECT id, user_id, portfolio_json, advice_json, charge, ts FROM analyses ORDER BY id DESC LIMIT ?",
              (limit,))
    rows = c.fetchall(); conn.close()
    return rows

# initialize DB
init_sqlite()

# ---------------------------
# Market data adapters (TwelveData optional, yfinance fallback)
# ---------------------------
@st.cache_data(ttl=20)
def fetch_history(ticker: str, period: str = "6mo", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical data via yfinance and return DataFrame with Date column."""
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

def quote_twelvedata(symbol: str):
    if not TWELVE_KEY:
        return None
    try:
        r = requests.get("https://api.twelvedata.com/quote", params={"symbol": symbol, "apikey": TWELVE_KEY}, timeout=5)
        j = r.json()
        if j.get("status") == "error":
            return None
        price = j.get("price")
        pct = j.get("percent_change")
        return {"price": float(price) if price else None, "pct": float(pct) if pct else None}
    except Exception:
        return None

@st.cache_data(ttl=10)
def quick_yf_quote(symbol: str):
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
    """Unified function: prefer TwelveData (if key) then yfinance fallback."""
    if TWELVE_KEY:
        q = quote_twelvedata(symbol)
        if q and q.get("price") is not None:
            return q
    return quick_yf_quote(symbol)

# ---------------------------
# Pathway/mock csv loader (optional)
# ---------------------------
def load_pathway_csv_bytes(uploaded_file) -> Dict[str, float]:
    """Accept uploaded file (BytesIO) and return dict mapping SYMBOL->PRICE"""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception:
        return {}
    # normalize lowercase columns
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    if 'symbol' in df.columns and 'price' in df.columns:
        df2 = df[['symbol','price']].dropna()
        df2['symbol'] = df2['symbol'].astype(str).str.upper()
        mapping = dict(zip(df2['symbol'], df2['price'].astype(float)))
        return mapping
    return {}

# ---------------------------
# Indicators & advice engine (simple, explainable)
# ---------------------------
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
    s = {}
    if not np.isnan(last.get('MA20', np.nan)) and not np.isnan(last.get('MA50', np.nan)):
        s['ma'] = 'bullish' if last['MA20'] > last['MA50'] else 'bearish'
    rsi = last.get('RSI')
    if rsi is not None:
        if rsi < 30: s['rsi'] = ('oversold', round(rsi,2))
        elif rsi > 70: s['rsi'] = ('overbought', round(rsi,2))
        else: s['rsi'] = ('neutral', round(rsi,2))
    # MACD optional could be added
    return s

def advice_for_stock(symbol: str, qty: float, buy_price: float, timeframe: str = 'long', live_override: float = None) -> dict:
    """Return advice dict for one stock."""
    df = fetch_history(symbol, period="1y")
    if df.empty:
        return {"symbol": symbol, "error": "no historical data"}
    df_ind = add_indicators(df)
    signals = compute_signals(df_ind)
    quote = get_quote(symbol)
    current = live_override if (live_override is not None) else quote.get("price")
    pnl_pct = ((current - buy_price) / buy_price * 100) if (current and buy_price and buy_price>0) else None

    score = 0
    reasons = []
    ma = signals.get('ma')
    if ma == 'bullish':
        score += 2; reasons.append("Short-term trend looks up (MA20 > MA50).")
    elif ma == 'bearish':
        score -= 2; reasons.append("Short-term trend looks down (MA20 ≤ MA50).")
    rsi_info = signals.get('rsi')
    if rsi_info:
        tag,val = rsi_info
        if tag == 'oversold':
            score += 1; reasons.append(f"RSI {val} (oversold).")
        elif tag == 'overbought':
            score -= 1; reasons.append(f"RSI {val} (overbought).")
    # pnl logic
    if pnl_pct is not None:
        if timeframe == 'short':
            if pnl_pct <= -5:
                score -= 1; reasons.append("Short-term loss >5% — consider stop-loss.")
            if pnl_pct >= 8:
                score += 1; reasons.append("Short-term gain >8% — consider booking partial profits.")
        else:
            if pnl_pct <= -20:
                score -= 2; reasons.append("Long-term loss >20% — review your position.")
            if pnl_pct >= 50:
                score += 2; reasons.append("Long-term gain >50% — consider rebalancing / taking profit.")

    # final mapping
    if score >= 3: rec = "Strong Buy"
    elif score == 2: rec = "Buy"
    elif score >= 0: rec = "Hold"
    elif score >= -2: rec = "Reduce / Take Profit"
    else: rec = "Sell / Cut Loss"
    explanation = " ".join(reasons) if reasons else "No strong technical signal. Consider fundamentals & horizon."
    return {"symbol": symbol, "quantity": qty, "buyPrice": buy_price, "current": round(current,2) if current else None,
            "pnl_pct": round(pnl_pct,2) if pnl_pct is not None else None, "signals": signals,
            "recommendation": rec, "explanation": explanation}

# ---------------------------
# Plotly Figures (always return Figure)
# ---------------------------
def fig_price_with_ma(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', mode='lines'))
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20'))
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50'))
    fig.update_layout(title=f"{symbol} Price (with MA)", xaxis_title="Date", yaxis_title="Price", height=450)
    return fig

def fig_candlestick(df: pd.DataFrame, symbol: str) -> go.Figure:
    fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                                         open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])
    fig.update_layout(title=f"{symbol} Candlestick", xaxis_rangeslider_visible=False, height=450)
    return fig

def fig_rsi(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if 'RSI' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name='RSI'))
        fig.add_hline(y=70, line_dash="dash", line_color="red")
        fig.add_hline(y=30, line_dash="dash", line_color="green")
        fig.update_layout(title="RSI (14)", height=250)
    return fig

# ---------------------------
# Invoice PDF and voice helpers
# ---------------------------
def make_invoice_pdf(user_id: str, analysis_id: int, portfolio: dict, advices: List[dict], charge: float) -> BytesIO:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Stock Consultant Agent - Invoice", ln=True, align='C')
    pdf.ln(4)
    pdf.cell(200, 8, txt=f"Analysis ID: {analysis_id}   User: {user_id}", ln=True)
    pdf.cell(200, 8, txt=f"Timestamp: {datetime.utcnow().isoformat()}", ln=True)
    pdf.cell(200, 8, txt=f"Charge (mock): {charge:.2f}", ln=True)
    pdf.ln(6)
    pdf.cell(200, 8, txt="Portfolio:", ln=True)
    for s in portfolio.get('stocks', []):
        pdf.cell(200, 6, txt=f"- {s['name']} | qty: {s['quantity']} | buyPrice: {s['buyPrice']}", ln=True)
    pdf.ln(4)
    pdf.cell(200, 8, txt="Advice Summary:", ln=True)
    for a in advices:
        pdf.multi_cell(0, 6, txt=f"{a['symbol']}: {a['recommendation']} -- {a['explanation']}")
    buf = BytesIO(); pdf.output(buf); buf.seek(0)
    return buf

def speak_text(text: str):
    if not HAVE_GTTS:
        return None
    try:
        tts = gTTS(text=text, lang='en')
        b = BytesIO(); tts.write_to_fp(b); b.seek(0)
        return b
    except Exception:
        return None

# ---------------------------
# UI helpers
# ---------------------------
def beginner_tips_md() -> str:
    return ("**Beginner steps:**\n\n"
            "1. Define goal & horizon (short/long).\n"
            "2. Paper-trade first.\n"
            "3. Diversify; avoid >30-40% in one stock.\n"
            "4. Use stop-loss for short-term trades.\n"
            "5. For long-term investing focus on fundamentals (earnings).\n")

def demo_script_md() -> str:
    return ("**Demo script (2-3 min):**\n"
            "1. Show Watchlist (live prices).\n"
            "2. Upload example portfolio -> Analyze.\n"
            "3. Show per-stock advice (plain English) and play voice tip.\n"
            "4. Save analysis -> Download PDF invoice (shows billing).\n"
            "5. Lookup a single ticker -> show charts and recommendation.\n")

# ---------------------------
# App: layout & logic
# ---------------------------
def main():
    st.title("Stock Consultant Agent — Beginner Friendly (Fixed)")
    st.markdown("**Problem:** Beginners need plain-English actionable advice and simpler UI. This demo uses live-ish data and explains what to do in simple steps.")
    st.info("Tip: For better near-real-time pricing set `TWELVEDATA_API_KEY` in environment; otherwise yfinance is used (delayed quotes).")

    # Top controls / Watchlist
    col1, col2 = st.columns([3,1])
    with col2:
        st.header("Controls")
        st.write("Provider:", "TwelveData (fast)" if TWELVE_KEY else "yfinance (fallback)")
        st.write("Persistence: SQLite (local)")
        refresh_hint = st.slider("Polling hint (seconds) — use Refresh now to update display", min_value=5, max_value=60, value=15, step=5)
        if st.button("Refresh now"):
            # clear caches and redraw; safe to call once on button press
            st.cache_data.clear()
            st.experimental_rerun()

    with col1:
        st.subheader("Watchlist")
        wl = st.text_input("Tickers (comma separated)", value="AAPL, MSFT, INFY.NS")
        tickers = [t.strip().upper() for t in wl.split(",") if t.strip()]
        if tickers:
            rows = []
            for s in tickers:
                q = get_quote(s)
                rows.append({"Ticker": s, "Price": round(q.get("price"),2) if q.get("price") else None, "Change%": q.get("pct")})
            st.table(pd.DataFrame(rows))

    st.markdown("---")

    # Pathway/mock CSV upload (optional)
    st.subheader("Optional: Upload Pathway/mock CSV (symbol,price) to use as a demo live feed")
    pathway_file = st.file_uploader("Pathway CSV (optional)", type=["csv"])
    pathway_map = {}
    if pathway_file is not None:
        pathway_map = load_pathway_csv_bytes(pathway_file)
        if pathway_map:
            st.success("Pathway/mock data loaded — these prices will override live quotes in analysis.")
        else:
            st.warning("Pathway CSV must contain columns 'symbol' and 'price' (case-insensitive).")

    st.markdown("---")

    # Portfolio input and analysis
    left, right = st.columns([2,1])
    with left:
        st.header("Portfolio — enter holdings (simple)")
        st.write("Upload CSV (name,quantity,buyPrice) or use the example and edit.")
        with st.form("portfolio_form"):
            user_id = st.text_input("Your name / user id", value="demo_user")
            uploaded = st.file_uploader("Portfolio CSV (optional)", type=["csv"])
            example = st.checkbox("Load example portfolio", value=True)
            if uploaded:
                try:
                    df_port = pd.read_csv(uploaded)[['name','quantity','buyPrice']]
                except Exception:
                    st.error("CSV must include columns: name,quantity,buyPrice")
                    df_port = pd.DataFrame(columns=['name','quantity','buyPrice'])
            else:
                if example:
                    df_port = pd.DataFrame([{"name":"AAPL","quantity":5,"buyPrice":150},{"name":"INFY.NS","quantity":10,"buyPrice":1200}])
                else:
                    df_port = pd.DataFrame(columns=['name','quantity','buyPrice'])
            st.dataframe(df_port)
            timeframe = st.selectbox("Advice timeframe", options=['short','long'], index=1, help="Short = days/weeks; Long = months/years")
            analyze = st.form_submit_button("Analyze portfolio")
        if analyze:
            portfolio = {"user_id": user_id, "stocks": df_port.to_dict(orient='records')}
            advices = []
            total_value = 0.0
            for pos in portfolio['stocks']:
                sym = pos['name']; qty = float(pos['quantity'] or 0); bp = float(pos['buyPrice'] or 0)
                override_price = pathway_map.get(sym.upper()) if pathway_map else None
                advice = advice_for_stock(sym, qty, bp, timeframe, live_override=override_price)
                advices.append(advice)
                cur = advice.get('current')
                # if current None, attempt to fetch non-overridden quote
                if cur is None:
                    q = get_quote(sym); cur = q.get('price') or 0.0
                total_value += cur * qty
            total_value = round(total_value,2)
            st.metric("Portfolio value (approx)", f"{total_value}")
            # allocation table
            rows_alloc=[]
            for a in advices:
                val = (a.get('current') or 0) * a.get('quantity', 0)
                rows_alloc.append({"symbol": a.get('symbol'), "value": round(val,2)})
            if rows_alloc:
                df_alloc = pd.DataFrame(rows_alloc)
                total_alloc = df_alloc['value'].sum() if not df_alloc.empty else 0
                if total_alloc > 0:
                    df_alloc['weight_pct'] = (df_alloc['value'] / total_alloc * 100).round(2)
                st.subheader("Allocation")
                st.table(df_alloc)
                heavy = df_alloc[df_alloc['weight_pct'] >= 40] if 'weight_pct' in df_alloc.columns else pd.DataFrame()
                if not heavy.empty:
                    st.warning("Concentration >40% in a holding. Consider diversifying.")
            # advice table & per-stock expanders
            if advices:
                st.subheader("Per-stock advice")
                df_adv = pd.DataFrame(advices)
                show_cols = [c for c in ['symbol','quantity','buyPrice','current','pnl_pct','recommendation'] if c in df_adv.columns]
                st.dataframe(df_adv[show_cols])
                for a in advices:
                    with st.expander(f"{a.get('symbol')} — {a.get('recommendation')}"):
                        st.write("Current price:", a.get('current'))
                        st.write("P/L %:", a.get('pnl_pct'))
                        st.write("Signals:", a.get('signals'))
                        st.write("Why:", a.get('explanation'))
                        hist = fetch_history(a.get('symbol'), period="6mo")
                        if not hist.empty:
                            hist_i = add_indicators(hist)
                            st.plotly_chart(fig_price_with_ma(hist_i, a.get('symbol')), use_container_width=True)
                            st.plotly_chart(fig_candlestick(hist_i, a.get('symbol')), use_container_width=True)
                            st.plotly_chart(fig_rsi(hist_i), use_container_width=True)
                        if HAVE_GTTS and st.button(f"Play voice for {a.get('symbol')}", key=f"voice_{a.get('symbol')}"):
                            txt = f"{a.get('symbol')}: Recommendation {a.get('recommendation')}. {a.get('explanation')}"
                            audio = speak_text(txt)
                            if audio:
                                st.audio(audio.read(), format="audio/mp3")
                            else:
                                st.warning("Voice not available")
            # billing & save
            n_adv = sum(1 for a in advices if not a.get('error'))
            charge = round(PER_ANALYSIS + PER_ADVICE * n_adv,2)
            st.write(f"Estimated charge (mock): {charge}")
            if st.button("Save analysis & download invoice"):
                save_analysis_sql(user_id, portfolio, advices, charge)
                log_usage("analysis_saved", f"user={user_id},count={n_adv}", charge)
                st.success("Analysis saved.")
                recent = get_recent_analyses_sql(1)
                if recent:
                    aid = recent[0][0]
                    pdfb = make_invoice_pdf(user_id, aid, portfolio, advices, charge)
                    st.download_button("Download invoice (PDF)", data=pdfb, file_name=f"invoice_{aid}.pdf", mime="application/pdf")
                st.download_button("Download analysis (JSON)", data=json.dumps({"portfolio":portfolio,"advice":advices,"total":total_value}, indent=2), file_name="analysis.json")
    # right column: lookup single ticker
    with right:
        st.header("Lookup & Learn — Single ticker")
        t = st.text_input("Ticker (e.g. AAPL or INFY.NS)", value="AAPL")
        if st.button("Lookup"):
            hist = fetch_history(t, period="1y")
            if hist.empty:
                st.error("No data for this ticker.")
            else:
                hist_i = add_indicators(hist)
                st.plotly_chart(fig_price_with_ma(hist_i, t), use_container_width=True)
                st.plotly_chart(fig_candlestick(hist_i, t), use_container_width=True)
                st.plotly_chart(fig_rsi(hist_i), use_container_width=True)
                q = get_quote(t)
                st.metric("Latest price (approx)", q.get("price"))
                # quick advice
                quick = advice_for_stock(t, 1, hist['Close'].iloc[-1], timeframe='long')
                st.write("Recommendation:", quick.get('recommendation'))
                st.write("Plain English:", quick.get('explanation'))
                if HAVE_GTTS and st.button("Play voice guidance for this ticker"):
                    audio = speak_text(f"{t}. Recommendation {quick.get('recommendation')}. {quick.get('explanation')}")
                    if audio:
                        st.audio(audio.read(), format="audio/mp3")
                    else:
                        st.warning("Voice not available")
    # usage & beginner guide
    st.markdown("---")
    st.header("Usage & Billing (mock)")
    rows = get_usage_summary_sql()
    if rows:
        dfu = pd.DataFrame(rows, columns=['action','count','total_charge'])
        st.table(dfu)
        try:
            st.write("Total mock charges:", round(dfu['total_charge'].sum(),2))
        except Exception:
            pass
    else:
        st.info("No usage recorded yet.")
    st.markdown("---")
    st.header("Beginner's Guide & Demo script")
    st.markdown(beginner_tips_md())
    st.markdown("**Demo script:**")
    st.code(demo_script_md())

if __name__ == "__main__":
    main()
