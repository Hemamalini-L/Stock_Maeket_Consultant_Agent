# app.py
import os
import io
import json
import math
import time
import requests
import logging
from datetime import datetime, date
from typing import List, Dict, Any

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objs as go

# OPTIONAL: technical indicators (if installed)
try:
    import ta
except Exception:
    ta = None

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("StockConsultantApp")

# ---------- Config ----------
APP_TITLE = "Stock Market Consultant Agent — Beginner Friendly"
COST_PER_ADVICE = 5.0   # currency units per advice generation
COST_PER_PORTFOLIO = 10.0

# ---------- Helpers: Secrets & OpenAI ----------
def get_openai_key():
    # Prefer Streamlit secrets
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    # Next check nested "general" or "openai" keys (common patterns)
    if "general" in st.secrets and "OPENAI_API_KEY" in st.secrets["general"]:
        return st.secrets["general"]["OPENAI_API_KEY"]
    # Lastly, allow environment variable for local dev
    return os.environ.get("OPENAI_API_KEY", "")

OPENAI_API_KEY = get_openai_key()
USE_OPENAI = bool(OPENAI_API_KEY)

if USE_OPENAI:
    import openai
    openai.api_key = OPENAI_API_KEY

# ---------- Persistence: usage logs ----------
USAGE_LOG_FILE = "usage_log.json"

def load_usage_log():
    try:
        if os.path.exists(USAGE_LOG_FILE):
            with open(USAGE_LOG_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning("Could not load usage log: %s", e)
    return {}

def save_usage_log(log):
    try:
        with open(USAGE_LOG_FILE, "w") as f:
            json.dump(log, f, indent=2, default=str)
    except Exception as e:
        logger.warning("Could not save usage log: %s", e)

usage_log = load_usage_log()
today_str = date.today().isoformat()
if today_str not in usage_log:
    usage_log[today_str] = {"portfolios_analyzed": 0, "advices_generated": 0}
save_usage_log(usage_log)

# ---------- Streamlit Page ----------
st.set_page_config(APP_TITLE, layout="wide")
st.title("📈 " + APP_TITLE)
st.caption("AI + Data powered guidance for beginners — plain English advice, charts, simulated billing, and a chat assistant.")
st.write("Tip: Add your OpenAI API key to Streamlit Secrets as `OPENAI_API_KEY` for AI chat features. (Settings → Secrets)")

# ---------- Sidebar: Data source & API key status ----------
st.sidebar.header("Data & Account")
data_source = st.sidebar.selectbox("Data Source", ["Live (yfinance)", "Pathway / CSV (mock)"])
st.sidebar.write("OpenAI Key: " + ("Available" if USE_OPENAI else "Not found — chat disabled"))
st.sidebar.markdown("---")

# ---------- Sidebar: Billing & usage info ----------
st.sidebar.header("Billing / Usage (Simulated)")
st.sidebar.write(f"Cost per advice: {COST_PER_ADVICE}")
st.sidebar.write(f"Cost per portfolio analyzed: {COST_PER_PORTFOLIO}")
# Show today's usage
today_usage = usage_log.get(today_str, {"portfolios_analyzed": 0, "advices_generated": 0})
st.sidebar.write(f"Portfolios analyzed today: {today_usage['portfolios_analyzed']}")
st.sidebar.write(f"Advices generated today: {today_usage['advices_generated']}")
st.sidebar.markdown("---")

# ---------- Input: Portfolio via manual or CSV ----------
st.header("1) Provide your portfolio")
input_mode = st.radio("How do you want to provide your portfolio?", ("Manual entry", "Upload CSV"))

if input_mode == "Manual entry":
    with st.form("manual_portfolio_form"):
        cols = st.columns([2,1,1,1])
        with cols[0]:
            tick_input = st.text_area("Tickers (comma-separated)", value="RELIANCE.NS, TCS.NS")
        with cols[1]:
            qty_input = st.text_area("Quantities (comma-separated)", value="1, 1")
        with cols[2]:
            buy_input = st.text_area("Buy Prices (optional, comma-separated)", value="")
        with cols[3]:
            add_btn = st.form_submit_button("Load Portfolio")
    if add_btn:
        tickers = [t.strip().upper() for t in tick_input.split(",") if t.strip()]
        quantities = [int(q.strip()) for q in qty_input.split(",") if q.strip()]
        buy_prices = []
        if buy_input.strip():
            buy_prices = [float(x.strip()) for x in buy_input.split(",") if x.strip()]
        else:
            buy_prices = [None] * len(tickers)
        if len(tickers) != len(quantities) or len(buy_prices) != len(tickers):
            st.error("Tickers, quantities and buy prices must match in length (or leave buy prices empty).")
        else:
            portfolio_df = pd.DataFrame({
                "Symbol": tickers,
                "Quantity": quantities,
                "BuyPrice": buy_prices
            })
            st.success("Portfolio loaded (manual).")
            st.session_state["portfolio_df"] = portfolio_df
else:
    uploaded = st.file_uploader("Upload CSV (columns: Symbol,Quantity,BuyPrice optional)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            # Normalize columns
            df.columns = [c.strip() for c in df.columns]
            if "Symbol" not in df.columns or "Quantity" not in df.columns:
                st.error("CSV must contain at least Symbol and Quantity columns.")
            else:
                df["Symbol"] = df["Symbol"].astype(str).str.upper()
                if "BuyPrice" not in df.columns:
                    df["BuyPrice"] = None
                st.session_state["portfolio_df"] = df[["Symbol","Quantity","BuyPrice"]]
                st.success("Portfolio uploaded from CSV.")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))

# Quick show portfolio if present
portfolio_df = st.session_state.get("portfolio_df", pd.DataFrame(columns=["Symbol","Quantity","BuyPrice"]))
if not portfolio_df.empty:
    st.subheader("Your Portfolio")
    st.dataframe(portfolio_df)

# ---------- Helper: fetch price data (Live or Pathway) ----------
def fetch_live_quotes(symbols: List[str], period="60d", interval="1d") -> Dict[str, pd.DataFrame]:
    out = {}
    for s in symbols:
        try:
            df = yf.download(s, period=period, interval=interval, progress=False, threads=False)
            # keep 'Close' column if exists
            if "Close" in df.columns:
                out[s] = df.copy()
            else:
                out[s] = pd.DataFrame()
        except Exception as e:
            logger.warning("yfinance fetch failed for %s: %s", s, e)
            out[s] = pd.DataFrame()
    return out

def fetch_pathway_csv_from_url(url: str) -> pd.DataFrame:
    # Expect CSV with Symbol, Date, Close (and optional Open/High/Low/Volume)
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return pd.read_csv(io.StringIO(r.text))
    except Exception as e:
        st.error("Failed to fetch CSV from URL: " + str(e))
        return pd.DataFrame()

def build_quotes_from_pathway(df: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    out = {}
    if df.empty:
        return {s: pd.DataFrame() for s in symbols}
    df["Symbol"] = df["Symbol"].astype(str).str.upper()
    df["Date"] = pd.to_datetime(df["Date"])
    for s in symbols:
        dfi = df[df["Symbol"] == s].sort_values("Date")
        if "Close" in dfi.columns:
            dfi = dfi.set_index("Date")
            out[s] = dfi
        else:
            out[s] = pd.DataFrame()
    return out

# ---------- Utility: compute technical indicators & advice ----------
def compute_features_for_symbol(df: pd.DataFrame) -> Dict[str, Any]:
    res = {}
    if df is None or df.empty or "Close" not in df.columns:
        return res
    # ensure sorted
    df = df.sort_index()
    closes = df["Close"].dropna()
    if len(closes) == 0:
        return res
    # latest price
    latest_price = float(closes.iloc[-1])
    res["latest_price"] = latest_price
    # returns & volatility
    returns = closes.pct_change().dropna()
    res["volatility_30d"] = float(returns.tail(30).std()) if len(returns) >= 30 else float(returns.std())
    # momentum (7 day)
    if len(closes) >= 8:
        prev = float(closes.iloc[-8])
        res["pct_change_7d"] = (latest_price - prev) / prev * 100
    else:
        res["pct_change_7d"] = 0.0
    # SMA and RSI using simple calculations if ta not available
    if ta:
        try:
            # ta requires 'Close' with datetime index
            df_local = df.copy()
            df_local["SMA20"] = ta.trend.sma_indicator(df_local["Close"], window=20)
            df_local["SMA50"] = ta.trend.sma_indicator(df_local["Close"], window=50)
            df_local["RSI"] = ta.momentum.rsi(df_local["Close"], window=14)
            res["sma20"] = float(df_local["SMA20"].dropna().iloc[-1]) if df_local["SMA20"].notna().any() else None
            res["sma50"] = float(df_local["SMA50"].dropna().iloc[-1]) if df_local["SMA50"].notna().any() else None
            res["rsi"] = float(df_local["RSI"].dropna().iloc[-1]) if df_local["RSI"].notna().any() else None
        except Exception as e:
            logger.debug("ta calculation failed: %s", e)
            res["sma20"] = None; res["sma50"] = None; res["rsi"] = None
    else:
        # fallback SMA simple
        res["sma20"] = float(closes.rolling(window=20).mean().dropna().iloc[-1]) if len(closes) >= 20 else None
        res["sma50"] = float(closes.rolling(window=50).mean().dropna().iloc[-1]) if len(closes) >= 50 else None
        # rsi simple approx
        delta = closes.diff()
        up = delta.clip(lower=0).rolling(14).mean()
        down = -delta.clip(upper=0).rolling(14).mean()
        rs = (up / down).replace([float("inf"), -float("inf")], 0).fillna(0)
        rsi = 100 - (100 / (1 + rs))
        res["rsi"] = float(rsi.dropna().iloc[-1]) if rsi.dropna().size > 0 else None
    return res

def advice_for_stock(symbol: str, features: Dict[str, Any], qty: int, buy_price: float=None) -> Dict[str, Any]:
    # Produce simple rule-based advice with explanation (beginner-friendly)
    reason = []
    action = "Hold"
    if not features:
        return {"symbol": symbol, "action": "Data Unavailable", "reason": ["No valid price data."]}
    price = features.get("latest_price")
    rsi = features.get("rsi")
    pct7 = features.get("pct_change_7d", 0.0)
    vol = features.get("volatility_30d", 0.0)
    sma20 = features.get("sma20")
    sma50 = features.get("sma50")

    # rules
    if rsi is not None and rsi > 70:
        action = "Consider Reducing (Overbought)"
        reason.append(f"RSI is high ({rsi:.1f}) — often signals short-term overbought conditions.")
    if rsi is not None and rsi < 30:
        action = "Consider Buying (Oversold)"
        reason.append(f"RSI is low ({rsi:.1f}) — often signals oversold opportunity.")
    if pct7 is not None and pct7 > 10:
        action = "Consider Booking Profit"
        reason.append(f"Recent 7-day gain {pct7:.1f}% — price moved up sharply.")
    if pct7 is not None and pct7 < -10:
        # only buy if also oversold or fundamentals good; but for beginner keep simple
        if action.startswith("Consider Buying"):
            action = "Consider Buying (Oversold & Falling)"
        else:
            action = "Consider Buying / Monitor"
        reason.append(f"Recent 7-day drop {pct7:.1f}% — price has fallen significantly.")
    # SMA cross check
    if sma20 and sma50:
        if sma20 > sma50:
            reason.append("Short-term trend (SMA20 > SMA50) looks bullish.")
        else:
            reason.append("Short-term trend (SMA20 <= SMA50) not strongly bullish.")

    # volatility note
    if vol and vol > 0.03:
        reason.append(f"Volatility (30d std) is {vol:.3f} — higher volatility means higher risk.")

    # buy price comparison
    if buy_price and price:
        pnl_pct = (price - buy_price) / buy_price * 100
        reason.append(f"Position P/L ~ {pnl_pct:.2f}% (based on buy price).")
        if pnl_pct > 20:
            reason.append("You have >20% gain on this position — consider taking some profit.")

    # Default to Hold if no aggressive signals
    if not reason:
        reason.append("No strong signals — holding and monitoring is reasonable for beginners.")

    return {
        "symbol": symbol,
        "action": action,
        "current_price": price,
        "qty": qty,
        "reasons": reason
    }

# ---------- Render analysis: main action ----------
st.markdown("---")
st.header("2) Analyze portfolio (real-time or Pathway CSV)")

# Pathway CSV input
pathway_df = pd.DataFrame()
if data_source == "Pathway / CSV (mock)":
    st.info("Upload a CSV with columns: Symbol, Date, Close (Date format YYYY-MM-DD). The app will use this as a Pathway dataset.")
    pathway_file = st.file_uploader("Upload Pathway CSV", type=["csv"], key="pathway")
    if pathway_file:
        try:
            pathway_df = pd.read_csv(pathway_file)
            st.success("Pathway CSV loaded.")
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
    pathway_url = st.text_input("Or (optional) provide a CSV URL for Pathway-like dataset", "")

analyze_clicked = st.button("🔎 Generate Advice & Charts")

if analyze_clicked:
    if portfolio_df.empty:
        st.error("Please provide a portfolio first (manual or CSV).")
    else:
        # list of distinct symbols
        symbols = list(portfolio_df["Symbol"].unique())
        # fetch quotes
        if data_source == "Live (yfinance)":
            st.info("Fetching live data (yfinance)... This may take a few seconds.")
            quotes = fetch_live_quotes(symbols, period="90d", interval="1d")
        else:
            # Pathway: use uploaded CSV or URL
            df_source = pathway_df.copy()
            if df_source.empty and pathway_url.strip():
                df_source = fetch_pathway_csv_from_url(pathway_url)
            if df_source.empty:
                st.warning("No Pathway CSV provided. Live fallback to yfinance.")
                quotes = fetch_live_quotes(symbols, period="90d", interval="1d")
            else:
                quotes = build_quotes_from_pathway(df_source, symbols)

        # compute features + advices
        advices = []
        allocation_map = {}  # symbol -> market value (qty * price)
        total_value = 0.0
        # iterate portfolio rows
        for idx, row in portfolio_df.iterrows():
            sym = row["Symbol"]
            qty = int(row["Quantity"])
            buy_price = row["BuyPrice"] if "BuyPrice" in row and not pd.isna(row["BuyPrice"]) else None
            df_sym = quotes.get(sym, pd.DataFrame())
            features = compute_features_for_symbol(df_sym)
            advice_obj = advice_for_stock(sym, features, qty, buy_price)
            advices.append(advice_obj)
            price = features.get("latest_price", 0.0) or 0.0
            allocation_map[sym] = allocation_map.get(sym, 0.0) + price * qty
            total_value += price * qty

        # diversification check (sector data not reliably available from Pathway)
        # simple check: if any symbol > 40% allocation
        heavy = [s for s,v in allocation_map.items() if total_value>0 and (v/total_value) > 0.4]
        if heavy:
            diversification_note = f"Portfolio concentrated: {', '.join(heavy)} make up >40%."
        else:
            diversification_note = "Portfolio allocation is reasonably diversified."

        # update usage log and counters
        usage_log[today_str]["portfolios_analyzed"] += 1
        usage_log[today_str]["advices_generated"] += len(advices)
        save_usage_log(usage_log)

        # increment Streamlit session counters (transient)
        if "portfolio_count" not in st.session_state:
            st.session_state.portfolio_count = 0
        if "advice_count" not in st.session_state:
            st.session_state.advice_count = 0
        st.session_state.portfolio_count += 1
        st.session_state.advice_count += len(advices)

        # Billing calculation (simulate)
        bill_amount = COST_PER_PORTFOLIO * 1 + COST_PER_ADVICE * len(advices)

        # ---------- Display advices ----------
        st.subheader("🔔 Advice (Plain English)")
        for a in advices:
            st.markdown(f"**{a['symbol']}** — **{a['action']}**")
            for r in a["reasons"]:
                st.markdown(f"- {r}")

        st.info(diversification_note)

        # ---------- Charts: allocation pie ----------
        st.subheader("📊 Portfolio Snapshot & Charts")
        alloc_items = [{"Symbol":k,"Value":v} for k,v in allocation_map.items()]
        if alloc_items:
            alloc_df = pd.DataFrame(alloc_items)
            fig_pie = go.Figure(go.Pie(labels=alloc_df["Symbol"], values=alloc_df["Value"], hole=0.35))
            fig_pie.update_layout(title="Portfolio Allocation by Market Value")
            st.plotly_chart(fig_pie, use_container_width=True)

        # ---------- Price charts per symbol ----------
        for s in symbols:
            df_sym = quotes.get(s, pd.DataFrame())
            if df_sym is not None and not df_sym.empty and "Close" in df_sym.columns:
                st.markdown(f"### {s} — Price Chart (last {len(df_sym)} days)")
                # Plotly candlestick if OHLC present, else line
                if all(c in df_sym.columns for c in ["Open","High","Low","Close"]):
                    ohlc = df_sym.reset_index()
                    fig = go.Figure(data=[go.Candlestick(x=ohlc["Date"],
                                                         open=ohlc["Open"], high=ohlc["High"],
                                                         low=ohlc["Low"], close=ohlc["Close"])])
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # simple line chart of Close
                    st.line_chart(df_sym["Close"])

        # ---------- Downloadable report & invoice ----------
        st.subheader("Export")
        # Advice CSV
        advice_csv = pd.DataFrame([{"Symbol":a["symbol"], "Action":a["action"], "Reasons": " | ".join(a["reasons"]), "CurrentPrice": a.get("current_price", "") ,"Qty":a.get("qty","")} for a in advices])
        csv_buf = io.StringIO()
        advice_csv.to_csv(csv_buf, index=False)
        st.download_button("Download advice CSV", data=csv_buf.getvalue(), file_name=f"advice_{datetime.now().strftime('%Y%m%d_%H%M')}.csv", mime="text/csv")

        # Invoice (simple txt)
        invoice_text = f"INVOICE\nDate: {datetime.now().isoformat()}\nPortfolios analyzed: 1\nAdvices generated: {len(advices)}\nAmount due: {bill_amount}\n\nDetails:\n"
        for a in advices:
            invoice_text += f"- {a['symbol']}: {a['action']}\n"
        st.download_button("Download invoice (txt)", data=invoice_text, file_name=f"invoice_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain")

# ---------- Chat Assistant ----------
st.markdown("---")
st.header("3) AI Chat Assistant (Ask for guidance)")

if not USE_OPENAI:
    st.warning("OpenAI API key not found in Streamlit Secrets. Add OPENAI_API_KEY to use chat assistant.")
else:
    # chat UI
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**You:** {chat['content']}")
        else:
            st.markdown(f"**Assistant:** {chat['content']}")

    user_msg = st.text_input("Ask a question (e.g., 'When to buy RELIANCE.NS?')", key="chat_input")
    if st.button("Send", key="send_msg"):
        if not user_msg.strip():
            st.warning("Type something.")
        else:
            # Build context: include latest advices & selected tickers
            context = "I am a helpful assistant for beginner investors. Provide simple actionable guidance and explain the reasons in short bullets.\n\n"
            if not portfolio_df.empty:
                context += "User portfolio:\n"
                for _, r in portfolio_df.iterrows():
                    context += f"- {r['Symbol']} qty {r['Quantity']} buy_price {r.get('BuyPrice')}\n"
            # If message mentions tickers, fetch their latest quick stats to include
            mentioned = []
            tokens = [w.upper().strip(".,") for w in user_msg.split()]
            for w in tokens:
                if any(ch.isdigit() for ch in w):  # simple heuristic
                    pass
                if "." in w or w.isalpha():
                    # try to treat as ticker if uppercase and length reasonable
                    if len(w) <= 10 and w.isupper():
                        mentioned.append(w)
            # Fetch quick stats for mentioned tickers
            stats_text = ""
            if mentioned:
                stats_text += "\nQuick stats:\n"
                quotes = fetch_live_quotes(mentioned, period="30d", interval="1d")
                for s in mentioned:
                    df_s = quotes.get(s, pd.DataFrame())
                    features = compute_features_for_symbol(df_s)
                    if features:
                        stats_text += f"- {s}: price {features.get('latest_price')}, 7d% {features.get('pct_change_7d'):.2f}, RSI {features.get('rsi')}\n"
            # Compose prompt
            prompt = f"{context}\n{stats_text}\nUser question: {user_msg}\nAnswer simply with actionable advice and short bullet reasons."
            # Call OpenAI ChatCompletion
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role":"system","content":"You are a friendly investment helper for beginners."},
                              {"role":"user","content":prompt}],
                    max_tokens=400,
                    temperature=0.2
                )
                ans = resp.choices[0].message["content"]
            except Exception as e:
                ans = f"AI query failed: {e}"
            st.session_state.chat_history.append({"role":"user","content":user_msg})
            st.session_state.chat_history.append({"role":"assistant","content":ans})
            # update usage & billing for chat if we count it as advice
            usage_log[today_str]["advices_generated"] += 1
            save_usage_log(usage_log)
            if "advice_count" not in st.session_state:
                st.session_state.advice_count = 0
            st.session_state.advice_count += 1
            st.success("Assistant replied — see conversation above.")

# ---------- Footer: show quick run commands & requirements ----------
st.markdown("---")
st.caption("App built to combine live data, Pathway mock CSV, rule-based advice, charts, chat assistant, usage logging, and billing simulation.")
