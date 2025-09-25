# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime
from io import BytesIO

# -----------------------
# Configuration / defaults
# -----------------------
DB_FILE = "usage.db"               # SQLite file saved in working dir
MOCK_PRICE_CSV = "mock_prices.csv" # CSV (symbol,price) if available
PER_ADVICE_CHARGE = 0.05           # Rupees/dollars per advice (mock)
PER_ANALYSIS_CHARGE = 0.50         # Rupees/dollars per full portfolio analysis (mock)

# -----------------------
# Database helpers
# -----------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            detail TEXT,
            charge REAL,
            timestamp TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            portfolio_json TEXT,
            advice_json TEXT,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_usage(action: str, detail: str, charge: float=0.0):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO usage (action, detail, charge, timestamp) VALUES (?, ?, ?, ?)',
              (action, detail, charge, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def save_analysis(user_id: str, portfolio: dict, advice: list):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('INSERT INTO analyses (user_id, portfolio_json, advice_json, timestamp) VALUES (?, ?, ?, ?)',
              (user_id, json.dumps(portfolio), json.dumps(advice), datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()

def get_usage_summary():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT action, COUNT(*), SUM(charge) FROM usage GROUP BY action')
    rows = c.fetchall()
    conn.close()
    return rows

def get_recent_analyses(limit=10):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('SELECT id, user_id, portfolio_json, advice_json, timestamp FROM analyses ORDER BY id DESC LIMIT ?', (limit,))
    rows = c.fetchall()
    conn.close()
    return rows

# -----------------------
# Mock price fetcher (Pathway simulation)
# -----------------------
@st.cache_data
def load_mock_prices():
    # Try to load CSV from repo; if not available, generate mock prices
    try:
        df = pd.read_csv(MOCK_PRICE_CSV)
        if 'symbol' in df.columns and 'price' in df.columns:
            df = df[['symbol','price']].drop_duplicates().set_index('symbol')
            return df['price'].to_dict()
    except Exception:
        pass
    # fallback: generate random prices for common tickers
    base = {'AAPL': 170, 'GOOGL': 2800, 'TCS': 3200, 'INFY': 1500, 'MSFT': 340, 'RELIANCE': 2350}
    prices = {k: float(v * (0.9 + 0.2*np.random.rand())) for k,v in base.items()}
    return prices

def get_price(symbol: str, price_map: dict):
    symbol = symbol.upper().strip()
    return float(price_map.get(symbol, np.round(10 + 990*np.random.rand(),2)))

# -----------------------
# Advice engine (simple heuristics)
# -----------------------
def analyze_portfolio(portfolio: dict, price_map: dict):
    """
    portfolio: { "userId":str, "stocks":[{"name":str,"quantity":int,"buyPrice":float}, ...] }
    returns: list of advice dicts
    """
    advices = []
    total_value = 0.0
    # compute current values and basic metrics
    for s in portfolio.get('stocks', []):
        name = s.get('name','').upper()
        qty = float(s.get('quantity',0) or 0)
        buy = float(s.get('buyPrice',0) or 0)
        current = get_price(name, price_map)
        value = current * qty
        pnl_pct = ((current - buy) / buy * 100) if buy>0 else None
        adv = {
            'stock': name,
            'quantity': qty,
            'buyPrice': buy,
            'currentPrice': current,
            'value': round(value,2),
            'pnl_pct': round(pnl_pct,2) if pnl_pct is not None else None
        }
        advices.append(adv)
        total_value += value

    # diversification and per-stock recommendation
    # compute weight of each holding
    for a in advices:
        weight = (a['value'] / total_value * 100) if total_value>0 else 0
        a['weight_pct'] = round(weight,2)
        # simple rules:
        # - If pnl_pct < -15% -> suggest consider selling (cut losses)
        # - If pnl_pct > 25% -> suggest consider taking profit
        # - If weight_pct > 40% -> suggest diversify
        # - If stock not in important sector (mock) -> suggest consider adding top IT (INFY/TCS)
        suggestion = "Hold"
        reasons = []
        if a['pnl_pct'] is not None:
            if a['pnl_pct'] <= -15:
                suggestion = "Reduce"
                reasons.append("Unrealized loss > 15%")
            elif a['pnl_pct'] >= 25:
                suggestion = "Consider Taking Profit"
                reasons.append("Unrealized gain > 25%")
        if a['weight_pct'] >= 40:
            suggestion = "Diversify"
            reasons.append("Single stock >40% portfolio")
        # if small pos and small qty, encourage buy if price low (demo logic)
        if a['quantity']>0 and a['pnl_pct'] is not None and abs(a['pnl_pct']) < 3 and weight < 5:
            reasons.append("Small position; consider increasing for diversification")
        a['suggestion'] = suggestion
        a['reason'] = "; ".join(reasons) if reasons else "No strong signal"

    # top-level portfolio advice (plain english)
    portfolio_advice = []
    if len(advices)==0:
        portfolio_advice.append("No stocks in portfolio. Add holdings to get advice.")
    else:
        # diversification check (count sectors via symbol heuristics)
        heavy = [a for a in advices if a['weight_pct']>=40]
        if heavy:
            portfolio_advice.append("Your portfolio has a concentration risk: consider reducing the largest positions.")
        # check missing IT if none in advices
        names = [a['stock'] for a in advices]
        if not any(x in names for x in ['INFY','TCS','WIPRO']):
            portfolio_advice.append("Consider adding IT stocks (e.g., INFY, TCS) to improve diversification.")
    # Return both detailed advices and top-level sentences
    return advices, portfolio_advice, round(total_value,2)

# -----------------------
# UI Components
# -----------------------
def render_header():
    st.title("Stock Consultant Agent — Streamlit Demo")
    st.caption("Simple demo of portfolio advice, usage and billing (mock Pathway & Flexprice integration)")

def portfolio_input_ui():
    st.header("Enter Portfolio")
    st.write("Add holdings manually or upload a CSV with columns: name,quantity,buyPrice")
    with st.form("portfolio_form", clear_on_submit=False):
        user_id = st.text_input("User ID", value="hema123")
        add_row = st.checkbox("Start with an example entry", value=True)
        # interactive table building (simple)
        rows = []
        if add_row:
            default_df = pd.DataFrame([{"name":"AAPL","quantity":10,"buyPrice":150},{"name":"GOOGL","quantity":5,"buyPrice":2800}])
        else:
            default_df = pd.DataFrame(columns=["name","quantity","buyPrice"])
        upfile = st.file_uploader("Upload CSV (optional)", type=['csv'])
        if upfile is not None:
            try:
                csv_df = pd.read_csv(upfile)
                default_df = csv_df[['name','quantity','buyPrice']]
            except Exception as e:
                st.error("CSV read error: "+str(e))
        st.write("Current portfolio (editable in CSV upload) — displayed for review:")
        st.dataframe(default_df)
        submitted = st.form_submit_button("Analyze Portfolio")
        if submitted:
            # build portfolio dict
            portfolio = {"userId": user_id, "stocks": default_df.to_dict(orient='records')}
            return portfolio
    return None

def display_advice_and_actions(portfolio, adv_result, portfolio_advice, total_value):
    st.subheader("Analysis Results")
    st.write(f"User: **{portfolio.get('userId','-')}** — Total Value (mock): **{total_value}**")
    st.write("Top-level portfolio advice:")
    for s in portfolio_advice:
        st.info(s)
    st.markdown("**Per-stock details & suggestions**")
    df = pd.DataFrame(adv_result)
    st.dataframe(df)
    # show download JSON of the analysis
    if st.button("Save this analysis & generate report"):
        save_analysis(portfolio.get('userId',''), portfolio, adv_result)
        log_usage("portfolio_analysis", f"user={portfolio.get('userId','')}", PER_ANALYSIS_CHARGE)
        st.success("Analysis saved. Billing recorded.")
        # also record per-advice charges
        n = len(adv_result)
        if n>0:
            log_usage("advice_batch", f"user={portfolio.get('userId','')},count={n}", PER_ADVICE_CHARGE * n)

def download_json(name: str, obj):
    b = BytesIO()
    b.write(json.dumps(obj, indent=2).encode('utf-8'))
    b.seek(0)
    st.download_button(label=f"Download {name}.json", data=b, file_name=f"{name}.json", mime="application/json")

def show_usage_and_billing():
    st.header("Usage & Billing (Flexprice Simulation)")
    rows = get_usage_summary()
    if not rows:
        st.info("No usage recorded yet.")
        return
    df = pd.DataFrame(rows, columns=['action','count','total_charge'])
    st.table(df)
    total_charge = df['total_charge'].sum()
    st.write("**Total charges (mock):** ", total_charge)
    # show recent analyses
    st.subheader("Recent Analyses")
    rows2 = get_recent_analyses(10)
    if rows2:
        for r in rows2:
            idx, user_id, portfolio_json, advice_json, ts = r
            st.markdown(f"**Analysis #{idx}** — user: {user_id} — {ts}")
            st.write("Portfolio:")
            st.json(json.loads(portfolio_json))
            st.write("Advice:")
            st.json(json.loads(advice_json))
            if st.button(f"Download report #{idx}"):
                report = {'id': idx, 'user': user_id, 'portfolio': json.loads(portfolio_json), 'advice': json.loads(advice_json), 'timestamp': ts}
                download_json(f"analysis_{idx}", report)

# -----------------------
# Main app
# -----------------------
def main():
    st.set_page_config(page_title="Stock Consultant Agent", layout="wide")
    init_db()
    render_header()
    price_map = load_mock_prices()

    col1, col2 = st.columns([2,1])
    with col1:
        portfolio = portfolio_input_ui()
        if portfolio:
            adv_result, portfolio_advice, total_value = analyze_portfolio(portfolio, price_map)
            display_advice_and_actions(portfolio, adv_result, portfolio_advice, total_value)
            # allow immediate download as JSON
            if st.button("Download analysis JSON"):
                analysis = {"portfolio": portfolio, "advice": adv_result, "summary": portfolio_advice, "total_value": total_value}
                download_json("analysis", analysis)
    with col2:
        st.sidebar.title("Quick Controls")
        st.sidebar.write("Mock data source: Pathway (CSV fallback used)")
        st.sidebar.write(f"Per-advice charge: {PER_ADVICE_CHARGE}")
        st.sidebar.write(f"Per-analysis charge: {PER_ANALYSIS_CHARGE}")
        if st.sidebar.button("Show Usage & Billing"):
            show_usage_and_billing()

    st.markdown("---")
    st.markdown("**Notes:** This is a demo. Pathway & Flexprice are simulated. For production, replace mock price fetching with a real market-data API and integrate real billing SDKs.")

if __name__ == "__main__":
    main()
