import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import feedparser
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------------------------------------------------
# Configuration & Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Peer Analysis Dashboard",
    page_icon="📈",
    layout="wide",
)

# Custom CSS for dark theme adjustments (Plotly dark + Streamlit dark)
st.markdown("""
<style>
    /* Adjust spacing to match the dense dashboard feel */
    .stApp > header {visibility: hidden;}
    .block-container {
        padding-top: 1rem;
        padding-bottom: 3rem;
    }

    /* Glass effect cards */
    div[data-testid="stContainer"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 10px;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* Smooth hover */
    div[data-testid="stContainer"]:hover {
        transform: scale(1.01);
        transition: 0.3s ease;
    }

    /* Metric font improve */
    div[data-testid="stMetric"] {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Constants & Data
# -----------------------------------------------------------------------------
STOCKS = [
    "AAPL", "ABBV", "ACN", "ADBE", "ADP", "AMD", "AMGN", "AMT", "AMZN", "APD", 
    "AVGO", "AXP", "BA", "BK", "BKNG", "BMY", "BRK-B", "BSX", "C", "CAT", 
    "CI", "CL", "CMCSA", "COST", "CRM", "CSCO", "CVX", "DE", "DHR", "DIS", 
    "DUK", "ELV", "EOG", "EQR", "FDX", "GD", "GE", "GILD", "GOOG", "GOOGL", 
    "HD", "HON", "HUM", "IBM", "ICE", "INTC", "ISRG", "JNJ", "JPM", "KO", 
    "LIN", "LLY", "LMT", "LOW", "MA", "MCD", "MDLZ", "META", "MMC", "MO", 
    "MRK", "MSFT", "NEE", "NFLX", "NKE", "NOW", "NVDA", "ORCL", "PEP", "PFE", 
    "PG", "PLD", "PM", "PSA", "REGN", "RTX", "SBUX", "SCHW", "SLB", "SO", 
    "SPGI", "T", "TJX", "TMO", "TSLA", "TXN", "UNH", "UNP", "UPS", "V", "VZ", 
    "WFC", "WM", "WMT", "XOM"
]

DEFAULT_STOCKS = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMZN", "TSLA", "META"]

HORIZON_MAP = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "5 Years": "5y",
    "10 Years": "10y",
    "YTD": "ytd",
    "Max": "max"
}

# -----------------------------------------------------------------------------
# State & Query Params Management
# -----------------------------------------------------------------------------
def stocks_to_str(stocks):
    return ",".join(stocks)

if "tickers_input" not in st.session_state:
    stocks_param = st.query_params.get("stocks",  stocks_to_str(DEFAULT_STOCKS))
    # Handle single string or list return from query_params depending on streamlit version
    if isinstance(stocks_param, list):
         stocks_param = stocks_param[0] if stocks_param else stocks_to_str(DEFAULT_STOCKS)
    elif stocks_param is None:
        stocks_param = stocks_to_str(DEFAULT_STOCKS)
    
    st.session_state.tickers_input = stocks_param.split(",")

# -----------------------------------------------------------------------------
# Layout Construction
# -----------------------------------------------------------------------------
st.title("📈 Stock Peer Analysis")
st.markdown("Easily compare stocks against others in their peer group.")
st.write("") # Spacer

# Main Layout: metrics on left, charts on right
col_left, col_right = st.columns([1, 3], gap="medium")

# --- Sidebar: Controls ---
with st.sidebar:
    st.markdown("## ⚙️ Controls")

    # Theme Toggle
    theme = st.toggle("Dark Mode", True)
    plot_template = "plotly_dark" if theme else "plotly_white"

    st.divider()

    # Stock Search Filter
    search = st.text_input("Search Stock in Directory", placeholder="e.g., TSLA")
    if search:
        filtered = [s for s in STOCKS if search.upper() in s]
        if filtered:
            st.caption(f"Matches: {', '.join(filtered)}")

    # Ticker Selection
    selected_tickers_new = st.multiselect(
        "Stock Tickers",
        options=sorted(set(STOCKS) | set(st.session_state.tickers_input)),
        default=st.session_state.tickers_input,
        placeholder="Select tickers...",
        key="ticker_multiselect"
    )

    # Update session state if changed
    if selected_tickers_new != st.session_state.tickers_input:
        st.session_state.tickers_input = selected_tickers_new

    # Update query params logic
    if st.session_state.tickers_input:
        st.query_params["stocks"] = stocks_to_str(st.session_state.tickers_input)
    else:
        if "stocks" in st.query_params:
            del st.query_params["stocks"]

    st.divider()

    # Date Range Picker
    start_date = st.date_input("Start Date", value=pd.to_datetime("2024-01-01"))
    end_date = st.date_input("End Date")

    st.divider()

    # Advanced Options
    st.markdown("### Options")
    use_log_returns = st.toggle("Log Returns", False)

# -----------------------------------------------------------------------------
# Data Logic
# -----------------------------------------------------------------------------
@st.cache_data(ttl="1h", show_spinner=False)
def load_data(tickers, start, end):
    try:
        data = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            return pd.DataFrame()

        # SAFE extraction
        if isinstance(data.columns, pd.MultiIndex):
            df = data.xs('Close', level=1, axis=1)
        else:
            df = data[['Close']]
            df.columns = tickers

        return df.dropna(how="all")

    except Exception as e:
        st.write("DEBUG ERROR:", e)
        return pd.DataFrame()


def lstm_predict(df, ticker, days=30):
    data = df[[ticker]].dropna()
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    window = 60

    for i in range(window, len(scaled_data)):
        X.append(scaled_data[i-window:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    # Predict future
    last_window = scaled_data[-window:]
    preds = []

    for _ in range(days):
        pred = model.predict(last_window.reshape(1, window, 1), verbose=0)
        preds.append(pred[0][0])
        last_window = np.append(last_window[1:], pred, axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1,1))
    return preds

if not st.session_state.tickers_input:
    col_left.warning("Please select at least one stock.")
    st.stop()

with st.spinner("Loading financial data..."):
    # Always sort tickers to maintain consistent column order/colors
    sorted_s_tickers = sorted(st.session_state.tickers_input)
    raw_df = load_data(sorted_s_tickers, start_date, end_date)

if raw_df is None or raw_df.empty:
    col_left.error("Failed to load data. Please check connection or ticker validity.")
    st.stop()

# Calculations
if use_log_returns:
    # Normalized Log Returns: ln(Pt / P0)
    # This aligns 0 as the starting point.
    normalized_df = np.log(raw_df / raw_df.iloc[0])
    y_axis_label = "Log Return"
    val_formatting = ".3f"
else:
    # Normalized Price: Pt / P0
    normalized_df = raw_df / raw_df.iloc[0]
    y_axis_label = "Normalized Price"
    val_formatting = ".2f"

# Metrics logic
# Sort by performance for stable best/worst identification
final_values = normalized_df.iloc[-1]
final_values_sorted = final_values.sort_values(ascending=False)

best_ticker = final_values_sorted.index[0]
best_val = final_values_sorted.iloc[0]
worst_ticker = final_values_sorted.index[-1]
worst_val = final_values_sorted.iloc[-1]

if use_log_returns:
    # Use just the value for log return
    best_disp = f"{best_val:.3f}" 
    worst_disp = f"{worst_val:.3f}"
    metric_delta_suffix = ""
else:
    # Show % change from 1.0
    best_disp = f"{(best_val - 1) * 100:.1f}%" 
    worst_disp = f"{(worst_val - 1) * 100:.1f}%"
    metric_delta_suffix = "%"

# --- Left Column: Metrics ---
with col_left:
    st.markdown("## 📊 Market Snapshot")

    # KPI calculations
    avg_perf = (final_values.mean() - 1) * 100
    returns = raw_df.pct_change().dropna()
    volatility = returns.std().mean() * np.sqrt(252)
    sharpe_ratio = (returns.mean() / returns.std()).mean() * np.sqrt(252)
    latest_price = raw_df[best_ticker].iloc[-1]

    k1, k2 = st.columns(2)
    k1.metric("📈 Avg Return", f"{avg_perf:.2f}%")
    k2.metric("⚡ Volatility", f"{volatility:.2f}")

    k3, k4 = st.columns(2)
    k3.metric("🧠 Sharpe Ratio", f"{sharpe_ratio:.2f}")
    k4.metric("🔥 Top Stock", best_ticker)

    st.markdown("<br>", unsafe_allow_html=True)

    # Current Price Box
    container_price = st.container(border=True)
    with container_price:
        st.metric(f"💲 Current Price ({best_ticker})", f"${latest_price:.2f}")

        c1, c2 = st.columns(2)
        with c1:
            st.write("🏆 Best")
            st.success(best_ticker)
        with c2:
            st.write("📉 Worst")
            st.error(worst_ticker)

# -----------------------------------------------------------------------------
# Main Visualizations (Right Column)
# -----------------------------------------------------------------------------
with col_right:

    # 📈 Price Chart
    container_main = st.container(border=True)
    with container_main:
        st.markdown("## 📈 Price Comparison")

        # Build fig_main
        fig_main = go.Figure()
        colors = [
            "#00CC96", "#636EFA", "#EF553B",
            "#AB63FA", "#FFA15A", "#19D3F3", "#FF6692"
        ]

        for i, col in enumerate(normalized_df.columns):
            c = colors[i % len(colors)]
            fig_main.add_trace(go.Scatter(
                x=normalized_df.index,
                y=normalized_df[col],
                mode='lines',
                name=col,
                line=dict(width=3, color=c, shape='spline'),
                fill='tozeroy',
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.05)",
                hovertemplate=f"<b>{col}</b><br>Value: %{{y:{val_formatting}}}<extra></extra>"
            ))

        fig_main.update_layout(
            title=dict(text="Normalized Price Comparison", x=0, font=dict(size=16)),
            template=plot_template,
            margin=dict(l=0, r=0, t=40, b=0),
            height=425,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            hovermode="x unified",
            xaxis_title=None,
            yaxis_title=y_axis_label,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
            transition_duration=500
        )
        st.plotly_chart(fig_main, use_container_width=True)

# 🏆 Ranking + 📊 Bar Chart — FULL WIDTH below the columns
st.markdown("<br>", unsafe_allow_html=True)
container_bottom = st.container(border=True)
with container_bottom:
    c1, c2 = st.columns(2)

    # LEFT → Ranking
    with c1:
        st.markdown("## 🏆 Stock Ranking")
        ranking = final_values.sort_values(ascending=False)
        styled_ranking = ranking.rename("Performance").to_frame()
        st.dataframe(
            styled_ranking.style
            .background_gradient(cmap="viridis")
            .format("{:.2f}"),
            use_container_width=True,
            height=450
        )

    # RIGHT → Bar Chart
    with c2:
        st.markdown("## 📊 Performance Comparison")

        sorted_vals = normalized_df.iloc[-1].sort_values(ascending=False).reset_index()
        sorted_vals.columns = ["Stock", "Performance"]

        fig_bar = px.bar(
            sorted_vals,
            x="Stock",
            y="Performance",
            color="Performance",
            color_continuous_scale="viridis"
        )

        fig_bar.update_layout(
            template=plot_template,
            margin=dict(t=20, b=0, l=0, r=0),
            showlegend=False,
            height=450,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )

        fig_bar.update_traces(marker_line_width=0, opacity=0.9)

        st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------------------------------------------------------
# AI Insights & Context
# -----------------------------------------------------------------------------
st.subheader("🤖 AI Insights")

st.markdown("### 🏆 Top Picks")
top_stocks = final_values.sort_values(ascending=False).head(3)
for stock in top_stocks.index:
    st.success(f"✔ {stock}")

if best_val > 1:
    momentum_msg = "Market is showing upward momentum 📈"
else:
    momentum_msg = "Market is under pressure 📉"
    
returns_temp = raw_df.pct_change().dropna()
sharpe_mean = ((returns_temp.mean() / returns_temp.std()) * np.sqrt(252)).mean()
risk_msg = "Good risk-adjusted returns detected 🛡️" if sharpe_mean > 1 else "Volatile risk profile detected ⚠️"

st.write(f"""
- Best performing stock: **{best_ticker}**
- Worst performing stock: **{worst_ticker}**
- Momentum: **{momentum_msg}**
- Risk Profile: **{risk_msg}**
""")

st.markdown(f"### 📌 Fundamentals for {best_ticker}")
try:
    ticker_obj = yf.Ticker(best_ticker)
    info = ticker_obj.info
    
    f1, f2 = st.columns(2)
    f1.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
    
    mcap = info.get('marketCap', 'N/A')
    if mcap != 'N/A' and mcap is not None:
        mcap = f"${mcap:,.0f}"
    f2.write(f"**Market Cap:** {mcap}")
except:
    st.write("Fundamental data not currently available for this ticker.")
    
st.markdown("### 📢 Trading Signals (All Stocks)")
signals = []

for ticker in raw_df.columns:
    current_price = raw_df[ticker].iloc[-1]
    current_sma = raw_df[ticker].rolling(20).mean().iloc[-1]

    if pd.isna(current_sma):
        signal = "WAIT ⚪"
    elif current_price > current_sma:
        signal = "BUY 🟢"
    else:
        signal = "SELL 🔴"

    signals.append({
        "Stock": ticker,
        "Price": round(current_price, 2),
        "SMA(20)": round(current_sma, 2) if not pd.isna(current_sma) else None,
        "Signal": signal
    })

signals_df = pd.DataFrame(signals)
filter_signal = st.selectbox("Filter Signal", ["All", "BUY", "SELL"])

if filter_signal != "All":
    signals_df = signals_df[signals_df["Signal"].str.contains(filter_signal)]

def color_signal(val):
    if pd.isna(val):
        return ""
    if "BUY" in str(val):
        return "color: #00CC96; font-weight: bold"
    elif "SELL" in str(val):
        return "color: #EF553B; font-weight: bold"
    return ""
    
# Safely handle newer pandas .map vs older .applymap
styler = signals_df.style.map(color_signal, subset=["Signal"]) if hasattr(signals_df.style, 'map') else signals_df.style.applymap(color_signal, subset=["Signal"])
st.dataframe(styler, use_container_width=True)

# -----------------------------------------------------------------------------
# Peer Analysis Section
# -----------------------------------------------------------------------------
st.markdown("### 🧬 Peer Performance")
st.markdown("Compare each stock against the average of its peers (excluding itself).")

if len(st.session_state.tickers_input) < 2:
    st.info("Select 2 or more tickers to see peer comparison.")
else:
    # Grid Layout for Small Multiples
    # We will use columns to create a grid. 
    # Logic: 4 columns wide.
    cols = st.columns(4)
    
    for i, ticker in enumerate(sorted_s_tickers):
        if ticker not in normalized_df.columns:
            continue
            
        # Calc logic
        peers = [t for t in normalized_df.columns if t != ticker]
        if not peers:
             continue
             
        peer_avg = normalized_df[peers].mean(axis=1)
        stock_line = normalized_df[ticker]
        delta = stock_line - peer_avg
        
        # Target Column in the grid
        dest_col = cols[i % 4]
        
        with dest_col:
            with st.container(border=True):
                # Mini Chart: Stock vs Peer
                fig_mini = go.Figure()
                
                # Peer Avg (Background)
                fig_mini.add_trace(go.Scatter(
                    x=normalized_df.index, y=peer_avg,
                    mode='lines', name='Peer Avg',
                    line=dict(color='#555555', dash='dot', width=1),
                    showlegend=False,
                    hoverinfo='skip' 
                ))
                
                # Stock (Foreground)
                is_outperforming = delta.iloc[-1] >= 0
                line_color = '#00CC96' if is_outperforming else '#EF553B'
                
                fig_mini.add_trace(go.Scatter(
                    x=normalized_df.index, y=stock_line,
                    mode='lines', name=ticker,
                    line=dict(color=line_color, width=2),
                    showlegend=False,
                    hovertemplate=f'%{{y:{val_formatting}}}'
                ))
                
                # Title & Layout
                fig_mini.update_layout(
                    title=dict(text=f"{ticker}", font=dict(size=14), x=0.5, xanchor='center'),
                    template=plot_template,
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=200,
                    xaxis=dict(showticklabels=False), # Cleaner mini charts
                    yaxis=dict(showticklabels=False),
                    hovermode="x unified"
                )
                st.plotly_chart(fig_mini, use_container_width=True, config={'displayModeBar': False})
                
                # Delta Metric text below chart
                final_delta = delta.iloc[-1]
                if use_log_returns:
                    delta_str = f"{final_delta:+.3f}"
                    delta_unit = ""
                else:
                    delta_str = f"{final_delta*100:+.1f}"
                    delta_unit = "%"
                
                if is_outperforming:
                     caret = "▲"
                     color_css = "#00CC96"
                else:
                     caret = "▼"
                     color_css = "#EF553B"
                     
                st.markdown(f"<div style='text-align: center; font-size: 0.9em;'>vs Peers: <span style='color: {color_css}; font-weight: bold;'>{caret} {delta_str}{delta_unit}</span></div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Raw Data Expansion
# -----------------------------------------------------------------------------
st.divider()

with st.expander("📊 Raw Data & Correlation"):
    t1, t2 = st.tabs(["Raw Data (Prices)", "Correlation Matrix"])
    
    with t1:
        st.dataframe(raw_df.style.highlight_max(axis=0), use_container_width=True)
        st.download_button("Download Analysis Report", raw_df.to_csv().encode('utf-8'), "report.csv", "text/csv")
        
    with t2:
        # Correlation
        corr = raw_df.pct_change().corr()
        fig_corr = px.imshow(
            corr, 
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig_corr.update_layout(height=450, template=plot_template)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.subheader("🧠 Insights")
        high_corr = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        pairs = high_corr.stack().sort_values(ascending=False)
        st.write("Highly Correlated Stocks:")
        st.write(pairs.head(5))

# -----------------------------------------------------------------------------
# Advanced AI Analysis Section
# -----------------------------------------------------------------------------
st.divider()
st.markdown("### 🤖 Advanced AI Analysis")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Portfolio Optimization", "Price Prediction", "Risk Metrics", "Moving Averages", "Sentiment Analysis", "Technical Indicators", "LSTM Prediction"
])

with tab1:
    st.subheader("🧠 Portfolio Optimization (Efficient Frontier)")
    st.markdown("Best allocation of money across stocks → max return, min risk")
    def portfolio_optimization(df):
        returns = df.pct_change().dropna()
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        weights_record = []
        for i in range(num_portfolios):
            weights = np.random.random(len(mean_returns))
            weights /= np.sum(weights)
            weights_record.append(weights)
            portfolio_return = np.sum(weights * mean_returns) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
            sharpe_ratio = portfolio_return / portfolio_std
            results[0,i] = portfolio_return
            results[1,i] = portfolio_std
            results[2,i] = sharpe_ratio
        return results, weights_record
        
    if len(st.session_state.tickers_input) >= 2:
        results, weights_record = portfolio_optimization(raw_df)
        fig_ef = px.scatter(
            x=results[1], y=results[0],
            color=results[2],
            labels={'x': 'Risk (Volatility)', 'y': 'Return', 'color': 'Sharpe Ratio'},
            title="Efficient Frontier"
        )
        fig_ef.update_layout(template=plot_template)
        st.plotly_chart(fig_ef, use_container_width=True)
        
        max_sharpe_idx = np.argmax(results[2])
        best_return = results[0, max_sharpe_idx]
        best_risk = results[1, max_sharpe_idx]
        best_sharpe = results[2, max_sharpe_idx]
        
        st.success(f"""
        **🏆 Best Portfolio Found:**  
        Return: {best_return:.2f}  |  Risk: {best_risk:.2f}  |  Sharpe: {best_sharpe:.2f}
        """)
        
        st.subheader("💼 Optimal Portfolio Allocation")
        best_weights = weights_record[max_sharpe_idx]
        allocation = pd.DataFrame({
            "Stock": raw_df.columns,
            "Weight (%)": (best_weights * 100).round(2)
        }).sort_values('Weight (%)', ascending=False)
        st.dataframe(allocation, use_container_width=True)
    else:
        st.info("Select 2 or more tickers to see portfolio optimization.")

with tab2:
    st.subheader("📈 Price Prediction")
    
    selected_stock = st.selectbox("Select Stock for Prediction", raw_df.columns)
    
    def predict_stock(df, ticker):
        data = df[[ticker]].dropna()
        data['Days'] = np.arange(len(data))
        X = data[['Days']]
        y = data[ticker]
        model = LinearRegression()
        model.fit(X, y)
        future_days = np.arange(len(data), len(data)+30).reshape(-1,1)
        predictions = model.predict(future_days)
        return predictions

    pred = predict_stock(raw_df, selected_stock)
    
    last_date = raw_df.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30)
    
    std = np.std(pred)
    upper = pred + std
    lower = pred - std
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=raw_df.index, y=raw_df[selected_stock], mode='lines', name='Historical Price'))
    
    fig_pred.add_trace(go.Scatter(x=future_dates, y=upper.flatten(), name="Upper Bound", line=dict(dash='dot', color='rgba(255,165,0,0.5)')))
    fig_pred.add_trace(go.Scatter(x=future_dates, y=lower.flatten(), name="Lower Bound", line=dict(dash='dot', color='rgba(255,165,0,0.5)'), fill='tonexty'))
    
    fig_pred.add_trace(go.Scatter(x=future_dates, y=pred.flatten(), mode='lines', name='Prediction', line=dict(dash='dash', color='orange')))
    
    fig_pred.update_layout(template=plot_template, title=f"{selected_stock} ML Prediction with Confidence Band")
    st.plotly_chart(fig_pred, use_container_width=True)

with tab3:
    st.subheader("📉 Risk Metrics (Sharpe Ratio + Volatility)")
    def risk_metrics(df):
        returns = df.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        return volatility, sharpe

    vol, sharpe = risk_metrics(raw_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Volatility (Annualized Risk)")
        st.dataframe(vol, use_container_width=True)
    with col2:
        st.markdown("#### Sharpe Ratio")
        st.dataframe(sharpe, use_container_width=True)
        
    st.markdown("#### Max Drawdown")
    drawdown = (raw_df / raw_df.cummax()) - 1
    fig_dd = px.line(drawdown, title="Drawdown Over Time")
    fig_dd.update_layout(template=plot_template, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_dd, use_container_width=True)
    
    st.divider()
    st.subheader("📊 Strategy Backtest (Momentum Rule)")
    bt_stock = st.selectbox("Select stock to backtest", raw_df.columns, key='bt_stock')
    bt_returns = raw_df[bt_stock].pct_change()
    strategy = bt_returns.copy()
    strategy[bt_returns < 0] = 0  
    cumulative = (1 + strategy).cumprod()
    fig_bt = px.line(cumulative, title=f"Simple Momentum Backtest: {bt_stock}")
    fig_bt.update_layout(template=plot_template)
    st.plotly_chart(fig_bt, use_container_width=True)

with tab4:
    st.subheader(f"📊 Moving Averages (Trend Indicator) for {best_ticker}")
    sma_20 = raw_df[best_ticker].rolling(20).mean()
    ema_20 = raw_df[best_ticker].ewm(span=20).mean()

    fig_ma = go.Figure()
    fig_ma.add_trace(go.Scatter(x=raw_df.index, y=raw_df[best_ticker], name="Price"))
    fig_ma.add_trace(go.Scatter(x=raw_df.index, y=sma_20, name="SMA 20"))
    fig_ma.add_trace(go.Scatter(x=raw_df.index, y=ema_20, name="EMA 20"))
    fig_ma.update_layout(template=plot_template, title=f"{best_ticker} Trends")
    st.plotly_chart(fig_ma, use_container_width=True)

with tab5:
    st.subheader("📰 Live News Sentiment")
    
    selected_stock_sent = st.selectbox("Select Stock for Sentiment", raw_df.columns, key="sentiment_picker")

    st.markdown(f"Fetching latest news for **{selected_stock_sent}** via RSS...")
    
    url = f"https://finance.yahoo.com/rss/headline?s={selected_stock_sent}"
    feed = feedparser.parse(url)

    if feed.entries:
        sentiments = []

        for entry in feed.entries[:5]:
            title = entry.title
            link = entry.link
            
            score = TextBlob(title).sentiment.polarity
            sentiments.append(score)

            if score > 0.1:
                st.success(f"🟢 **{title}**  \n[Read full article]({link})")
            elif score < -0.1:
                st.error(f"🔴 **{title}**  \n[Read full article]({link})")
            else:
                st.info(f"⚪ **{title}**  \n[Read full article]({link})")

        avg = np.mean(sentiments)

        st.markdown("### Overall Sentiment")

        if avg > 0.1:
            st.success("Market Sentiment: Positive 📈")
        elif avg < -0.1:
            st.error("Market Sentiment: Negative 📉")
        else:
            st.info("Market Sentiment: Neutral 😐")
            
        fig_sent = px.bar(
            x=[f"News {i+1}" for i in range(len(sentiments))],
            y=sentiments,
            title=f"News Sentiment Scores ({selected_stock_sent})"
        )
        fig_sent.update_layout(
            template=plot_template,
            margin=dict(t=30, b=0, l=0, r=0),
            xaxis_title="Articles",
            yaxis_title="Sentiment Polarity"
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    else:
        st.error("No news found")

with tab6:
    st.subheader("📊 RSI & MACD Analysis")
    
    tech_stock = st.selectbox("Select Stock for Technicals", raw_df.columns, key="tech_stock")

    # ================= RSI =================
    def calculate_rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    # ================= MACD =================
    def calculate_macd(series):
        ema12 = series.ewm(span=12).mean()
        ema26 = series.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
        
    rsi = calculate_rsi(raw_df[tech_stock])
    macd, signal = calculate_macd(raw_df[tech_stock])

    fig_rsi = go.Figure()
    fig_rsi.add_trace(go.Scatter(x=raw_df.index, y=rsi, name="RSI"))
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
    fig_rsi.update_layout(title="RSI Indicator", template=plot_template, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_rsi, use_container_width=True)

    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=raw_df.index, y=macd, name="MACD"))
    fig_macd.add_trace(go.Scatter(x=raw_df.index, y=signal, name="Signal"))
    fig_macd.update_layout(title="MACD Indicator", template=plot_template, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_macd, use_container_width=True)

    st.subheader("🤖 Smart Trading Signal")
    price_tech = raw_df[tech_stock].iloc[-1]
    sma_tech = raw_df[tech_stock].rolling(20).mean().iloc[-1]
    rsi_latest = rsi.iloc[-1]

    if price_tech > sma_tech and rsi_latest < 70:
        st.success(f"🟢 STRONG BUY ({tech_stock}) - Upside potential with healthy RSI")
    elif price_tech < sma_tech and rsi_latest > 30:
        st.error(f"🔴 STRONG SELL ({tech_stock}) - Downward trend with room to fall")
    else:
        st.info(f"⚪ HOLD ({tech_stock}) - Mixed or neutral technical alignment")

with tab7:
    st.subheader("🧠 LSTM Deep Learning Prediction")

    lstm_stock = st.selectbox("Select Stock for LSTM", raw_df.columns, key="lstm_stock")

    preds = lstm_predict(raw_df, lstm_stock)

    future_dates = pd.date_range(
        start=raw_df.index[-1] + pd.Timedelta(days=1),
        periods=30
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=raw_df.index,
        y=raw_df[lstm_stock],
        name="Historical"
    ))

    fig.add_trace(go.Scatter(
        x=future_dates,
        y=preds.flatten(),
        name="LSTM Prediction",
        line=dict(color="cyan", dash="dash")
    ))

    fig.update_layout(
        template="plotly_dark",
        title=f"{lstm_stock} LSTM Forecast"
    )

    st.plotly_chart(fig, use_container_width=True)
