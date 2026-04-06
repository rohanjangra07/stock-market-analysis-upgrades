# Stock Peer Analysis Dashboard

A **premium‑grade** Streamlit dashboard for comparing and analysing peer stocks.

## ✨ Features
- **Price Comparison** chart with smooth spline lines, custom colour palette and glow fill.
- **Market Snapshot** KPI cards (average return, volatility, Sharpe ratio, top stock).
- **Stock Ranking** table with a viridis gradient background.
- **Performance Comparison** bar chart (full‑width, responsive).
- **Glass‑morphism** UI styling for a modern, premium look.
- **Sidebar controls** for theme toggle, ticker selection, date range and advanced options.
- **LSTM‑based price forecasting** (TensorFlow) and portfolio optimisation utilities.
- **AI‑generated insights** powered by a lightweight language model.

## 🛠️ Installation
```bash
# Clone the repo (already done) and navigate to the project folder
cd "c:/Users/rohan/stock edit 3"

# Create a virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## ▶️ Running the Dashboard
```bash
streamlit run app.py
```
Open the URL shown in the terminal (usually `http://localhost:8504`).

## 📸 Screenshots
*(The dashboard is rendered with glass‑morphism, gradient tables and smooth charts.)*

## 📦 Dependencies
Key libraries are listed in `requirements.txt`:
- streamlit
- pandas, numpy, yfinance
- plotly
- tensorflow (for LSTM forecasting)
- scikit‑learn, textblob, feedparser (sentiment & news)

## 🤝 Contributing
Feel free to fork the repo, open a PR, or suggest enhancements via Issues.

---
*Created by Rohan Jangra – © 2026*
