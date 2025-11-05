import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="TradeFlow Demo", layout="wide")

st.title("📈 TradeFlow — Automated ML Trading Demo")
st.markdown("""
Demo of the ML trading system predicting short-term market movements on BTC.  
The model predicts hourly direction, evaluates confidence, and simulates risk-controlled trading.
""")

# --- Try to load demo data automatically ---
csv_path = Path("tradeflow_demo.csv")

if csv_path.exists():
    df = pd.read_csv(csv_path)
    st.success(f"✅ Loaded demo data automatically: `{csv_path.name}`")
else:
    st.error("❌ CSV file not found. Please make sure `tradeflow_demo.csv` is in the same folder.")
    st.stop()

# --- Prepare data ---
df['datetime'] = pd.to_datetime(df['datetime'])
df['equity'] = (1 + df['pnl']).cumprod() * 10000
accuracy = (df['pred'] == df['actual']).mean() * 100
avg_prob = df['prob'].mean() * 100
total_pnl = df['pnl'].sum() * 100

# --- Metrics ---
st.subheader("📊 Model Performance Summary")
c1, c2, c3 = st.columns(3)
c1.metric("Accuracy", f"{accuracy:.2f}%")
c2.metric("Avg Confidence", f"{avg_prob:.1f}%")
c3.metric("Total Return", f"{total_pnl:.1f}%")

# --- Price Chart with signals ---
st.subheader("📉 BTC Price & Model Predictions")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['datetime'], y=df['price'],
    mode='lines', name='BTC Price', line=dict(color='gray')
))

correct = df[df['pred'] == df['actual']]
wrong = df[df['pred'] != df['actual']]

fig.add_trace(go.Scatter(
    x=correct['datetime'], y=correct['price'],
    mode='markers', name='✅ Correct', marker=dict(color='green', size=8)
))
fig.add_trace(go.Scatter(
    x=wrong['datetime'], y=wrong['price'],
    mode='markers', name='❌ Wrong', marker=dict(color='red', size=8)
))

st.plotly_chart(fig, use_container_width=True)

# --- Equity curve ---
st.subheader("💰 Equity Curve (Simulated Portfolio)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df['datetime'], y=df['equity'],
    mode='lines', name='Equity', line=dict(color='teal')
))
st.plotly_chart(fig2, use_container_width=True)

# --- Optional: dynamic slider ---
st.subheader("⏱️ Step-by-step simulation")
step = st.slider("Show data up to step:", 1, len(df), len(df))
subset = df.iloc[:step]
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=subset['datetime'], y=subset['price'],
                          mode='lines', name='BTC Price', line=dict(color='gray')))
fig3.add_trace(go.Scatter(x=subset['datetime'], y=subset['equity'],
                          mode='lines', name='Equity', line=dict(color='teal')))
st.plotly_chart(fig3, use_container_width=True)

st.subheader("🧾 Recent Predictions")
st.dataframe(df.tail(20).sort_values('datetime', ascending=False))
