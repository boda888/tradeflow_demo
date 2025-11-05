import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from pathlib import Path

st.set_page_config(page_title="TradeFlow Demo", layout="wide")

st.title("📈 TradeFlow — Automated ML Trading Demo")
st.markdown("""
This demo illustrates how the **TradeFlow ML model** predicts short-term BTC price movements.  
The system updates every 15 minutes and forecasts the **next 1-hour direction** (Up / Down / No trade).  
The focus here is on **prediction accuracy and model confidence** rather than profit.
""")

# --- Load demo data automatically ---
csv_path = Path("tradeflow_demo.csv")


if csv_path.exists():
    df = pd.read_csv(csv_path)
    st.success(f"✅ Loaded demo data automatically: `{csv_path.name}`")
else:
    st.error("❌ CSV file not found. Please make sure `tradeflow_demo.csv` is in the same folder.")
    st.stop()

# --- Prepare data ---
df['datetime'] = pd.to_datetime(df['datetime'])
df['is_trade'] = df['pred'] != 'no_trade'

# --- Метрики только по up/down ---
df_trades = df[df['is_trade']]
accuracy = (df_trades['pred'] == df_trades['actual']).mean() * 100
avg_prob = df_trades['prob'].mean() * 100

# --- Rolling accuracy по сделкам ---
df['rolling_acc'] = (
    (df['pred'] == df['actual'])
    .where(df['is_trade'])
    .rolling(50)
    .mean() * 100
)

# --- KPI ---
st.subheader("📊 Model Accuracy Overview")
c1, c2 = st.columns(2)
c1.metric("Overall Accuracy (trades only)", f"{accuracy:.2f}%")
c2.metric("Avg Confidence", f"{avg_prob:.1f}%")

# --- Price chart: добавляем no_trade серым цветом ---
st.subheader("📉 BTC Price & Model Predictions")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['datetime'], y=df['price'],
    mode='lines', name='BTC Price',
    line=dict(color='lightgray', width=2)
))

# no_trade
no_trade = df[df['pred'] == 'no_trade']
fig.add_trace(go.Scatter(
    x=no_trade['datetime'], y=no_trade['price'],
    mode='markers', name='No trade',
    marker=dict(color='gray', size=6, symbol='circle')
))

# correct / wrong trades
correct = df_trades[df_trades['pred'] == df_trades['actual']]
wrong = df_trades[df_trades['pred'] != df_trades['actual']]

fig.add_trace(go.Scatter(
    x=correct['datetime'], y=correct['price'],
    mode='markers', name='✅ Correct', marker=dict(color='green', size=8)
))
fig.add_trace(go.Scatter(
    x=wrong['datetime'], y=wrong['price'],
    mode='markers', name='❌ Wrong', marker=dict(color='red', size=8)
))
st.plotly_chart(fig, use_container_width=True)

# --- Таблица с подсветкой ---
st.subheader("🧾 Recent Predictions")

def highlight_no_trade(row):
    color = '#f0f0f0' if row['pred'] == 'no_trade' else 'white'
    return ['background-color: {}'.format(color)] * len(row)

styled_df = df.tail(25).sort_values('datetime', ascending=False).style.apply(highlight_no_trade, axis=1)
st.dataframe(styled_df, use_container_width=True)
