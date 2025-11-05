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
df = df[df['pred'] != 'no_trade']  # exclude no-trade for accuracy focus
accuracy = (df['pred'] == df['actual']).mean() * 100
avg_prob = df['prob'].mean() * 100

# --- Rolling accuracy (moving window of 50 predictions) ---
df['rolling_acc'] = (df['pred'] == df['actual']).rolling(50).mean() * 100

# --- KPI metrics ---
st.subheader("📊 Model Accuracy Overview")
c1, c2 = st.columns(2)
c1.metric("Overall Accuracy", f"{accuracy:.2f}%")
c2.metric("Avg Confidence", f"{avg_prob:.1f}%")

# --- Accuracy over time chart ---
st.subheader("📈 Rolling Accuracy (window=50 predictions)")
fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(
    x=df['datetime'], y=df['rolling_acc'],
    mode='lines', name='Rolling Accuracy', line=dict(color='limegreen', width=3)
))
fig_acc.update_layout(yaxis_title="Accuracy (%)", template="plotly_white")
st.plotly_chart(fig_acc, use_container_width=True)

# --- Price chart with colored confidence ---
st.subheader("📉 BTC Price & Model Predictions")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['datetime'], y=df['price'], mode='lines', name='BTC Price',
    line=dict(color='lightgray', width=2)
))

fig.add_trace(go.Scatter(
    x=df['datetime'], y=df['price'],
    mode='markers',
    marker=dict(
        color=df['prob'],
        colorscale='Viridis',
        size=8,
        colorbar=dict(title="Confidence"),
        showscale=True
    ),
    name="Predicted signal (color = confidence)"
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

# --- Interactive simulation ---
st.subheader("▶ Live Prediction Simulation")
play = st.button("Play simulation (step through time)")

if play:
    placeholder_chart = st.empty()
    placeholder_metric = st.empty()

    for step in range(10, len(df), 10):
        subset = df.iloc[:step]
        acc = (subset['pred'] == subset['actual']).mean() * 100

        fig_live = go.Figure()
        fig_live.add_trace(go.Scatter(
            x=subset['datetime'], y=subset['price'], mode='lines', name='BTC Price',
            line=dict(color='lightgray', width=2)
        ))
        fig_live.add_trace(go.Scatter(
            x=subset.loc[subset['pred'] == subset['actual'], 'datetime'],
            y=subset.loc[subset['pred'] == subset['actual'], 'price'],
            mode='markers', name='✅ Correct', marker=dict(color='green', size=6)
        ))
        fig_live.add_trace(go.Scatter(
            x=subset.loc[subset['pred'] != subset['actual'], 'datetime'],
            y=subset.loc[subset['pred'] != subset['actual'], 'price'],
            mode='markers', name='❌ Wrong', marker=dict(color='red', size=6)
        ))
        fig_live.update_layout(title=f"Step {step}/{len(df)}", template="plotly_white")

        placeholder_chart.plotly_chart(fig_live, use_container_width=True)
        placeholder_metric.metric("Accuracy so far", f"{acc:.2f}%")
        time.sleep(0.2)

st.subheader("🧾 Recent Predictions")
st.dataframe(df.tail(20).sort_values('datetime', ascending=False))
