
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time

# --- Настройки страницы ---
st.set_page_config(page_title="TradeFlow Demo", layout="wide")

st.title("📈 TradeFlow — Automated ML Trading Demo")
st.markdown("""
This demo illustrates how the **TradeFlow ML model** predicts short-term BTC price movements.  
The system updates every 15 minutes and forecasts the next 1-hour direction (Up / Down / No trade).  
The focus here is on **prediction accuracy and model confidence**, rather than profit.
""")

# --- Загрузка CSV ---
csv_path = Path("tradeflow_demo.csv")

if not csv_path.exists():
    st.error("❌ File `tradeflow_demo.csv` not found. Please upload it to the same directory.")
    st.stop()

df = pd.read_csv(csv_path)
st.success(f"✅ Loaded demo data automatically: `{csv_path.name}`")

# --- Очистка данных ---
df.columns = [c.strip().lower() for c in df.columns]
for col in ['pred', 'actual']:
    if col in df:
        df[col] = df[col].astype(str).str.strip().str.lower()

df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
df = df.dropna(subset=['datetime', 'price'])
df = df.sort_values('datetime').reset_index(drop=True)

# --- Equity ---
df['equity'] = (1 + df.get('pnl', 0)).cumprod() * 10000

# --- Проверка пустых данных ---
if df.empty:
    st.warning("⚠️ CSV загружен, но нет данных для отображения.")
    st.stop()

# --- Метрики ---
trades = df[df['pred'] != 'no_trade']
accuracy = (trades['pred'] == trades['actual']).mean() * 100 if len(trades) > 0 else 0
avg_prob = df['prob'].mean() * 100 if 'prob' in df else 0
total_pnl = df.get('pnl', pd.Series(0)).sum() * 100

st.subheader("📊 Model Accuracy Overview")
c1, c2, c3 = st.columns(3)
c1.metric("Overall Accuracy (trades only)", f"{accuracy:.2f}%")
c2.metric("Avg Confidence", f"{avg_prob:.1f}%")
c3.metric("Total Return (simulated)", f"{total_pnl:.2f}%")

# --- Основной график ---
st.subheader("📉 BTC Price & Model Predictions")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['datetime'], y=df['price'],
    mode='lines', name='BTC Price', line=dict(color='lightgray', width=1)
))

correct = trades[trades['pred'] == trades['actual']]
wrong = trades[trades['pred'] != trades['actual']]
no_trade = df[df['pred'] == 'no_trade']

if not correct.empty:
    fig.add_trace(go.Scatter(
        x=correct['datetime'], y=correct['price'],
        mode='markers', name='✅ Correct',
        marker=dict(color='green', size=6)
    ))
if not wrong.empty:
    fig.add_trace(go.Scatter(
        x=wrong['datetime'], y=wrong['price'],
        mode='markers', name='❌ Wrong',
        marker=dict(color='red', size=6, symbol='x')
    ))
if not no_trade.empty:
    fig.add_trace(go.Scatter(
        x=no_trade['datetime'], y=no_trade['price'],
        mode='markers', name='⚪ No Trade',
        marker=dict(color='orange', size=5, symbol='circle-open')
    ))

fig.update_layout(
    height=500,
    margin=dict(l=30, r=30, t=40, b=30),
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
)
st.plotly_chart(fig, use_container_width=True)


# --- Rolling Accuracy Chart ---
st.subheader("📈 Rolling Accuracy Over Time (excl. No Trade)")
window = 30  # кол-во сделок для сглаживания
df_trades = df[df['pred'] != 'no_trade'].copy()
df_trades['is_correct'] = (df_trades['pred'] == df_trades['actual']).astype(int)
df_trades['rolling_acc'] = df_trades['is_correct'].rolling(window).mean() * 100

fig_acc = go.Figure()
fig_acc.add_trace(go.Scatter(
    x=df_trades['datetime'],
    y=df_trades['rolling_acc'],
    mode='lines',
    line=dict(color='blue', width=2),
    name=f'Rolling Accuracy ({window} trades)'
))
fig_acc.update_layout(height=300, margin=dict(l=30, r=30, t=40, b=30))
st.plotly_chart(fig_acc, use_container_width=True)



# --- Equity Curve ---
st.subheader("💰 Simulated Equity Curve")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df['datetime'], y=df['equity'],
    mode='lines', name='Equity', line=dict(color='teal', width=2)
))
fig2.update_layout(height=400, margin=dict(l=30, r=30, t=40, b=30))
st.plotly_chart(fig2, use_container_width=True)


# --- Live Simulation ---
st.subheader("🎬 Live Prediction Simulation")
st.markdown("Smooth playback of model predictions over time (TradingView-style).")

speed = st.slider("Speed (seconds per step)", 0.1, 1.0, 0.25)

if st.button("▶️ Start Simulation"):
    placeholder = st.empty()
    window = 20  # для скользящей средней
    df["sma"] = df["price"].rolling(window).mean()

    for i in range(window, len(df), 10):
        subset = df.iloc[:i]

        sim_fig = go.Figure()

        # --- Свечной график ---
        sim_fig.add_trace(go.Candlestick(
            x=subset["datetime"],
            open=subset["price"].shift(1).fillna(subset["price"]),
            high=subset["price"].rolling(3).max(),
            low=subset["price"].rolling(3).min(),
            close=subset["price"],
            name="BTC/USDT",
            increasing_line_color="green",
            decreasing_line_color="red",
            showlegend=False
        ))

        # --- Скользящая средняя ---
        sim_fig.add_trace(go.Scatter(
            x=subset["datetime"],
            y=subset["sma"],
            mode="lines",
            name=f"SMA {window}",
            line=dict(color="orange", width=1.5)
        ))

        # --- Сигналы модели ---
        correct_live = subset[(subset["pred"] == subset["actual"]) & (subset["pred"] != "no_trade")]
        wrong_live = subset[(subset["pred"] != subset["actual"]) & (subset["pred"] != "no_trade")]
        no_trade_live = subset[subset["pred"] == "no_trade"]

        # Точки сделок
        sim_fig.add_trace(go.Scatter(
            x=correct_live["datetime"], y=correct_live["price"],
            mode="markers", name="✅ Correct", marker=dict(color="limegreen", size=7, symbol="triangle-up")
        ))
        sim_fig.add_trace(go.Scatter(
            x=wrong_live["datetime"], y=wrong_live["price"],
            mode="markers", name="❌ Wrong", marker=dict(color="red", size=7, symbol="x")
        ))
        sim_fig.add_trace(go.Scatter(
            x=no_trade_live["datetime"], y=no_trade_live["price"],
            mode="markers", name="⚪ No Trade", marker=dict(color="gray", size=5, symbol="circle")
        ))

        # --- Внешний вид графика ---
        sim_fig.update_layout(
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=500,
            margin=dict(l=30, r=30, t=40, b=30),
            yaxis_title="BTC Price",
            xaxis_title="Time",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        placeholder.plotly_chart(sim_fig, use_container_width=True)
        time.sleep(speed)

# --- Таблица ---
st.subheader("🧾 Recent Predictions")
st.dataframe(df.tail(30).sort_values('datetime', ascending=False))
