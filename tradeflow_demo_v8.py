
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

# --- Фильтр по датам ---
min_date, max_date = df["datetime"].min(), df["datetime"].max()
st.markdown("Use the slider below to select the date range to visualize:")
date_range = st.slider(
    "Date range:",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    step=pd.Timedelta(days=1),
    format="YYYY-MM-DD"
)

# Фильтруем данные
mask = (df["datetime"] >= pd.Timestamp(date_range[0])) & (df["datetime"] <= pd.Timestamp(date_range[1]))
filtered_df = df.loc[mask]
filtered_trades = filtered_df[filtered_df["pred"] != "no_trade"]

fig = go.Figure()

# Линия цены
fig.add_trace(go.Scatter(
    x=filtered_df["datetime"], y=filtered_df["price"],
    mode="lines", name="BTC Price", line=dict(color="lightgray", width=1)
))

# Сигналы
correct = filtered_trades[filtered_trades["pred"] == filtered_trades["actual"]]
wrong = filtered_trades[filtered_trades["pred"] != filtered_trades["actual"]]
no_trade = filtered_df[filtered_df["pred"] == "no_trade"]

if not correct.empty:
    fig.add_trace(go.Scatter(
        x=correct["datetime"], y=correct["price"],
        mode="markers", name="✅ Correct",
        marker=dict(color="green", size=7, symbol="triangle-up")
    ))
if not wrong.empty:
    fig.add_trace(go.Scatter(
        x=wrong["datetime"], y=wrong["price"],
        mode="markers", name="❌ Wrong",
        marker=dict(color="red", size=7, symbol="x")
    ))
if not no_trade.empty:
    fig.add_trace(go.Scatter(
        x=no_trade["datetime"], y=no_trade["price"],
        mode="markers", name="⚪ No Trade",
        marker=dict(color="orange", size=6, symbol="circle-open")
    ))

fig.update_layout(
    height=500,
    margin=dict(l=30, r=30, t=40, b=30),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    xaxis=dict(
        rangeslider=dict(visible=True, bgcolor="#1e1e1e", thickness=0.08),
        type="date",
        showgrid=False
    ),
    yaxis=dict(showgrid=False),
    template="plotly_dark"
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
st.markdown("Interactive playback of model predictions over time (TradingView-style).")

speed = st.slider("Speed (seconds per step)", 0.1, 1.0, 0.25)
placeholder = st.empty()
metric_placeholder = st.empty()

start = st.button("▶️ Start Simulation")
stop_flag = st.session_state.get("stop_sim", False)

if st.button("⏹ Stop Simulation"):
    st.session_state["stop_sim"] = True

if start:
    st.session_state["stop_sim"] = False  # сбрасываем стоп при новом запуске

    df["sma"] = df["price"].rolling(20).mean()
    total_steps = len(df)
    trade_df = df[df["pred"] != "no_trade"]

    for i in range(30, total_steps, 10):
        if st.session_state.get("stop_sim"):
            st.warning("⏸ Simulation stopped.")
            break

        subset = df.iloc[:i]
        trades = subset[subset["pred"] != "no_trade"]
        if len(trades) > 0:
            live_acc = (trades["pred"] == trades["actual"]).mean() * 100
        else:
            live_acc = 0

        # --- построение графика ---
        sim_fig = go.Figure()

        sim_fig.add_trace(go.Candlestick(
            x=subset["datetime"],
            open=subset["price"].shift(1).fillna(subset["price"]),
            high=subset["price"].rolling(3).max(),
            low=subset["price"].rolling(3).min(),
            close=subset["price"],
            name="BTC/USDT",
            increasing_line_color="#26A69A",
            decreasing_line_color="#EF5350",
            showlegend=False
        ))

        # SMA
        sim_fig.add_trace(go.Scatter(
            x=subset["datetime"], y=subset["sma"],
            mode="lines", name="SMA 20", line=dict(color="#FFA726", width=1.5)
        ))

        # Сигналы
        correct_live = subset[(subset["pred"] == subset["actual"]) & (subset["pred"] != "no_trade")]
        wrong_live = subset[(subset["pred"] != subset["actual"]) & (subset["pred"] != "no_trade")]
        no_trade_live = subset[subset["pred"] == "no_trade"]

        sim_fig.add_trace(go.Scatter(
            x=correct_live["datetime"], y=correct_live["price"],
            mode="markers", name="✅ Correct",
            marker=dict(color="#00E676", size=10, symbol="triangle-up")
        ))
        sim_fig.add_trace(go.Scatter(
            x=wrong_live["datetime"], y=wrong_live["price"],
            mode="markers", name="❌ Wrong",
            marker=dict(color="#FF1744", size=10, symbol="x")
        ))
        sim_fig.add_trace(go.Scatter(
            x=no_trade_live["datetime"], y=no_trade_live["price"],
            mode="markers", name="⚪ No Trade",
            marker=dict(color="#FFD600", size=8, symbol="circle-open")
        ))

        sim_fig.update_layout(
            template="plotly_dark",
            height=500,
            margin=dict(l=30, r=30, t=40, b=30),
            xaxis_rangeslider_visible=False,
            yaxis_title="BTC Price",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        metric_placeholder.metric("📊 Live Accuracy", f"{live_acc:.2f}%")
        placeholder.plotly_chart(sim_fig, use_container_width=True)
        time.sleep(speed)


# --- Таблица ---
st.subheader("🧾 Recent Predictions")
st.dataframe(df.tail(30).sort_values('datetime', ascending=False))
