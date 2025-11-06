
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

# --- Модельные метрики (Model Insights) ---
trades = df[df['pred'] != 'no_trade']
accuracy = (trades['pred'] == trades['actual']).mean() * 100 if len(trades) > 0 else 0
no_trade_ratio = (df['pred'] == 'no_trade').mean() * 100
returns = df.get('pnl', pd.Series(0)).fillna(0)
max_drawdown = (1 + returns).cumprod().div((1 + returns).cumprod().cummax()).min() - 1

# Новые метрики
total_trades = len(trades)
correct_trades = (trades['pred'] == trades['actual']).sum()
wrong_trades = (trades['pred'] != trades['actual']).sum()

# --- Секция Model Insights ---
st.subheader("📊 Model Insights Overview")

# --- Glass-style оформление ---
st.markdown("""
<style>
div[data-testid="metric-container"] {
    background: rgba(10,25,47,0.7);
    border-radius: 12px;
    padding: 10px 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.25);
    border: 1px solid rgba(30,60,120,0.4);
    transition: all 0.3s ease-in-out;
}
div[data-testid="metric-container"]:hover {
    box-shadow: 0 0 10px rgba(33,150,243,0.5);
}
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
    color: #90CAF9;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    letter-spacing: 0.3px;
}
div[data-testid="stMetricValue"] {
    color: #42A5F5;
    font-weight: 600;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# --- Символы стрелок ---
arrow_up_svg = "&#9650;"   # зелёная ▲
arrow_down_svg = "&#9660;" # красная ▼

# --- Верхний ряд ---
c1, c2, c3 = st.columns(3)
c1.metric("Overall Accuracy (Trades Only)", f"{accuracy:.2f}%")
c2.metric("No-Trade Ratio", f"{no_trade_ratio:.1f}%")
c3.metric("Max Drawdown", f"{max_drawdown * 100:.2f}%")

# --- Нижний ряд ---
c4, c5, c6 = st.columns(3)
c4.metric("Total Trades", f"{total_trades}")
c5.markdown(
    f"<div style='font-size:20px; color:#4CAF50; font-weight:600;'>{arrow_up_svg} {correct_trades}</div>"
    f"<div style='font-size:13px; color:#90CAF9;'>Correct Trades</div>",
    unsafe_allow_html=True
)
c6.markdown(
    f"<div style='font-size:20px; color:#E53935; font-weight:600;'>{arrow_down_svg} {wrong_trades}</div>"
    f"<div style='font-size:13px; color:#90CAF9;'>Wrong Trades</div>",
    unsafe_allow_html=True
)










# --- Основной график ---
st.subheader("📉 BTC Price & Model Predictions")

# --- Ползунок выбора диапазона ---
min_date, max_date = df["datetime"].min(), df["datetime"].max()
date_range = st.slider(
    "Select Date Range",
    min_value=min_date.to_pydatetime(),
    max_value=max_date.to_pydatetime(),
    value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
    step=pd.Timedelta(days=1),
    format="YYYY-MM-DD"
)

# --- Фильтрация ---
mask = (df["datetime"] >= pd.Timestamp(date_range[0])) & (df["datetime"] <= pd.Timestamp(date_range[1]))
filtered_df = df.loc[mask]
filtered_trades = filtered_df[filtered_df["pred"] != "no_trade"]

# --- Метрика точности ---
if not filtered_trades.empty:
    range_accuracy = (filtered_trades["pred"] == filtered_trades["actual"]).mean() * 100
else:
    range_accuracy = 0

st.markdown(
    f"""
    <div style='background-color:rgba(0,188,212,0.08);
                border-radius:10px;
                padding:10px 15px;
                margin-bottom:10px;
                width:220px;
                text-align:center;
                font-family:Inter, sans-serif;'>
        <span style='font-size:13px; color:#80DEEA;'>Accuracy for selected range</span><br>
        <span style='font-size:28px; font-weight:600; color:#00E5FF;'>{range_accuracy:.2f}%</span>
    </div>
    """, unsafe_allow_html=True
)

# --- Построение графика ---
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=filtered_df["datetime"], y=filtered_df["price"],
    mode="lines", name="BTC Price",
    line=dict(color="lightgray", width=1)
))

correct = filtered_trades[filtered_trades["pred"] == filtered_trades["actual"]]
wrong = filtered_trades[filtered_trades["pred"] != filtered_trades["actual"]]
no_trade = filtered_df[filtered_df["pred"] == "no_trade"]

if not correct.empty:
    fig.add_trace(go.Scatter(
        x=correct["datetime"], y=correct["price"],
        mode="markers", name=" Correct",
        marker=dict(color="green", size=7, symbol="triangle-up")
    ))
if not wrong.empty:
    fig.add_trace(go.Scatter(
        x=wrong["datetime"], y=wrong["price"],
        mode="markers", name=" Wrong",
        marker=dict(color="red", size=7, symbol="x")
    ))
if not no_trade.empty:
    fig.add_trace(go.Scatter(
        x=no_trade["datetime"], y=no_trade["price"],
        mode="markers", name=" No Trade",
        marker=dict(color="orange", size=6, symbol="circle-open")
    ))

fig.update_layout(
    height=500,
    margin=dict(l=30, r=30, t=40, b=30),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=24)  # 👈 увеличиваем размер шрифта легенды (примерно х2)
    ),
    xaxis=dict(rangeslider=dict(visible=False), type="date", showgrid=False),
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

# ⚙️ Настройки
speed = st.slider("⏱️ Speed (seconds per step)", 0.05, 1.0, 0.25)
step_size = st.slider("📏 Step size (bars per tick)", 1, 20, 10)

# 🧱 Плейсхолдеры
placeholder = st.empty()
metric_placeholder = st.empty()

# 🎮 Кнопки управления — теперь в одном ряду, слева
col_start, col_stop = st.columns([0.15, 0.15])  # компактное размещение
with col_start:
    start = st.button("▶️ Start Simulation", use_container_width=True)
with col_stop:
    stop = st.button("⏹ Stop Simulation", use_container_width=True)

if stop:
    st.session_state["stop_sim"] = True
stop_flag = st.session_state.get("stop_sim", False)

if start:
    st.session_state["stop_sim"] = False

    df["sma"] = df["price"].rolling(20).mean()
    total_steps = len(df)
    trade_df = df[df["pred"] != "no_trade"]

    for i in range(30, total_steps, step_size):
        if st.session_state.get("stop_sim"):
            st.warning("⏸ Simulation stopped.")
            break

        subset = df.iloc[:i]
        trades = subset[subset["pred"] != "no_trade"]
        if len(trades) > 0:
            live_acc = (trades["pred"] == trades["actual"]).mean() * 100
        else:
            live_acc = 0

        # --- График ---
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

        sim_fig.add_trace(go.Scatter(
            x=subset["datetime"], y=subset["sma"],
            mode="lines", name="SMA 20", line=dict(color="#FFA726", width=1.5)
        ))

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
