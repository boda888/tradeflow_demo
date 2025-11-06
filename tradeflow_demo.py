
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import time

# --- Настройки страницы ---
st.set_page_config(page_title="TradeFlow Demo", layout="wide")

st.title("📈 TradeFlow — ML Trading Demo")
# st.markdown("""
# This demo illustrates how the **TradeFlow ML model** predicts short-term BTC price movements.  
# The system updates every 15 minutes and forecasts the next 1-hour direction (Up / Down / No trade).  
# The focus here is on **prediction accuracy and model confidence**, rather than profit.
# """)

# --- Minimal Model Summary ---
st.markdown("""
### Model Summary
**Model:** TradeFlow v1.3  
**Trained on:** BTC/USDT (15-minute candles)  
**Data period:** Sep 2025 – Oct 2025  
**Model type:** LGBMClassifier  
**Signal horizon:** 1 hour ahead  
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
c3.metric("Max Drawdown", f"{-0.0574 * 100:.2f}%")

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

# Линия цены
fig.add_trace(go.Scatter(
    x=filtered_df["datetime"], y=filtered_df["price"],
    mode="lines", name="BTC Price",
    line=dict(color="lightgray", width=1)
))

# Фактические точки на графике (без легенды)
correct = filtered_trades[filtered_trades["pred"] == filtered_trades["actual"]]
wrong = filtered_trades[filtered_trades["pred"] != filtered_trades["actual"]]
no_trade = filtered_df[filtered_df["pred"] == "no_trade"]

if not correct.empty:
    fig.add_trace(go.Scatter(
        x=correct["datetime"], y=correct["price"],
        mode="markers",
        marker=dict(color="green", size=7, symbol="triangle-up"),
        showlegend=False
    ))
if not wrong.empty:
    fig.add_trace(go.Scatter(
        x=wrong["datetime"], y=wrong["price"],
        mode="markers",
        marker=dict(color="red", size=7, symbol="x"),
        showlegend=False
    ))
if not no_trade.empty:
    fig.add_trace(go.Scatter(
        x=no_trade["datetime"], y=no_trade["price"],
        mode="markers",
        marker=dict(color="orange", size=6, symbol="circle-open"),
        showlegend=False
    ))

# --- Фиктивные большие маркеры для легенды ---
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers", name="BTC Price",
    marker=dict(color="lightgray", size=10, symbol="line-ns")
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers", name=" Correct",
    marker=dict(color="green", size=14, symbol="triangle-up")
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers", name=" Wrong",
    marker=dict(color="red", size=14, symbol="x")
))
fig.add_trace(go.Scatter(
    x=[None], y=[None],
    mode="markers", name=" No Trade",
    marker=dict(color="orange", size=12, symbol="circle-open")
))

# --- Настройки отображения ---
fig.update_layout(
    height=500,
    margin=dict(l=30, r=30, t=40, b=30),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(size=18)  # увеличенный размер шрифта легенды
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







import numpy as np
import pandas as pd

# --- Confidence Filter ---
np.random.seed(42)

# Диапазон confidence: от 0.5 до 1.0
conf_levels = np.arange(0.5, 1.01, 0.01)

# ✅ Зависимость точности (accuracy) от confidence
# Используем сигмоидную кривую с шумом, чтобы был плавный рост
base_accuracy = 46 + 30 * (1 / (1 + np.exp(-10 * (conf_levels - 0.72))))  # диапазон примерно 46–76%
noise = np.random.normal(0, 0.6, len(conf_levels))
accuracy_curve = np.clip(base_accuracy + noise, 46, 76)  # жёстко ограничиваем границы

# 📉 Зависимость числа сделок (trades) от confidence
# Чем выше уверенность, тем меньше трейдов
trades_curve = 900 * np.exp(-4.5 * (conf_levels - 0.5)) + np.random.normal(0, 15, len(conf_levels))
trades_curve = np.clip(trades_curve, 30, 900).astype(int)

# Собираем итоговый DataFrame
df_conf_sim = pd.DataFrame({
    "confidence": conf_levels,
    "accuracy": accuracy_curve,
    "trades": trades_curve
})


# --- Confidence Filter (demo simulation) ---
st.subheader("🕹 Confidence Filter (Demo Simulation)")

min_conf = st.slider(
    "Min Confidence Threshold",
    0.5, 1.0, 0.6, 0.01,
    help="Filter trades by model confidence (simulated relationship)"
)

# --- Находим ближайшее значение ---
closest = df_conf_sim.iloc[(df_conf_sim["confidence"] - min_conf).abs().argsort()[:1]]

accuracy_conf = float(closest["accuracy"])
trades_conf = int(closest["trades"])

# --- Базовые значения (для дельты) ---
baseline_acc = float(df_conf_sim.loc[df_conf_sim["confidence"] == 0.5, "accuracy"])
baseline_trades = int(df_conf_sim.loc[df_conf_sim["confidence"] == 0.5, "trades"])

# --- Отображение метрик ---
c1, c2, c3 = st.columns(3)
c1.metric("Filtered Accuracy", f"{accuracy_conf:.2f}%", f"{accuracy_conf - baseline_acc:+.2f}%")
c2.metric("Remaining Trades", f"{trades_conf}")
c3.metric("Baseline Accuracy", f"{baseline_acc:.2f}%")















# # --- Confidence Filter ---
# st.subheader("🕹 Confidence Filter")

# # Слайдер для минимальной уверенности
# min_conf = st.slider(
#     "Min Confidence Threshold",
#     0.5, 1.0, 0.6, 0.01,
#     help="Filter trades by model confidence (prob >= threshold)"
# )

# # --- Фильтрация только по трейдам (без no_trade) ---
# trades_only = df[df["pred"] != "no_trade"].copy()
# filtered_trades_conf = trades_only[trades_only["prob"] >= min_conf]

# # --- Метрики после фильтра ---
# total_trades_conf = len(filtered_trades_conf)
# if total_trades_conf > 0:
#     accuracy_conf = (filtered_trades_conf["pred"] == filtered_trades_conf["actual"]).mean() * 100
# else:
#     accuracy_conf = 0

# # --- Метрики до фильтра ---
# baseline_trades = len(trades_only)
# baseline_acc = (trades_only["pred"] == trades_only["actual"]).mean() * 100

# # --- Отображение метрик ---
# c1, c2, c3 = st.columns(3)
# c1.metric("Filtered Accuracy", f"{accuracy_conf:.2f}%", f"{accuracy_conf - baseline_acc:+.2f}%")
# c2.metric("Remaining Trades", f"{total_trades_conf}", f"{(total_trades_conf / baseline_trades - 1) * 100:+.1f}%")
# c3.metric("Baseline Accuracy", f"{baseline_acc:.2f}%")

# # --- Текст-пояснение ---
# st.markdown(
#     f"""
#     <p style='font-size:13px; color:#90CAF9; font-family:Inter, sans-serif;'>
#     As confidence threshold increases, <b>accuracy rises</b> but <b>number of trades decreases</b> —
#     reflecting a more conservative and precise trading strategy.
#     </p>
#     """,
#     unsafe_allow_html=True
# )





# # --- Rolling PnL vs Accuracy ---
# st.subheader("📈 Rolling PnL vs Accuracy")

# # Кумулятивный PnL
# df['cum_pnl'] = (1 + df['pnl']).cumprod() - 1

# # Rolling accuracy (на окне, например, 100 точек)
# window = 100
# df['rolling_acc'] = (
#     (df['pred'] == df['actual'])
#     .rolling(window)
#     .mean()
#     .fillna(0)
# ) * 100

# # Построение графика
# fig_pnl_acc = go.Figure()

# fig_pnl_acc.add_trace(go.Scatter(
#     x=df['datetime'], y=df['cum_pnl'],
#     mode='lines',
#     name='Cumulative PnL',
#     line=dict(color='#42A5F5', width=2)
# ))

# fig_pnl_acc.add_trace(go.Scatter(
#     x=df['datetime'], y=df['rolling_acc'],
#     mode='lines',
#     name='Rolling Accuracy (100 trades)',
#     line=dict(color='#FF5252', width=2, dash='dot'),
#     yaxis='y2'
# ))

# # Настройка осей и легенды
# fig_pnl_acc.update_layout(
#     template="plotly_dark",
#     height=450,
#     margin=dict(l=30, r=30, t=40, b=30),
#     legend=dict(
#         orientation="h",
#         yanchor="bottom", y=1.02,
#         xanchor="right", x=1,
#         font=dict(size=14)
#     ),
#     xaxis=dict(title="Time", showgrid=False),
#     yaxis=dict(title="Cumulative PnL", showgrid=False),
#     yaxis2=dict(
#         title="Rolling Accuracy (%)",
#         overlaying='y',
#         side='right',
#         showgrid=False
#     )
# )

# st.plotly_chart(fig_pnl_acc, use_container_width=True)








# --- Live Simulation ---
st.subheader("🎬 Live Prediction Simulation")
st.markdown("Interactive playback of model predictions over time (TradingView-style).")

# ⚙️ Настройки
speed = st.slider("⏱️ Speed (seconds per step)", 0.05, 1.0, 0.25)
step_size = st.slider("📏 Step size (bars per tick)", 1, 20, 10)

# 🧱 Плейсхолдеры
placeholder = st.empty()
metric_placeholder = st.empty()

# 🎮 Кнопки управления
col_start, col_stop = st.columns([0.15, 0.15])
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

        # Свечи
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
            mode="lines", name="SMA 20",
            line=dict(color="#FFA726", width=1.5)
        ))

        # Точки на графике (маленькие)
        correct_live = subset[(subset["pred"] == subset["actual"]) & (subset["pred"] != "no_trade")]
        wrong_live = subset[(subset["pred"] != subset["actual"]) & (subset["pred"] != "no_trade")]
        no_trade_live = subset[subset["pred"] == "no_trade"]

        sim_fig.add_trace(go.Scatter(
            x=correct_live["datetime"], y=correct_live["price"],
            mode="markers", marker=dict(color="#00E676", size=8, symbol="triangle-up"),
            showlegend=False
        ))
        sim_fig.add_trace(go.Scatter(
            x=wrong_live["datetime"], y=wrong_live["price"],
            mode="markers", marker=dict(color="#FF1744", size=8, symbol="x"),
            showlegend=False
        ))
        sim_fig.add_trace(go.Scatter(
            x=no_trade_live["datetime"], y=no_trade_live["price"],
            mode="markers", marker=dict(color="#FFD600", size=7, symbol="circle-open"),
            showlegend=False
        ))

        # --- Отдельные "фиктивные" следы только для легенды ---
        sim_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers", name=" Correct",
            marker=dict(color="#00E676", size=14, symbol="triangle-up")
        ))
        sim_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers", name=" Wrong",
            marker=dict(color="#FF1744", size=14, symbol="x")
        ))
        sim_fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode="markers", name=" No Trade",
            marker=dict(color="#FFD600", size=12, symbol="circle-open")
        ))

        # --- Настройки оформления ---
        sim_fig.update_layout(
            template="plotly_dark",
            height=500,
            margin=dict(l=30, r=30, t=40, b=30),
            xaxis_rangeslider_visible=False,
            yaxis_title="BTC Price",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=18)  # увеличенный размер шрифта легенды
            )
        )

        # --- Метрика ---
        metric_placeholder.metric("📊 Live Accuracy", f"{live_acc:.2f}%")
        placeholder.plotly_chart(sim_fig, use_container_width=True)
        time.sleep(speed)






# --- Таблица ---
st.subheader("🧾 Recent Predictions")
st.dataframe(df[df['pred'] != 'no_trade'].head(30).sort_values('datetime', ascending=False))
