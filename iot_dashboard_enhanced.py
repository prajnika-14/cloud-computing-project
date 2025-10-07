# iot_dashboard_enhanced.py
# Real-Time IoT + Big-Data Dashboard with Predictive Insights
# Author: Prajnika 🌸

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import time

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="IoT Dashboard", layout="wide")
st.title("🌡️ Real-Time IoT + Big-Data Dashboard with Predictive Insights")
st.markdown(
    """
This interactive dashboard simulates IoT devices sending **temperature** and **humidity** data in real time  
and provides **predictive insights** using a simple machine learning model (Linear Regression).  
It visually highlights **predicted trends** to help anticipate environmental changes.
"""
)

# ----------------------------
# INITIALIZATION
# ----------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["timestamp", "temperature", "humidity"])

# ----------------------------
# SIDEBAR CONTROLS
# ----------------------------
st.sidebar.header("⚙️ Simulation Controls")
run = st.sidebar.checkbox("Run Simulation", value=False)
reset = st.sidebar.button("Reset Data")

# Allow user to control speed and number of steps
speed = st.sidebar.slider("Simulation Speed (seconds per update)", 0.1, 2.0, 0.8)
steps = st.sidebar.number_input("Number of Simulation Steps", 10, 500, 100)

if reset:
    st.session_state.data = pd.DataFrame(columns=["timestamp", "temperature", "humidity"])
    st.toast("Simulation data has been reset.", icon="♻️")

placeholder = st.empty()

# ----------------------------
# FUNCTIONS
# ----------------------------
def simulate_iot_data():
    """Simulate one step of IoT sensor data."""
    new_data = {
        "timestamp": pd.Timestamp.now(),
        "temperature": round(np.random.uniform(20, 35), 2),
        "humidity": round(np.random.uniform(40, 70), 2)
    }
    st.session_state.data = pd.concat(
        [st.session_state.data, pd.DataFrame([new_data])],
        ignore_index=True
    )

def predict_temperature():
    """Predict next 5 temperature readings using linear regression."""
    if len(st.session_state.data) > 10:
        df_temp = st.session_state.data[["temperature"]].tail(20)
        X = np.arange(len(df_temp)).reshape(-1, 1)
        y = df_temp["temperature"].values
        model = LinearRegression().fit(X, y)

        X_future = np.arange(len(df_temp), len(df_temp) + 5).reshape(-1, 1)
        y_pred = model.predict(X_future)
        return [round(val, 2) for val in y_pred]
    return []

# ----------------------------
# SIMULATION LOOP
# ----------------------------
if run:
    for _ in range(int(steps)):
        simulate_iot_data()

        with placeholder.container():
            st.markdown("### 📈 Live IoT Data Feed")
            latest = st.session_state.data.tail(5)

            # -------------------
            # METRICS
            # -------------------
            col1, col2, col3 = st.columns(3)
            col1.metric("Average Temperature (°C)", f"{st.session_state.data['temperature'].mean():.2f}")
            col2.metric("Average Humidity (%)", f"{st.session_state.data['humidity'].mean():.2f}")
            col3.metric("Latest Temperature (°C)", f"{st.session_state.data['temperature'].iloc[-1]:.2f}")

            st.markdown("### 🔹 Recent Sensor Readings")
            st.dataframe(latest, use_container_width=True)

            # -------------------
            # PREDICTIONS
            # -------------------
            predictions = predict_temperature()

            # Create temperature plot with predictions
            fig_temp = go.Figure()

            # Actual data (blue line)
            fig_temp.add_trace(go.Scatter(
                x=st.session_state.data["timestamp"],
                y=st.session_state.data["temperature"],
                mode="lines+markers",
                name="Actual Temperature",
                line=dict(color="blue", width=2),
                marker=dict(symbol="circle", size=6)
            ))

            # Predicted data (red dashed)
            if predictions:
                last_time = st.session_state.data["timestamp"].iloc[-1]
                future_times = pd.date_range(last_time, periods=6, freq="s")[1:]
                fig_temp.add_trace(go.Scatter(
                    x=future_times,
                    y=predictions,
                    mode="lines+markers",
                    name="Predicted Temperature",
                    line=dict(color="red", dash="dash", width=2),
                    marker=dict(symbol="diamond", size=8)
                ))

            fig_temp.update_layout(
                title="Temperature (Actual vs Predicted)",
                xaxis_title="Timestamp",
                yaxis_title="Temperature (°C)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                template="plotly_white"
            )
            st.plotly_chart(fig_temp, use_container_width=True)

            # -------------------
            # HUMIDITY CHART
            # -------------------
            fig_hum = go.Figure()
            fig_hum.add_trace(go.Scatter(
                x=st.session_state.data["timestamp"],
                y=st.session_state.data["humidity"],
                mode="lines+markers",
                name="Humidity",
                line=dict(color="green", width=2),
                marker=dict(symbol="square", size=6)
            ))
            fig_hum.update_layout(
                title="Humidity Levels Over Time",
                xaxis_title="Timestamp",
                yaxis_title="Humidity (%)",
                template="plotly_white"
            )
            st.plotly_chart(fig_hum, use_container_width=True)

            # -------------------
            # PREDICTED VALUES TABLE
            # -------------------
            if predictions:
                st.markdown("### 🔮 Predicted Temperatures (Next 5 Seconds)")
                pred_df = pd.DataFrame({
                    "Future Timestamp": future_times,
                    "Predicted Temperature (°C)": predictions
                })
                st.dataframe(pred_df.style.highlight_max(axis=0, color="#f7cac9"), use_container_width=True)

            st.caption("💡 Powered by Linear Regression | Real-time simulated IoT environment")
        time.sleep(speed)

else:
    st.info("👈 Turn ON 'Run Simulation' in the sidebar to start streaming IoT data.")
