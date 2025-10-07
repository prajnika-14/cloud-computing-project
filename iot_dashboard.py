# iot_dashboard.py
# Real-Time IoT + Big-Data Dashboard with Predictive Insights

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import time

# ----------------------------
# PAGE SETUP
# ----------------------------
st.set_page_config(page_title="IoT Dashboard", layout="wide")
st.title("🌡️ Real-Time IoT + Big-Data Dashboard")
st.markdown(
    """
This dashboard simulates IoT devices sending temperature and humidity data in real time.
It also predicts the next 5 temperature readings using linear regression.
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
st.sidebar.header("Simulation Controls")
run = st.sidebar.checkbox("Run Simulation", value=False)
reset = st.sidebar.button("Reset Data")

if reset:
    st.session_state.data = pd.DataFrame(columns=["timestamp", "temperature", "humidity"])
    st.toast("Simulation reset!")

# ----------------------------
# SIMULATION & DASHBOARD
# ----------------------------
placeholder = st.empty()

def simulate_iot_data():
    """Simulate one step of IoT sensor data."""
    new_data = {
        "timestamp": pd.Timestamp.now(),
        "temperature": round(np.random.uniform(20, 35), 2),
        "humidity": round(np.random.uniform(40, 70), 2)
    }
    st.session_state.data = pd.concat([st.session_state.data, pd.DataFrame([new_data])], ignore_index=True)

def predict_temperature():
    """Predict next 5 temperature readings using linear regression."""
    if len(st.session_state.data) > 10:
        df_temp = st.session_state.data[["temperature"]].tail(10)
        X = np.array(range(len(df_temp))).reshape(-1,1)
        y = df_temp["temperature"].values
        model = LinearRegression()
        model.fit(X, y)
        X_future = np.array(range(len(df_temp), len(df_temp)+5)).reshape(-1,1)
        y_pred = model.predict(X_future)
        return [round(val,2) for val in y_pred]
    return []

if run:
    for _ in range(100):  # simulate 100 time steps
        simulate_iot_data()

        with placeholder.container():
            st.subheader("Latest IoT Readings")
            st.dataframe(st.session_state.data.tail(5))

            # Temperature chart
            fig_temp = px.line(st.session_state.data, x="timestamp", y="temperature",
                               title="Temperature Over Time")
            st.plotly_chart(fig_temp, use_container_width=True)

            # Humidity chart
            fig_hum = px.line(st.session_state.data, x="timestamp", y="humidity",
                              title="Humidity Over Time")
            st.plotly_chart(fig_hum, use_container_width=True)

            # Predictive insights
            predictions = predict_temperature()
            if predictions:
                st.subheader("Predicted Temperature for Next 5 Steps")
                st.write(predictions)

        time.sleep(1)

else:
    st.info("👈 Turn ON 'Run Simulation' in the sidebar to start streaming data")
