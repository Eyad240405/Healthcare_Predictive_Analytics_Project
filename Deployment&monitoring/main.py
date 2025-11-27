import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
st.set_page_config(page_title="Diabetes App", page_icon="🩺")
st.title("🩺 Diabetes Prediction App + Monitoring Dashboard")

LOG_FILE = "prediction_logs.csv"
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=[
        "timestamp", "age", "bmi", "hbA1c", "glucose",
        "gender", "race", "smoking", "prediction", "probability"
    ]).to_csv(LOG_FILE, index=False)

def log_prediction(data, prediction, probability):
    df = pd.read_csv(LOG_FILE)
    df.loc[len(df)] = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        data["age"],
        data["bmi"],
        data["hbA1c"],
        data["blood_glucose"],
        data["gender"],
        data["race"],
        data["smoking"],
        prediction,
        probability
    ]
    df.to_csv(LOG_FILE, index=False)

# =========================================================
# Sidebar Navigation
# =========================================================
page = st.sidebar.selectbox(
    "Navigation",
    ["Diabetes Prediction", "Monitoring Dashboard"]
)

# =========================================================
# Page 1 — Prediction Page
# =========================================================
if page == "Diabetes Prediction":

    st.subheader("Enter Patient Information")

    try:
        with open("diabetes_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model_columns.pkl", "rb") as f:
            model_columns = pickle.load(f)
    except:
        st.error("Missing model files!")
        st.stop()

    # Inputs
    age = st.number_input("Age", 1, 120, 30)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    hbA1c = st.number_input("HbA1c Level", 3.5, 15.0, 5.5)
    glucose = st.number_input("Blood Glucose Level", 50, 300, 120)

    gender = st.selectbox("Gender", ["Male", "Female"])
    race = st.selectbox("Race", ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"])
    smoking = st.selectbox("Smoking History", ["Current", "Ever", "Former", "Never", "Not current", "No Info"])

    # Prepare one-hot encoding
    gender_F = 1 if gender == "Female" else 0
    gender_M = 1 if gender == "Male" else 0

    race_dict = {f"race:{r}": int(r == race)
                 for r in ["AfricanAmerican", "Asian", "Caucasian", "Hispanic", "Other"]}

    smoking_dict = {
        f"smoking_history_{s.lower().replace(' ','_')}": int(s == smoking)
        for s in ["Current", "Ever", "Former", "Never", "Not current", "No Info"]
    }

    input_data = {
        "year": 2020,
        "age": age,
        "gender_Female": gender_F,
        "gender_Male": gender_M,
        "hypertension": 0,
        "bmi": bmi,
        "hbA1c_level": hbA1c,
        "blood_glucose_level": glucose
    }
    input_data.update(race_dict)
    input_data.update(smoking_dict)

    df_input = pd.DataFrame([input_data])

    # Match model training columns
    for col in model_columns:
        if col not in df_input.columns:
            df_input[col] = 0

    df_input = df_input[model_columns]
    scaled_input = scaler.transform(df_input)

    if st.button("Predict"):

        prediction = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1] * 100

        st.subheader("Prediction Result")
        st.write(f"Prediction: **{prediction}**")
        st.write(f"Risk Score: **{probability:.2f}%**")

        if prediction == 1:
            st.error("High Diabetes Risk")
        else:
            st.success("Low Diabetes Risk")
        log_prediction(
            {
                "age": age,
                "bmi": bmi,
                "hbA1c": hbA1c,
                "blood_glucose": glucose,
                "gender": gender,
                "race": race,
                "smoking": smoking
            },
            prediction,
            probability
        )


# =========================================================
# Page 2 — Monitoring Dashboard
# =========================================================
if page == "Monitoring Dashboard":

    st.header("Monitoring Dashboard")

    df = pd.read_csv(LOG_FILE)

    if df.empty:
        st.warning("No predictions logged yet.")
        st.stop()
    st.subheader("Total Predictions")
    st.metric("Predictions Count", len(df))
    st.subheader("Predictions Over Time")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    st.line_chart(df.groupby(df["timestamp"].dt.date).size())
    st.subheader("Class Distribution")
    st.bar_chart(df["prediction"].value_counts())
    st.subheader("Drift Detection")
    numeric_cols = ["age", "bmi", "hbA1c", "glucose"]
    drift_data = df[numeric_cols]
    st.write("Summary Statistics:")
    st.write(drift_data.describe())
    if len(df) >= 20:
        first10 = drift_data.head(10).mean()
        last10 = drift_data.tail(10).mean()
        drift_score = abs(last10 - first10)
        st.write("Drift Magnitude:")
        st.write(drift_score)
        if drift_score.max() > drift_data.std().mean() * 0.5:
            st.error("Possible Data Drift Detected!")
        else:
            st.success("No Data Drift Detected.")
    # ---- 5. Logs Table ----
    st.subheader("Raw Logs")
    st.dataframe(df)
