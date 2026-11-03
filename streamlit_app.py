import streamlit as st
import numpy as np
import joblib
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Disease Prediction System", layout="wide")

st.title("🩺 Disease Prediction System")
st.write("Predict Diabetes and Heart Disease using Machine Learning")

# ---------------- LOAD MODELS ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DIABETES_MODEL_PATH = os.path.join(BASE_DIR, "models", "diabetes_model.joblib")
HEART_MODEL_PATH = os.path.join(BASE_DIR, "models", "heart_model.joblib")

diabetes_model = joblib.load(DIABETES_MODEL_PATH)
heart_model = joblib.load(HEART_MODEL_PATH)

# ---------------- DISEASE SELECTION ----------------
disease = st.selectbox("Select Disease", ["Diabetes", "Heart Disease"])

# ===================== DIABETES =====================
if disease == "Diabetes":
    st.subheader("🧪 Diabetes Prediction")

    glucose = st.number_input("Glucose Level", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    skin = st.number_input("Skin Thickness", min_value=0)
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=0)

    if st.button("Predict Diabetes"):
        input_data = np.array([[glucose, bp, skin, insulin, bmi, dpf, age]])
        prediction = diabetes_model.predict(input_data)

        if prediction[0] == 1:
            st.error("⚠️ Person is likely Diabetic")
        else:
            st.success("✅ Person is Not Diabetic")

# ===================== HEART =====================
elif disease == "Heart Disease":
    st.subheader("❤️ Heart Disease Prediction")

    age = st.number_input("Age", min_value=0)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
    cp = st.number_input("Chest Pain Type", min_value=0)
    trestbps = st.number_input("Resting Blood Pressure", min_value=0)
    chol = st.number_input("Cholesterol", min_value=0)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.number_input("Resting ECG", min_value=0)
    thalach = st.number_input("Maximum Heart Rate", min_value=0)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", min_value=0.0)
    slope = st.number_input("Slope", min_value=0)
    ca = st.number_input("CA", min_value=0)
    thal = st.number_input("Thal", min_value=0)

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs,
                                 restecg, thalach, exang, oldpeak,
                                 slope, ca, thal]])
        prob = heart_model.predict_proba(input_data)[0][1]  # probability of disease

        st.write(f"🧪 Risk Probability: **{prob*100:.2f}%**")

        if prob > 0.6:
          st.error("⚠️ High Risk of Heart Disease")
        elif prob > 0.4:
         st.warning("⚠️ Moderate Risk of Heart Disease")
        else:
         st.success("✅ Low Risk / Healthy Heart")

