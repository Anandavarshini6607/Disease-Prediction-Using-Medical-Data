
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("heart_model.pkl")

st.title("Heart Disease Prediction System")

st.write("Enter sample dataset attributes")

age = st.number_input("Age", 1, 120, 45)
sex = st.selectbox("Sex coding", [0, 1])
cp = st.number_input("Chest pain type code", 0, 3, 0)
trestbps = st.number_input("Resting blood pressure", 50, 250, 130)
chol = st.number_input("Cholesterol", 50, 600, 220)
fbs = st.selectbox("Fasting blood sugar code", [0, 1])
restecg = st.number_input("Resting ECG code", 0, 2, 1)
thalach = st.number_input("Maximum heart rate", 50, 250, 150)
exang = st.selectbox("Exercise angina code", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.number_input("Slope code", 0, 2, 1)
ca = st.number_input("Major vessels code", 0, 4, 0)
thal = st.number_input("Thal code", 0, 3, 2)

if st.button("Predict"):

    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    result = model.predict(patient)[0]

    if result == 1:
        st.warning("Heart disease indicated by the model.")
    else:
        st.success("No heart disease indicated by the model.")

    st.caption("Educational demonstration only. Not a medical diagnosis.")
