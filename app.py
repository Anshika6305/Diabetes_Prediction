import joblib
import pandas as pd

import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load(
    "models/random_forest_diabetes_model.pkl"
)

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

st.title("Diabetes Prediction System")
st.write(
    "Educational demonstration using a Random Forest model."
)

st.subheader("Enter the input values")

pregnancies = st.number_input(
    "Pregnancies", min_value=0, max_value=20, value=2
)

glucose = st.number_input(
    "Glucose", min_value=0.0, value=120.0
)

blood_pressure = st.number_input(
    "Blood Pressure", min_value=0.0, value=70.0
)

skin_thickness = st.number_input(
    "Skin Thickness", min_value=0.0, value=25.0
)

insulin = st.number_input(
    "Insulin", min_value=0.0, value=80.0
)

bmi = st.number_input(
    "BMI", min_value=0.0, value=28.5
)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function",
    min_value=0.0,
    value=0.45
)

age = st.number_input(
    "Age", min_value=1, max_value=120, value=35
)

if st.button("Predict"):
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.warning("Model predicted class: Diabetic (1)")
    else:
        st.success("Model predicted class: Non-Diabetic (0)")

    st.write(
        "Model estimated probability for class 1:",
        f"{probability * 100:.2f}%"
    )

st.caption(
    "Educational demonstration only. "
    "This model is not a medical diagnostic tool."
)
