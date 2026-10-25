
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Enter the patient's details to get a model prediction.")

# -------------------------------
# Load saved files
# -------------------------------
BASE_DIR = Path(__file__).resolve().parent

try:
    model = joblib.load(BASE_DIR / "KNN_heart.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    expected_columns = joblib.load(BASE_DIR / "columns.pkl")

    if not isinstance(expected_columns, (list, tuple)):
        expected_columns = list(expected_columns)

except FileNotFoundError as e:
    st.error(f"Required file not found: {e.filename}")
    st.info(
        "Keep app.py, KNN_heart.pkl, scaler.pkl, "
        "and columns.pkl in the same folder."
    )
    st.stop()

except Exception as e:
    st.error(f"Could not load the saved model files: {e}")
    st.stop()

# -------------------------------
# User input
# -------------------------------
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=45
    )

    sex = st.selectbox(
        "Sex",
        ["Male", "Female"]
    )

    chest_pain = st.selectbox(
        "Chest Pain Type",
        ["Typical angina", "Atypical angina",
         "Non-anginal pain", "Asymptomatic"]
    )

    resting_bp = st.number_input(
        "Resting Blood Pressure",
        min_value=0,
        max_value=300,
        value=120
    )

    cholesterol = st.number_input(
        "Cholesterol",
        min_value=0,
        max_value=700,
        value=200
    )

    fasting_bs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        ["No", "Yes"]
    )

with col2:
    resting_ecg = st.selectbox(
        "Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.number_input(
        "Maximum Heart Rate",
        min_value=0,
        max_value=250,
        value=150
    )

    exercise_angina = st.selectbox(
        "Exercise-Induced Angina",
        ["No", "Yes"]
    )

    oldpeak = st.number_input(
        "Oldpeak",
        min_value=-5.0,
        max_value=10.0,
        value=0.0,
        step=0.1
    )

    st_slope = st.selectbox(
        "ST Slope",
        ["Up", "Flat", "Down"]
    )

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict", type="primary", use_container_width=True):

    # Create input data using common heart disease dataset names.
    input_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": 1 if fasting_bs == "Yes" else 0,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_M": 1 if sex == "Male" else 0,
        "ChestPainType_ATA": 1 if chest_pain == "Atypical angina" else 0,
        "ChestPainType_NAP": 1 if chest_pain == "Non-anginal pain" else 0,
        "ChestPainType_TA": 1 if chest_pain == "Typical angina" else 0,
        "RestingECG_Normal": 1 if resting_ecg == "Normal" else 0,
        "RestingECG_ST": 1 if resting_ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if exercise_angina == "Yes" else 0,
        "ST_Slope_Flat": 1 if st_slope == "Flat" else 0,
        "ST_Slope_Up": 1 if st_slope == "Up" else 0
    }

    # Arrange features in the exact order used during training.
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(
        columns=expected_columns,
        fill_value=0
    )

    try:
        # Transform the input using the saved scaler.
        scaled_input = scaler.transform(input_df)

        # Predict using the saved model.
        prediction = model.predict(scaled_input)

        st.subheader("Prediction Result")

        result = prediction[0]

        if result == 1:
            st.warning(
                "The model predicts a positive heart disease result."
            )
        else:
            st.success(
                "The model predicts a negative heart disease result."
            )

        st.caption(
            "This is a machine-learning prediction, not a medical "
            "diagnosis. Please consult a qualified healthcare professional."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the saved scaler, feature columns, and model "
            "match the same training pipeline."
        )
