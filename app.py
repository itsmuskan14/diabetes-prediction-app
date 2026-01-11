import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ------------------ SETUP ------------------
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Initialize page state
if "page" not in st.session_state:
    st.session_state.page = "gender"

# ------------------ LOAD & TRAIN MODEL ------------------
data = pd.read_csv("diabetes.csv")

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ------------------ PAGE 1: GENDER SELECTION ------------------
if st.session_state.page == "gender":

    st.title("🩺 Diabetes Prediction System")
    st.subheader("Please select your gender to continue")

    gender = st.radio(
        "Gender",
        ("Female", "Male"),
        horizontal=True
    )

    if st.button("➡️ Continue"):
        st.session_state.gender = gender
        st.session_state.page = "predict"
        st.rerun()

# ------------------ PAGE 2: PREDICTION FORM ------------------
elif st.session_state.page == "predict":

    st.title("🧪 Medical Details & Prediction")
    st.write(f"**Selected Gender:** {st.session_state.gender}")

    st.markdown("---")

    # Pregnancy handling
    if st.session_state.gender == "Female":
        preg = st.number_input(
            "Pregnancies (Females only)",
            min_value=0, max_value=20, value=1
        )
    else:
        st.info("Pregnancy not applicable for males (set to 0)")
        preg = 0

    glucose = st.number_input(
        "Glucose Level (mg/dL) | Normal: 70–140",
        min_value=50, max_value=300, value=120
    )

    bp = st.number_input(
        "Blood Pressure (Diastolic mm Hg) | Normal: ~80",
        min_value=40, max_value=200, value=80
    )

    skin = st.number_input(
        "Skin Thickness (mm) | Normal: 10–40",
        min_value=0, max_value=100, value=20
    )

    insulin = st.number_input(
        "Insulin (µU/mL) | Normal: 16–166",
        min_value=0, max_value=900, value=80
    )

    bmi = st.number_input(
        "BMI | Normal: 18.5–24.9",
        min_value=10.0, max_value=70.0, value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function | Normal: 0.1–1.0",
        min_value=0.0, max_value=3.0, value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1, max_value=120, value=30
    )

    st.markdown("---")

    if st.button("🔍 Predict Diabetes"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        result = model.predict(input_data)
        prob = model.predict_proba(input_data)[0][1] * 100

        if result[0] == 1:
            st.error(f"⚠️ Person is likely Diabetic ({prob:.2f}% probability)")
        else:
            st.success(f"✅ Person is likely Not Diabetic ({100-prob:.2f}% confidence)")

        if st.session_state.gender == "Male":
            st.warning(
                "Note: Prediction for males is an approximation because "
                "the original dataset was collected from female patients only."
            )

    # Back button
    if st.button("⬅️ Change Gender"):
        st.session_state.page = "gender"
        st.rerun()
