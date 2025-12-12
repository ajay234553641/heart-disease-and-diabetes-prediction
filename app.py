import streamlit as st
import numpy as np
import pickle

# Load models and scalers
heart_model = pickle.load(open('heart_model.pkl', 'rb'))
diabetes_model = pickle.load(open('diabetes_model.pkl', 'rb'))
heart_scaler = pickle.load(open('heart_scaler.pkl', 'rb'))
diabetes_scaler = pickle.load(open('diabetes_scaler.pkl', 'rb'))

st.title("🩺 Smart Health Predictor")
st.write("Predict your risk for **Heart Disease** or **Diabetes** using Machine Learning.")

disease = st.radio("Select a Disease to Predict:", ["Heart Disease", "Diabetes"])

# ---------------- Heart Disease Section ----------------
if disease == "Heart Disease":
    st.header("💓 Heart Disease Prediction")
    age = st.number_input("Age", 1, 120, 45)
    sex = st.selectbox("Sex (1 = Male, 0 = Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)", [1, 0])
    restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [1, 0])
    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST", [0, 1, 2])
    ca = st.selectbox("Major Vessels Colored (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thal (0 = Normal, 1 = Fixed Defect, 2 = Reversible)", [0, 1, 2])

    if st.button("Predict Heart Disease"):
        input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                                
                                thalach, exang, oldpeak, slope, ca, thal]])
        scaled_data = heart_scaler.transform(input_data)
        prediction = heart_model.predict(scaled_data)
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Heart Disease!")
        else:
            st.success("✅ No Risk of Heart Disease Detected.")

# ---------------- Diabetes Section ----------------
elif disease == "Diabetes":
    st.header("💉 Diabetes Prediction")

    # Ask for gender first
    gender = st.selectbox("Gender", ["Male", "Female"])

    # Pregnancies only for females
    if gender == "Female":
        preg = st.number_input("Number of Pregnancies", 0, 20, 2)
    else:
        preg = 0  # Automatically 0 for males

    glucose = st.number_input("Glucose Level", 0, 300, 120)
    bp = st.number_input("Blood Pressure", 0, 200, 70)
    skin = st.number_input("Skin Thickness", 0, 99, 20)
    insulin = st.number_input("Insulin Level", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = st.number_input("Age", 1, 120, 33)

    if st.button("Predict Diabetes"):
        input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
        scaled_data = diabetes_scaler.transform(input_data)
        prediction = diabetes_model.predict(scaled_data)
        if prediction[0] == 1:
            st.error("⚠️ High Risk of Diabetes!")
        else:
            st.success("✅ No Risk of Diabetes Detected.")
    