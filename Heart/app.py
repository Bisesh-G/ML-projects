import streamlit as st
import pandas as pd
import joblib

# Load model and columns
model = joblib.load("E:/ML/Heart/KNN_heart.pkl")
expected_columns = joblib.load("E:/ML/Heart/columns.pkl")

# Page config
st.set_page_config(page_title="Heart Risk Predictor", page_icon="❤️", layout="centered")

# Header
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("### *AI-powered health assessment tool*")
st.markdown("---")

# Section title
st.subheader("📋 Patient Information")

# Layout
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)

with col2:
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

st.markdown("---")

# Predict button
if st.button("🔍 Analyze Risk", use_container_width=True):

    with st.spinner("Analyzing patient data..."):

        # Input dataframe
        input_df = pd.DataFrame([{
            'Age': age,
            'Sex': sex,
            'ChestPainType': chest_pain,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'RestingECG': resting_ecg,
            'MaxHR': max_hr,
            'ExerciseAngina': exercise_angina,
            'Oldpeak': oldpeak,
            'ST_Slope': st_slope
        }])

        # Encoding
        input_df = pd.get_dummies(input_df)

        # Align columns
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

    st.markdown("## 📊 Prediction Result")

    # Progress bar
    st.progress(float(probability))

    # Result message
    if prediction == 1:
        st.error(f"⚠️ High Risk Detected\n\n**Risk Probability:** {round(probability*100, 2)}%")
    else:
        st.success(f"✅ Low Risk\n\n**Risk Probability:** {round(probability*100, 2)}%")

# Footer
st.markdown("---")
st.caption("Developed with ❤️ using Machine Learning & Streamlit")