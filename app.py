import streamlit as st
import joblib
import numpy as np

# Load the trained LightGBM model
model = joblib.load("credit_risk_model.pkl")

# 🎨 Streamlit UI Styling
st.set_page_config(page_title="Credit Risk Prediction", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4CAF50;'>💳 Credit Risk Prediction 🚀</h1>", unsafe_allow_html=True)
st.write("Fill in the details below to check the credit risk status.")

# 📌 User Input Fields
LIMIT_BAL = st.number_input("💰 Credit Limit ($)", min_value=10000, max_value=1000000, step=5000)
AGE = st.number_input("🎂 Age", min_value=18, max_value=80, step=1)
EDUCATION = st.selectbox("🎓 Education Level", ["Graduate School", "University", "High School", "Others"])
MARRIAGE = st.selectbox("💍 Marital Status", ["Married", "Single", "Others"])
PAY_1 = st.slider("📆 Last Month's Repayment Status", -2, 8, 0)
BILL_AMT1 = st.number_input("🧾 Latest Bill Amount ($)", min_value=0, max_value=100000, step=500)
PAY_AMT1 = st.number_input("💳 Last Payment Amount ($)", min_value=0, max_value=100000, step=500)

# 🔄 Convert User Input to Model Format
education_map = {"Graduate School": 1, "University": 2, "High School": 3, "Others": 4}
marriage_map = {"Married": 1, "Single": 2, "Others": 3}

input_data = np.array([
    LIMIT_BAL, AGE, education_map[EDUCATION], marriage_map[MARRIAGE], PAY_1, BILL_AMT1, PAY_AMT1
]).reshape(1, -1)

# 🔮 Prediction Button
if st.button("🔍 Predict Credit Risk"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: This applicant is likely to default.")
    else:
        st.success("✅ Low Risk: This applicant is unlikely to default.")

st.write("🚀 Built with LightGBM & Streamlit")
