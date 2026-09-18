import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page config
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 Customer Churn Predictor")
st.write("Predict which telecom customers are at risk of churning.")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
   return joblib.load("churn_model.pkl")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

st.subheader("📝 Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col2:
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    payment = st.selectbox("Payment Method", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    senior = st.selectbox("Senior Citizen", [0, 1])

if st.button("🎯 Predict Churn", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "gender": ["Male"],
        "SeniorCitizen": [senior],
        "Partner": ["No"],
        "Dependents": ["No"],
        "tenure": [tenure],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": [internet],
        "OnlineSecurity": [online_security],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["No"],
        "TechSupport": [tech_support],
        "StreamingTV": ["No"],
        "StreamingMovies": ["No"],
        "Contract": [contract],
        "PaperlessBilling": [paperless],
        "PaymentMethod": [payment],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
    })

    prob = model.predict_proba(input_data)[0][1]
    pct = prob * 100

    st.markdown("---")
    st.subheader("📊 Prediction Result")

    if prob > 0.5:
        st.error(f"⚠️ **High Churn Risk: {pct:.1f}%**")
        st.write("This customer is likely to leave. Consider a retention offer.")
    else:
        st.success(f"✅ **Low Churn Risk: {pct:.1f}%**")
        st.write("This customer is likely to stay.")

    st.progress(float(prob))

    with st.expander("ℹ️ What drives churn?"):
        st.write("""
        Based on SHAP analysis, the top churn drivers are:
        1. **Month-to-month contract** — flexible contracts are easy to leave
        2. **Low tenure** — newer customers churn more
        3. **High monthly charges** — higher bills = higher churn
        4. **No online security / tech support** — add-ons increase stickiness
        """)

st.markdown("---")
st.caption("Built with XGBoost + SHAP · Model ROC-AUC: 0.842")
