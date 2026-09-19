import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

hide_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_style, unsafe_allow_html=True)


st.title("🔮 Customer Churn Predictor")
st.write("Predict which telecom customers are at risk of churning — and understand **why**.")
st.markdown("---")

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model.named_steps["clf"])

try:
    model = load_model()
    explainer = load_explainer(model)
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()
num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod"
]


st.subheader("📝 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)

with col2:
    st.markdown("**Services**")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

with col3:
    st.markdown("**Billing & Contract**")
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)


if st.button("🎯 Predict Churn", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
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

    colA, colB = st.columns([1, 2])
    with colA:
        if prob > 0.5:
            st.error(f"⚠️ **High Risk**\n\n### {pct:.1f}%")
        else:
            st.success(f"✅ **Low Risk**\n\n### {pct:.1f}%")
    with colB:
        st.write("**Churn Probability**")
        st.progress(float(prob))
        if prob > 0.5:
            st.write("This customer is likely to leave. Consider a retention offer.")
        else:
            st.write("This customer is likely to stay.")

    
    st.markdown("---")
    st.subheader("🔍 Why this prediction?")

    with st.spinner("Calculating SHAP values..."):
        try:
            X_transformed = model.named_steps["prep"].transform(input_data)
            shap_values = explainer.shap_values(X_transformed)

            
            ohe = model.named_steps["prep"].named_transformers_["cat"]
            cat_names = ohe.get_feature_names_out(cat_cols)
            feature_names = num_cols + list(cat_names)

            
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=explainer.expected_value,
                    data=X_transformed[0],
                    feature_names=feature_names
                ),
                show=False
            )
            st.pyplot(fig, use_container_width=True)

            st.caption("🔴 Red bars push toward churn · 🔵 Blue bars push against churn")

        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")

st.markdown("---")
st.caption("Built with XGBoost + SHAP · Model ROC-AUC: 0.842 · [GitHub](https://github.com/Nir-bitcoin/Customer_Churn_Predictor)")
