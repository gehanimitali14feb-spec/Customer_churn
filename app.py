import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🔮",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")
st.write("Enter customer attributes below to evaluate their risk of churn.")

# Load trained assets
@st.cache_resource
def load_assets():
    model = joblib.load('churn_model.pkl')
    model_columns = joblib.load('model_columns.pkl')
    return model, model_columns

try:
    model, model_columns = load_assets()
    st.sidebar.success("Model & Columns Loaded")
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Layout inputs into two main columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Account & Financial Info")
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=float(tenure * monthly_charges))
    
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col2:
    st.subheader("Services & Demographics")
    internet_service = st.selectbox("Internet Service Type", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security Service", ["No", "Yes", "No internet service"])
    tech_support = st.selectbox("Tech Support Service", ["No", "Yes", "No internet service"])
    
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

st.markdown("---")

# Prediction action
if st.button("Run Risk Assessment", type="primary", use_container_width=True):
    # Initialize zeroed DataFrame matching training column structure
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)
    
    # Map numerical variables
    if 'tenure' in input_df.columns:
        input_df['tenure'] = tenure
    if 'MonthlyCharges' in input_df.columns:
        input_df['MonthlyCharges'] = monthly_charges
    if 'TotalCharges' in input_df.columns:
        input_df['TotalCharges'] = total_charges
    if 'SeniorCitizen' in input_df.columns:
        input_df['SeniorCitizen'] = 1 if senior_citizen == "Yes" else 0
        
    # Helper function to set encoded category values
    def set_category(col_prefix, val):
        col_name = f"{col_prefix}_{val}"
        if col_name in input_df.columns:
            input_df[col_name] = 1

    set_category('gender', gender)
    set_category('Partner', partner)
    set_category('Dependents', dependents)
    set_category('Contract', contract)
    set_category('PaperlessBilling', paperless_billing)
    set_category('PaymentMethod', payment_method)
    set_category('InternetService', internet_service)
    set_category('OnlineSecurity', online_security)
    set_category('TechSupport', tech_support)

    # Make predictions
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    churn_probability = probabilities[1] * 100

    # Display Results
    st.header("Prediction Results")
    res_col1, res_col2 = st.columns(2)

    with res_col1:
        st.metric(label="Calculated Churn Probability", value=f"{churn_probability:.1f}%")

    with res_col2:
        if prediction == 1:
            st.error("🚨 **High Risk of Churn**\n\nThis customer is likely to cancel their subscription.")
        else:
            st.success("✅ **Low Risk**\n\nThis customer is likely to stay retained.")