import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))
from inference import load_model, engineer_features, predict

st.set_page_config(page_title="Single Prediction", page_icon="🔍", layout="wide")

# Load model
model = load_model(os.path.join(os.path.dirname(__file__), '../../models/lgb_model.pkl'))

st.title("🔍 Single Customer Prediction")
st.markdown("Enter the customer's details below to get a default risk assessment.")
st.divider()

# Demographic Information
st.subheader("👤 Demographic Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    sex = st.selectbox("Gender", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")

with col2:
    education = st.selectbox("Education", options=[1, 2, 3, 0],
                             format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 0: "Others"}[x])

with col3:
    marriage = st.selectbox("Marital Status", options=[1, 2, 3],
                            format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])

with col4:
    age = st.slider("Age", min_value=21, max_value=80, value=35)

st.divider()

# Credit Information
st.subheader("💳 Credit Information")
limit_bal = st.slider("Credit Limit (NT$)", min_value=10000, max_value=1000000, value=100000, step=10000)

st.divider()

# Payment Status
st.subheader("📅 Payment Status (last 6 months)")
st.caption("Payment status: -2 = No consumption, -1 = Paid in full, 0 = Revolving credit, 1-3 = Months delayed")
col1, col2, col3 = st.columns(3)

pay_options = [-2, -1, 0, 2, 3]
pay_labels = {-2: "No consumption", -1: "Paid in full", 0: "Revolving credit", 2: "1-2 months delay", 3: "3+ months delay"}

with col1:
    pay_0 = st.selectbox("September 2005 (PAY_0)", options=pay_options, format_func=lambda x: pay_labels[x])
    pay_3 = st.selectbox("June 2005 (PAY_3)", options=pay_options, format_func=lambda x: pay_labels[x])

with col2:
    pay_2 = st.selectbox("August 2005 (PAY_2)", options=pay_options, format_func=lambda x: pay_labels[x])
    pay_4 = st.selectbox("May 2005 (PAY_4)", options=pay_options, format_func=lambda x: pay_labels[x])

with col3:
    pay_5 = st.selectbox("April 2005 (PAY_5)", options=pay_options, format_func=lambda x: pay_labels[x])
    pay_6 = st.selectbox("March 2005 (PAY_6)", options=pay_options, format_func=lambda x: pay_labels[x])

st.divider()

# Bill Amounts
st.subheader("🧾 Bill Statement Amounts (NT$)")
col1, col2, col3 = st.columns(3)

with col1:
    bill_amt1 = st.number_input("September 2005", value=50000, step=1000)
    bill_amt4 = st.number_input("June 2005", value=35000, step=1000)

with col2:
    bill_amt2 = st.number_input("August 2005", value=45000, step=1000)
    bill_amt5 = st.number_input("May 2005", value=30000, step=1000)

with col3:
    bill_amt3 = st.number_input("July 2005", value=40000, step=1000)
    bill_amt6 = st.number_input("April 2005", value=25000, step=1000)

st.divider()

# Payment Amounts
st.subheader("💰 Previous Payment Amounts (NT$)")
col1, col2, col3 = st.columns(3)

with col1:
    pay_amt1 = st.number_input("September 2005 ", value=5000, step=500)
    pay_amt4 = st.number_input("June 2005 ", value=2000, step=500)

with col2:
    pay_amt2 = st.number_input("August 2005 ", value=4000, step=500)
    pay_amt5 = st.number_input("May 2005 ", value=1000, step=500)

with col3:
    pay_amt3 = st.number_input("July 2005 ", value=3000, step=500)
    pay_amt6 = st.number_input("April 2005 ", value=500, step=500)

st.divider()

# Threshold Slider
st.subheader("⚙️ Risk Threshold")
threshold = st.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=0.5,
    step=0.05
)

if threshold < 0.4:
    st.warning(
        "⚠️ Low threshold — catches more defaulters but generates more false alarms. Suitable for high-value loans where missing a default is very costly.")
elif threshold > 0.6:
    st.info(
        "ℹ️ High threshold — fewer false alarms but may miss some defaulters. Suitable when customer retention is also a priority.")
else:
    st.success("✅ Balanced threshold — good trade-off between catching defaulters and minimising false alarms.")

st.divider()

# Predict Button
if st.button("🔍 Predict Default Risk", type="primary", use_container_width=True):

    # Build input dataframe
    input_data = pd.DataFrame({
        'SEX': [sex], 'EDUCATION': [education], 'MARRIAGE': [marriage], 'AGE': [age],
        'PAY_0': [pay_0], 'PAY_2': [pay_2], 'PAY_3': [pay_3],
        'PAY_4': [pay_4], 'PAY_5': [pay_5], 'PAY_6': [pay_6],
        'LIMIT_BAL': [limit_bal],
        'BILL_AMT1': [bill_amt1], 'BILL_AMT2': [bill_amt2], 'BILL_AMT3': [bill_amt3],
        'BILL_AMT4': [bill_amt4], 'BILL_AMT5': [bill_amt5], 'BILL_AMT6': [bill_amt6],
        'PAY_AMT1': [pay_amt1], 'PAY_AMT2': [pay_amt2], 'PAY_AMT3': [pay_amt3],
        'PAY_AMT4': [pay_amt4], 'PAY_AMT5': [pay_amt5], 'PAY_AMT6': [pay_amt6]
    })

    # Engineer features
    input_engineered = engineer_features(input_data)

    # Predict
    probability, prediction = predict(model, input_engineered, threshold)

    st.divider()
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Default Probability", f"{probability[0]:.1%}")

    with col2:
        if prediction[0] == 1:
            st.error("⚠️ HIGH RISK — Likely to Default")
        else:
            st.success("✅ LOW RISK — Unlikely to Default")

    # Risk explanation
    st.markdown("### What does this mean?")
    if prediction[0] == 1:
        st.markdown(f"""
        The model predicts this customer has a **{probability[0]:.1%} probability of defaulting** 
        on their next payment. Based on the selected threshold of **{threshold}**, this customer 
        is classified as **high risk**. Consider reviewing their credit limit or requiring 
        additional documentation before approving new credit.
        """)
    else:
        st.markdown(f"""
        The model predicts this customer has a **{probability[0]:.1%} probability of defaulting** 
        on their next payment. Based on the selected threshold of **{threshold}**, this customer 
        is classified as **low risk**. They appear to be a reliable credit customer.
        """)