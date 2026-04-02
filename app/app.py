import streamlit as st

st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Credit Card Default Prediction")
st.markdown("""
Welcome to the Credit Card Default Prediction app. This tool uses a trained 
LightGBM model to predict the likelihood of a credit card client defaulting 
on their next payment.

Use the navigation on the left to:
- **Single Prediction** — enter a customer's details and get an instant risk assessment
- **Batch Prediction** — upload a CSV file to get predictions for multiple customers
- **Model Dashboard** — explore model performance metrics and feature importance
""")

st.divider()

st.info("👈 Select a page from the sidebar to get started!")