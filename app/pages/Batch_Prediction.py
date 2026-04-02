import streamlit as st
import pandas as pd
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from inference import load_model, engineer_features, predict

st.set_page_config(page_title="Batch Prediction", page_icon="📂", layout="wide")

model = load_model(os.path.join(ROOT_DIR, 'models/lgb_model.pkl'))

st.title("📂 Batch Prediction")
st.markdown("Upload a CSV file containing multiple customers to get default risk predictions for all of them at once.")
st.divider()

# Template download
st.subheader("📥 Download Template")
st.markdown("Not sure about the format? Download the sample CSV template below:")

template_path = os.path.join(ROOT_DIR, 'app/assets/sample_batch.csv')
with open(template_path, 'rb') as f:
    st.download_button(
        label="Download Sample CSV Template",
        data=f,
        file_name="sample_batch.csv",
        mime="text/csv"
    )

st.divider()

# Threshold
st.subheader("⚙️ Risk Threshold")
threshold = st.slider("Decision Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)

if threshold < 0.4:
    st.warning("⚠️ Low threshold — catches more defaulters but generates more false alarms.")
elif threshold > 0.6:
    st.info("ℹ️ High threshold — fewer false alarms but may miss some defaulters.")
else:
    st.success("✅ Balanced threshold.")

st.divider()

# File upload
st.subheader("📤 Upload Customer Data")
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown(f"**{len(df)} customers loaded successfully!**")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("🔍 Run Batch Prediction", type="primary", use_container_width=True):
        with st.spinner("Running predictions..."):
            df_engineered = engineer_features(df.copy())
            probabilities, predictions = predict(model, df_engineered, threshold)

        # Add results to dataframe
        results = df.copy()
        results['Default Probability'] = [f"{p:.1%}" for p in probabilities]
        results['Risk Verdict'] = ['HIGH RISK' if p == 1 else 'LOW RISK' for p in predictions]

        st.divider()
        st.subheader("📊 Prediction Results")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Customers", len(results))
        with col2:
            st.metric("High Risk", sum(predictions))
        with col3:
            st.metric("Low Risk", len(predictions) - sum(predictions))

        st.dataframe(results, use_container_width=True)

        # Download results
        csv = results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name="batch_predictions.csv",
            mime="text/csv"
        )