import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from inference import load_model, engineer_features, predict

from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             RocCurveDisplay, roc_auc_score,
                             recall_score, precision_score,
                             f1_score, matthews_corrcoef)

st.set_page_config(page_title="Model Dashboard", page_icon="📊", layout="wide")

model = load_model(os.path.join(ROOT_DIR, 'models/lgb_model.pkl'))

st.title("📊 Model Dashboard")
st.markdown("Model performance metrics, evaluation plots and feature importance for the final LightGBM model.")
st.divider()

# Load test data and generate predictions
@st.cache_data
def load_test_data():
    featured = pd.read_csv(os.path.join(ROOT_DIR, 'data/processed/data_featured.csv'))
    from sklearn.model_selection import train_test_split
    X = featured.drop('default.payment.next.month', axis=1)
    y = featured['default.payment.next.month']
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_test, y_test

X_test, y_test = load_test_data()
y_prob, y_pred = predict(model, X_test)

# Model Performance Metrics
st.subheader("🎯 Model Performance — Test Set")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("AUC-ROC", f"{roc_auc_score(y_test, y_prob):.4f}")
with col2:
    st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
with col3:
    st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
with col4:
    st.metric("F1-Score", f"{f1_score(y_test, y_pred):.4f}")
with col5:
    st.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.4f}")

st.divider()

# Confusion Matrix and ROC Curve side by side
st.subheader("📈 Evaluation Plots")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Confusion Matrix**")
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Default', 'Default'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — LightGBM (Tuned)')
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.markdown("**ROC Curve**")
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, name='LightGBM (Tuned)', ax=ax, color='teal')
    ax.plot([0, 1], [0, 1], 'k--', label='Random classifier')
    ax.set_title('ROC Curve — LightGBM (Tuned)')
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# SHAP Summary Plot
st.subheader("🔍 Feature Importance — SHAP Values")
st.markdown("The SHAP summary plot shows which features have the most impact on the model's predictions.")

try:
    import shap
    @st.cache_data
    def get_shap_values(_model, _X_test):
        lgb_model = _model.named_steps['model']
        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(_X_test)
        return shap_values

    shap_values = get_shap_values(model, X_test)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title('SHAP Summary Plot — LightGBM (Tuned)')
    plt.tight_layout()
    st.pyplot(fig)

except Exception as e:
    st.warning(f"SHAP plot could not be generated: {e}")

st.divider()

# Key Findings
st.subheader("💡 Key Findings")
st.markdown("""
- **PAY_0** (most recent payment status) is the single most important feature — confirmed by both EDA (Cramér's V = 0.42) and SHAP analysis
- **TOTAL_PAID_6MONTHS** and **BILL_LIMIT_RATIO** rank 2nd and 3rd — validating the feature engineering decisions
- **Demographic features** (SEX, AGE, EDUCATION, MARRIAGE) have negligible impact on predictions
- The model achieves **AUC-ROC of 0.7734** on the test set — consistent with published research on this dataset (typical range 0.75–0.82)
- The small gap between CV scores and test scores (~0.01) confirms the model generalises well with no overfitting
""")