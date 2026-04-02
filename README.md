# 🏦 Credit Card Default Prediction

A machine learning project to predict whether a credit card client will default on their next payment, based on demographic information, credit history, and payment behaviour.

---

## 📋 Project Overview
This project builds and compares multiple machine learning models to predict credit card default risk using the [UCI Default of Credit Card Clients dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset). The final model is deployed as an interactive web application built with Streamlit, allowing users to input customer details and receive a real-time default risk prediction.

---

## ✨ Project Highlights

- **Rigorous evaluation** — 12 models compared across a three-stage ablation study (baseline → feature engineering → hyperparameter tuning), isolating the contribution of each decision rather than just reporting a final number
- **Justified feature engineering** — every engineered feature is motivated by EDA findings, with careful handling of edge cases (division by zero, negative bills, extreme outliers) documented throughout
- **No data leakage** — strict train/test separation with all preprocessing, scaling and class imbalance handling inside sklearn pipelines fitted only on training folds
- **Explainable predictions** — SHAP analysis validates EDA findings and provides feature-level explanations for individual predictions, exposed in the Streamlit app
- **Adjustable risk threshold** — the Streamlit app exposes the decision threshold as an interactive slider, allowing loan officers to balance recall and precision based on their business context
- **Production-ready structure** — modular `src/` pipeline scripts, end-to-end `main.py`, deployed Streamlit app with single prediction, batch prediction and model dashboard
---


## 🌐 Live Demo
**[credit-card-default-prediction-vs.streamlit.app](https://credit-card-default-prediction-vs.streamlit.app)**

The app includes three pages:
- **Single Prediction** — enter a customer's details and get an instant default risk assessment with adjustable threshold
- **Batch Prediction** — upload a CSV file to get predictions for multiple customers at once
- **Model Dashboard** — explore model performance metrics, confusion matrix, ROC curve and SHAP feature importance

---

## 📂 Project Structure
```
Credit_Risk_Prediction/
│
├── data/
│   ├── raw/                        # Original dataset (UCI_Credit_Card.csv) — not committed
│   └── processed/                  # Cleaned and featured datasets — committed
│       ├── data_preprocessed.csv
│       ├── data_baseline.csv
│       └── data_featured.csv
│
├── models/
│   └── lgb_model.pkl              # Trained LightGBM model — committed
│
├── notebooks/
│   ├── eda.ipynb                   # Exploratory Data Analysis
│   ├── feature_engineering.ipynb  # Feature Engineering
│   └── modeling.ipynb             # Model Training & Evaluation
│
├── src/
│   ├── data_preprocessing.py      # Data cleaning pipeline
│   ├── feature_engineering.py     # Feature engineering pipeline
│   ├── model_training.py          # Model training pipeline
│   └── inference.py               # Inference pipeline for Streamlit app
│
├── app/
│   ├── app.py                     # Main Streamlit entry point
│   ├── assets/
│   │   └── sample_batch.csv       # Template CSV for batch predictions
│   └── pages/
│       ├── 1_Single_Prediction.py
│       ├── 2_Batch_Prediction.py
│       └── 3_Model_Dashboard.py
│
├── main.py                        # End-to-end pipeline
├── requirements.txt               # Python dependencies
├── packages.txt                   # System dependencies for Streamlit Cloud
├── runtime.txt                    # Python version for Streamlit Cloud
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.11+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/VishalSaravanan02/Default_Credit_Card_Prediction.git
cd Default_Credit_Card_Prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the App
Since the processed datasets and trained model are included in the repository, you can run the Streamlit app directly after installation:
```bash
streamlit run app/app.py
```

### Retraining from Scratch (Optional)
If you want to retrain the model from scratch using fresh data:

1. Download `UCI_Credit_Card.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
2. Create the raw data folder and place the dataset there:
```bash
mkdir -p data/raw
# Place UCI_Credit_Card.csv inside data/raw/
```
3. Run the full pipeline with a single command:
```bash
python main.py
```

---

## 📊 Dataset
**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

- **Rows:** 30,000 (29,965 after removing duplicates)
- **Features:** 24 (after dropping ID column)
- **Target variable:** `default.payment.next.month` (1 = default, 0 = no default)
- **Class imbalance:** ~78% non-default, ~22% default

> **Note:** The original Kaggle data dictionary contains several inaccuracies. Corrections were sourced from a [Kaggle discussion](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608) where the dataset creator confirmed the true variable definitions. These corrections informed several preprocessing decisions.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Univariate and bivariate analysis of all features
- Pearson correlation heatmap for numerical features
- Cramér's V heatmap for categorical features
- Key finding: PAY_0 (most recent payment status) and LIMIT_BAL are the strongest predictors of default

#### EDA Key Findings

##### Target Variable
- Significant class imbalance: 77.9% non-default vs 22.1% default
- Addressed using class_weight='balanced' for all models and scale_pos_weight=3.52 for XGBoost

##### Most Predictive Features
- **PAY_0** (most recent payment status) — strongest predictor overall (Cramér's V = 0.42 with target). Recent payment behaviour is the single most reliable signal of default risk
- **LIMIT_BAL** — customers with lower credit limits default at significantly higher rates (Pearson correlation = -0.15). Lower limits reflect higher risk assessments by the bank at the point of credit assignment

##### Categorical Features (Cramér's V Analysis)
- PAY columns show moderate-to-strong association with the target, decreasing for older months: PAY_0 (0.42) → PAY_2 (0.34) → PAY_3 (0.30) → PAY_4 (0.28) → PAY_5 (0.27) → PAY_6 (0.25)
- Demographic features (SEX, EDUCATION, MARRIAGE) show negligible association with the target (all below 0.07) — retained for modelling but confirmed as unimportant by SHAP analysis

##### Numerical Features (Pearson Correlation & Bivariate Analysis)
- **BILL_AMT1–6** are extremely intercorrelated (0.80–0.95), making them largely redundant as raw features
- **PAY_AMT1–6** show a consistent and clear pattern — non-defaulters make significantly higher payments across all 6 months
- **AGE** shows negligible correlation with both the target and other features — weakest raw feature in the dataset

##### Feature Engineering Implications
- `BILL_LIMIT_RATIO` — BILL_AMT1 / LIMIT_BAL: captures credit utilisation
- `PAYMENT_RATIO` — PAY_AMT1 / BILL_AMT1: captures how much of the bill was actually paid
- `TOTAL_PAID_6MONTHS` — sum of all PAY_AMT columns: aggregates consistent payment behaviour signal
- `LOG_LIMIT_BAL` — log transform of LIMIT_BAL to handle right skew

### 2. Feature Engineering
Two datasets were created to measure the impact of feature engineering through an ablation study:

**Baseline dataset** (`data_baseline.csv`) — cleaned data with multicollinearity handled and log transform applied:
- Dropped BILL_AMT2–6 (intercorrelation of 0.80–0.95 with BILL_AMT1)
- Replaced LIMIT_BAL with LOG_LIMIT_BAL (log transform — right skewed, no zero/negative values)

**Featured dataset** (`data_featured.csv`) — engineered features replacing raw columns:
- `BILL_LIMIT_RATIO` = BILL_AMT1 / LIMIT_BAL — captures credit utilisation
- `PAYMENT_RATIO` = PAY_AMT1 / BILL_AMT1 — captures proportion of bill paid (with careful edge case handling for zero/negative bills)
- `TOTAL_PAID_6MONTHS` = sum(PAY_AMT1–6) — aggregates consistent payment behaviour signal across all 6 months
- `LOG_LIMIT_BAL` = log(LIMIT_BAL) — log transform applied upfront (safe — no zero/negative values)
- Dropped BILL_AMT1–6, PAY_AMT1–6 and LIMIT_BAL (replaced by engineered features)

**Skew & outlier handling strategy:**
- Upfront (feature engineering): log transform for LIMIT_BAL only
- Inside pipeline (modeling): Yeo-Johnson transformation for Logistic Regression
- Capping: applied to PAYMENT_RATIO only where division by near-zero values produced mathematical artifacts

### 3. Models Trained
Four models were trained across three stages (12 models total) using 5-Fold Stratified Cross-Validation:

- **Logistic Regression** — interpretable baseline. Pipeline: Yeo-Johnson → StandardScaler → LR. `class_weight='balanced'`
- **Random Forest** — handles non-linearity and outliers well. `class_weight='balanced'`
- **XGBoost** — gradient boosting, strong on tabular data with class imbalance. `scale_pos_weight=3.52`
- **LightGBM** — fast gradient boosting, best overall performer. `class_weight='balanced'`

**Three stage ablation study:**

| Stage | Dataset | Tuning | Purpose |
|---|---|---|---|
| Stage 1 — Baseline | data_baseline.csv | None | Performance floor with no feature engineering |
| Stage 2 — Featured | data_featured.csv | None | Measure impact of feature engineering alone |
| Stage 3 — Tuned | data_featured.csv | GridSearchCV (AUC-ROC) | Measure impact of hyperparameter tuning |

### 4. Results

#### Cross-Validation Results (All 12 Models)

| Model | Stage | AUC-ROC | Recall | Precision | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | Stage 1 - Baseline | 0.7426 | 0.6495 | 0.4041 | 0.4981 | 0.3272 |
| Random Forest | Stage 1 - Baseline | 0.7620 | 0.3380 | 0.6249 | 0.4386 | 0.3584 |
| XGBoost | Stage 1 - Baseline | 0.7542 | 0.5764 | 0.4586 | 0.5108 | 0.3548 |
| LightGBM | Stage 1 - Baseline | 0.7784 | 0.6307 | 0.4717 | 0.5397 | 0.3910 |
| Logistic Regression | Stage 2 - Featured | 0.7303 | 0.6514 | 0.3814 | 0.4811 | 0.3007 |
| Random Forest | Stage 2 - Featured | 0.7646 | 0.3401 | 0.6143 | 0.4377 | 0.3536 |
| XGBoost | Stage 2 - Featured | 0.7566 | 0.5839 | 0.4604 | 0.5148 | 0.3598 |
| LightGBM | Stage 2 - Featured | 0.7803 | 0.6225 | 0.4696 | 0.5353 | 0.3854 |
| Logistic Regression | Stage 3 - Tuned | 0.7304 | 0.6518 | 0.3807 | 0.4806 | 0.2999 |
| Random Forest | Stage 3 - Tuned | 0.7815 | 0.5733 | 0.5246 | 0.5479 | 0.4127 |
| XGBoost | Stage 3 - Tuned | 0.7824 | 0.6303 | 0.4660 | 0.5358 | 0.3853 |
| **LightGBM** | **Stage 3 - Tuned** | **0.7832** | **0.6299** | **0.4712** | **0.5391** | **0.3902** |

#### Best Model — LightGBM (Stage 3 - Tuned)
Selected based on highest AUC-ROC (0.7832) and competitive Recall (0.6299) across all 12 models.

**Best hyperparameters:** n_estimators=300, max_depth=7, learning_rate=0.01, num_leaves=31

**Final test set evaluation (held-out, evaluated once):**

| Metric | CV Score | Test Score |
|---|---|---|
| AUC-ROC | 0.7832 | 0.7734 |
| Recall | 0.6299 | 0.6229 |
| Precision | 0.4712 | 0.4620 |
| F1 | 0.5391 | 0.5305 |
| MCC | 0.3902 | 0.3781 |

The small gap between CV and test scores (~0.01) confirms the model generalises well to unseen data with no overfitting.

#### Key Findings from Modelling
- **LightGBM** was the strongest model across all three stages
- **Hyperparameter tuning** had more impact than feature engineering alone — particularly for Random Forest (Recall 0.34 → 0.57)
- **Logistic Regression** showed minimal improvement across all stages — confirms the relationships in this dataset are fundamentally non-linear
- **SHAP analysis** confirmed EDA findings — PAY_0, TOTAL_PAID_6MONTHS and BILL_LIMIT_RATIO are the three most important features. Demographic features (SEX, AGE, EDUCATION, MARRIAGE) have negligible impact

#### Threshold Tuning
The default decision threshold of 0.5 is used for the final model. The Streamlit app exposes this as an adjustable slider:
- **Lower threshold (0.3–0.4):** Higher Recall, more false alarms — suitable when minimising financial loss is the priority
- **Default threshold (0.5):** Balanced approach
- **Higher threshold (0.5–0.6):** Fewer false alarms, lower Recall — suitable when customer retention is also important

---

## 📈 Key Findings
- **PAY_0** (most recent payment status) is the single strongest predictor of default — confirmed by both Cramér's V (0.42) and SHAP analysis
- **Feature engineering improved model performance** — BILL_LIMIT_RATIO and TOTAL_PAID_6MONTHS rank 2nd and 3rd in SHAP importance
- **Tree-based models significantly outperform Logistic Regression** — confirming non-linear relationships dominate this dataset
- **Final model (LightGBM tuned) achieves AUC-ROC of 0.7734** on the held-out test set — consistent with published research on this dataset (typical range 0.75–0.82)

---

## 🙏 Acknowledgements
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) — I-Cheng Yeh
- Variable definition corrections: [Kaggle Discussion Forum](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608)