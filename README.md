# 🏦 Credit Card Default Prediction

A machine learning project to predict whether a credit card client will default on their next payment, based on demographic information, credit history, and payment behaviour.

---

## 📋 Project Overview
This project builds and compares multiple machine learning models to predict credit card default risk using the [UCI Default of Credit Card Clients dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset). The final model is deployed as an interactive web application built with Streamlit, allowing users to input customer details and receive a real-time default risk prediction.

---

## 📂 Project Structure
```
Credit_Risk_Prediction/
│
├── data/
│   ├── raw/                        # Original dataset (UCI_Credit_Card.csv) — not committed
│   └── processed/                  # Cleaned and featured datasets — not committed
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
├── models/                        # Saved trained models (to be added)
│
├── app/                           # Streamlit application (to be added)
│
├── main.py                        # End-to-end pipeline
├── requirements.txt               # Dependencies (to be added)
└── README.md
```

---

## 📥 Data Setup
The dataset is not included in this repository. Follow these steps to set it up:

1. Download `UCI_Credit_Card.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)
2. Create the following folder structure in the root of the project:
```
mkdir -p data/raw
```
3. Place `UCI_Credit_Card.csv` inside `data/raw/`
4. Run the preprocessing script to generate the cleaned dataset:
```
cd src
python data_preprocessing.py
```
This will automatically create `data/processed/` and save `data_preprocessed.csv` there.

---

## 📊 Dataset
**Source:** [UCI Machine Learning Repository — Default of Credit Card Clients](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset)

- **Rows:** 30,000 (29,965 after removing duplicates)
- **Features:** 24 (after dropping ID column)
- **Target variable:** `default.payment.next.month` (1 = default, 0 = no default)
- **Class imbalance:** ~78% non-default, ~22% default

> **Note:** The original Kaggle data dictionary contains several inaccuracies. Corrections were sourced from a [Kaggle discussion](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608) where the dataset creator confirmed the true variable definitions. These corrections informed several preprocessing decisions.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- Virtual environment (recommended)

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd Credit_Risk_Prediction

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

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
- Strategies to address this will be explored during modelling: class weights, SMOTE, and threshold tuning

##### Most Predictive Features
- **PAY_0** (most recent payment status) — strongest predictor overall (Cramér's V = 0.42 with target). Recent payment behaviour is the single most reliable signal of default risk
- **LIMIT_BAL** — customers with lower credit limits default at significantly higher rates (Pearson correlation = -0.15). Lower limits reflect higher risk assessments by the bank at the point of credit assignment

##### Categorical Features (Cramér's V Analysis)
- PAY columns show moderate-to-strong association with the target, decreasing for older months: PAY_0 (0.42) → PAY_2 (0.34) → PAY_3 (0.30) → PAY_4 (0.28) → PAY_5 (0.27) → PAY_6 (0.25)
- Demographic features (SEX, EDUCATION, MARRIAGE) show negligible association with the target (all below 0.07) — retained for modelling but not expected to be important

##### Numerical Features (Pearson Correlation & Bivariate Analysis)
- **BILL_AMT1–6** are extremely intercorrelated (0.80–0.95), making them largely redundant as raw features. Defaulters tend to have slightly lower bill amounts — likely because they have stopped using their card
- **PAY_AMT1–6** show a consistent and clear pattern — non-defaulters make significantly higher payments across all 6 months, suggesting payment behaviour is a persistent habit
- **AGE** shows negligible correlation with both the target and other features — weakest raw feature in the dataset

##### Feature Engineering Implications
- `bill_to_limit_ratio` — BILL_AMT1 / LIMIT_BAL: captures credit utilisation
- `payment_ratio` — PAY_AMT1 / BILL_AMT1: captures how much of the bill was actually paid
- `total_paid_6months` — sum of all PAY_AMT columns: aggregates consistent payment behaviour signal
- `log_limit_bal` — log transform of LIMIT_BAL to handle right skew

##### Modelling Expectations
- The weak linear correlations between raw numerical features and the target suggest tree-based models (XGBoost, Random Forest) will outperform Logistic Regression
- Logistic Regression will be retained as an interpretable baseline

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
- Inside pipeline (modeling): Yeo-Johnson transformation for all other numerical features
- Capping: applied to PAYMENT_RATIO only where division by near-zero values produced mathematical artifacts
- 

### 3. Models Trained
*(To be updated)*

### 4. Results
*(To be updated)*

---

## 📈 Key Findings
*(To be updated after modelling)*

---

## 🌐 Streamlit App
*(To be updated after app development)*

---

## 🙏 Acknowledgements
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) — I-Cheng Yeh
- Variable definition corrections: [Kaggle Discussion Forum](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608)