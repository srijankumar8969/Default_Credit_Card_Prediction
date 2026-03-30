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
│   ├── raw/                        # Original dataset (UCI_Credit_Card.csv)
│   └── processed/                  # Cleaned and featured datasets
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

### Running the preprocessing pipeline
```bash
cd src
python data_preprocessing.py
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- Univariate and bivariate analysis of all features
- Pearson correlation heatmap for numerical features
- Cramér's V heatmap for categorical features
- Key finding: PAY_0 (most recent payment status) and LIMIT_BAL are the strongest predictors of default

### EDA Key Findings

#### Target Variable
- Significant class imbalance: 77.9% non-default vs 22.1% default
- Strategies to address this will be explored during modelling: class weights, SMOTE, and threshold tuning

#### Most Predictive Features
- **PAY_0** (most recent payment status) — strongest predictor overall (Cramér's V = 0.42 with target). Recent payment behaviour is the single most reliable signal of default risk
- **LIMIT_BAL** — customers with lower credit limits default at significantly higher rates (Pearson correlation = -0.15). Lower limits reflect higher risk assessments by the bank at the point of credit assignment

#### Categorical Features (Cramér's V Analysis)
- PAY columns show moderate-to-strong association with the target, decreasing for older months: PAY_0 (0.42) → PAY_2 (0.34) → PAY_3 (0.30) → PAY_4 (0.28) → PAY_5 (0.27) → PAY_6 (0.25)
- Demographic features (SEX, EDUCATION, MARRIAGE) show negligible association with the target (all below 0.07) — retained for modelling but not expected to be important

#### Numerical Features (Pearson Correlation & Bivariate Analysis)
- **BILL_AMT1–6** are extremely intercorrelated (0.80–0.95), making them largely redundant as raw features. Defaulters tend to have slightly lower bill amounts — likely because they have stopped using their card
- **PAY_AMT1–6** show a consistent and clear pattern — non-defaulters make significantly higher payments across all 6 months, suggesting payment behaviour is a persistent habit
- **AGE** shows negligible correlation with both the target and other features — weakest raw feature in the dataset

### 2. Feature Engineering
*(To be updated)*

### 3. Models Trained
*(To be updated)*

### 4. Results
*(To be updated)*

---

## 📈 Key Findings
*(To be updated after modeling)*

---

## 🌐 Streamlit App
*(To be updated after app development)*

---

## 🙏 Acknowledgements
- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients) — I-Cheng Yeh
- Variable definition corrections: [Kaggle Discussion Forum](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset/discussion/34608)