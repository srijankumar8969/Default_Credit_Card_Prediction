import numpy as np
import pandas as pd
import joblib

def load_model(filepath):
    model = joblib.load(filepath)
    return model

CAP_VALUE = 16.88  # 99th percentile of PAYMENT_RATIO calculated during feature engineering

def engineer_features(df):
    df = df.copy()

    PAY_AMT_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    BILL_AMT_cols = [f'BILL_AMT{i}' for i in range(1, 7)]

    df['BILL_LIMIT_RATIO'] = df['BILL_AMT1'] / df['LIMIT_BAL']
    df['PAYMENT_RATIO'] = df['PAY_AMT1'] / df['BILL_AMT1']
    df['TOTAL_PAID_6MONTHS'] = df[PAY_AMT_cols].sum(axis=1)

    df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].fillna(0)
    df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].replace([np.inf, -np.inf], CAP_VALUE)
    df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].clip(0, CAP_VALUE)

    df['LOG_LIMIT_BAL'] = np.log(df['LIMIT_BAL'])
    df = df.drop(columns=['LIMIT_BAL'] + BILL_AMT_cols + PAY_AMT_cols)

    return df


def predict(model, df, threshold=0.5):
    probability = model.predict_proba(df)[:, 1]

    prediction = (probability >= threshold).astype(int)
    return probability, prediction

if __name__ == "__main__":
    import os

    model_path = "../models/lgb_model.pkl"

    # Load model
    model = load_model(model_path)
    print("Model loaded successfully!")

    # Create a dummy customer for testing
    test_customer = pd.DataFrame({
        'SEX': [2], 'EDUCATION': [2], 'MARRIAGE': [2], 'AGE': [30],
        'PAY_0': [0], 'PAY_2': [0], 'PAY_3': [0], 'PAY_4': [0], 'PAY_5': [0], 'PAY_6': [0],
        'LIMIT_BAL': [100000],
        'BILL_AMT1': [50000], 'BILL_AMT2': [45000], 'BILL_AMT3': [40000],
        'BILL_AMT4': [35000], 'BILL_AMT5': [30000], 'BILL_AMT6': [25000],
        'PAY_AMT1': [5000], 'PAY_AMT2': [4000], 'PAY_AMT3': [3000],
        'PAY_AMT4': [2000], 'PAY_AMT5': [1000], 'PAY_AMT6': [500]
    })

    # Engineer features
    test_customer = engineer_features(test_customer)
    print(f"Features engineered! Shape: {test_customer.shape}")

    # Predict
    probability, prediction = predict(model, test_customer)
    print(f"Default probability: {probability[0]:.4f}")
    print(f"Prediction: {'Default' if prediction[0] == 1 else 'No Default'}")