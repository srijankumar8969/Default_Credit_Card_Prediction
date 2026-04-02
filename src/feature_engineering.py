import pandas as pd
import numpy as np
import os

# Loading Dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def log_transform_limit_bal(df):
    """Creates LOG_LIMIT_BAL and drops LIMIT_BAL"""
    df['LOG_LIMIT_BAL'] = np.log(df['LIMIT_BAL'])
    df = df.drop(columns=['LIMIT_BAL'])
    return df

def drop_BILL_AMT2_6(df):
    """Drops BILL_AMT2–6"""
    df_baseline = df.drop(columns=['BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'])
    return df_baseline

def create_engineered_features(df):
    """Creates BILL_LIMIT_RATIO, PAYMENT_RATIO and TOTAL_PAID_6MONTHS"""
    PAY_AMT_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['BILL_LIMIT_RATIO'] = df['BILL_AMT1'] / df['LIMIT_BAL']
    df['PAYMENT_RATIO'] = df['PAY_AMT1'] / df['BILL_AMT1']
    df['TOTAL_PAID_6MONTHS'] = df[PAY_AMT_cols].sum(axis=1)
    return df

def fix_payment_ratio(df):
    """Handles NaN, inf and extreme values in PAYMENT_RATIO"""
    valid_ratio = df['PAYMENT_RATIO'][
        ~np.isinf(df['PAYMENT_RATIO']) &
        ~df['PAYMENT_RATIO'].isna()
    ]
    cap_value = valid_ratio.quantile(0.99)
    df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].fillna(0)
    df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].replace([np.inf, -np.inf], cap_value)
    df['PAYMENT_RATIO'] = df['PAYMENT_RATIO'].clip(0, cap_value)
    return df

def drop_raw_columns(df):
    """Drops raw columns replaced by engineered features"""
    BILL_AMT_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    PAY_AMT_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df = df.drop(columns=BILL_AMT_cols + PAY_AMT_cols)
    return df

def create_baseline(df):
    """Creates baseline dataset"""
    df_baseline = df.copy()
    print(f'Shape of dataset: {df_baseline.shape}')

    print('Creating LOG_LIMIT_BAL and dropping LIMIT_BAL...')
    df_baseline = log_transform_limit_bal(df_baseline)

    print('Dropping BILL_AMT2–6...')
    df_baseline = drop_BILL_AMT2_6(df_baseline)

    print('Baseline dataset created!')
    return df_baseline

def create_featured(df):
    """Creates featured dataset"""
    df_featured = df.copy()
    print(f'Shape of dataset: {df_featured.shape}')

    print('Creating engineered features...')
    df_featured = create_engineered_features(df_featured)

    print('Fixing PAYMENT_RATIO edge cases...')
    df_featured = fix_payment_ratio(df_featured)

    print('Applying log transform to LIMIT_BAL...')
    df_featured = log_transform_limit_bal(df_featured)

    print('Dropping raw columns...')
    df_featured = drop_raw_columns(df_featured)

    print('Featured dataset created!')
    return df_featured


def save_dataset(df, output_filepath):
    """Saves dataset to output filepath"""
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Saved cleaned data to {output_filepath}")

if __name__ == "__main__":
    input_path = "../data/processed/data_preprocessed.csv"
    baseline_output_path = "../data/processed/data_baseline.csv"
    featured_output_path = "../data/processed/data_featured.csv"

    df = load_data(input_path)

    df_baseline = create_baseline(df)
    save_dataset(df_baseline, baseline_output_path)

    df_featured = create_featured(df)
    save_dataset(df_featured, featured_output_path)
