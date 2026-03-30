import pandas as pd
import os

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def drop_id(df):
    return df.drop('ID', axis=1)

def remove_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)

def education_update(df):
    df['EDUCATION'] = df['EDUCATION'].replace({0: 0, 4: 0, 5: 0, 6: 0})
    return df

def bin_pay_columns(df):
    pay_cols = ['PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    bin_mapping = {-2: -2, -1: -1, 0: 0, 1: 2, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3, 8: 3}

    for col in pay_cols:
        df[col] = df[col].map(bin_mapping)
    return df

def preprocess(filepath):
    print("Loading data...")
    df = load_data(filepath)
    print(f"  Raw shape: {df.shape}")

    print("Dropping ID column...")
    df = drop_id(df)

    print("Removing duplicates...")
    df = remove_duplicates(df)
    print(f"  Shape after deduplication: {df.shape}")

    print("Fixing EDUCATION column...")
    df = education_update(df)

    print("Binning PAY columns...")
    df = bin_pay_columns(df)

    print("Preprocessing complete!")
    return df

def save_data(df, output_filepath):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    df.to_csv(output_filepath, index=False)
    print(f"Saved cleaned data to {output_filepath}")

if __name__ == "__main__":
    input_path = "../data/raw/UCI_Credit_Card.csv"
    output_path = "../data/processed/data_preprocessed.csv"

    df = preprocess(input_path)
    save_data(df, output_path)