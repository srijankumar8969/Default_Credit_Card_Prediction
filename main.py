import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import preprocess, save_data
from feature_engineering import create_baseline, create_featured, save_dataset
from model_training import run_pipeline

if __name__ == "__main__":
    # Step 1: Preprocess
    df = preprocess('data/raw/UCI_Credit_Card.csv')
    save_data(df, 'data/processed/data_preprocessed.csv')

    # Step 2: Feature Engineering
    df_baseline = create_baseline(df)
    save_dataset(df_baseline, 'data/processed/data_baseline.csv')
    df_featured = create_featured(df)
    save_dataset(df_featured, 'data/processed/data_featured.csv')

    # Step 3: Train Model
    run_pipeline('data/processed/data_featured.csv', 'models/lgb_model.pkl')

