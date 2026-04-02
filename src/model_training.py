import pandas as pd
import numpy as np
import os
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef
)

import lightgbm as lgb

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    return df

def split_data(df):
    X = df.drop('default.payment.next.month', axis=1)
    y = df['default.payment.next.month']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def build_pipeline():
    lgb_pipeline = Pipeline([
        ('model', lgb.LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1, n_estimators=300, max_depth=7, learning_rate=0.01, num_leaves=31))
    ])
    return lgb_pipeline

def train_model(pipeline, X_train, y_train):
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("Final Model Evaluation — LightGBM (Tuned)")
    print("=" * 50)
    print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")

def save_model(pipeline, output_filepath):
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    joblib.dump(pipeline, output_filepath)
    print(f"Model saved to {output_filepath}")


def run_pipeline(input_path, output_path):
    print("Loading data...")
    df = load_dataset(input_path)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Building pipeline...")
    pipeline = build_pipeline()

    print("Training model...")
    lgb_pipeline = train_model(pipeline, X_train, y_train)

    print("Evaluating model...")
    evaluate_model(lgb_pipeline, X_test, y_test)

    print("Saving model...")
    save_model(lgb_pipeline, output_path)

    print("Process complete, model saved!")


if __name__ == "__main__":
    input_path = "../data/processed/data_featured.csv"
    output_path = "../models/lgb_model.pkl"

    run_pipeline(input_path, output_path)