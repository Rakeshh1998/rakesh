"""
AI-Powered Financial Fraud Detection System
Summer Training Project 2026
Author: Rakesh Kumar 
"""

import pandas as pd
import numpy as np
import warnings
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

warnings.filterwarnings('ignore')


def generate_synthetic_fraud_data():
    """
    Generates synthetic fraud dataset.
    """
    print("[*] Generating synthetic transaction data...")

    X, y = make_classification(
        n_samples=10000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        weights=[0.99],  # 1% fraud
        flip_y=0.0,
        random_state=42
    )

    feature_names = [f'Feature_{i}' for i in range(1, 21)]
    df = pd.DataFrame(X, columns=feature_names)
    df['Is_Fraud'] = y

    print(f"[*] Dataset generated. Shape: {df.shape}")

    # Fixed: Proper formatting for class distribution
    class_dist = df['Is_Fraud'].value_counts(normalize=True) * 100
    print("[*] Class Distribution:")
    print(class_dist.round(2).to_string())

    return df


def preprocess_data(df):
    print("\n[*] Preprocessing data...")

    X = df.drop('Is_Fraud', axis=1)
    y = df['Is_Fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train_evaluate_model(X_train, X_test, y_train, y_test):
    print("\n[*] Training Random Forest Classifier...")

    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    rf_model.fit(X_train, y_train)

    print("[*] Generating predictions...")
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

    print("\n" + "="*60)
    print(" MODEL EVALUATION REPORT ".center(60))
    print("="*60)

    cm = confusion_matrix(y_test, y_pred)
    print(f"True Negatives  : {cm[0][0]}")
    print(f"False Positives : {cm[0][1]}")
    print(f"False Negatives : {cm[1][0]}")
    print(f"True Positives  : {cm[1][1]}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

    print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("="*60)


if __name__ == "__main__":
    print(" Starting AI-Powered Fraud Detection Training Pipeline...\n")

    data = generate_synthetic_fraud_data()
    X_train, X_test, y_train, y_test = preprocess_data(data)
    train_evaluate_model(X_train, X_test, y_train, y_test)
    print("\n Pipeline execution completed successfully!")

    print("\n Pipeline execution completed successfully!")
