#!/usr/bin/env python3
"""
Train a local XGBoost model for fraud detection.
This script can be used for local testing before deploying to SageMaker.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb

class FraudDetectionModel:
    # Decline rate feature names
    DECLINE_RATE_FEATURES = [
        'country_decline_rate_7d', 'country_decline_rate_14d',
        'bank_decline_rate_7d', 'bank_decline_rate_14d',
        'bin_decline_rate_7d', 'bin_decline_rate_14d',
        'sdk_type_decline_rate_7d', 'sdk_type_decline_rate_14d',
        'ip_decline_rate_7d', 'ip_decline_rate_14d',
        'card_network_decline_rate_7d', 'card_network_decline_rate_14d',
    ]

    def __init__(self, use_decline_rates=True):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = ['country', 'bank', 'bin', 'sdk_type', 'ip_address', 'card_network']
        self.categorical_columns = ['country', 'bank', 'sdk_type', 'card_network']
        self.use_decline_rates = use_decline_rates

    def preprocess_features(self, df, fit=False):
        """
        Preprocess features for the model.
        - Encode categorical variables
        - Extract features from IP address
        - Convert BIN to numeric
        - Include decline rate features if available
        """
        df = df.copy()

        # Encode categorical columns
        for col in self.categorical_columns:
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen labels
                le = self.label_encoders[col]
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )

        # Convert BIN to numeric
        df['bin_numeric'] = pd.to_numeric(df['bin'], errors='coerce').fillna(0)

        # Extract features from IP address
        df['ip_first_octet'] = df['ip_address'].apply(
            lambda x: int(x.split('.')[0]) if pd.notna(x) and '.' in str(x) else 0
        )
        df['ip_is_private'] = df['ip_address'].apply(
            lambda x: 1 if str(x).startswith(('10.', '192.168.', '172.')) else 0
        )

        # Base feature columns (7 features)
        feature_cols = [
            'country_encoded', 'bank_encoded', 'sdk_type_encoded',
            'card_network_encoded', 'bin_numeric', 'ip_first_octet', 'ip_is_private'
        ]

        # Add decline rate features if enabled and available (12 additional features)
        if self.use_decline_rates:
            for feature in self.DECLINE_RATE_FEATURES:
                if feature in df.columns:
                    # Ensure the feature is numeric and fill NaN with 0
                    df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)
                    feature_cols.append(feature)
                elif fit:
                    # If fitting and feature not present, don't include decline rates
                    print(f"  Warning: Decline rate feature '{feature}' not found in data")

        return df[feature_cols]

    def train(self, train_path, test_path=None):
        """Train the XGBoost model."""
        print("Loading training data...")
        train_df = pd.read_csv(train_path)

        # Split features and target
        X = self.preprocess_features(train_df, fit=True)
        y = train_df['is_fraud']

        # Split for validation if no test set provided
        if test_path:
            test_df = pd.read_csv(test_path)
            X_test = self.preprocess_features(test_df, fit=False)
            y_test = test_df['is_fraud']
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

        print(f"Training set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['auc', 'logloss'],
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            'random_state': 42
        }

        print("\nTraining XGBoost model...")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=10,
            verbose_eval=10
        )

        # Evaluate on test set
        print("\n" + "=" * 50)
        print("Model Evaluation on Test Set")
        print("=" * 50)

        dtest = xgb.DMatrix(X_test)
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)

        print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"Precision: {precision_score(y_test, y_pred):.4f}")
        print(f"Recall: {recall_score(y_test, y_pred):.4f}")
        print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
        print(f"ROC AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))

        # Feature importance
        print("\nFeature Importance:")
        importance = self.model.get_score(importance_type='gain')
        for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"  {feature}: {score:.4f}")

        return self.model

    def save(self, model_dir):
        """Save the model and encoders."""
        os.makedirs(model_dir, exist_ok=True)

        # Save XGBoost model
        model_path = os.path.join(model_dir, 'xgboost-model')
        self.model.save_model(model_path)
        print(f"Model saved to: {model_path}")

        # Save label encoders
        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        with open(encoders_path, 'wb') as f:
            pickle.dump(self.label_encoders, f)
        print(f"Label encoders saved to: {encoders_path}")

        # Save feature info
        feature_info = {
            'feature_columns': self.feature_columns,
            'categorical_columns': self.categorical_columns,
            'encoder_classes': {k: list(v.classes_) for k, v in self.label_encoders.items()},
            'use_decline_rates': self.use_decline_rates,
            'decline_rate_features': self.DECLINE_RATE_FEATURES if self.use_decline_rates else [],
            'total_features': 19 if self.use_decline_rates else 7
        }
        feature_info_path = os.path.join(model_dir, 'feature_info.json')
        with open(feature_info_path, 'w') as f:
            json.dump(feature_info, f, indent=2)
        print(f"Feature info saved to: {feature_info_path}")

    def load(self, model_dir):
        """Load the model and encoders."""
        model_path = os.path.join(model_dir, 'xgboost-model')
        self.model = xgb.Booster()
        self.model.load_model(model_path)

        encoders_path = os.path.join(model_dir, 'label_encoders.pkl')
        with open(encoders_path, 'rb') as f:
            self.label_encoders = pickle.load(f)

        print("Model loaded successfully")

    def predict(self, features_dict, decline_rates=None):
        """
        Make a prediction for a single transaction.

        Args:
            features_dict: Must contain: country, bank, bin, sdk_type, ip_address, card_network
            decline_rates: Optional dict with decline rate features (12 features)
                          If not provided and model uses decline rates, zeros will be used.

        Returns:
            Dict with score, prediction, and confidence
        """
        # Merge features with decline rates
        combined = features_dict.copy()
        if self.use_decline_rates:
            if decline_rates:
                combined.update(decline_rates)
            else:
                # Use zeros for missing decline rates
                for feature in self.DECLINE_RATE_FEATURES:
                    combined[feature] = 0.0

        df = pd.DataFrame([combined])
        X = self.preprocess_features(df, fit=False)
        dmatrix = xgb.DMatrix(X)
        score = float(self.model.predict(dmatrix)[0])
        return {
            'score': round(score, 4),
            'prediction': 'fraud' if score > 0.5 else 'legitimate',
            'confidence': round(abs(score - 0.5) * 200, 1)
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Train fraud detection model')
    parser.add_argument('--with-decline-rates', action='store_true',
                        help='Use training data with decline rate features')
    parser.add_argument('--no-decline-rates', action='store_true',
                        help='Train without decline rate features (legacy mode)')
    args = parser.parse_args()

    print("=" * 50)
    print("Fraud Detection Model Training")
    print("=" * 50)

    # Paths
    script_dir = os.path.dirname(__file__)
    data_dir = os.path.join(script_dir, 'data')
    model_dir = os.path.join(script_dir, 'model')

    # Determine which data files to use
    if args.with_decline_rates:
        train_path = os.path.join(data_dir, 'train_with_decline_rates.csv')
        test_path = os.path.join(data_dir, 'test_with_decline_rates.csv')
        use_decline_rates = True
        print("Training with decline rate features (19 total features)")
    else:
        train_path = os.path.join(data_dir, 'train.csv')
        test_path = os.path.join(data_dir, 'test.csv')
        use_decline_rates = not args.no_decline_rates
        print(f"Training {'with' if use_decline_rates else 'without'} decline rate features")

    # Check if data exists
    if not os.path.exists(train_path):
        print(f"Training data not found at {train_path}")
        if args.with_decline_rates:
            print("Please run generate_training_data_with_decline_rates.py first.")
        else:
            print("Please run generate_sample_data.py first.")
        return

    # Train model
    model = FraudDetectionModel(use_decline_rates=use_decline_rates)
    model.train(train_path, test_path if os.path.exists(test_path) else None)
    model.save(model_dir)

    # Test prediction
    print("\n" + "=" * 50)
    print("Sample Predictions")
    print("=" * 50)

    test_cases = [
        {
            'country': 'US',
            'bank': 'chase',
            'bin': '411111',
            'sdk_type': 'ios',
            'ip_address': '192.168.1.1',
            'card_network': 'visa'
        },
        {
            'country': 'NG',
            'bank': 'santander',
            'bin': '400000',
            'sdk_type': 'web',
            'ip_address': '10.0.0.1',
            'card_network': 'visa'
        },
        {
            'country': 'DE',
            'bank': 'deutsche_bank',
            'bin': '512345',
            'sdk_type': 'android',
            'ip_address': '85.123.45.67',
            'card_network': 'mastercard'
        }
    ]

    # Sample decline rates for testing (high-risk vs low-risk)
    sample_decline_rates = {
        'country_decline_rate_7d': 0.25,
        'country_decline_rate_14d': 0.22,
        'bank_decline_rate_7d': 0.10,
        'bank_decline_rate_14d': 0.09,
        'bin_decline_rate_7d': 0.08,
        'bin_decline_rate_14d': 0.07,
        'sdk_type_decline_rate_7d': 0.12,
        'sdk_type_decline_rate_14d': 0.11,
        'ip_decline_rate_7d': 0.15,
        'ip_decline_rate_14d': 0.14,
        'card_network_decline_rate_7d': 0.06,
        'card_network_decline_rate_14d': 0.05,
    }

    for i, case in enumerate(test_cases, 1):
        if model.use_decline_rates:
            result = model.predict(case, decline_rates=sample_decline_rates)
        else:
            result = model.predict(case)
        print(f"\nTest Case {i}:")
        print(f"  Input: {case}")
        if model.use_decline_rates:
            print(f"  Decline Rates: (sample high-risk rates)")
        print(f"  Result: {result}")


if __name__ == '__main__':
    main()
