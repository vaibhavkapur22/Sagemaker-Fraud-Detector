#!/usr/bin/env python3
"""
XGBoost training script for SageMaker.
This script is used as the entry point for SageMaker training jobs.

Supports both:
- Original 7 features (legacy mode)
- Extended 19 features (with decline rates)
"""

import argparse
import os
import json
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Feature definitions
CATEGORICAL_COLUMNS = ['country', 'bank', 'sdk_type', 'card_network']
DECLINE_RATE_FEATURES = [
    'country_decline_rate_7d', 'country_decline_rate_14d',
    'bank_decline_rate_7d', 'bank_decline_rate_14d',
    'bin_decline_rate_7d', 'bin_decline_rate_14d',
    'sdk_type_decline_rate_7d', 'sdk_type_decline_rate_14d',
    'ip_decline_rate_7d', 'ip_decline_rate_14d',
    'card_network_decline_rate_7d', 'card_network_decline_rate_14d',
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument('--objective', type=str, default='binary:logistic')
    parser.add_argument('--num_round', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=6)
    parser.add_argument('--eta', type=float, default=0.1)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample_bytree', type=float, default=0.8)
    parser.add_argument('--eval_metric', type=str, default='auc')

    # SageMaker specific arguments
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))

    return parser.parse_args()


def load_data(data_dir):
    """Load CSV data from directory."""
    files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not files:
        raise ValueError(f"No CSV files found in {data_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f), header=None)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)

    # First column is target, rest are features
    y = data.iloc[:, 0]
    X = data.iloc[:, 1:]

    return X, y


def train():
    """Main training function."""
    args = parse_args()

    print("Loading training data...")
    X_train, y_train = load_data(args.train)
    print(f"Training data shape: {X_train.shape}")

    print("Loading validation data...")
    X_val, y_val = load_data(args.validation)
    print(f"Validation data shape: {X_val.shape}")

    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # Set up parameters
    params = {
        'objective': args.objective,
        'max_depth': args.max_depth,
        'eta': args.eta,
        'subsample': args.subsample,
        'colsample_bytree': args.colsample_bytree,
        'eval_metric': args.eval_metric,
    }

    print(f"Training parameters: {params}")
    print(f"Number of boosting rounds: {args.num_round}")

    # Train model
    watchlist = [(dtrain, 'train'), (dval, 'validation')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round,
        evals=watchlist,
        early_stopping_rounds=10,
        verbose_eval=10
    )

    # Evaluate
    y_pred_proba = model.predict(dval)
    y_pred = (y_pred_proba > 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    print(f"\nValidation Accuracy: {accuracy:.4f}")
    print(f"Validation AUC: {auc:.4f}")

    # Save model
    model_path = os.path.join(args.model_dir, 'xgboost-model')
    model.save_model(model_path)
    print(f"Model saved to: {model_path}")


def model_fn(model_dir):
    """Load model for inference."""
    model_path = os.path.join(model_dir, 'xgboost-model')
    model = xgb.Booster()
    model.load_model(model_path)
    return model


def preprocess_inference_input(data):
    """
    Preprocess input data for inference.

    Handles two input formats:
    1. Pre-encoded features (numeric array) - passes through directly
    2. Raw features (dict with country, bank, etc.) - encodes them

    For raw features, also handles optional decline rate features.
    """
    df = pd.DataFrame(data if isinstance(data, list) else [data])

    # Check if data is already encoded (all numeric)
    if df.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all():
        return df

    # Raw features need encoding
    # Load encoders if available (saved during training)
    model_dir = os.environ.get('SM_MODEL_DIR', '/opt/ml/model')
    encoders_path = os.path.join(model_dir, 'label_encoders.pkl')

    if os.path.exists(encoders_path):
        with open(encoders_path, 'rb') as f:
            label_encoders = pickle.load(f)
    else:
        # Create simple encoders based on known categories
        label_encoders = {}

    # Encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                df[f'{col}_encoded'] = df[col].apply(
                    lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                )
            else:
                # Fallback: use hash encoding
                df[f'{col}_encoded'] = df[col].apply(lambda x: hash(str(x)) % 1000)

    # Convert BIN to numeric
    if 'bin' in df.columns:
        df['bin_numeric'] = pd.to_numeric(df['bin'], errors='coerce').fillna(0)

    # Extract IP features
    if 'ip_address' in df.columns:
        df['ip_first_octet'] = df['ip_address'].apply(
            lambda x: int(str(x).split('.')[0]) if pd.notna(x) and '.' in str(x) else 0
        )
        df['ip_is_private'] = df['ip_address'].apply(
            lambda x: 1 if str(x).startswith(('10.', '192.168.', '172.')) else 0
        )

    # Build feature columns
    feature_cols = [
        'country_encoded', 'bank_encoded', 'sdk_type_encoded',
        'card_network_encoded', 'bin_numeric', 'ip_first_octet', 'ip_is_private'
    ]

    # Add decline rate features if present
    for feature in DECLINE_RATE_FEATURES:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0.0)
            feature_cols.append(feature)

    # Select only the feature columns that exist
    available_cols = [c for c in feature_cols if c in df.columns]
    return df[available_cols]


def input_fn(request_body, request_content_type):
    """Parse input data for inference."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        if 'instances' in data:
            processed = preprocess_inference_input(data['instances'])
        else:
            processed = preprocess_inference_input(data)
        return xgb.DMatrix(processed)
    elif request_content_type == 'text/csv':
        # CSV assumes pre-encoded features
        return xgb.DMatrix(pd.read_csv(pd.io.common.StringIO(request_body), header=None))
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Make predictions."""
    predictions = model.predict(input_data)
    return predictions


def output_fn(prediction, response_content_type):
    """Format prediction output."""
    if response_content_type == 'application/json':
        return json.dumps({'predictions': prediction.tolist()})
    raise ValueError(f"Unsupported response content type: {response_content_type}")


if __name__ == '__main__':
    train()
