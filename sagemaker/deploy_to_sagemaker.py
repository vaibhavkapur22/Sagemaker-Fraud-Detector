#!/usr/bin/env python3
"""
Deploy the fraud detection model to AWS SageMaker.
This script:
1. Uploads training data to S3
2. Trains the model using SageMaker's built-in XGBoost
3. Deploys the model as a real-time endpoint
"""

import os
import json
import time
import boto3
import sagemaker
from sagemaker import get_execution_role, image_uris
from sagemaker.inputs import TrainingInput
from sagemaker.estimator import Estimator
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer
import pandas as pd
import argparse

class SageMakerDeployer:
    def __init__(self, region='us-east-1', bucket_name=None, prefix='fraud-detection'):
        """Initialize SageMaker deployer."""
        self.region = region
        self.prefix = prefix

        # Initialize AWS clients
        self.session = boto3.Session(region_name=region)
        self.sm_session = sagemaker.Session(boto_session=self.session)

        # S3 bucket for training data and model artifacts
        self.bucket = bucket_name or self.sm_session.default_bucket()

        # Try to get execution role, fall back to configured role
        try:
            self.role = get_execution_role()
        except ValueError:
            # Not running in SageMaker environment, use configured role
            self.role = os.environ.get('SAGEMAKER_ROLE_ARN')
            if not self.role:
                raise ValueError(
                    "SAGEMAKER_ROLE_ARN environment variable must be set when not running in SageMaker"
                )

        print(f"Region: {self.region}")
        print(f"Bucket: {self.bucket}")
        print(f"Role: {self.role}")

    # Decline rate features (must match training data)
    DECLINE_RATE_FEATURES = [
        'country_decline_rate_7d', 'country_decline_rate_14d',
        'bank_decline_rate_7d', 'bank_decline_rate_14d',
        'bin_decline_rate_7d', 'bin_decline_rate_14d',
        'sdk_type_decline_rate_7d', 'sdk_type_decline_rate_14d',
        'ip_decline_rate_7d', 'ip_decline_rate_14d',
        'card_network_decline_rate_7d', 'card_network_decline_rate_14d',
    ]

    def prepare_data_for_sagemaker(self, train_path, test_path, use_decline_rates=True):
        """
        Prepare data in format required by SageMaker XGBoost.
        SageMaker XGBoost expects CSV with target as first column.
        """
        print("\nPreparing data for SageMaker...")
        print(f"Using decline rates: {use_decline_rates}")

        # Load data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Encode categorical features
        from sklearn.preprocessing import LabelEncoder

        categorical_cols = ['country', 'bank', 'sdk_type', 'card_network']
        encoders = {}

        for col in categorical_cols:
            encoders[col] = LabelEncoder()
            # Fit on combined data to handle all categories
            all_values = pd.concat([train_df[col], test_df[col]]).astype(str)
            encoders[col].fit(all_values)
            train_df[col] = encoders[col].transform(train_df[col].astype(str))
            test_df[col] = encoders[col].transform(test_df[col].astype(str))

        # Convert BIN to numeric
        train_df['bin'] = pd.to_numeric(train_df['bin'], errors='coerce').fillna(0)
        test_df['bin'] = pd.to_numeric(test_df['bin'], errors='coerce').fillna(0)

        # Extract IP features
        for df in [train_df, test_df]:
            df['ip_first_octet'] = df['ip_address'].apply(
                lambda x: int(str(x).split('.')[0]) if pd.notna(x) else 0
            )
            df['ip_is_private'] = df['ip_address'].apply(
                lambda x: 1 if str(x).startswith(('10.', '192.168.', '172.')) else 0
            )

        # Base feature columns (7 features)
        feature_cols = ['country', 'bank', 'bin', 'sdk_type', 'card_network',
                        'ip_first_octet', 'ip_is_private']

        # Add decline rate features if available and requested (12 additional features)
        if use_decline_rates:
            for feature in self.DECLINE_RATE_FEATURES:
                if feature in train_df.columns:
                    # Ensure numeric
                    train_df[feature] = pd.to_numeric(train_df[feature], errors='coerce').fillna(0.0)
                    test_df[feature] = pd.to_numeric(test_df[feature], errors='coerce').fillna(0.0)
                    feature_cols.append(feature)
                else:
                    print(f"  Warning: {feature} not found in data")

        print(f"Total features: {len(feature_cols)}")

        train_df = train_df[['is_fraud'] + feature_cols]
        test_df = test_df[['is_fraud'] + feature_cols]

        # Save processed data
        output_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
        os.makedirs(output_dir, exist_ok=True)

        train_processed_path = os.path.join(output_dir, 'train.csv')
        test_processed_path = os.path.join(output_dir, 'test.csv')

        # Save without header (SageMaker XGBoost requirement)
        train_df.to_csv(train_processed_path, index=False, header=False)
        test_df.to_csv(test_processed_path, index=False, header=False)

        # Save encoders info
        encoder_info = {col: list(enc.classes_) for col, enc in encoders.items()}
        encoder_path = os.path.join(output_dir, 'encoders.json')
        with open(encoder_path, 'w') as f:
            json.dump(encoder_info, f, indent=2)

        print(f"Processed training data: {train_processed_path}")
        print(f"Processed test data: {test_processed_path}")
        print(f"Encoder info: {encoder_path}")

        return train_processed_path, test_processed_path, encoder_info

    def upload_data_to_s3(self, train_path, test_path):
        """Upload training and test data to S3."""
        print("\nUploading data to S3...")

        train_s3_path = self.sm_session.upload_data(
            train_path,
            bucket=self.bucket,
            key_prefix=f'{self.prefix}/data/train'
        )

        test_s3_path = self.sm_session.upload_data(
            test_path,
            bucket=self.bucket,
            key_prefix=f'{self.prefix}/data/test'
        )

        print(f"Training data uploaded to: {train_s3_path}")
        print(f"Test data uploaded to: {test_s3_path}")

        return train_s3_path, test_s3_path

    def train_model(self, train_s3_path, test_s3_path, instance_type='ml.m5.large'):
        """Train XGBoost model using SageMaker's built-in algorithm."""
        print("\nStarting SageMaker training job...")

        # Get the built-in XGBoost container image
        container = image_uris.retrieve(
            framework='xgboost',
            region=self.region,
            version='1.5-1'
        )
        print(f"Using container: {container}")

        # Configure XGBoost estimator using built-in algorithm
        xgb_estimator = Estimator(
            image_uri=container,
            instance_type=instance_type,
            instance_count=1,
            role=self.role,
            sagemaker_session=self.sm_session,
            hyperparameters={
                'objective': 'binary:logistic',
                'num_round': 100,
                'max_depth': 6,
                'eta': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'auc'
            },
            output_path=f's3://{self.bucket}/{self.prefix}/output'
        )

        # Set up training inputs
        train_input = TrainingInput(
            train_s3_path,
            content_type='text/csv'
        )
        test_input = TrainingInput(
            test_s3_path,
            content_type='text/csv'
        )

        # Start training
        job_name = f'fraud-detection-{int(time.time())}'
        xgb_estimator.fit(
            {'train': train_input, 'validation': test_input},
            job_name=job_name,
            wait=True
        )

        print(f"Training job completed: {job_name}")
        return xgb_estimator

    def deploy_endpoint(self, estimator, endpoint_name='fraud-detector-endpoint',
                        instance_type='ml.t2.medium'):
        """Deploy trained model as a real-time endpoint."""
        print(f"\nDeploying model to endpoint: {endpoint_name}")

        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            serializer=CSVSerializer(),
            deserializer=JSONDeserializer()
        )

        print(f"Endpoint deployed successfully: {endpoint_name}")
        return predictor

    def test_endpoint(self, endpoint_name, num_features=19):
        """Test the deployed endpoint."""
        print(f"\nTesting endpoint: {endpoint_name}")

        runtime = self.session.client('sagemaker-runtime')

        # Test payload - CSV format for built-in XGBoost
        # Features: country, bank, bin, sdk_type, card_network, ip_first_octet, ip_is_private
        # + 12 decline rate features (all zeros for test)
        if num_features == 19:
            # 7 base features + 12 decline rates
            test_values = [0, 0, 411111, 0, 0, 192, 1] + [0.1] * 12
        else:
            # 7 base features only
            test_values = [0, 0, 411111, 0, 0, 192, 1]

        test_payload = ','.join(map(str, test_values))

        response = runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_payload
        )

        result = response['Body'].read().decode()
        print(f"Test prediction result: {result}")
        return result


def main():
    parser = argparse.ArgumentParser(description='Deploy fraud detection model to SageMaker')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--bucket', help='S3 bucket name (uses default if not specified)')
    parser.add_argument('--endpoint-name', default='fraud-detector-endpoint',
                        help='Name for the SageMaker endpoint')
    parser.add_argument('--instance-type', default='ml.m5.large',
                        help='Instance type for training')
    parser.add_argument('--deploy-instance-type', default='ml.t2.medium',
                        help='Instance type for endpoint')
    parser.add_argument('--skip-training', action='store_true',
                        help='Skip training, only deploy from existing model')
    parser.add_argument('--with-decline-rates', action='store_true', default=True,
                        help='Use training data with decline rate features (default: True)')
    parser.add_argument('--no-decline-rates', action='store_true',
                        help='Train without decline rate features (legacy 7-feature mode)')

    args = parser.parse_args()

    use_decline_rates = args.with_decline_rates and not args.no_decline_rates

    print("=" * 60)
    print("SageMaker Fraud Detection Model Deployment")
    print("=" * 60)
    print(f"Model type: {'19 features (with decline rates)' if use_decline_rates else '7 features (legacy)'}")

    # Check for required environment variables
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    if not all(os.environ.get(var) for var in required_vars):
        print("\nError: AWS credentials not found.")
        print("Please set the following environment variables:")
        for var in required_vars:
            print(f"  - {var}")
        print("\nOptionally set:")
        print("  - SAGEMAKER_ROLE_ARN (required if not running in SageMaker)")
        return

    # Initialize deployer
    deployer = SageMakerDeployer(
        region=args.region,
        bucket_name=args.bucket
    )

    # Paths - use decline rate data if available
    script_dir = os.path.dirname(__file__)
    if use_decline_rates:
        train_path = os.path.join(script_dir, 'data', 'train_with_decline_rates.csv')
        test_path = os.path.join(script_dir, 'data', 'test_with_decline_rates.csv')
    else:
        train_path = os.path.join(script_dir, 'data', 'train.csv')
        test_path = os.path.join(script_dir, 'data', 'test.csv')

    # Check if data exists
    if not os.path.exists(train_path):
        print(f"\nTraining data not found at {train_path}")
        if use_decline_rates:
            print("Please run: python3 generate_training_data_with_decline_rates.py --synthetic")
        else:
            print("Please run generate_sample_data.py first.")
        return

    if not args.skip_training:
        # Prepare and upload data
        train_processed, test_processed, _ = deployer.prepare_data_for_sagemaker(
            train_path, test_path, use_decline_rates=use_decline_rates
        )
        train_s3, test_s3 = deployer.upload_data_to_s3(train_processed, test_processed)

        # Train model
        estimator = deployer.train_model(
            train_s3, test_s3,
            instance_type=args.instance_type
        )

        # Deploy endpoint
        deployer.deploy_endpoint(
            estimator,
            endpoint_name=args.endpoint_name,
            instance_type=args.deploy_instance_type
        )

    # Test endpoint
    deployer.test_endpoint(args.endpoint_name)

    print("\n" + "=" * 60)
    print("Deployment Complete!")
    print("=" * 60)
    print(f"\nEndpoint Name: {args.endpoint_name}")
    print(f"\nAdd these to your Rails .env file:")
    print(f"  SAGEMAKER_ENDPOINT_NAME={args.endpoint_name}")
    print(f"  AWS_REGION={args.region}")
    print(f"  SAGEMAKER_MOCK_MODE=false")


if __name__ == '__main__':
    main()
