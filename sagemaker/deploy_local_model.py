#!/usr/bin/env python3
"""
Deploy a locally trained XGBoost model to AWS SageMaker.
This script:
1. Packages the local model into a tar.gz file
2. Uploads it to S3
3. Creates a SageMaker model
4. Deploys it as a real-time endpoint
"""

import os
import json
import tarfile
import boto3
import sagemaker
from sagemaker.xgboost import XGBoostModel
import argparse


def create_model_tarball(model_dir, output_path):
    """Package the model into a tar.gz file for SageMaker."""
    print(f"Creating model tarball from {model_dir}...")

    with tarfile.open(output_path, "w:gz") as tar:
        # Add the xgboost model file
        model_file = os.path.join(model_dir, 'xgboost-model')
        if os.path.exists(model_file):
            tar.add(model_file, arcname='xgboost-model')
            print(f"  Added: xgboost-model")

        # Add label encoders
        encoders_file = os.path.join(model_dir, 'label_encoders.pkl')
        if os.path.exists(encoders_file):
            tar.add(encoders_file, arcname='label_encoders.pkl')
            print(f"  Added: label_encoders.pkl")

        # Add feature info
        feature_info_file = os.path.join(model_dir, 'feature_info.json')
        if os.path.exists(feature_info_file):
            tar.add(feature_info_file, arcname='feature_info.json')
            print(f"  Added: feature_info.json")

    print(f"Model tarball created: {output_path}")
    return output_path


def upload_to_s3(local_path, bucket, key):
    """Upload a file to S3."""
    print(f"Uploading {local_path} to s3://{bucket}/{key}...")
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket, key)
    s3_uri = f"s3://{bucket}/{key}"
    print(f"Uploaded to: {s3_uri}")
    return s3_uri


def deploy_model(model_data_s3, role, region, endpoint_name, instance_type='ml.t2.medium'):
    """Deploy the model to SageMaker endpoint."""
    print(f"\nDeploying model to endpoint: {endpoint_name}")

    boto_session = boto3.Session(region_name=region)
    sm_session = sagemaker.Session(boto_session=boto_session)

    # Create XGBoost model
    xgb_model = XGBoostModel(
        model_data=model_data_s3,
        role=role,
        framework_version='1.7-1',
        sagemaker_session=sm_session,
        py_version='py3'
    )

    # Deploy to endpoint
    predictor = xgb_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )

    print(f"Endpoint deployed successfully: {endpoint_name}")
    return predictor


def test_endpoint(endpoint_name, region):
    """Test the deployed endpoint."""
    print(f"\nTesting endpoint: {endpoint_name}")

    runtime = boto3.client('sagemaker-runtime', region_name=region)

    # Test with CSV format (SageMaker XGBoost default)
    # Features: country_encoded, bank_encoded, sdk_type_encoded, card_network_encoded, bin_numeric, ip_first_octet, ip_is_private
    test_data = "0,0,0,0,411111,192,1"

    response = runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='text/csv',
        Body=test_data
    )

    result = response['Body'].read().decode()
    print(f"Test input: {test_data}")
    print(f"Prediction result: {result}")
    return result


def main():
    parser = argparse.ArgumentParser(description='Deploy local model to SageMaker')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--bucket', help='S3 bucket name (uses default if not specified)')
    parser.add_argument('--endpoint-name', default='fraud-detector-endpoint',
                        help='Name for the SageMaker endpoint')
    parser.add_argument('--instance-type', default='ml.t2.medium',
                        help='Instance type for endpoint')
    parser.add_argument('--skip-deploy', action='store_true',
                        help='Skip deployment, only upload model')

    args = parser.parse_args()

    print("=" * 60)
    print("Deploy Local Model to SageMaker")
    print("=" * 60)

    # Check for required environment variables
    required_vars = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY']
    if not all(os.environ.get(var) for var in required_vars):
        print("\nError: AWS credentials not found.")
        print("Please set the following environment variables:")
        for var in required_vars:
            print(f"  - {var}")
        return

    # Get role
    role = os.environ.get('SAGEMAKER_ROLE_ARN')
    if not role:
        print("\nError: SAGEMAKER_ROLE_ARN environment variable must be set")
        return

    # Paths
    script_dir = os.path.dirname(__file__)
    model_dir = os.path.join(script_dir, 'model')
    tarball_path = os.path.join(script_dir, 'model.tar.gz')

    # Check if model exists
    if not os.path.exists(os.path.join(model_dir, 'xgboost-model')):
        print(f"\nModel not found at {model_dir}")
        print("Please run train_model.py first.")
        return

    # Initialize SageMaker session
    boto_session = boto3.Session(region_name=args.region)
    sm_session = sagemaker.Session(boto_session=boto_session)
    bucket = args.bucket or sm_session.default_bucket()

    print(f"Region: {args.region}")
    print(f"Bucket: {bucket}")
    print(f"Role: {role}")

    # Create model tarball
    create_model_tarball(model_dir, tarball_path)

    # Upload to S3
    s3_key = 'fraud-detection/model/model.tar.gz'
    model_s3_uri = upload_to_s3(tarball_path, bucket, s3_key)

    if args.skip_deploy:
        print("\nSkipping deployment (--skip-deploy flag set)")
        print(f"Model uploaded to: {model_s3_uri}")
        return

    # Deploy model
    deploy_model(
        model_data_s3=model_s3_uri,
        role=role,
        region=args.region,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type
    )

    # Test endpoint
    test_endpoint(args.endpoint_name, args.region)

    print("\n" + "=" * 60)
    print("Deployment Complete!")
    print("=" * 60)
    print(f"\nEndpoint Name: {args.endpoint_name}")
    print(f"\nAdd these to your Rails .env file:")
    print(f"  SAGEMAKER_ENDPOINT_NAME={args.endpoint_name}")
    print(f"  AWS_REGION={args.region}")
    print(f"  SAGEMAKER_MOCK_MODE=false")

    # Cleanup
    if os.path.exists(tarball_path):
        os.remove(tarball_path)
        print(f"\nCleaned up temporary file: {tarball_path}")


if __name__ == '__main__':
    main()
