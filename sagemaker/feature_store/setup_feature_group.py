#!/usr/bin/env python3
"""
Set up SageMaker Feature Store feature groups for decline rate features.

This script creates a feature group that stores:
- Transaction identifiers
- Original transaction features
- Computed decline rate features (7-day and 14-day for each dimension)

Usage:
    python setup_feature_group.py --role-arn <sagemaker-role-arn>
    python setup_feature_group.py --role-arn <arn> --bucket <s3-bucket> --region us-west-2
"""

import argparse
import os
import time
import boto3
import sagemaker
from sagemaker.feature_store.feature_group import FeatureGroup
from sagemaker.feature_store.feature_definition import (
    FeatureDefinition,
    FeatureTypeEnum,
)


def create_feature_definitions():
    """Define all features for the decline rate feature group."""
    return [
        # Record identifier and event time (required by Feature Store)
        FeatureDefinition(feature_name='record_id', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='event_time', feature_type=FeatureTypeEnum.FRACTIONAL),

        # Original transaction features for lookup
        FeatureDefinition(feature_name='country', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='bank', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='bin', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='sdk_type', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='ip_address', feature_type=FeatureTypeEnum.STRING),
        FeatureDefinition(feature_name='card_network', feature_type=FeatureTypeEnum.STRING),

        # Decline rate features - 7 day window
        FeatureDefinition(feature_name='country_decline_rate_7d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='bank_decline_rate_7d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='bin_decline_rate_7d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='sdk_type_decline_rate_7d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='ip_decline_rate_7d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='card_network_decline_rate_7d', feature_type=FeatureTypeEnum.FRACTIONAL),

        # Decline rate features - 14 day window
        FeatureDefinition(feature_name='country_decline_rate_14d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='bank_decline_rate_14d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='bin_decline_rate_14d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='sdk_type_decline_rate_14d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='ip_decline_rate_14d', feature_type=FeatureTypeEnum.FRACTIONAL),
        FeatureDefinition(feature_name='card_network_decline_rate_14d', feature_type=FeatureTypeEnum.FRACTIONAL),

        # Target variable for training
        FeatureDefinition(feature_name='is_fraud', feature_type=FeatureTypeEnum.INTEGRAL),
        FeatureDefinition(feature_name='status', feature_type=FeatureTypeEnum.STRING),
    ]


def create_decline_rate_feature_group(
    sagemaker_session,
    role_arn,
    feature_group_name='fraud-detection-decline-rates',
    offline_store_s3_uri=None,
    enable_online_store=True
):
    """
    Create a feature group for decline rate features.

    Args:
        sagemaker_session: SageMaker session
        role_arn: IAM role ARN for SageMaker
        feature_group_name: Name for the feature group
        offline_store_s3_uri: S3 URI for offline store (e.g., s3://bucket/prefix)
        enable_online_store: Whether to enable online store for real-time lookup

    Returns:
        FeatureGroup object
    """
    print(f"Creating feature group: {feature_group_name}")

    feature_definitions = create_feature_definitions()

    feature_group = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=sagemaker_session,
        feature_definitions=feature_definitions
    )

    # Create the feature group
    try:
        feature_group.create(
            s3_uri=offline_store_s3_uri,
            record_identifier_name='record_id',
            event_time_feature_name='event_time',
            role_arn=role_arn,
            enable_online_store=enable_online_store,
            description='Decline rate features for fraud detection model'
        )
        print(f"Feature group '{feature_group_name}' creation initiated")

        # Wait for feature group to be created
        print("Waiting for feature group to be active...")
        status = None
        while status != 'Created':
            time.sleep(5)
            status = feature_group.describe().get('FeatureGroupStatus')
            print(f"  Status: {status}")
            if status == 'CreateFailed':
                raise Exception(f"Feature group creation failed: {feature_group.describe()}")

        print(f"Feature group '{feature_group_name}' is now active")

    except Exception as e:
        if 'ResourceInUse' in str(e):
            print(f"Feature group '{feature_group_name}' already exists")
        else:
            raise

    return feature_group


def delete_feature_group(sagemaker_session, feature_group_name):
    """Delete an existing feature group."""
    print(f"Deleting feature group: {feature_group_name}")

    feature_group = FeatureGroup(
        name=feature_group_name,
        sagemaker_session=sagemaker_session
    )

    try:
        feature_group.delete()
        print(f"Feature group '{feature_group_name}' deleted")
    except Exception as e:
        print(f"Error deleting feature group: {e}")


def main():
    parser = argparse.ArgumentParser(description='Set up SageMaker Feature Store for decline rates')
    parser.add_argument('--role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--bucket', help='S3 bucket for offline store')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--feature-group-name', default='fraud-detection-decline-rates',
                        help='Name for the feature group')
    parser.add_argument('--delete', action='store_true', help='Delete existing feature group')
    parser.add_argument('--no-online-store', action='store_true',
                        help='Disable online store (offline only)')

    args = parser.parse_args()

    # Set up AWS session
    boto_session = boto3.Session(region_name=args.region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    # Determine S3 bucket
    bucket = args.bucket or sagemaker_session.default_bucket()
    offline_store_s3_uri = f's3://{bucket}/fraud-detection/feature-store'

    print(f"Using S3 bucket: {bucket}")
    print(f"Offline store URI: {offline_store_s3_uri}")
    print(f"Region: {args.region}")

    if args.delete:
        delete_feature_group(sagemaker_session, args.feature_group_name)
        return

    # Create the feature group
    feature_group = create_decline_rate_feature_group(
        sagemaker_session=sagemaker_session,
        role_arn=args.role_arn,
        feature_group_name=args.feature_group_name,
        offline_store_s3_uri=offline_store_s3_uri,
        enable_online_store=not args.no_online_store
    )

    # Print feature group info
    print("\nFeature Group Details:")
    description = feature_group.describe()
    print(f"  Name: {description['FeatureGroupName']}")
    print(f"  ARN: {description['FeatureGroupArn']}")
    print(f"  Status: {description['FeatureGroupStatus']}")
    print(f"  Online Store: {'Enabled' if description.get('OnlineStoreConfig', {}).get('EnableOnlineStore') else 'Disabled'}")
    print(f"  Offline Store S3 URI: {description.get('OfflineStoreConfig', {}).get('S3StorageConfig', {}).get('S3Uri', 'N/A')}")

    print("\nFeatures:")
    for feature in description['FeatureDefinitions']:
        print(f"  - {feature['FeatureName']}: {feature['FeatureType']}")


if __name__ == '__main__':
    main()
