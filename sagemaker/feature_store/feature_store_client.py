#!/usr/bin/env python3
"""
Client for interacting with SageMaker Feature Store.

Provides methods for:
- Putting records (single and batch)
- Getting records for online inference
- Querying offline store for training data
"""

import os
import time
from typing import Dict, List, Optional, Any
import boto3
import pandas as pd


class FeatureStoreClient:
    """Client for SageMaker Feature Store operations."""

    DECLINE_RATE_FEATURES = [
        'country_decline_rate_7d', 'country_decline_rate_14d',
        'bank_decline_rate_7d', 'bank_decline_rate_14d',
        'bin_decline_rate_7d', 'bin_decline_rate_14d',
        'sdk_type_decline_rate_7d', 'sdk_type_decline_rate_14d',
        'ip_decline_rate_7d', 'ip_decline_rate_14d',
        'card_network_decline_rate_7d', 'card_network_decline_rate_14d',
    ]

    def __init__(
        self,
        region: str = None,
        feature_group_name: str = 'fraud-detection-decline-rates'
    ):
        """
        Initialize the Feature Store client.

        Args:
            region: AWS region (defaults to AWS_REGION env var or us-east-1)
            feature_group_name: Name of the feature group
        """
        self.region = region or os.environ.get('AWS_REGION', 'us-east-1')
        self.feature_group_name = feature_group_name

        self.featurestore_runtime = boto3.client(
            'sagemaker-featurestore-runtime',
            region_name=self.region
        )

        self.sagemaker = boto3.client(
            'sagemaker',
            region_name=self.region
        )

    def get_record(self, record_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single record from the online store.

        Args:
            record_id: The record identifier

        Returns:
            Dictionary of feature name -> value, or None if not found
        """
        try:
            response = self.featurestore_runtime.get_record(
                FeatureGroupName=self.feature_group_name,
                RecordIdentifierValueAsString=record_id
            )

            if 'Record' not in response:
                return None

            return {
                f['FeatureName']: self._parse_value(f)
                for f in response['Record']
            }

        except self.featurestore_runtime.exceptions.ResourceNotFoundException:
            return None
        except Exception as e:
            print(f"Error fetching record {record_id}: {e}")
            return None

    def get_decline_rates(self, record_id: str) -> Dict[str, float]:
        """
        Fetch only decline rate features for a record.

        Args:
            record_id: The record identifier

        Returns:
            Dictionary of decline rate feature name -> value
        """
        record = self.get_record(record_id)

        if not record:
            # Return zeros if record not found
            return {feature: 0.0 for feature in self.DECLINE_RATE_FEATURES}

        return {
            feature: float(record.get(feature, 0.0))
            for feature in self.DECLINE_RATE_FEATURES
        }

    def put_record(self, record: Dict[str, Any]) -> bool:
        """
        Write a single record to the feature store.

        Args:
            record: Dictionary containing all feature values
                   Must include 'record_id' and 'event_time'

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert record to Feature Store format
            feature_records = [
                {'FeatureName': k, 'ValueAsString': str(v)}
                for k, v in record.items()
            ]

            self.featurestore_runtime.put_record(
                FeatureGroupName=self.feature_group_name,
                Record=feature_records
            )
            return True

        except Exception as e:
            print(f"Error putting record: {e}")
            return False

    def put_records_batch(self, records: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Write multiple records to the feature store.

        Args:
            records: List of record dictionaries

        Returns:
            Dictionary with 'success' and 'failed' counts
        """
        success_count = 0
        failed_count = 0

        for record in records:
            if self.put_record(record):
                success_count += 1
            else:
                failed_count += 1

        return {'success': success_count, 'failed': failed_count}

    def build_record(
        self,
        record_id: str,
        features: Dict[str, str],
        decline_rates: Dict[str, float],
        is_fraud: int = 0,
        status: str = 'pending'
    ) -> Dict[str, Any]:
        """
        Build a complete record for the feature store.

        Args:
            record_id: Unique identifier for the record
            features: Original transaction features (country, bank, etc.)
            decline_rates: Computed decline rate features
            is_fraud: Whether the transaction is fraud (0 or 1)
            status: Transaction status (approved, declined, pending)

        Returns:
            Complete record dictionary ready for put_record
        """
        record = {
            'record_id': record_id,
            'event_time': time.time(),
            'is_fraud': is_fraud,
            'status': status,
        }

        # Add original features
        for key in ['country', 'bank', 'bin', 'sdk_type', 'ip_address', 'card_network']:
            record[key] = features.get(key, '')

        # Add decline rate features
        for feature in self.DECLINE_RATE_FEATURES:
            record[feature] = decline_rates.get(feature, 0.0)

        return record

    def describe_feature_group(self) -> Dict[str, Any]:
        """
        Get information about the feature group.

        Returns:
            Feature group description
        """
        try:
            return self.sagemaker.describe_feature_group(
                FeatureGroupName=self.feature_group_name
            )
        except Exception as e:
            print(f"Error describing feature group: {e}")
            return {}

    def _parse_value(self, feature: Dict) -> Any:
        """Parse a feature value from the Feature Store response."""
        value = feature.get('ValueAsString', '')

        # Try to parse as number
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except ValueError:
            return value


def create_training_dataset_query(feature_group_name: str, database: str, table: str) -> str:
    """
    Generate an Athena query to create a training dataset from the offline store.

    Args:
        feature_group_name: Name of the feature group
        database: Glue database name
        table: Glue table name

    Returns:
        SQL query string
    """
    return f"""
    SELECT
        country, bank, bin, sdk_type, ip_address, card_network,
        country_decline_rate_7d, country_decline_rate_14d,
        bank_decline_rate_7d, bank_decline_rate_14d,
        bin_decline_rate_7d, bin_decline_rate_14d,
        sdk_type_decline_rate_7d, sdk_type_decline_rate_14d,
        ip_decline_rate_7d, ip_decline_rate_14d,
        card_network_decline_rate_7d, card_network_decline_rate_14d,
        is_fraud
    FROM "{database}"."{table}"
    WHERE is_deleted = false
    ORDER BY event_time DESC
    """


if __name__ == '__main__':
    # Example usage
    client = FeatureStoreClient()

    # Describe the feature group
    print("Feature Group Info:")
    info = client.describe_feature_group()
    if info:
        print(f"  Name: {info.get('FeatureGroupName')}")
        print(f"  Status: {info.get('FeatureGroupStatus')}")

    # Example: Get decline rates for a record
    print("\nExample get_decline_rates:")
    rates = client.get_decline_rates('test_record_123')
    for feature, value in rates.items():
        print(f"  {feature}: {value}")
