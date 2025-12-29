#!/usr/bin/env python3
"""
ETL script to sync transaction data from PostgreSQL to SageMaker Feature Store.

This script:
1. Connects to PostgreSQL (AWS RDS)
2. Computes point-in-time decline rates for each transaction
3. Ingests records into SageMaker Feature Store

Usage:
    python sync_postgres_to_feature_store.py --db-url <postgres-url>
    python sync_postgres_to_feature_store.py --db-url <url> --batch-size 500 --limit 10000
"""

import argparse
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_store.feature_store_client import FeatureStoreClient


DIMENSIONS = ['country', 'bank', 'bin', 'sdk_type', 'ip_address', 'card_network']
WINDOWS = [7, 14]


class PostgresDeclineRateCalculator:
    """Calculate decline rates from PostgreSQL transaction data."""

    def __init__(self, connection_string: str):
        self.conn_string = connection_string
        self.conn = None

    def connect(self):
        """Establish database connection."""
        self.conn = psycopg2.connect(self.conn_string)
        print("Connected to PostgreSQL")

    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            print("Disconnected from PostgreSQL")

    def get_transactions(self, limit: Optional[int] = None, offset: int = 0) -> pd.DataFrame:
        """
        Fetch transactions from the database.

        Args:
            limit: Maximum number of transactions to fetch
            offset: Number of transactions to skip

        Returns:
            DataFrame with transaction data
        """
        query = """
            SELECT
                transaction_id, country, bank, bin, sdk_type,
                ip_address, card_network, status, is_fraud, processed_at
            FROM transactions
            WHERE status IN ('approved', 'declined')
            ORDER BY processed_at ASC
        """

        if limit:
            query += f" LIMIT {limit}"
        if offset:
            query += f" OFFSET {offset}"

        return pd.read_sql(query, self.conn)

    def get_transaction_count(self) -> int:
        """Get total count of transactions."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM transactions WHERE status IN ('approved', 'declined')")
            return cur.fetchone()[0]

    def compute_decline_rate(
        self,
        dimension: str,
        value: str,
        before_time: datetime,
        days: int
    ) -> float:
        """
        Compute historical decline rate for a dimension/value pair.

        Args:
            dimension: The dimension (country, bank, etc.)
            value: The value to filter by
            before_time: Calculate rate before this time (point-in-time)
            days: Number of days to look back

        Returns:
            Decline rate as a float (0.0 to 1.0)
        """
        start_time = before_time - timedelta(days=days)

        query = f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'declined' THEN 1 ELSE 0 END) as declined
            FROM transactions
            WHERE {dimension} = %s
              AND processed_at >= %s
              AND processed_at < %s
              AND status IN ('approved', 'declined')
        """

        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, (value, start_time, before_time))
            result = cur.fetchone()

        total = result['total'] or 0
        declined = result['declined'] or 0

        if total == 0:
            return 0.0

        return round(declined / total, 4)

    def compute_all_decline_rates(
        self,
        features: Dict[str, str],
        before_time: datetime
    ) -> Dict[str, float]:
        """
        Compute all decline rate features for a transaction.

        Args:
            features: Transaction features (country, bank, etc.)
            before_time: Calculate rates before this time

        Returns:
            Dictionary of decline rate feature name -> value
        """
        rates = {}

        for dimension in DIMENSIONS:
            value = features.get(dimension)
            if not value:
                continue

            for days in WINDOWS:
                feature_name = f'{dimension}_decline_rate_{days}d'
                rates[feature_name] = self.compute_decline_rate(
                    dimension, value, before_time, days
                )

        return rates


def sync_to_feature_store(
    db_url: str,
    feature_group_name: str = 'fraud-detection-decline-rates',
    batch_size: int = 100,
    limit: Optional[int] = None,
    region: str = 'us-east-1'
):
    """
    Sync PostgreSQL transactions to SageMaker Feature Store.

    Args:
        db_url: PostgreSQL connection string
        feature_group_name: Name of the feature group
        batch_size: Number of records to process in each batch
        limit: Maximum number of transactions to process
        region: AWS region
    """
    calculator = PostgresDeclineRateCalculator(db_url)
    fs_client = FeatureStoreClient(region=region, feature_group_name=feature_group_name)

    try:
        calculator.connect()

        total_count = calculator.get_transaction_count()
        process_count = min(total_count, limit) if limit else total_count

        print(f"Total transactions: {total_count}")
        print(f"Processing: {process_count}")
        print(f"Batch size: {batch_size}")

        processed = 0
        success = 0
        failed = 0
        start_time = time.time()

        offset = 0
        while processed < process_count:
            # Fetch batch of transactions
            batch_limit = min(batch_size, process_count - processed)
            df = calculator.get_transactions(limit=batch_limit, offset=offset)

            if df.empty:
                break

            records = []

            for _, row in df.iterrows():
                features = {
                    'country': row['country'],
                    'bank': row['bank'],
                    'bin': row['bin'],
                    'sdk_type': row['sdk_type'],
                    'ip_address': row['ip_address'],
                    'card_network': row['card_network'],
                }

                # Compute point-in-time decline rates
                decline_rates = calculator.compute_all_decline_rates(
                    features,
                    row['processed_at']
                )

                # Build feature store record
                record = fs_client.build_record(
                    record_id=row['transaction_id'],
                    features=features,
                    decline_rates=decline_rates,
                    is_fraud=1 if row['is_fraud'] else 0,
                    status=row['status']
                )

                records.append(record)

            # Batch write to Feature Store
            result = fs_client.put_records_batch(records)
            success += result['success']
            failed += result['failed']

            processed += len(df)
            offset += len(df)

            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"Processed {processed}/{process_count} ({rate:.1f}/sec) - Success: {success}, Failed: {failed}")

        elapsed = time.time() - start_time
        print(f"\nSync complete in {elapsed:.1f}s")
        print(f"  Total processed: {processed}")
        print(f"  Success: {success}")
        print(f"  Failed: {failed}")

    finally:
        calculator.close()


def main():
    parser = argparse.ArgumentParser(
        description='Sync PostgreSQL transactions to SageMaker Feature Store'
    )
    parser.add_argument(
        '--db-url',
        required=True,
        help='PostgreSQL connection string (e.g., postgresql://user:pass@host:5432/db)'
    )
    parser.add_argument(
        '--feature-group-name',
        default='fraud-detection-decline-rates',
        help='Name of the feature group'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=100,
        help='Number of records per batch'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Maximum number of transactions to process'
    )
    parser.add_argument(
        '--region',
        default='us-east-1',
        help='AWS region'
    )

    args = parser.parse_args()

    sync_to_feature_store(
        db_url=args.db_url,
        feature_group_name=args.feature_group_name,
        batch_size=args.batch_size,
        limit=args.limit,
        region=args.region
    )


if __name__ == '__main__':
    main()
