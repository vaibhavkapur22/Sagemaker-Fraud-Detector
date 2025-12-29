#!/usr/bin/env python3
"""
Generate training data with decline rate features.

This script either:
1. Generates synthetic data with simulated decline rates (for testing)
2. Computes historical decline rates from PostgreSQL (for production)

The output includes all 19 features:
- 7 original encoded features
- 12 decline rate features (7-day and 14-day for each dimension)

Usage:
    # Generate synthetic data
    python generate_training_data_with_decline_rates.py --synthetic

    # Generate from PostgreSQL
    python generate_training_data_with_decline_rates.py --db-url postgresql://user:pass@host/db
"""

import argparse
import os
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

# Try to import psycopg2 for PostgreSQL support
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False


# Constants
# Note: We use 'ip' instead of 'ip_address' for the decline rate feature name
# to match the model's expected feature names (ip_decline_rate_7d, etc.)
DIMENSIONS = ['country', 'bank', 'bin', 'sdk_type', 'ip_address', 'card_network']
DECLINE_RATE_DIMENSIONS = ['country', 'bank', 'bin', 'sdk_type', 'ip', 'card_network']
WINDOWS = [7, 14]

COUNTRIES = ['US', 'UK', 'CA', 'DE', 'FR', 'IN', 'AU', 'BR', 'JP', 'MX', 'NG', 'RU', 'CN']
BANKS = ['chase', 'bank_of_america', 'wells_fargo', 'citibank', 'capital_one', 'us_bank',
         'pnc_bank', 'td_bank', 'hsbc', 'barclays', 'goldman_sachs', 'morgan_stanley',
         'deutsche_bank', 'credit_suisse', 'santander', 'bnp_paribas']
SDK_TYPES = ['web', 'ios', 'android', 'java', 'python', 'ruby']
CARD_NETWORKS = ['visa', 'mastercard', 'amex', 'discover', 'jcb']

# Risk profiles for synthetic data
HIGH_RISK_COUNTRIES = ['NG', 'RU', 'CN']
MEDIUM_RISK_COUNTRIES = ['BR', 'MX', 'IN']


def generate_bin():
    """Generate a realistic BIN."""
    prefixes = ['4111', '4000', '4242', '5100', '5200', '5300', '3782', '6011', '3528']
    return f"{random.choice(prefixes)}{random.randint(10, 99)}"


def generate_ip():
    """Generate a random IP address."""
    if random.random() < 0.2:
        # Private IP
        patterns = [
            f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}",
            f"192.168.{random.randint(0, 255)}.{random.randint(0, 255)}",
            f"172.{random.randint(16, 31)}.{random.randint(0, 255)}.{random.randint(0, 255)}"
        ]
        return random.choice(patterns)
    else:
        return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"


def calculate_base_decline_rate(dimension: str, value: str) -> float:
    """Calculate base decline rate for a dimension value."""
    if dimension == 'country':
        if value in HIGH_RISK_COUNTRIES:
            return random.uniform(0.25, 0.40)
        elif value in MEDIUM_RISK_COUNTRIES:
            return random.uniform(0.12, 0.20)
        else:
            return random.uniform(0.03, 0.08)

    elif dimension == 'sdk_type':
        if value == 'web':
            return random.uniform(0.10, 0.18)
        elif value in ['ios', 'android']:
            return random.uniform(0.02, 0.06)
        else:
            return random.uniform(0.05, 0.10)

    elif dimension == 'bank':
        if value in ['santander', 'bnp_paribas', 'credit_suisse']:
            return random.uniform(0.08, 0.15)
        else:
            return random.uniform(0.04, 0.10)

    elif dimension == 'card_network':
        if value == 'amex':
            return random.uniform(0.02, 0.05)  # Stricter verification
        else:
            return random.uniform(0.04, 0.10)

    else:
        return random.uniform(0.05, 0.12)


def generate_synthetic_decline_rates(features: Dict[str, str]) -> Dict[str, float]:
    """
    Generate synthetic decline rates for a transaction.

    The rates are correlated with fraud probability - higher risk
    dimensions have higher decline rates.
    """
    rates = {}

    # Map from actual feature keys to decline rate feature name prefixes
    dimension_to_feature_prefix = {
        'country': 'country',
        'bank': 'bank',
        'bin': 'bin',
        'sdk_type': 'sdk_type',
        'ip_address': 'ip',  # Use 'ip' not 'ip_address' to match model expectations
        'card_network': 'card_network',
    }

    for dimension in DIMENSIONS:
        value = features.get(dimension)
        if not value:
            continue

        # Calculate base rate for this dimension
        base_rate = calculate_base_decline_rate(dimension, value)
        feature_prefix = dimension_to_feature_prefix.get(dimension, dimension)

        for days in WINDOWS:
            feature_name = f'{feature_prefix}_decline_rate_{days}d'
            # Add some variance, with 14-day rate slightly more stable
            variance = 0.03 if days == 7 else 0.02
            rate = max(0.0, min(1.0, base_rate + random.uniform(-variance, variance)))
            rates[feature_name] = round(rate, 4)

    return rates


def calculate_fraud_probability(features: Dict[str, str], decline_rates: Dict[str, float]) -> float:
    """Calculate fraud probability based on features and decline rates."""
    prob = 0.05  # Base probability

    # Country risk
    if features.get('country') in HIGH_RISK_COUNTRIES:
        prob += 0.25
    elif features.get('country') in MEDIUM_RISK_COUNTRIES:
        prob += 0.10

    # SDK risk
    if features.get('sdk_type') == 'web':
        prob += 0.08
    elif features.get('sdk_type') in ['ios', 'android']:
        prob -= 0.02

    # Private IP risk
    ip = features.get('ip_address', '')
    if ip.startswith(('10.', '192.168.', '172.')):
        prob += 0.12

    # Decline rate influence - higher decline rates increase fraud probability
    avg_7d_rate = np.mean([v for k, v in decline_rates.items() if '7d' in k])
    avg_14d_rate = np.mean([v for k, v in decline_rates.items() if '14d' in k])

    prob += avg_7d_rate * 0.3
    prob += avg_14d_rate * 0.2

    return min(0.8, max(0.0, prob))


def generate_synthetic_dataset(num_samples: int = 10000, fraud_rate: float = 0.15) -> pd.DataFrame:
    """
    Generate a synthetic dataset with decline rate features.

    Args:
        num_samples: Number of samples to generate
        fraud_rate: Target fraud rate

    Returns:
        DataFrame with all features including decline rates
    """
    print(f"Generating {num_samples} synthetic samples...")

    records = []

    for i in range(num_samples):
        # Generate base features
        features = {
            'country': random.choice(COUNTRIES),
            'bank': random.choice(BANKS),
            'bin': generate_bin(),
            'sdk_type': random.choice(SDK_TYPES),
            'ip_address': generate_ip(),
            'card_network': random.choice(CARD_NETWORKS),
        }

        # Generate decline rates
        decline_rates = generate_synthetic_decline_rates(features)

        # Calculate fraud probability and determine label
        fraud_prob = calculate_fraud_probability(features, decline_rates)
        is_fraud = 1 if random.random() < fraud_prob else 0

        # Combine all features
        record = {**features, **decline_rates, 'is_fraud': is_fraud}
        records.append(record)

        if (i + 1) % 2000 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")

    df = pd.DataFrame(records)

    # Print statistics
    actual_fraud_rate = df['is_fraud'].mean()
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(df)}")
    print(f"  Fraud samples: {df['is_fraud'].sum()}")
    print(f"  Fraud rate: {actual_fraud_rate:.2%}")

    return df


def compute_decline_rates_from_postgres(db_url: str, limit: Optional[int] = None) -> pd.DataFrame:
    """
    Compute decline rates from PostgreSQL transaction history.

    Args:
        db_url: PostgreSQL connection string
        limit: Maximum number of transactions to process

    Returns:
        DataFrame with all features including computed decline rates
    """
    if not HAS_PSYCOPG2:
        raise ImportError("psycopg2 is required for PostgreSQL support. Install with: pip install psycopg2-binary")

    print("Connecting to PostgreSQL...")
    conn = psycopg2.connect(db_url)

    try:
        # Fetch transactions
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

        df = pd.read_sql(query, conn)
        print(f"Loaded {len(df)} transactions")

        # Compute decline rates for each transaction
        print("Computing decline rates...")

        # Map from dimension to feature name prefix
        dimension_to_feature_prefix = {
            'country': 'country',
            'bank': 'bank',
            'bin': 'bin',
            'sdk_type': 'sdk_type',
            'ip_address': 'ip',  # Use 'ip' not 'ip_address'
            'card_network': 'card_network',
        }

        for dimension in DIMENSIONS:
            feature_prefix = dimension_to_feature_prefix.get(dimension, dimension)
            for days in WINDOWS:
                feature_name = f'{feature_prefix}_decline_rate_{days}d'
                print(f"  Computing {feature_name}...")

                rates = []
                for _, row in df.iterrows():
                    rate = compute_single_decline_rate(
                        conn, dimension, row[dimension],
                        row['processed_at'], days
                    )
                    rates.append(rate)

                df[feature_name] = rates

        # Convert is_fraud to int
        df['is_fraud'] = df['is_fraud'].astype(int)

        return df

    finally:
        conn.close()


def compute_single_decline_rate(conn, dimension: str, value: str, before_time: datetime, days: int) -> float:
    """Compute a single decline rate from the database."""
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

    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(query, (value, start_time, before_time))
        result = cur.fetchone()

    total = result['total'] or 0
    declined = result['declined'] or 0

    if total == 0:
        return 0.0

    return round(declined / total, 4)


def save_dataset(df: pd.DataFrame, output_dir: str, train_ratio: float = 0.8):
    """
    Save the dataset as train/test CSV files.

    Args:
        df: DataFrame with all features
        output_dir: Directory to save files
        train_ratio: Ratio of data to use for training
    """
    os.makedirs(output_dir, exist_ok=True)

    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)

    train_df = df[:split_idx]
    test_df = df[split_idx:]

    # Save files
    train_path = os.path.join(output_dir, 'train_with_decline_rates.csv')
    test_path = os.path.join(output_dir, 'test_with_decline_rates.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved datasets:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Test: {test_path} ({len(test_df)} samples)")

    # Print feature columns
    print(f"\nFeature columns ({len(df.columns)}):")
    for col in df.columns:
        print(f"  - {col}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate training data with decline rate features'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Generate synthetic data (default if no --db-url)'
    )
    parser.add_argument(
        '--db-url',
        help='PostgreSQL connection string for computing from real data'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=10000,
        help='Number of synthetic samples to generate'
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Output directory (defaults to ./data)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        help='Limit number of transactions from database'
    )

    args = parser.parse_args()

    # Determine output directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, 'data')

    if args.db_url:
        # Generate from PostgreSQL
        df = compute_decline_rates_from_postgres(args.db_url, limit=args.limit)
    else:
        # Generate synthetic data
        df = generate_synthetic_dataset(num_samples=args.num_samples)

    save_dataset(df, output_dir)


if __name__ == '__main__':
    main()
