#!/usr/bin/env python3
"""
Generate synthetic fraud detection training data.
Creates a CSV file with labeled transactions for training the XGBoost model.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Feature value pools
COUNTRIES = ['US', 'UK', 'CA', 'DE', 'FR', 'IN', 'AU', 'BR', 'JP', 'MX', 'NG', 'RU', 'CN']
HIGH_RISK_COUNTRIES = ['NG', 'RU', 'CN']
MEDIUM_RISK_COUNTRIES = ['BR', 'MX', 'IN']

SDK_TYPES = ['web', 'ios', 'android', 'java', 'python', 'ruby']
CARD_NETWORKS = ['visa', 'mastercard', 'amex', 'discover', 'jcb']

BANKS = [
    'chase', 'bank_of_america', 'wells_fargo', 'citibank', 'capital_one',
    'us_bank', 'pnc_bank', 'td_bank', 'hsbc', 'barclays',
    'goldman_sachs', 'morgan_stanley', 'deutsche_bank', 'credit_suisse',
    'santander', 'bnp_paribas'
]

# BIN ranges (first 6 digits)
VISA_BINS = ['411111', '422222', '433333', '444444', '455555']
MASTERCARD_BINS = ['512345', '523456', '534567', '545678', '556789']
AMEX_BINS = ['371234', '372345', '373456', '374567']
DISCOVER_BINS = ['601111', '602222', '603333']
UNKNOWN_BINS = ['400000', '500000', '600000']  # Higher risk

def generate_ip_address(is_fraud=False):
    """Generate an IP address. Fraudulent transactions may have suspicious patterns."""
    if is_fraud and random.random() < 0.3:
        # VPN/Proxy-like IPs (certain ranges)
        return f"10.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
    return f"{random.randint(1, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"

def get_bin_for_network(network):
    """Get a BIN number for the given card network."""
    bin_map = {
        'visa': VISA_BINS,
        'mastercard': MASTERCARD_BINS,
        'amex': AMEX_BINS,
        'discover': DISCOVER_BINS,
        'jcb': ['356789', '357890']
    }
    return random.choice(bin_map.get(network, UNKNOWN_BINS))

def calculate_fraud_probability(country, sdk_type, card_network, bank, bin_number):
    """
    Calculate base fraud probability based on features.
    This simulates the underlying pattern the model will learn.
    """
    prob = 0.05  # Base fraud rate of 5%

    # Country risk
    if country in HIGH_RISK_COUNTRIES:
        prob += 0.25
    elif country in MEDIUM_RISK_COUNTRIES:
        prob += 0.10

    # SDK type risk
    if sdk_type == 'web':
        prob += 0.08
    elif sdk_type in ['ios', 'android']:
        prob -= 0.02  # Native apps are slightly safer
    elif sdk_type in ['java', 'python', 'ruby']:
        prob += 0.03  # Server-side SDKs slightly higher risk

    # Card network
    if card_network == 'amex':
        prob -= 0.03  # Amex has stricter verification

    # Bank risk - no unknown banks in our list now, but keep pattern for edge cases
    if bank in ['santander', 'bnp_paribas']:
        prob += 0.02  # European banks slightly different risk profile

    # BIN risk (unknown/test BINs)
    if bin_number.startswith('40000') or bin_number.startswith('50000'):
        prob += 0.15

    return min(max(prob, 0.01), 0.95)  # Clamp between 1% and 95%

def generate_transaction(is_fraud=None):
    """Generate a single transaction record."""
    country = random.choice(COUNTRIES)
    sdk_type = random.choice(SDK_TYPES)
    card_network = random.choice(CARD_NETWORKS)
    bank = random.choice(BANKS)
    bin_number = get_bin_for_network(card_network)
    ip_address = generate_ip_address(is_fraud if is_fraud is not None else False)

    # Calculate fraud probability if not predetermined
    if is_fraud is None:
        fraud_prob = calculate_fraud_probability(country, sdk_type, card_network, bank, bin_number)
        is_fraud = random.random() < fraud_prob

    return {
        'country': country,
        'bank': bank,
        'bin': bin_number,
        'sdk_type': sdk_type,
        'ip_address': ip_address,
        'card_network': card_network,
        'is_fraud': int(is_fraud)
    }

def generate_dataset(n_samples=10000, fraud_ratio=0.15):
    """
    Generate a balanced dataset with specified fraud ratio.
    """
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    print(f"Generating {n_samples} samples ({n_fraud} fraud, {n_legit} legitimate)...")

    transactions = []

    # Generate fraudulent transactions
    for _ in range(n_fraud):
        transactions.append(generate_transaction(is_fraud=True))

    # Generate legitimate transactions
    for _ in range(n_legit):
        transactions.append(generate_transaction(is_fraud=False))

    # Shuffle the dataset
    random.shuffle(transactions)

    return pd.DataFrame(transactions)

def main():
    # Generate training data
    print("=" * 50)
    print("Fraud Detection Sample Data Generator")
    print("=" * 50)

    # Generate datasets
    train_df = generate_dataset(n_samples=10000, fraud_ratio=0.15)
    test_df = generate_dataset(n_samples=2000, fraud_ratio=0.15)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(output_dir, exist_ok=True)

    # Save to CSV
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nTraining data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")

    # Print sample statistics
    print("\n" + "=" * 50)
    print("Dataset Statistics")
    print("=" * 50)
    print(f"\nTraining set:")
    print(f"  Total samples: {len(train_df)}")
    print(f"  Fraud cases: {train_df['is_fraud'].sum()} ({train_df['is_fraud'].mean()*100:.1f}%)")

    print(f"\nTest set:")
    print(f"  Total samples: {len(test_df)}")
    print(f"  Fraud cases: {test_df['is_fraud'].sum()} ({test_df['is_fraud'].mean()*100:.1f}%)")

    print("\nFeature value distributions (training set):")
    print(f"  Countries: {train_df['country'].nunique()} unique")
    print(f"  Banks: {train_df['bank'].nunique()} unique")
    print(f"  SDK Types: {train_df['sdk_type'].nunique()} unique")
    print(f"  Card Networks: {train_df['card_network'].nunique()} unique")

    print("\nSample records:")
    print(train_df.head(5).to_string())

if __name__ == '__main__':
    main()
