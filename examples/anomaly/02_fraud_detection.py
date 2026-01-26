#!/usr/bin/env python3
"""Fraud Detection: Real-World Transaction Monitoring

This example shows how to build a fraud detection system using
LlamaFarm's anomaly detection. Demonstrates:

1. Training on historical normal transactions
2. Saving the model for production use
3. Loading and scoring new transactions in real-time
4. Using different backends (ECOD for speed, Isolation Forest for accuracy)

Requirements:
    - Universal Runtime running: nx start universal-runtime

Run:
    cd /path/to/llamafarm
    uv run python examples/anomaly/02_fraud_detection.py
"""

import os
import random
import httpx
from pathlib import Path

# Configuration - uses environment variable or .env file, falls back to default
def get_llamafarm_url():
    """Get LlamaFarm server URL from environment or .env file."""
    if url := os.environ.get("LLAMAFARM_URL"):
        return url.rstrip("/")
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("LLAMAFARM_URL="):
                    return line.split("=", 1)[1].strip().strip('"\'').rstrip("/")
    return "http://localhost:8000"

BASE_URL = get_llamafarm_url()
client = httpx.Client(timeout=60)


def generate_normal_transactions(n: int) -> list[list[float]]:
    """Generate synthetic normal transaction data."""
    transactions = []
    for _ in range(n):
        # Normal transactions: amount $10-$500, time 8am-10pm
        amount = random.gauss(150, 80)  # Mean $150, std $80
        hour = random.gauss(14, 4)  # Peak at 2pm
        merchant_risk = random.uniform(0, 0.3)  # Low risk merchants

        transactions.append([
            max(10, min(500, amount)),  # Clamp to reasonable range
            max(0, min(23, hour)),
            merchant_risk,
        ])
    return transactions


def generate_fraudulent_transactions(n: int) -> list[list[float]]:
    """Generate synthetic fraudulent transaction patterns."""
    frauds = []
    for _ in range(n):
        fraud_type = random.choice(["high_amount", "odd_hour", "high_risk"])

        if fraud_type == "high_amount":
            # Unusually large transaction
            amount = random.uniform(2000, 10000)
            hour = random.gauss(14, 4)
            merchant_risk = random.uniform(0, 0.3)
        elif fraud_type == "odd_hour":
            # Transaction at unusual time
            amount = random.gauss(150, 80)
            hour = random.uniform(1, 5)  # 1am-5am
            merchant_risk = random.uniform(0, 0.3)
        else:
            # High risk merchant
            amount = random.gauss(150, 80)
            hour = random.gauss(14, 4)
            merchant_risk = random.uniform(0.7, 1.0)

        frauds.append([max(10, amount), max(0, min(23, hour)), merchant_risk])
    return frauds


def main():
    print("=" * 60)
    print("FRAUD DETECTION EXAMPLE")
    print("Using LlamaFarm Anomaly Detection")
    print("=" * 60)
    print()

    # Step 1: Generate training data (normal transactions only)
    print("Step 1: Generating 500 normal transactions for training...")
    random.seed(42)
    training_data = generate_normal_transactions(500)
    print(f"  Features: [amount, hour_of_day, merchant_risk_score]")
    print(f"  Sample: {training_data[0]}")
    print()

    # Step 2: Train the fraud detector using ECOD (fast, parameter-free)
    print("Step 2: Training fraud detector with ECOD backend...")
    response = client.post(
        f"{BASE_URL}/v1/ml/anomaly/fit",
        json={
            "model": "fraud-detector",
            "backend": "ecod",
            "data": training_data,
            "contamination": 0.05,  # Expect 5% fraud rate
        },
    )
    fit_result = response.json()
    print(f"  Trained on {fit_result['samples_fitted']} samples")
    print(f"  Training time: {fit_result['training_time_ms']:.1f}ms")
    print(f"  Model saved to: {fit_result['saved_path']}")
    print()

    # Step 3: Generate test data (mix of normal and fraudulent)
    print("Step 3: Generating test transactions...")
    test_normal = generate_normal_transactions(45)
    test_fraud = generate_fraudulent_transactions(5)
    test_data = test_normal + test_fraud
    random.shuffle(test_data)

    # Remember which are actual frauds (last 5 before shuffle, but we shuffled)
    # For demo, we'll just look at detection results
    print(f"  Generated 45 normal + 5 fraudulent transactions")
    print()

    # Step 4: Score transactions in real-time
    print("Step 4: Scoring transactions for fraud...")
    response = client.post(
        f"{BASE_URL}/v1/ml/anomaly/score",
        json={
            "model": "fraud-detector",
            "backend": "ecod",
            "data": test_data,
        },
    )
    score_result = response.json()

    # Show flagged transactions
    flagged = [d for d in score_result["data"] if d["is_anomaly"]]
    print(f"\n  FLAGGED TRANSACTIONS ({len(flagged)} detected):")
    print("  " + "-" * 56)
    for item in flagged[:5]:  # Show first 5
        tx = test_data[item["index"]]
        print(f"  Index {item['index']:3d}: "
              f"Amount=${tx[0]:,.2f}, Hour={tx[1]:.1f}, "
              f"Risk={tx[2]:.2f} | Score: {item['score']:.3f}")

    print()
    print(f"Summary: Detected {score_result['summary']['anomaly_count']} anomalies "
          f"({score_result['summary']['anomaly_rate']*100:.1f}% of transactions)")
    print()

    # Step 5: Compare with Isolation Forest
    print("Step 5: Comparing with Isolation Forest backend...")

    # Train Isolation Forest model
    response = client.post(
        f"{BASE_URL}/v1/ml/anomaly/fit",
        json={
            "model": "fraud-detector-iforest",
            "backend": "isolation_forest",
            "data": training_data,
            "contamination": 0.05,
            "n_estimators": 100,
        },
    )
    iforest_fit = response.json()

    # Score with Isolation Forest
    response = client.post(
        f"{BASE_URL}/v1/ml/anomaly/score",
        json={
            "model": "fraud-detector-iforest",
            "backend": "isolation_forest",
            "data": test_data,
        },
    )
    iforest_result = response.json()

    print(f"\n  Backend Comparison:")
    print(f"  {'Backend':<20} {'Anomalies':<12} {'Rate':<10} {'Train Time'}")
    print("  " + "-" * 56)
    print(f"  {'ECOD':<20} {score_result['summary']['anomaly_count']:<12} "
          f"{score_result['summary']['anomaly_rate']*100:.1f}%      "
          f"{fit_result['training_time_ms']:.1f}ms")
    print(f"  {'Isolation Forest':<20} {iforest_result['summary']['anomaly_count']:<12} "
          f"{iforest_result['summary']['anomaly_rate']*100:.1f}%      "
          f"{iforest_fit['training_time_ms']:.1f}ms")
    print()

    # Step 6: Production usage - load saved model
    print("Step 6: Loading saved model for production...")
    response = client.post(
        f"{BASE_URL}/v1/ml/anomaly/load",
        json={
            "model": "fraud-detector",
            "backend": "ecod",
        },
    )
    load_result = response.json()
    print(f"  Loaded: {load_result['filename']}")
    print(f"  Threshold: {load_result['threshold']:.4f}")
    print()

    print("=" * 60)
    print("DONE! The fraud detector is ready for production use.")
    print("=" * 60)


if __name__ == "__main__":
    main()
