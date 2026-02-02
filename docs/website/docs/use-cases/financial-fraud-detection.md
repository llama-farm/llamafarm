---
title: "Financial Fraud Detection"
sidebar_position: 3
---

# Financial Fraud Detection with Anomaly Detection

This use case demonstrates how to build a real-time fraud detection system for financial transactions using LlamaFarm's anomaly detection APIs.

## Scenario

You're building a fraud detection system for a payment processor that handles:
- Thousands of transactions per second
- Multiple data types (amounts, locations, device fingerprints)
- Evolving fraud patterns
- Real-time alerting requirements

## Architecture

```
┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Transaction  │───▶│  LlamaFarm  │───▶│   Fraud      │
│   Stream     │    │  Streaming  │    │   Review     │
└──────────────┘    │  + Batch    │    │   Queue      │
                    └─────────────┘    └──────────────┘
                          │
                    ┌─────┴─────┐
                    │  Trained  │
                    │  Models   │
                    │  (daily)  │
                    └───────────┘
```

## Backend Selection for Fraud Detection

| Backend | Use Case | Strengths |
|---------|----------|-----------|
| `isolation_forest` | **Primary choice** - complex transaction patterns | Handles high-dimensional data, robust |
| `suod` | Maximum accuracy (ensemble) | Combines multiple algorithms |
| `autoencoder` | Deep pattern learning | Best for complex non-linear fraud patterns |
| `ecod` | Real-time streaming | Fastest, good for pre-screening |

**For this use case, we'll use a hybrid approach:**
1. **Streaming with `ecod`** for real-time pre-screening
2. **Batch with `isolation_forest`** for daily model training
3. **`suod` ensemble** for high-risk transaction review

## Step 1: Feature Engineering with Polars

Fraud detection requires rich features. Use Polars buffers for per-user behavioral features:

```python
import httpx
import asyncio

LLAMAFARM_URL = "http://localhost:14345/v1/ml"

async def setup_user_buffer(user_id: str):
    """Create a per-user behavior buffer."""
    async with httpx.AsyncClient() as client:
        await client.post(f"{LLAMAFARM_URL}/polars/buffers", json={
            "buffer_id": f"user-{user_id}",
            "window_size": 1000  # Keep last 1000 transactions
        })


async def add_transaction_features(user_id: str, transaction: dict):
    """Add transaction and compute behavioral features."""
    async with httpx.AsyncClient() as client:
        # Append transaction
        await client.post(f"{LLAMAFARM_URL}/polars/append", json={
            "buffer_id": f"user-{user_id}",
            "data": transaction
        })

        # Get behavioral features
        response = await client.post(f"{LLAMAFARM_URL}/polars/features", json={
            "buffer_id": f"user-{user_id}",
            "rolling_windows": [5, 10, 50],  # Last 5, 10, 50 transactions
            "include_rolling_stats": ["mean", "std", "min", "max"],
            "include_lags": True,
            "lag_periods": [1, 5, 10],
            "tail": 1  # Just the latest with features
        })

        if response.json()["rows"] > 0:
            return response.json()["data"][0]
        return transaction  # Not enough history yet
```

## Step 2: Real-Time Streaming Detection

For immediate fraud screening:

```python
async def screen_transaction(enhanced_transaction: dict):
    """Fast pre-screening with streaming detector."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/stream",
            json={
                "model": "fraud-realtime",
                "data": enhanced_transaction,
                "backend": "ecod",  # Fast pre-screening
                "min_samples": 100,
                "retrain_interval": 1000,
                "window_size": 10000,
                "threshold": 0.4,  # Lower threshold = catch more
                "contamination": 0.01  # Expect 1% fraud
            }
        )
        result = response.json()

        if result["status"] != "collecting":
            for r in result["results"]:
                if r["is_anomaly"]:
                    return {
                        "action": "review",
                        "score": r["score"],
                        "reason": "realtime_anomaly"
                    }
            # Use last result's score if available
            if result["results"]:
                return {"action": "approve", "score": result["results"][-1].get("score", 0)}

        return {"action": "approve", "score": 0}
```

## Step 3: Batch Model Training (Daily)

Train more accurate models on historical data:

```python
async def train_daily_model(training_data: list[dict]):
    """Train daily fraud detection model on historical data."""
    async with httpx.AsyncClient() as client:
        # Define schema for mixed data types
        schema = {
            "amount": "numeric",
            "amount_rolling_mean_10": "numeric",
            "amount_rolling_std_10": "numeric",
            "merchant_id": "hash",      # High cardinality
            "device_type": "label",     # Low cardinality
            "country": "label",
            "hour_of_day": "numeric",
            "is_weekend": "binary",
            "days_since_last_tx": "numeric"
        }

        # Train with isolation_forest for complex patterns
        response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/fit",
            json={
                "model": "fraud-daily",
                "backend": "isolation_forest",
                "data": training_data,
                "schema": schema,
                "contamination": 0.01,  # 1% expected fraud rate
                "normalization": "standardization",
                "overwrite": False  # Version models
            }
        )

        result = response.json()
        print(f"Trained model: {result['model']}")
        print(f"Samples: {result['samples_fitted']}")

        # Save for production use
        await client.post(f"{LLAMAFARM_URL}/anomaly/save", json={
            "model": result['model'],
            "backend": "isolation_forest"
        })

        return result['model']
```

## Step 4: Ensemble Scoring for High-Risk

For transactions flagged by real-time screening:

```python
async def deep_fraud_analysis(transaction: dict) -> dict:
    """Run transaction through ensemble model for high accuracy."""
    async with httpx.AsyncClient() as client:
        # Score with ensemble backend
        response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/score",
            json={
                "model": "fraud-ensemble",
                "backend": "suod",  # Ensemble of multiple algorithms
                "data": [transaction],
                "normalization": "standardization"
            }
        )
        result = response.json()

        if result["data"]:
            score = result["data"][0]
            return {
                "is_fraud": score["is_anomaly"],
                "confidence": score["score"],
                "raw_score": score["raw_score"],
                "recommendation": (
                    "block" if score["score"] > 0.8 else
                    "review" if score["score"] > 0.5 else
                    "approve"
                )
            }

        return {"is_fraud": False, "confidence": 0}
```

## Complete Pipeline

```python
async def process_transaction(transaction: dict):
    """Complete fraud detection pipeline."""

    user_id = transaction["user_id"]

    # 1. Add to user's behavior buffer
    enhanced = await add_transaction_features(user_id, {
        "amount": transaction["amount"],
        "merchant_id": transaction["merchant_id"],
        "device_type": transaction["device_type"],
        "country": transaction["country"],
        "timestamp": transaction["timestamp"]
    })

    # 2. Real-time screening (sub-10ms)
    screening = await screen_transaction(enhanced)

    if screening["action"] == "approve":
        return {"decision": "approve", "latency_ms": 10}

    # 3. Deep analysis for flagged transactions
    deep = await deep_fraud_analysis(enhanced)

    if deep["recommendation"] == "block":
        return {
            "decision": "block",
            "reason": "high_fraud_confidence",
            "score": deep["confidence"]
        }

    return {
        "decision": "review",
        "reason": "flagged_for_review",
        "score": deep["confidence"]
    }
```

## Feature Engineering Patterns

### Velocity Features

Detect rapid transaction sequences:

```python
# With Polars rolling features
{
    "rolling_windows": [3, 5, 10],  # Last 3, 5, 10 transactions
    "include_rolling_stats": ["mean", "std", "max"],
}

# Produces:
# - amount_rolling_mean_3 (avg of last 3)
# - amount_rolling_std_5 (volatility of last 5)
# - amount_rolling_max_10 (max in last 10)
```

### Time-Based Features

Detect unusual timing:

```python
def extract_time_features(timestamp):
    """Extract time-based features."""
    dt = datetime.fromisoformat(timestamp)
    return {
        "hour_of_day": dt.hour,
        "day_of_week": dt.weekday(),
        "is_weekend": 1 if dt.weekday() >= 5 else 0,
        "is_night": 1 if dt.hour < 6 or dt.hour > 22 else 0
    }
```

### Geographic Features

Detect impossible travel:

```python
def geographic_features(current_country, last_country, minutes_between):
    """Detect geographic impossibilities."""
    return {
        "country_changed": 1 if current_country != last_country else 0,
        "minutes_since_last": minutes_between,
        # If country changed in < 120 minutes, suspicious
        "impossible_travel": 1 if (
            current_country != last_country and
            minutes_between < 120
        ) else 0
    }
```

## Schema Design

```python
# Complete fraud detection schema
FRAUD_SCHEMA = {
    # Transaction attributes
    "amount": "numeric",
    "merchant_id": "hash",         # Millions of merchants
    "merchant_category": "label",  # ~50 categories
    "device_type": "label",        # mobile, desktop, tablet
    "device_fingerprint": "hash",  # High cardinality

    # Geographic
    "country": "label",
    "city": "hash",
    "ip_prefix": "hash",

    # Time
    "hour_of_day": "numeric",
    "is_weekend": "binary",
    "is_night": "binary",

    # Velocity (from Polars)
    "amount_rolling_mean_5": "numeric",
    "amount_rolling_std_5": "numeric",
    "amount_rolling_max_10": "numeric",
    "tx_count_rolling_mean_5": "numeric",

    # Lags
    "amount_lag_1": "numeric",
    "minutes_since_last": "numeric",

    # Derived
    "amount_vs_avg": "numeric",  # Current / user's average
    "impossible_travel": "binary"
}
```

## Performance Optimization

### Batch Processing

For bulk scoring (end-of-day review):

```python
async def batch_score(transactions: list[dict]):
    """Score many transactions at once."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/score",
            json={
                "model": "fraud-daily-latest",
                "backend": "isolation_forest",
                "data": transactions,
                "normalization": "standardization"
            }
        )
        return response.json()["data"]

# Score 10,000 transactions at once
results = await batch_score(pending_transactions)
```

### Caching User Buffers

```python
# Instead of creating buffer on every transaction
# Pre-create buffers for active users

async def warmup_user_buffers(active_user_ids: list[str]):
    """Pre-create buffers for expected active users."""
    async with httpx.AsyncClient() as client:
        for user_id in active_user_ids:
            await client.post(f"{LLAMAFARM_URL}/polars/buffers", json={
                "buffer_id": f"user-{user_id}",
                "window_size": 500
            })
```

## Monitoring and Tuning

### Track Model Performance

```python
async def log_decision(transaction_id: str, decision: dict, actual_fraud: bool = None):
    """Log decisions for model tuning."""
    log = {
        "transaction_id": transaction_id,
        "decision": decision["decision"],
        "score": decision.get("score"),
        "timestamp": datetime.now().isoformat()
    }

    if actual_fraud is not None:
        # After manual review or chargeback
        log["actual_fraud"] = actual_fraud
        log["correct"] = (
            (decision["decision"] == "block" and actual_fraud) or
            (decision["decision"] == "approve" and not actual_fraud)
        )

    # Store for analysis
    await store_decision_log(log)
```

### Tune Thresholds

```python
# Based on your business requirements:

# High security (catch more fraud, more false positives)
threshold = 0.3
contamination = 0.02

# Balanced
threshold = 0.5
contamination = 0.01

# Customer-friendly (fewer false positives, might miss some fraud)
threshold = 0.7
contamination = 0.005
```

## Complete Example

```python
#!/usr/bin/env python3
"""Complete fraud detection example."""

import asyncio
import httpx
import random
from datetime import datetime, timedelta

LLAMAFARM_URL = "http://localhost:14345/v1/ml"

async def simulate_transaction() -> dict:
    """Generate realistic transaction data."""
    # Normal transaction
    tx = {
        "amount": random.lognormvariate(3, 1),  # Log-normal amounts
        "merchant_category": random.choice([
            "grocery", "gas", "restaurant", "retail", "online"
        ]),
        "device_type": random.choice(["mobile", "desktop", "tablet"]),
        "hour": datetime.now().hour,
        "is_weekend": 1 if datetime.now().weekday() >= 5 else 0
    }

    # Occasionally simulate fraud patterns
    if random.random() < 0.02:  # 2% fraud rate
        tx["amount"] = random.uniform(1000, 5000)  # Unusually large
        tx["merchant_category"] = "electronics"  # High-risk category
        tx["_is_fraud"] = True
    else:
        tx["_is_fraud"] = False

    return tx


async def main():
    """Run fraud detection demo."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        print("💳 Starting Fraud Detection Demo")
        print("-" * 50)

        tx_count = 0
        flagged_count = 0
        fraud_count = 0

        while tx_count < 200:
            tx = await simulate_transaction()
            is_actual_fraud = tx.pop("_is_fraud")

            # Stream to detector
            response = await client.post(
                f"{LLAMAFARM_URL}/anomaly/stream",
                json={
                    "model": "fraud-demo",
                    "data": tx,
                    "backend": "ecod",
                    "min_samples": 50,
                    "retrain_interval": 100,
                    "window_size": 500,
                    "threshold": 0.5,
                    "contamination": 0.02
                }
            )
            result = response.json()
            tx_count += 1

            if is_actual_fraud:
                fraud_count += 1

            if result["status"] == "collecting":
                if tx_count % 10 == 0:
                    print(f"⏳ Collecting: {result['samples_until_ready']} to go")
            else:
                for r in result["results"]:
                    if r["is_anomaly"]:
                        flagged_count += 1
                        status = "✅ TRUE POSITIVE" if is_actual_fraud else "⚠️  FALSE POSITIVE"
                        print(f"🚨 Flagged tx #{tx_count}: ${tx['amount']:.2f} "
                              f"@ {tx['merchant_category']} | score={r['score']:.3f} | {status}")
                    elif is_actual_fraud:
                        print(f"❌ MISSED FRAUD: ${tx['amount']:.2f} @ {tx['merchant_category']}")

            await asyncio.sleep(0.05)  # 20 TPS

        print("-" * 50)
        print(f"Summary: {tx_count} transactions, "
              f"{fraud_count} actual fraud, "
              f"{flagged_count} flagged")


if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

- [Streaming Anomaly Detection](../models/streaming-anomaly-detection.md) - API reference
- [Polars Buffer API](../models/polars-buffers.md) - Feature engineering
- [IoT Sensor Monitoring](./iot-sensor-monitoring.md) - Another use case
