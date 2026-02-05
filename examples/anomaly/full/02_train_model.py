#!/usr/bin/env python3
"""
Step 2: Train Anomaly Detection Model

Loads the generated training data and trains an anomaly detection model.
Demonstrates:
- Training with different backends (ECOD, Isolation Forest, HBOS)
- Saving and loading models
- Listing available models

Prerequisites:
- Run 01_generate_training_data.py first
- LlamaFarm servers running (nx start universal-runtime && nx start server)
"""

import json
import asyncio
import os
import httpx
from pathlib import Path

# Configuration - uses environment variable or .env file, falls back to default
def get_llamafarm_url():
    """Get LlamaFarm server URL from environment or .env file."""
    # Check environment variable first
    if url := os.environ.get("LLAMAFARM_URL"):
        return url.rstrip("/") + "/v1/ml"

    # Try to load from .env file
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line.startswith("LLAMAFARM_URL="):
                    url = line.split("=", 1)[1].strip().strip('"\'')
                    return url.rstrip("/") + "/v1/ml"

    # Default to standard LlamaFarm server port
    return "http://localhost:8000/v1/ml"

LLAMAFARM_URL = get_llamafarm_url()
MODEL_NAME = "factory-sensors"
BACKEND = "ecod"  # Try: isolation_forest, hbos, loda, autoencoder


async def train_model():
    """Train anomaly detection model on generated data."""
    print("=" * 60)
    print("Step 2: Train Anomaly Detection Model")
    print("=" * 60)
    print()

    # Load training data
    data_file = Path(__file__).parent / "training_data.json"
    if not data_file.exists():
        print("❌ Error: training_data.json not found!")
        print("   Run 01_generate_training_data.py first.")
        return

    with open(data_file) as f:
        training_data = json.load(f)

    print(f"Loaded {len(training_data)} samples from {data_file}")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Step 1: Train the model
        print(f"Training model '{MODEL_NAME}' with backend '{BACKEND}'...")
        print()

        response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/fit",
            json={
                "model": MODEL_NAME,
                "backend": BACKEND,
                "data": training_data,
                "schema": {
                    "temperature": "numeric",
                    "humidity": "numeric",
                    "pressure": "numeric",
                    "motor_rpm": "numeric"
                },
                "contamination": 0.05,  # Expect 5% anomalies
                "normalization": "standardization",
                "overwrite": True,  # Overwrite if exists
                "description": "Factory sensor anomaly detector"
            }
        )

        if response.status_code != 200:
            print(f"❌ Training failed: {response.text}")
            return

        result = response.json()
        print("✅ Training complete!")
        print(f"   Model: {result['model']}")
        print(f"   Backend: {result['backend']}")
        print(f"   Samples fitted: {result['samples_fitted']}")
        print(f"   Training time: {result['training_time_ms']:.2f}ms")
        print()

        # Step 2: Save the model
        print("Saving model to disk...")
        save_response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/save",
            json={
                "model": MODEL_NAME,
                "backend": BACKEND,
                "normalization": "standardization",
                "description": "Factory sensor anomaly detector"
            }
        )

        if save_response.status_code == 200:
            save_result = save_response.json()
            print(f"✅ Model saved!")
            print(f"   Path: {save_result.get('path', 'N/A')}")
        else:
            print(f"⚠️  Save failed: {save_response.text}")
        print()

        # Step 3: List available models
        print("Listing saved models...")
        models_response = await client.get(f"{LLAMAFARM_URL}/anomaly/models")

        if models_response.status_code == 200:
            models = models_response.json()
            print(f"Found {models['total']} saved models:")
            for model in models.get("data", []):
                print(f"   - {model['filename']} ({model['size_bytes']} bytes)")
        print()

        # Step 4: Test loading the model
        print("Testing model load...")
        load_response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/load",
            json={
                "model": MODEL_NAME,
                "backend": BACKEND
            }
        )

        if load_response.status_code == 200:
            load_result = load_response.json()
            print(f"✅ Model loaded successfully!")
            print(f"   Status: {load_result['status']}")
        else:
            print(f"⚠️  Load failed: {load_response.text}")
        print()

        # Step 5: Quick test with a few samples
        print("Testing model with sample data...")
        test_data = [
            {"temperature": 72.5, "humidity": 45.0, "pressure": 1013, "motor_rpm": 3000},  # Normal
            {"temperature": 150.0, "humidity": 10.0, "pressure": 900, "motor_rpm": 5000},  # Anomaly
            {"temperature": 73.1, "humidity": 46.2, "pressure": 1015, "motor_rpm": 2980},  # Normal
        ]

        score_response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/score",
            json={
                "model": MODEL_NAME,
                "backend": BACKEND,
                "data": test_data,
                "schema": {
                    "temperature": "numeric",
                    "humidity": "numeric",
                    "pressure": "numeric",
                    "motor_rpm": "numeric"
                },
                "normalization": "standardization"
            }
        )

        if score_response.status_code == 200:
            scores = score_response.json()
            print("Test results:")
            for i, (data, score) in enumerate(zip(test_data, scores["data"])):
                status = "🚨 ANOMALY" if score["is_anomaly"] else "✅ NORMAL"
                print(f"   {i+1}. temp={data['temperature']}, rpm={data['motor_rpm']} -> "
                      f"score={score['score']:.3f} {status}")
        else:
            print(f"⚠️  Scoring failed: {score_response.text}")

        print()
        print("Model training complete! Run 03_streaming_detection.py next.")


def main():
    asyncio.run(train_model())


if __name__ == "__main__":
    main()
