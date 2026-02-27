#!/usr/bin/env python3
"""
Step 5: SHAP Explainability Demo

Demonstrates SHAP (SHapley Additive exPlanations) for anomaly detection:
- Batch anomaly scoring with feature explanations
- Streaming anomaly detection with real-time explanations
- Understanding WHY a data point is flagged as anomalous

SHAP provides interpretable machine learning by showing:
- Which features contributed most to the anomaly score
- Whether each feature increased or decreased the score
- A human-readable summary of the anomaly

Prerequisites:
- LlamaFarm servers running (nx start universal-runtime && nx start server)
"""

import asyncio
import random

import httpx

# Configuration
LLAMAFARM_URL = "http://localhost:14345/v1/ml"


def generate_normal_transaction():
    """Generate a normal transaction."""
    return {
        "amount": round(random.uniform(10, 200), 2),
        "frequency": round(random.uniform(1, 10), 2),
        "time_since_last": round(random.uniform(1, 100), 2),
    }


def generate_anomalous_transaction():
    """Generate an anomalous transaction."""
    return {
        "amount": round(random.uniform(5000, 15000), 2),  # Very high amount
        "frequency": round(random.uniform(50, 100), 2),  # Very high frequency
        "time_since_last": round(random.uniform(0.01, 0.5), 2),  # Very short time
    }


async def demo_batch_shap():
    """Demonstrate batch anomaly detection with SHAP explanations."""
    print("=" * 60)
    print("Part 1: Batch Anomaly Detection with SHAP")
    print("=" * 60)
    print()
    print("SHAP (SHapley Additive exPlanations) breaks down each anomaly score")
    print("to show which features contributed to the detection and how much.")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Generate training data
        print("Step 1: Training anomaly detector on 100 normal transactions...")
        training_data = [generate_normal_transaction() for _ in range(100)]

        fit_response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/fit",
            json={
                "model": "shap-demo-model",
                "backend": "isolation_forest",
                "data": training_data,
                "schema": {
                    "amount": "numeric",
                    "frequency": "numeric",
                    "time_since_last": "numeric",
                },
                "contamination": 0.1,
            },
        )

        if fit_response.status_code != 200:
            print(f"   Fit failed: {fit_response.text}")
            return False

        print("   Model trained successfully!")
        print()

        # Generate test data with anomalies
        print("Step 2: Generating test data (5 normal + 2 anomalous)...")
        test_data = [generate_normal_transaction() for _ in range(5)]
        test_data.append(generate_anomalous_transaction())
        test_data.append(generate_anomalous_transaction())
        print()

        # Score with SHAP explanations
        print("Step 3: Scoring with SHAP explanations enabled...")
        print()
        score_response = await client.post(
            f"{LLAMAFARM_URL}/anomaly/score",
            json={
                "model": "shap-demo-model",
                "backend": "isolation_forest",
                "data": test_data,
                "schema": {
                    "amount": "numeric",
                    "frequency": "numeric",
                    "time_since_last": "numeric",
                },
                "explain": True,  # Enable SHAP explanations
                "feature_names": ["amount", "frequency", "time_since_last"],
            },
        )

        if score_response.status_code != 200:
            print(f"   Score failed: {score_response.text}")
            return False

        results = score_response.json()
        print(f"Scored {len(results['data'])} data points:")
        print()

        # Display results with explanations
        anomaly_count = 0
        explanation_count = 0
        for item in results["data"]:
            status = "ANOMALY" if item["is_anomaly"] else "normal"
            print(f"Point {item['index']}: {status} (score: {item['score']:.4f})")

            if item["is_anomaly"]:
                anomaly_count += 1
                if item.get("explanation"):
                    explanation_count += 1
                    exp = item["explanation"]
                    print(f"   Summary: {exp.get('summary', 'N/A')}")
                    print("   Feature contributions:")
                    for c in exp.get("contributions", [])[:3]:
                        direction = "+" if c["direction"] == "increases" else "-"
                        print(
                            f"      {c['feature']}: {direction}{abs(c['shap_value']):.4f} "
                            f"(value={c['value']:.2f})"
                        )
                print()

        print("-" * 40)
        print(f"Summary: {anomaly_count} anomalies, {explanation_count} with SHAP explanations")
        return explanation_count > 0


async def demo_streaming_shap():
    """Demonstrate streaming anomaly detection with SHAP explanations."""
    print()
    print("=" * 60)
    print("Part 2: Streaming Anomaly Detection with SHAP")
    print("=" * 60)
    print()
    print("Streaming detection can also generate SHAP explanations in real-time.")
    print("This is useful for monitoring systems where you need to understand")
    print("why an alert was triggered immediately.")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Clean up any existing detector
        await client.delete(f"{LLAMAFARM_URL}/anomaly/stream/shap-stream-demo")

        # Cold start phase
        print("Step 1: Cold start - streaming 60 normal transactions...")
        for i in range(60):
            data = generate_normal_transaction()
            response = await client.post(
                f"{LLAMAFARM_URL}/anomaly/stream",
                json={
                    "model": "shap-stream-demo",
                    "data": data,
                    "backend": "ecod",
                    "min_samples": 50,
                    "threshold": 0.6,
                    "explain": True,
                    "feature_names": ["amount", "frequency", "time_since_last"],
                },
            )
            if response.status_code != 200:
                print(f"   Stream failed: {response.text}")
                return False

            result = response.json()
            if i == 0 or i == 49 or i == 59:
                print(f"   Point {i + 1}: status={result['status']}")

        print(f"   Model ready! Status: {result['status']}")
        print()

        # Stream anomalies
        print("Step 2: Streaming anomalous transactions...")
        print()
        anomaly_count = 0
        explanation_count = 0

        for i in range(5):
            data = generate_anomalous_transaction()
            response = await client.post(
                f"{LLAMAFARM_URL}/anomaly/stream",
                json={
                    "model": "shap-stream-demo",
                    "data": data,
                    "backend": "ecod",
                    "explain": True,
                    "feature_names": ["amount", "frequency", "time_since_last"],
                },
            )

            if response.status_code != 200:
                continue

            result = response.json()
            for r in result["results"]:
                if r.get("is_anomaly"):
                    anomaly_count += 1
                    print(f"ALERT: Anomaly detected! (score: {r['score']:.4f})")
                    if r.get("explanation"):
                        explanation_count += 1
                        exp = r["explanation"]
                        print(f"   {exp.get('summary', 'N/A')}")
                        if exp.get("contributions"):
                            print("   Top factors:")
                            for c in exp["contributions"][:2]:
                                print(f"      - {c['feature']}: {c['direction']}")
                    print()

        # Cleanup
        await client.delete(f"{LLAMAFARM_URL}/anomaly/stream/shap-stream-demo")

        print("-" * 40)
        print(f"Summary: {anomaly_count} anomalies, {explanation_count} with SHAP explanations")
        return explanation_count > 0


async def main():
    """Run the SHAP explanations demo."""
    print()
    print("SHAP Explainability for Anomaly Detection")
    print("=" * 60)
    print()
    print("This demo shows how to get interpretable explanations for")
    print("why data points are flagged as anomalies using SHAP values.")
    print()

    batch_success = await demo_batch_shap()
    streaming_success = await demo_streaming_shap()

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Batch SHAP: {'PASSED' if batch_success else 'FAILED'}")
    print(f"Streaming SHAP: {'PASSED' if streaming_success else 'FAILED'}")
    print()
    print("Key takeaways:")
    print("- Set explain=True to enable SHAP explanations")
    print("- Provide feature_names for readable output")
    print("- SHAP shows which features drive anomaly scores")
    print("- Works with both batch and streaming detection")


if __name__ == "__main__":
    asyncio.run(main())
