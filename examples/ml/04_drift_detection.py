#!/usr/bin/env python3
"""
Data Drift Detection with Alibi Detect

Demonstrates LlamaFarm's data drift detection powered by Alibi Detect:
- KS (Kolmogorov-Smirnov): Univariate distribution drift
- MMD (Maximum Mean Discrepancy): Multivariate drift
- Chi-squared: Categorical feature drift

Data drift occurs when the statistical properties of production data
differ from training data, causing model performance degradation.

Use cases:
- ML model monitoring
- Data quality assurance
- Feature distribution monitoring
- Concept drift detection

Prerequisites:
- LlamaFarm servers running (nx start universal-runtime && nx start server)
"""

import asyncio
import random

import httpx

# Configuration - Direct to Universal Runtime for drift detection
RUNTIME_URL = "http://localhost:11540"


def generate_reference_data(n_samples: int = 500) -> list[list[float]]:
    """Generate reference (training) data distribution."""
    data = []
    for _ in range(n_samples):
        # Feature 1: Normal distribution, mean=50, std=10
        f1 = random.gauss(50, 10)
        # Feature 2: Normal distribution, mean=100, std=20
        f2 = random.gauss(100, 20)
        # Feature 3: Uniform distribution, 0-100
        f3 = random.uniform(0, 100)
        data.append([round(f1, 2), round(f2, 2), round(f3, 2)])
    return data


def generate_no_drift_data(n_samples: int = 200) -> list[list[float]]:
    """Generate data from the SAME distribution (no drift)."""
    return generate_reference_data(n_samples)


def generate_drifted_data(n_samples: int = 200) -> list[list[float]]:
    """Generate data with distribution drift."""
    data = []
    for _ in range(n_samples):
        # Feature 1: SHIFTED mean (50 -> 70)
        f1 = random.gauss(70, 10)
        # Feature 2: INCREASED variance (std 20 -> 40)
        f2 = random.gauss(100, 40)
        # Feature 3: Changed distribution (uniform -> skewed)
        f3 = random.betavariate(2, 5) * 100
        data.append([round(f1, 2), round(f2, 2), round(f3, 2)])
    return data


def generate_gradual_drift_data(n_samples: int = 200, drift_ratio: float = 0.5) -> list[list[float]]:
    """Generate data with gradual drift (mix of original and drifted)."""
    data = []
    for _ in range(n_samples):
        if random.random() < drift_ratio:
            # Drifted sample
            f1 = random.gauss(70, 10)
            f2 = random.gauss(100, 40)
            f3 = random.betavariate(2, 5) * 100
        else:
            # Original distribution
            f1 = random.gauss(50, 10)
            f2 = random.gauss(100, 20)
            f3 = random.uniform(0, 100)
        data.append([round(f1, 2), round(f2, 2), round(f3, 2)])
    return data


async def demo_ks_drift_detection():
    """Demo: Kolmogorov-Smirnov test for univariate drift."""
    print("=" * 60)
    print("Demo 1: KS Test (Univariate Drift Detection)")
    print("=" * 60)
    print()
    print("The KS test compares cumulative distributions of each feature.")
    print("Good for: detecting shifts in individual feature distributions")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Generate reference data
        reference_data = generate_reference_data(500)
        print(f"Reference data: 500 samples, 3 features")
        print(f"  Feature 1: mean=50, std=10")
        print(f"  Feature 2: mean=100, std=20")
        print(f"  Feature 3: uniform 0-100")
        print()

        # Fit the drift detector
        print("Fitting KS drift detector on reference data...")
        fit_response = await client.post(
            f"{RUNTIME_URL}/v1/drift/fit",
            json={
                "model": "ks-detector",
                "detector": "ks",
                "reference_data": reference_data,
                "params": {"p_val": 0.05},
            },
        )

        if fit_response.status_code != 200:
            print(f"Fit failed: {fit_response.text}")
            return False

        print("Detector fitted!")
        print()

        # Test 1: No drift
        print("Test 1: Data from SAME distribution (no drift)...")
        no_drift_data = generate_no_drift_data(200)

        detect_response = await client.post(
            f"{RUNTIME_URL}/v1/drift/detect",
            json={
                "model": "ks-detector",
                "data": no_drift_data,
            },
        )

        if detect_response.status_code == 200:
            response = detect_response.json()
            result = response.get("result", {})
            drift_detected = result.get("is_drift", False)
            p_values = result.get("p_values", [])
            print(f"  Drift detected: {drift_detected}")
            if p_values:
                print(f"  P-values per feature: {[f'{p:.4f}' for p in p_values]}")
        print()

        # Test 2: With drift
        print("Test 2: Data with SHIFTED distribution (drift)...")
        drifted_data = generate_drifted_data(200)

        detect_response = await client.post(
            f"{RUNTIME_URL}/v1/drift/detect",
            json={
                "model": "ks-detector",
                "data": drifted_data,
            },
        )

        if detect_response.status_code == 200:
            response = detect_response.json()
            result = response.get("result", {})
            drift_detected = result.get("is_drift", False)
            p_values = result.get("p_values", [])
            print(f"  Drift detected: {drift_detected}")
            if p_values:
                print(f"  P-values per feature: {[f'{p:.4f}' for p in p_values]}")
                print(f"  (p < 0.05 indicates significant drift)")

        return True


async def demo_mmd_drift_detection():
    """Demo: Maximum Mean Discrepancy for multivariate drift."""
    print()
    print("=" * 60)
    print("Demo 2: MMD Test (Multivariate Drift Detection)")
    print("=" * 60)
    print()
    print("MMD measures distance between distributions in kernel space.")
    print("Good for: detecting drift in feature interactions/correlations")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Generate reference data
        reference_data = generate_reference_data(500)
        print(f"Reference data: 500 samples, 3 features")
        print()

        # Fit the drift detector
        print("Fitting MMD drift detector...")
        fit_response = await client.post(
            f"{RUNTIME_URL}/v1/drift/fit",
            json={
                "model": "mmd-detector",
                "detector": "mmd",
                "reference_data": reference_data,
                "params": {"p_val": 0.05},
            },
        )

        if fit_response.status_code != 200:
            print(f"Fit failed: {fit_response.text}")
            return False

        print("Detector fitted!")
        print()

        # Test with gradual drift
        drift_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
        print("Testing with increasing drift levels:")
        print("-" * 50)

        for drift_ratio in drift_levels:
            test_data = generate_gradual_drift_data(200, drift_ratio)

            detect_response = await client.post(
                f"{RUNTIME_URL}/v1/drift/detect",
                json={
                    "model": "mmd-detector",
                    "data": test_data,
                },
            )

            if detect_response.status_code == 200:
                response = detect_response.json()
                result = response.get("result", {})
                drift_detected = result.get("is_drift", False)
                p_value = result.get("p_value", 1.0)
                status = "DRIFT" if drift_detected else "OK"
                print(f"  {int(drift_ratio*100):>3}% drifted samples: {status} (p={p_value:.4f})")

        return True


async def demo_monitoring_workflow():
    """Demo: Production monitoring workflow."""
    print()
    print("=" * 60)
    print("Demo 3: Production Monitoring Workflow")
    print("=" * 60)
    print()
    print("Simulating a production ML monitoring scenario...")
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Generate reference data (from training)
        reference_data = generate_reference_data(500)

        # Fit detector
        print("Step 1: Fit drift detector on training data distribution")
        fit_response = await client.post(
            f"{RUNTIME_URL}/v1/drift/fit",
            json={
                "model": "production-monitor",
                "detector": "ks",
                "reference_data": reference_data,
            },
        )

        if fit_response.status_code != 200:
            print(f"Fit failed: {fit_response.text}")
            return False

        print("  Detector ready for monitoring")
        print()

        # Simulate batches over time
        print("Step 2: Monitor incoming data batches")
        print("-" * 50)

        batches = [
            ("Week 1", generate_no_drift_data(100)),
            ("Week 2", generate_no_drift_data(100)),
            ("Week 3", generate_gradual_drift_data(100, 0.3)),  # Slight drift
            ("Week 4", generate_gradual_drift_data(100, 0.5)),  # More drift
            ("Week 5", generate_drifted_data(100)),  # Full drift
        ]

        alerts = []
        for week, batch_data in batches:
            detect_response = await client.post(
                f"{RUNTIME_URL}/v1/drift/detect",
                json={
                    "model": "production-monitor",
                    "data": batch_data,
                },
            )

            if detect_response.status_code == 200:
                response = detect_response.json()
                result = response.get("result", {})
                drift_detected = result.get("is_drift", False)

                if drift_detected:
                    alerts.append(week)
                    print(f"  {week}: ALERT - Drift detected!")
                else:
                    print(f"  {week}: OK - No drift")

        print()
        print("Step 3: Analysis")
        print("-" * 50)
        if alerts:
            print(f"  Drift alerts: {', '.join(alerts)}")
            print("  Recommended actions:")
            print("    1. Investigate data quality issues")
            print("    2. Check for upstream data changes")
            print("    3. Consider model retraining")
        else:
            print("  No drift detected - model performing as expected")

        return len(alerts) > 0


async def demo_available_detectors():
    """List all available drift detectors."""
    print()
    print("=" * 60)
    print("Available Drift Detectors")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{RUNTIME_URL}/v1/drift/detectors")
        if response.status_code != 200:
            print(f"Failed to list detectors: {response.text}")
            return

        result = response.json()
        detectors = result.get("detectors", [])

        print(f"{'Detector':<10} | {'Multivariate':<12} | Description")
        print("-" * 60)
        for det in detectors:
            multi = "Yes" if det["multivariate"] else "No"
            print(f"{det['name']:<10} | {multi:<12} | {det['description']}")
        print()


async def main():
    """Run drift detection demos."""
    print()
    print("Data Drift Detection with Alibi Detect")
    print("=" * 60)
    print()
    print("Data drift occurs when production data differs from training data,")
    print("causing ML model performance to degrade over time.")
    print()

    await demo_available_detectors()

    ks_ok = await demo_ks_drift_detection()
    mmd_ok = await demo_mmd_drift_detection()
    monitor_ok = await demo_monitoring_workflow()

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"KS Univariate Drift: {'PASSED' if ks_ok else 'FAILED'}")
    print(f"MMD Multivariate Drift: {'PASSED' if mmd_ok else 'FAILED'}")
    print(f"Production Monitoring: {'PASSED' if monitor_ok else 'FAILED'}")
    print()
    print("When to use each detector:")
    print("- KS: Individual feature distribution shifts")
    print("- MMD: Multivariate/correlation changes")
    print("- Chi2: Categorical feature drift")
    print()
    print("Best practices:")
    print("- Monitor drift continuously in production")
    print("- Set up alerts for p-value thresholds")
    print("- Retrain models when significant drift is detected")


if __name__ == "__main__":
    asyncio.run(main())
