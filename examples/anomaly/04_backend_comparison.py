#!/usr/bin/env python3
"""Backend Comparison: Choosing the Right Algorithm

This example compares all available anomaly detection backends
to help you choose the right one for your use case.

Available Backends (12 total):

FAST (Parameter-free, best for most use cases):
  - ecod: Empirical Cumulative Distribution (RECOMMENDED)
  - hbos: Histogram-based, fastest algorithm
  - copod: Copula-based, interpretable

LEGACY (Good but need tuning):
  - isolation_forest: Tree-based, robust default
  - one_class_svm: SVM-based, small datasets
  - local_outlier_factor: Density-based, clustered data

ADVANCED:
  - knn: K-nearest neighbors
  - mcd: Minimum Covariance Determinant
  - cblof: Clustering-based
  - suod: Ensemble of multiple algorithms
  - loda: Lightweight, good for streaming
  - autoencoder: Neural network, complex patterns

Requirements:
    - Universal Runtime running: nx start universal-runtime

Run:
    cd /path/to/llamafarm
    uv run python examples/anomaly/04_backend_comparison.py
"""

import os
import random
import time
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
client = httpx.Client(timeout=120)


def generate_multimodal_data(n_normal: int, n_anomaly: int) -> tuple[list, list]:
    """Generate data with multiple normal clusters and scattered anomalies."""
    random.seed(42)

    # Normal data: two clusters
    normal = []
    for _ in range(n_normal // 2):
        # Cluster 1: centered at (10, 10)
        normal.append([random.gauss(10, 2), random.gauss(10, 2)])
    for _ in range(n_normal // 2):
        # Cluster 2: centered at (30, 30)
        normal.append([random.gauss(30, 2), random.gauss(30, 2)])

    # Anomalies: scattered
    anomalies = []
    for _ in range(n_anomaly):
        anomalies.append([random.uniform(-10, 50), random.uniform(-10, 50)])

    return normal, anomalies


def test_backend(backend: str, train_data: list, test_data: list) -> dict:
    """Test a single backend and return results."""
    try:
        # Fit
        start = time.perf_counter()
        response = client.post(
            f"{BASE_URL}/v1/ml/anomaly/fit",
            json={
                "model": f"compare-{backend}",
                "backend": backend,
                "data": train_data,
                "contamination": 0.1,
            },
        )
        fit_time = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            return {"backend": backend, "error": response.text}

        # Score
        start = time.perf_counter()
        response = client.post(
            f"{BASE_URL}/v1/ml/anomaly/score",
            json={
                "model": f"compare-{backend}",
                "backend": backend,
                "data": test_data,
            },
        )
        score_time = (time.perf_counter() - start) * 1000

        if response.status_code != 200:
            return {"backend": backend, "error": response.text}

        result = response.json()

        return {
            "backend": backend,
            "fit_time_ms": fit_time,
            "score_time_ms": score_time,
            "anomaly_count": result["summary"]["anomaly_count"],
            "anomaly_rate": result["summary"]["anomaly_rate"],
            "threshold": result["summary"]["threshold"],
        }
    except Exception as e:
        return {"backend": backend, "error": str(e)}


def main():
    print("=" * 70)
    print("BACKEND COMPARISON: Choosing the Right Anomaly Detection Algorithm")
    print("=" * 70)
    print()

    # List available backends
    print("Step 1: Listing available backends...")
    response = client.get(f"{BASE_URL}/v1/ml/anomaly/backends")
    backends_info = response.json()

    print(f"\nAvailable backends ({backends_info['total']} total):")
    print("-" * 70)
    for b in backends_info["data"]:
        print(f"  {b['backend']:<20} | {b['category']:<12} | "
              f"Speed: {b['speed']:<10} | {b['description'][:35]}...")
    print()

    # Generate test data
    print("Step 2: Generating test data...")
    train_normal, _ = generate_multimodal_data(400, 0)
    test_normal, test_anomalies = generate_multimodal_data(90, 10)
    test_data = test_normal + test_anomalies

    print(f"  Training: {len(train_normal)} normal samples")
    print(f"  Testing: {len(test_normal)} normal + {len(test_anomalies)} anomalies")
    print()

    # Test all backends
    print("Step 3: Testing all backends...")
    print("-" * 70)

    # Backends to test (ordered by expected speed)
    backends_to_test = [
        "hbos",            # Fastest
        "ecod",            # Fast, parameter-free
        "copod",           # Fast, interpretable
        "loda",            # Fast, streaming
        "isolation_forest",# Classic
        "knn",             # Distance-based
        "local_outlier_factor",
        "mcd",
        "cblof",
        "one_class_svm",   # Slow
        "suod",            # Ensemble (slowest)
        # "autoencoder",   # Requires special handling
    ]

    results = []
    for backend in backends_to_test:
        print(f"  Testing {backend}...", end=" ", flush=True)
        result = test_backend(backend, train_normal, test_data)
        results.append(result)

        if "error" in result:
            print(f"ERROR: {result['error'][:50]}")
        else:
            print(f"OK ({result['fit_time_ms']:.0f}ms fit, "
                  f"{result['anomaly_count']} detected)")

    print()

    # Results table
    print("Step 4: Results Summary")
    print("=" * 70)
    print(f"{'Backend':<22} {'Fit (ms)':<10} {'Score (ms)':<12} "
          f"{'Detected':<10} {'Rate':<8}")
    print("-" * 70)

    # Sort by fit time
    successful = [r for r in results if "error" not in r]
    successful.sort(key=lambda x: x["fit_time_ms"])

    for r in successful:
        print(f"{r['backend']:<22} {r['fit_time_ms']:>8.1f}  "
              f"{r['score_time_ms']:>10.1f}  "
              f"{r['anomaly_count']:>8}  "
              f"{r['anomaly_rate']*100:>6.1f}%")

    failed = [r for r in results if "error" in r]
    if failed:
        print("\nFailed backends:")
        for r in failed:
            print(f"  {r['backend']}: {r['error'][:60]}")

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print("""
For most use cases, start with ECOD:
  - Parameter-free (no tuning needed)
  - Fast (scales to millions of samples)
  - Robust (handles various data distributions)

Choose based on your needs:

SPEED CRITICAL (real-time, streaming):
  - hbos: Fastest, use for high-throughput
  - loda: Good for streaming data
  - ecod: Fast with good accuracy

ACCURACY CRITICAL (batch, offline):
  - suod: Ensemble of algorithms, most robust
  - isolation_forest: Well-tested, good default
  - local_outlier_factor: Good for clustered anomalies

INTERPRETABILITY:
  - copod: Scores based on empirical copulas
  - ecod: Scores based on cumulative distribution

SMALL DATASETS (<1000 samples):
  - one_class_svm: Works well with limited data
  - mcd: Robust covariance estimation

COMPLEX PATTERNS:
  - autoencoder: Neural network, learns complex representations
  - suod: Combines multiple approaches
""")


if __name__ == "__main__":
    main()
