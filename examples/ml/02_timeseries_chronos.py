#!/usr/bin/env python3
"""
Time-Series Forecasting with Chronos (Zero-Shot)

Demonstrates LlamaFarm's time-series forecasting powered by:
- Amazon Chronos: Zero-shot foundation model (no training needed!)
- Darts: ARIMA, Exponential Smoothing, Theta methods

Chronos is a T5-based model pre-trained on diverse time-series data,
enabling accurate forecasting without any model training.

Use cases:
- Demand forecasting
- Resource planning
- Capacity prediction
- Sales forecasting

Prerequisites:
- LlamaFarm servers running (nx start universal-runtime && nx start server)
"""

import asyncio
import math
import random

import httpx

# Configuration - Direct to Universal Runtime for time-series
RUNTIME_URL = "http://localhost:11540"


def generate_seasonal_data(n_points: int = 100, noise: float = 0.1) -> list[dict]:
    """Generate seasonal time-series data with trend as timestamped data."""
    from datetime import datetime, timedelta

    data = []
    base_date = datetime(2024, 1, 1)

    for i in range(n_points):
        # Base trend (slight upward)
        trend = 100 + i * 0.5
        # Seasonal component (weekly pattern, assuming daily data)
        seasonal = 20 * math.sin(2 * math.pi * i / 7)
        # Noise
        noise_val = random.gauss(0, noise * trend)
        timestamp = (base_date + timedelta(days=i)).isoformat()
        data.append({"timestamp": timestamp, "value": round(trend + seasonal + noise_val, 2)})
    return data


def generate_sales_data(n_points: int = 60) -> list[dict]:
    """Generate realistic sales data with weekly seasonality as timestamped data."""
    from datetime import datetime, timedelta

    data = []
    base_date = datetime(2024, 1, 1)

    for i in range(n_points):
        # Day of week effect (higher on weekends)
        day_of_week = i % 7
        weekend_boost = 1.5 if day_of_week >= 5 else 1.0

        # Base sales
        base = 1000

        # Gradual growth
        growth = i * 5

        # Weekly pattern
        weekly = 200 * math.sin(2 * math.pi * i / 7)

        # Random noise
        noise = random.gauss(0, 50)

        sales = (base + growth + weekly) * weekend_boost + noise
        timestamp = (base_date + timedelta(days=i)).isoformat()
        data.append({"timestamp": timestamp, "value": round(max(0, sales), 2)})

    return data


async def demo_chronos_forecast():
    """Demo: Zero-shot forecasting with Chronos."""
    print("=" * 60)
    print("Demo 1: Zero-Shot Forecasting with Chronos")
    print("=" * 60)
    print()
    print("Chronos is Amazon's foundation model for time-series.")
    print("It requires NO training - just provide your data!")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Generate sample data
        print("Generating 60 days of sales data...")
        sales_data = generate_sales_data(60)
        values = [d["value"] for d in sales_data]
        print(f"  Data range: {min(values):.0f} to {max(values):.0f}")
        print(f"  Last 5 values: {values[-5:]}")
        print()

        # Forecast with Chronos (zero-shot - no training!)
        print("Forecasting next 14 days with Chronos (zero-shot)...")
        response = await client.post(
            f"{RUNTIME_URL}/v1/timeseries/predict",
            json={
                "model": "sales-forecast-chronos",
                "horizon": 14,  # Forecast 14 days ahead
                "confidence_level": 0.9,  # 90% confidence intervals
                "data": sales_data,  # Zero-shot requires data in predict
            },
        )

        if response.status_code != 200:
            print(f"Chronos forecast failed: {response.text[:200]}")
            print()
            print("Note: Chronos requires the model to be downloaded first.")
            print("Falling back to traditional methods...")
            return False

        result = response.json()
        predictions = result.get("predictions", [])

        if not predictions:
            print("No predictions returned")
            return False

        print()
        print("Forecast Results:")
        print("-" * 60)
        print(f"{'Day':>4} | {'Forecast':>10} | {'90% CI':>20}")
        print("-" * 60)
        for i, pred in enumerate(predictions[:7], 1):  # Show first week
            value = pred.get("value", 0)
            lo = pred.get("lower")
            hi = pred.get("upper")
            if lo is not None and hi is not None:
                ci = f"[{lo:.0f}, {hi:.0f}]"
            else:
                ci = "N/A"
            print(f"{i:>4} | {value:>10.0f} | {ci:>20}")

        print()
        print(f"Inference time: {result.get('predict_time_ms', 0):.0f}ms")
        return True


async def demo_traditional_methods():
    """Demo: Compare with traditional methods (ARIMA, Exponential Smoothing)."""
    print()
    print("=" * 60)
    print("Demo 2: Traditional Methods (ARIMA, Theta)")
    print("=" * 60)
    print()
    print("For comparison, let's also try traditional forecasting methods")
    print("that require fitting to your specific data.")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Generate data
        data = generate_seasonal_data(100, noise=0.05)
        print(f"Generated 100 data points with weekly seasonality")
        print()

        # List available backends
        backends_response = await client.get(f"{RUNTIME_URL}/v1/timeseries/backends")
        if backends_response.status_code == 200:
            backends = backends_response.json().get("backends", [])
            print("Available backends:")
            for b in backends:
                training = "zero-shot" if not b["requires_training"] else "requires training"
                print(f"  - {b['name']}: {b['description'][:50]}... ({training})")
            print()

        # Try different methods
        methods = ["theta", "exponential_smoothing"]
        results = {}

        for method in methods:
            print(f"Testing {method}...")

            # First fit the model
            fit_response = await client.post(
                f"{RUNTIME_URL}/v1/timeseries/fit",
                json={
                    "model": f"test-{method}",
                    "backend": method,
                    "data": data,  # Structured timestamped data
                },
            )

            if fit_response.status_code != 200:
                print(f"  Fit failed: {fit_response.text[:100]}")
                continue

            fit_result = fit_response.json()
            fit_time = fit_result.get("training_time_ms", 0)

            # Then predict
            predict_response = await client.post(
                f"{RUNTIME_URL}/v1/timeseries/predict",
                json={
                    "model": f"test-{method}",
                    "horizon": 7,
                },
            )

            if predict_response.status_code != 200:
                print(f"  Predict failed: {predict_response.text[:100]}")
                continue

            pred_result = predict_response.json()
            predictions = pred_result.get("predictions", [])
            inference_time = pred_result.get("inference_time_ms", 0)

            results[method] = {
                "fit_time": fit_time,
                "inference_time": inference_time,
                "predictions": predictions,
            }
            print(f"  Fit: {fit_time:.0f}ms, Inference: {inference_time:.0f}ms")

        print()
        print("Method Comparison (7-day forecast):")
        print("-" * 60)
        print(f"{'Method':<25} | {'Day 1':>8} | {'Day 7':>8} | {'Total Time':>10}")
        print("-" * 60)
        for method, res_data in results.items():
            preds = res_data["predictions"]
            total_time = res_data["fit_time"] + res_data["inference_time"]
            if preds:
                first_val = preds[0].get("value", 0)
                last_val = preds[-1].get("value", 0)
                print(f"{method:<25} | {first_val:>8.1f} | {last_val:>8.1f} | {total_time:>8.0f}ms")

        return len(results) > 0


async def demo_resource_planning():
    """Demo: Practical use case - resource planning."""
    print()
    print("=" * 60)
    print("Demo 3: Practical Use Case - Server Capacity Planning")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        from datetime import datetime, timedelta

        # Simulate server CPU usage data (hourly for 7 days)
        print("Scenario: Predicting server CPU usage for capacity planning")
        print()

        cpu_data = []
        base_date = datetime(2024, 1, 1)
        for day in range(7):
            for hour in range(24):
                # Business hours have higher usage
                if 9 <= hour <= 17:
                    base = 60 + random.gauss(0, 10)
                else:
                    base = 25 + random.gauss(0, 5)

                # Weekends are lower
                if day >= 5:
                    base *= 0.4

                timestamp = (base_date + timedelta(days=day, hours=hour)).isoformat()
                cpu_data.append({"timestamp": timestamp, "value": round(max(5, min(95, base)), 1)})

        values = [d["value"] for d in cpu_data]
        print(f"Historical data: {len(cpu_data)} hours (7 days)")
        print(f"  Average CPU: {sum(values)/len(values):.1f}%")
        print(f"  Peak CPU: {max(values):.1f}%")
        print()

        # First fit a traditional model (Chronos has API issues)
        print("Fitting exponential smoothing model...")
        fit_response = await client.post(
            f"{RUNTIME_URL}/v1/timeseries/fit",
            json={
                "model": "cpu-forecast",
                "backend": "exponential_smoothing",
                "data": cpu_data,
            },
        )

        if fit_response.status_code != 200:
            print(f"Fit failed: {fit_response.text}")
            return False

        # Forecast next 24 hours
        print("Forecasting next 24 hours...")
        response = await client.post(
            f"{RUNTIME_URL}/v1/timeseries/predict",
            json={
                "model": "cpu-forecast",
                "horizon": 24,
            },
        )

        if response.status_code != 200:
            print(f"Forecast failed: {response.text}")
            return False

        result = response.json()
        predictions = result.get("predictions", [])
        pred_values = [p.get("value", 0) for p in predictions]
        pred_uppers = [p.get("upper") for p in predictions]

        print()
        print("24-Hour CPU Forecast:")
        print("-" * 50)

        # Group by time periods
        night = pred_values[0:6]  # 12am-6am
        morning = pred_values[6:12]  # 6am-12pm
        afternoon = pred_values[12:18]  # 12pm-6pm
        evening = pred_values[18:24]  # 6pm-12am

        print(f"  Night (12am-6am):    avg {sum(night)/len(night):.1f}% CPU")
        print(f"  Morning (6am-12pm):  avg {sum(morning)/len(morning):.1f}% CPU")
        print(f"  Afternoon (12pm-6pm): avg {sum(afternoon)/len(afternoon):.1f}% CPU")
        print(f"  Evening (6pm-12am):  avg {sum(evening)/len(evening):.1f}% CPU")
        print()

        peak_hour = pred_values.index(max(pred_values))
        peak_value = max(pred_values)
        peak_upper = pred_uppers[peak_hour] if pred_uppers[peak_hour] else peak_value

        print(f"Peak expected at hour {peak_hour}: {peak_value:.1f}% (95% CI: up to {peak_upper:.1f}%)")

        if peak_upper > 80:
            print()
            print("  *** RECOMMENDATION: Consider scaling up capacity ***")

        return True


async def main():
    """Run time-series demos."""
    print()
    print("Time-Series Forecasting Demo")
    print("=" * 60)
    print()
    print("LlamaFarm supports multiple forecasting methods:")
    print("- Chronos: Zero-shot foundation model (no training!)")
    print("- ARIMA: Classic statistical method")
    print("- Exponential Smoothing: Trend/seasonality decomposition")
    print("- Theta: Simple and robust")
    print()

    chronos_ok = await demo_chronos_forecast()
    traditional_ok = await demo_traditional_methods()
    planning_ok = await demo_resource_planning()

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Chronos Zero-Shot: {'PASSED' if chronos_ok else 'FAILED'}")
    print(f"Traditional Methods: {'PASSED' if traditional_ok else 'FAILED'}")
    print(f"Resource Planning: {'PASSED' if planning_ok else 'FAILED'}")
    print()
    print("Key takeaways:")
    print("- Chronos requires NO training - just provide data")
    print("- Traditional methods need fitting but may be more accurate")
    print("- Confidence intervals help with planning decisions")


if __name__ == "__main__":
    asyncio.run(main())
