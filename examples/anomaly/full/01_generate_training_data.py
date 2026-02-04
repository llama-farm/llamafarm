#!/usr/bin/env python3
"""
Step 1: Generate Training Data

Creates realistic factory sensor data for training an anomaly detection model.
The data simulates normal operating conditions with typical sensor noise.

Sensors:
- Temperature: 72°F ± 2°F (factory floor)
- Humidity: 45% ± 3% (climate controlled)
- Pressure: 1013 hPa ± 5 hPa (atmospheric)
- Motor RPM: 3000 ± 50 (industrial motor)

Output: training_data.json
"""

import json
import random
from pathlib import Path


def generate_normal_reading() -> dict:
    """Generate a single normal sensor reading."""
    return {
        "temperature": round(72 + random.gauss(0, 2), 2),
        "humidity": round(45 + random.gauss(0, 3), 2),
        "pressure": round(1013 + random.gauss(0, 5), 2),
        "motor_rpm": round(3000 + random.gauss(0, 50), 1),
    }


def main():
    print("=" * 60)
    print("Step 1: Generate Training Data")
    print("=" * 60)
    print()

    # Configuration
    num_samples = 500
    output_file = Path(__file__).parent / "training_data.json"

    print(f"Generating {num_samples} normal sensor readings...")
    print()

    # Generate training data
    training_data = [generate_normal_reading() for _ in range(num_samples)]

    # Show sample statistics
    temps = [d["temperature"] for d in training_data]
    humids = [d["humidity"] for d in training_data]
    rpms = [d["motor_rpm"] for d in training_data]

    print("Sample Statistics:")
    print(f"  Temperature: {min(temps):.1f} - {max(temps):.1f}°F (mean: {sum(temps)/len(temps):.1f})")
    print(f"  Humidity:    {min(humids):.1f} - {max(humids):.1f}% (mean: {sum(humids)/len(humids):.1f})")
    print(f"  Motor RPM:   {min(rpms):.0f} - {max(rpms):.0f} (mean: {sum(rpms)/len(rpms):.0f})")
    print()

    # Save to file
    with open(output_file, "w") as f:
        json.dump(training_data, f, indent=2)

    print(f"✅ Saved {num_samples} samples to {output_file}")
    print()

    # Show first few samples
    print("First 3 samples:")
    for i, sample in enumerate(training_data[:3]):
        print(f"  {i+1}. temp={sample['temperature']}, humid={sample['humidity']}, "
              f"pressure={sample['pressure']}, rpm={sample['motor_rpm']}")

    print()
    print("Training data ready! Run 02_train_model.py next.")


if __name__ == "__main__":
    main()
