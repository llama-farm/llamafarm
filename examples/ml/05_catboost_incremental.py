#!/usr/bin/env python3
"""
CatBoost Gradient Boosting with Incremental Learning

Demonstrates LlamaFarm's CatBoost integration:
- Native categorical feature handling (no one-hot encoding)
- Incremental learning (update model without full retrain)
- GPU acceleration (when available)
- Feature importance

Use cases:
- Tabular data classification
- Real-time model updates with streaming data
- Mixed numeric/categorical features
- Customer churn, fraud detection

Prerequisites:
- LlamaFarm servers running (nx start universal-runtime && nx start server)
"""

import asyncio
import random

import httpx

# Configuration - Direct to Universal Runtime
RUNTIME_URL = "http://localhost:11540"


def generate_customer_data(n_samples: int = 500, seed: int = 42) -> tuple[list[list], list[int]]:
    """Generate synthetic customer data for churn prediction.

    Features:
    - age (numeric)
    - tenure_months (numeric)
    - monthly_charges (numeric)
    - contract_type (categorical: 0=month-to-month, 1=one-year, 2=two-year)
    - payment_method (categorical: 0=credit, 1=bank, 2=electronic)
    """
    random.seed(seed)
    X = []
    y = []

    # Categorical value mappings (as strings for CatBoost)
    contract_types = ["month-to-month", "one-year", "two-year"]
    payment_methods = ["credit", "bank", "electronic"]

    for _ in range(n_samples):
        age = random.randint(18, 80)
        tenure = random.randint(0, 72)
        charges = round(random.uniform(20, 120), 2)
        contract_idx = random.choice([0, 1, 2])
        payment_idx = random.choice([0, 1, 2])

        # Churn logic: higher chance if month-to-month, low tenure, young
        churn_prob = 0.1
        if contract_idx == 0:  # month-to-month
            churn_prob += 0.3
        if tenure < 12:
            churn_prob += 0.2
        if age < 30:
            churn_prob += 0.1
        if charges > 80:
            churn_prob += 0.1

        churned = 1 if random.random() < churn_prob else 0

        # Use string values for categorical features (required by CatBoost)
        X.append([age, tenure, charges, contract_types[contract_idx], payment_methods[payment_idx]])
        y.append(churned)

    return X, y


def generate_new_customer_batch(n_samples: int = 50) -> tuple[list[list], list[int]]:
    """Generate a batch of new customer data (for incremental learning)."""
    return generate_customer_data(n_samples, seed=random.randint(1000, 9999))


async def demo_catboost_classifier():
    """Demo: Train and use CatBoost classifier."""
    print("=" * 60)
    print("Demo 1: CatBoost Classifier for Churn Prediction")
    print("=" * 60)
    print()
    print("CatBoost is a gradient boosting library with:")
    print("- Native categorical support (no one-hot encoding)")
    print("- Ordered boosting to reduce overfitting")
    print("- Fast training and inference")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check CatBoost availability
        info_response = await client.get(f"{RUNTIME_URL}/v1/catboost/info")
        if info_response.status_code != 200:
            print(f"CatBoost not available: {info_response.text}")
            return False

        info = info_response.json()
        if not info.get("available"):
            print(f"CatBoost not installed: {info.get('error')}")
            return False

        print(f"CatBoost available: GPU={info.get('gpu_available', False)}")
        print(f"Features: {[f for f in info.get('features', []) if f]}")
        print()

        # Generate training data
        X_train, y_train = generate_customer_data(500)

        print(f"Training data: {len(X_train)} samples")
        print(f"Features: age, tenure_months, monthly_charges, contract_type, payment_method")
        print(f"Churn rate: {sum(y_train)/len(y_train):.1%}")
        print()

        # Train classifier
        print("Training CatBoost classifier...")
        fit_response = await client.post(
            f"{RUNTIME_URL}/v1/catboost/fit",
            json={
                "model_id": "churn-classifier",
                "model_type": "classifier",
                "data": X_train,
                "labels": y_train,
                "feature_names": ["age", "tenure_months", "monthly_charges", "contract_type", "payment_method"],
                "cat_features": [3, 4],  # contract_type and payment_method are categorical
                "iterations": 200,
                "depth": 4,
            },
        )

        if fit_response.status_code != 200:
            print(f"Training failed: {fit_response.text}")
            return False

        result = fit_response.json()
        print(f"Training complete!")
        print(f"  Model: {result.get('model_id')}")
        print(f"  Iterations: {result.get('iterations')}")
        print(f"  Training time: {result.get('fit_time_ms', 0):.0f}ms")
        print()

        # Make predictions
        X_test, y_test = generate_customer_data(100, seed=999)

        print("Making predictions on 100 test samples...")
        predict_response = await client.post(
            f"{RUNTIME_URL}/v1/catboost/predict",
            json={
                "model_id": "churn-classifier",
                "data": X_test,
                "return_proba": True,
            },
        )

        if predict_response.status_code != 200:
            print(f"Prediction failed: {predict_response.text}")
            return False

        pred_result = predict_response.json()
        prediction_objects = pred_result.get("predictions", [])

        # Extract predictions and probabilities from the response objects
        predictions = [p.get("prediction") for p in prediction_objects]

        # Calculate accuracy
        correct = sum(1 for p, actual in zip(predictions, y_test) if p == actual)
        accuracy = correct / len(y_test)

        print(f"  Predictions: {len(predictions)} samples")
        print(f"  Accuracy: {accuracy:.1%}")
        print(f"  Inference time: {pred_result.get('predict_time_ms', 0):.0f}ms")
        print()

        # Show sample predictions
        print("Sample predictions:")
        print("-" * 50)
        for i in range(min(5, len(prediction_objects))):
            pred_obj = prediction_objects[i]
            pred = pred_obj.get("prediction")
            probas = pred_obj.get("probabilities", {})
            churn_proba = probas.get("1", probas.get(1, 0)) if probas else 0
            actual = "Churned" if y_test[i] == 1 else "Stayed"
            predicted = "Churned" if pred == 1 else "Stayed"
            print(f"  Customer {i+1}: {predicted} ({churn_proba:.1%} prob) - Actual: {actual}")

        return True


async def demo_incremental_learning():
    """Demo: Incrementally update CatBoost model with new data."""
    print()
    print("=" * 60)
    print("Demo 2: Incremental Learning (Model Update)")
    print("=" * 60)
    print()
    print("Update the model with new data without full retraining!")
    print("Useful for: streaming data, concept drift, continuous learning")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Generate new data batch
        X_new, y_new = generate_new_customer_batch(100)

        print(f"New data batch: {len(X_new)} samples")
        print(f"Churn rate in new data: {sum(y_new)/len(y_new):.1%}")
        print()

        # Update the existing model
        print("Incrementally updating model...")
        update_response = await client.post(
            f"{RUNTIME_URL}/v1/catboost/update",
            json={
                "model_id": "churn-classifier",
                "data": X_new,
                "labels": y_new,
            },
        )

        if update_response.status_code != 200:
            print(f"Update failed: {update_response.text}")
            return False

        result = update_response.json()
        print(f"Model updated!")
        print(f"  Samples added: {result.get('samples_added')}")
        print(f"  Trees before: {result.get('trees_before')}")
        print(f"  Trees after: {result.get('trees_after')}")
        print(f"  Update time: {result.get('update_time_ms', 0):.0f}ms")
        print()

        # Test updated model
        X_test, y_test = generate_customer_data(50, seed=888)

        predict_response = await client.post(
            f"{RUNTIME_URL}/v1/catboost/predict",
            json={
                "model_id": "churn-classifier",
                "data": X_test,
            },
        )

        if predict_response.status_code == 200:
            pred_result = predict_response.json()
            prediction_objects = pred_result.get("predictions", [])
            predictions = [p.get("prediction") for p in prediction_objects]
            correct = sum(1 for p, actual in zip(predictions, y_test) if p == actual)
            accuracy = correct / len(y_test)
            print(f"Updated model accuracy: {accuracy:.1%}")

        return True


async def demo_feature_importance():
    """Demo: Get feature importance from CatBoost model."""
    print()
    print("=" * 60)
    print("Demo 3: Feature Importance")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Getting feature importance...")
        importance_response = await client.get(
            f"{RUNTIME_URL}/v1/catboost/churn-classifier/importance",
        )

        if importance_response.status_code != 200:
            print(f"Failed to get importance: {importance_response.text}")
            return False

        result = importance_response.json()
        importances = result.get("importances", [])

        print()
        print("Feature Importance (sorted):")
        print("-" * 40)
        for item in importances:
            feat = item.get("feature", "unknown")
            imp = item.get("importance", 0)
            bar = "=" * int(imp / 5)
            print(f"  {feat:<20} {imp:>6.2f}% {bar}")

        return True


async def main():
    """Run CatBoost demos."""
    print()
    print("CatBoost Gradient Boosting Demo")
    print("=" * 60)
    print()
    print("CatBoost advantages over other boosting libraries:")
    print("- Native categorical handling (no preprocessing)")
    print("- Incremental learning (update without retrain)")
    print("- GPU acceleration")
    print("- Handles missing values automatically")
    print()

    classifier_ok = await demo_catboost_classifier()

    incremental_ok = False
    importance_ok = False
    if classifier_ok:
        incremental_ok = await demo_incremental_learning()
        importance_ok = await demo_feature_importance()

    print()
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"CatBoost Classifier: {'PASSED' if classifier_ok else 'FAILED'}")
    print(f"Incremental Learning: {'PASSED' if incremental_ok else 'FAILED'}")
    print(f"Feature Importance: {'PASSED' if importance_ok else 'FAILED'}")
    print()
    print("Key takeaways:")
    print("- CatBoost handles categorical features natively")
    print("- Incremental learning enables continuous model updates")
    print("- Feature importance helps with model interpretability")


if __name__ == "__main__":
    asyncio.run(main())
