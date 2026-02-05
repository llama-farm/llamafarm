#!/usr/bin/env python3
"""
Text Classification with SetFit (Few-Shot Learning)

Demonstrates LlamaFarm's text classifier powered by SetFit:
- Train with as few as 8-16 examples per class
- Uses contrastive learning on sentence-transformers
- Fast inference with high accuracy

Use cases:
- Intent classification
- Sentiment analysis
- Topic categorization
- Support ticket routing

Prerequisites:
- LlamaFarm servers running (nx start universal-runtime && nx start server)
"""

import asyncio

import httpx

# Configuration
LLAMAFARM_URL = "http://localhost:14345/v1/ml"


async def demo_intent_classifier():
    """Demo: Train and use an intent classifier."""
    print("=" * 60)
    print("Demo 1: Intent Classification")
    print("=" * 60)
    print()
    print("Training a customer service intent classifier with just")
    print("a few examples per class using SetFit few-shot learning.")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Training data - just a few examples per class!
        training_data = [
            # Booking intent
            {"text": "I need to book a flight to New York", "label": "booking"},
            {"text": "Can you reserve a hotel room for me?", "label": "booking"},
            {"text": "I want to schedule a meeting", "label": "booking"},
            {"text": "Book me a table at the restaurant", "label": "booking"},
            # Cancellation intent
            {"text": "I need to cancel my reservation", "label": "cancellation"},
            {"text": "Please cancel my flight", "label": "cancellation"},
            {"text": "I want to cancel my order", "label": "cancellation"},
            {"text": "Cancel the appointment please", "label": "cancellation"},
            # Information intent
            {"text": "What are your business hours?", "label": "information"},
            {"text": "How much does shipping cost?", "label": "information"},
            {"text": "What's your return policy?", "label": "information"},
            {"text": "Where is my order?", "label": "information"},
            # Complaint intent
            {"text": "I'm unhappy with my purchase", "label": "complaint"},
            {"text": "This product is defective", "label": "complaint"},
            {"text": "I want to speak to a manager", "label": "complaint"},
            {"text": "Your service is terrible", "label": "complaint"},
        ]

        print(f"Training data: {len(training_data)} examples")
        print(f"Classes: booking, cancellation, information, complaint")
        print()

        # Train the classifier
        print("Training classifier (this may take a minute)...")
        fit_response = await client.post(
            f"{LLAMAFARM_URL}/classifier/fit",
            json={
                "model": "intent-classifier",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "training_data": training_data,
                "num_iterations": 10,  # Fewer iterations for demo
                "overwrite": True,
            },
        )

        if fit_response.status_code != 200:
            print(f"Training failed: {fit_response.text}")
            return False

        result = fit_response.json()
        print(f"Training complete!")
        print(f"  Model: {result.get('model')}")
        print(f"  Training time: {result.get('training_time_ms', 0):.0f}ms")
        print(f"  Classes: {result.get('classes')}")
        print()

        # Test predictions
        test_texts = [
            "I'd like to book a flight to Paris",
            "Cancel my subscription immediately",
            "What time do you close?",
            "This is the worst experience ever",
            "Reserve a conference room for tomorrow",
            "How do I track my package?",
        ]

        print("Testing predictions...")
        print()
        predict_response = await client.post(
            f"{LLAMAFARM_URL}/classifier/predict",
            json={
                "model": "intent-classifier",
                "texts": test_texts,
            },
        )

        if predict_response.status_code != 200:
            print(f"Prediction failed: {predict_response.text}")
            return False

        predictions = predict_response.json()
        print("Results:")
        print("-" * 60)
        for i, pred in enumerate(predictions.get("data", [])):
            text = test_texts[i][:40] + "..." if len(test_texts[i]) > 40 else test_texts[i]
            label = pred.get("label", "unknown")
            confidence = pred.get("score", 0)
            print(f"  '{text}'")
            print(f"    -> {label} ({confidence:.1%} confidence)")
            print()

        return True


async def demo_sentiment_classifier():
    """Demo: Quick sentiment analysis."""
    print()
    print("=" * 60)
    print("Demo 2: Sentiment Analysis")
    print("=" * 60)
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Minimal training data
        training_data = [
            {"text": "I love this product!", "label": "positive"},
            {"text": "Amazing experience, highly recommend", "label": "positive"},
            {"text": "Best purchase I ever made", "label": "positive"},
            {"text": "Absolutely fantastic service", "label": "positive"},
            {"text": "This is terrible", "label": "negative"},
            {"text": "Worst experience of my life", "label": "negative"},
            {"text": "Complete waste of money", "label": "negative"},
            {"text": "I hate this so much", "label": "negative"},
            {"text": "It's okay, nothing special", "label": "neutral"},
            {"text": "Average product, meets expectations", "label": "neutral"},
            {"text": "Neither good nor bad", "label": "neutral"},
            {"text": "It works as described", "label": "neutral"},
        ]

        print(f"Training sentiment classifier with {len(training_data)} examples...")
        fit_response = await client.post(
            f"{LLAMAFARM_URL}/classifier/fit",
            json={
                "model": "sentiment-classifier",
                "base_model": "sentence-transformers/all-MiniLM-L6-v2",
                "training_data": training_data,
                "num_iterations": 10,
                "overwrite": True,
            },
        )

        if fit_response.status_code != 200:
            print(f"Training failed: {fit_response.text}")
            return False

        print("Training complete!")
        print()

        # Test
        reviews = [
            "The food was absolutely delicious!",
            "Meh, it was fine I guess",
            "Never coming back, horrible service",
            "Pretty good overall, would recommend",
        ]

        predict_response = await client.post(
            f"{LLAMAFARM_URL}/classifier/predict",
            json={"model": "sentiment-classifier", "texts": reviews},
        )

        if predict_response.status_code != 200:
            print(f"Prediction failed: {predict_response.text}")
            return False

        print("Sentiment Analysis Results:")
        print("-" * 60)
        for i, pred in enumerate(predict_response.json().get("data", [])):
            sentiment = pred.get("label", "unknown")
            confidence = pred.get("score", 0)
            emoji = {"positive": "😊", "negative": "😠", "neutral": "😐"}.get(sentiment, "❓")
            print(f"  {emoji} '{reviews[i]}'")
            print(f"     -> {sentiment} ({confidence:.1%})")
            print()

        return True


async def main():
    """Run classifier demos."""
    print()
    print("SetFit Text Classification Demo")
    print("=" * 60)
    print()
    print("SetFit enables few-shot text classification - train accurate")
    print("classifiers with just 8-16 examples per class!")
    print()

    intent_ok = await demo_intent_classifier()
    sentiment_ok = await demo_sentiment_classifier()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print(f"Intent Classifier: {'PASSED' if intent_ok else 'FAILED'}")
    print(f"Sentiment Classifier: {'PASSED' if sentiment_ok else 'FAILED'}")
    print()
    print("Key features:")
    print("- Few-shot learning (8-16 examples per class)")
    print("- Fast training (sentence-transformer fine-tuning)")
    print("- High accuracy for domain-specific tasks")
    print("- Auto-saves models for production use")


if __name__ == "__main__":
    asyncio.run(main())
