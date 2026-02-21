#!/usr/bin/env python3
"""ML API demo — exercise anomaly, classifier, timeseries, NLP, and other ML endpoints."""

from llamafarm import LlamaFarm


def safe_call(name: str, fn, *args, **kwargs):
    """Call an API and handle errors gracefully."""
    try:
        result = fn(*args, **kwargs)
        print(f"  ✅ {name}: {result}")
        return result
    except Exception as e:
        print(f"  ⚠️  {name}: {e}")
        return None


def main():
    lf = LlamaFarm()
    print(f"LlamaFarm: {lf}")
    print(f"OpenAI base URL: {lf.openai_base_url}\n")

    # --- NLP ---
    print("=== NLP API ===")
    safe_call("embeddings", lf.nlp.embeddings, "Hello world")
    safe_call("classify", lf.nlp.classify, "I love this product!", labels=["positive", "negative"])
    safe_call("ner", lf.nlp.ner, "John works at Google in Mountain View.")
    safe_call("rerank", lf.nlp.rerank, "machine learning", ["ML is great", "I like cats", "Deep learning rocks"])

    # --- Anomaly Detection ---
    print("\n=== Anomaly Detection API ===")
    sample_data = [[1.0, 2.0], [1.1, 2.1], [1.0, 1.9], [10.0, 10.0]]
    safe_call("backends", lf.anomaly.backends)
    safe_call("models", lf.anomaly.models)
    safe_call("fit", lf.anomaly.fit, sample_data[:3], algorithm="IsolationForest")
    safe_call("detect", lf.anomaly.detect, sample_data, explain=True)
    safe_call("score", lf.anomaly.score, sample_data)

    # --- Classifier ---
    print("\n=== Classifier API ===")
    safe_call("models", lf.classifier.models)
    safe_call("fit", lf.classifier.fit, [[1, 0], [0, 1], [1, 1]], [0, 1, 1])
    safe_call("predict", lf.classifier.predict, [[1, 0], [0, 1]])

    # --- Time Series ---
    print("\n=== Time Series API ===")
    safe_call("backends", lf.timeseries.backends)
    safe_call("models", lf.timeseries.models)
    ts_data = [{"timestamp": f"2024-01-{i+1:02d}", "value": float(i * 10 + 5)} for i in range(30)]
    safe_call("fit", lf.timeseries.fit, ts_data)
    safe_call("predict", lf.timeseries.predict, horizon=7)

    # --- ADTK ---
    print("\n=== ADTK API ===")
    safe_call("detectors", lf.adtk.detectors)
    safe_call("models", lf.adtk.models)

    # --- CatBoost ---
    print("\n=== CatBoost API ===")
    safe_call("info", lf.catboost.info)
    safe_call("models", lf.catboost.models)

    # --- Drift ---
    print("\n=== Drift Detection API ===")
    safe_call("detectors", lf.drift.detectors)
    safe_call("models", lf.drift.models)

    # --- Explainability ---
    print("\n=== Explainability API ===")
    safe_call("explainers", lf.explain.explainers)

    # --- Polars ---
    print("\n=== Polars API ===")
    safe_call("list_buffers", lf.polars.list_buffers)

    print("\n✅ ML demo complete!")


if __name__ == "__main__":
    main()
