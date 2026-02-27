#!/usr/bin/env python3
"""Comprehensive live integration test — hits every SDK endpoint against running servers.

Prerequisites:
    cd server && uv run python main.py           # port 14345
    cd runtimes/universal && uv run python server.py  # port 11540

    # Ensure sdk-test project exists:
    mkdir -p ~/.llamafarm/projects/default/sdk-test
    # with a llamafarm.yaml pointing to a GGUF model

Usage:
    cd packages/llamafarm-sdk
    uv run python examples/integration_test.py
"""

from __future__ import annotations

import asyncio
import sys
import time
from typing import Any

from llamafarm import LlamaFarm, ChatMessage, StreamChunk, build_project_config

NAMESPACE = "default"
PROJECT = "sdk-test"


class TestResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[tuple[str, str]] = []
        self.skipped: list[tuple[str, str]] = []

    def ok(self, name: str, detail: str = ""):
        self.passed.append(name)
        icon = "✅"
        print(f"  {icon} {name}" + (f" — {detail}" if detail else ""))

    def fail(self, name: str, error: str):
        self.failed.append((name, error))
        print(f"  ❌ {name} — {error}")

    def skip(self, name: str, reason: str):
        self.skipped.append((name, reason))
        print(f"  ⏭️  {name} — {reason}")

    def summary(self):
        total = len(self.passed) + len(self.failed) + len(self.skipped)
        print(f"\n{'=' * 60}")
        print(f"Results: {len(self.passed)}/{total} passed, "
              f"{len(self.failed)} failed, {len(self.skipped)} skipped")
        if self.failed:
            print("\nFailed:")
            for name, err in self.failed:
                print(f"  ❌ {name}: {err}")
        print()
        return len(self.failed) == 0


def safe(results: TestResult, name: str, fn, *args, timeout_s: int = 30, **kwargs) -> Any:
    """Run a test function, catching errors. Timeout via thread."""
    import concurrent.futures
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(fn, *args, **kwargs)
            result = future.result(timeout=timeout_s)
        return result
    except concurrent.futures.TimeoutError:
        results.fail(name, f"timed out ({timeout_s}s)")
        return None
    except Exception as e:
        results.fail(name, str(e)[:120])
        return None


def main():
    results = TestResult()
    lf = LlamaFarm()
    print(f"LlamaFarm SDK Integration Test")
    print(f"  Server: {lf.url}")
    print(f"  Runtime: {lf.runtime_url}")
    print(f"  OpenAI compat: {lf.openai_base_url}")
    print()

    # ── 1. Core ──────────────────────────────────────────────────────
    print("── Core ──")

    h = safe(results, "health", lf.health)
    if h:
        results.ok("health", f"status={h['status']}")
    
    models = safe(results, "models", lf.models)
    if models:
        count = len(models.get("data", models) if isinstance(models, dict) else models)
        results.ok("models", f"{count} models")

    # ── 2. Projects ──────────────────────────────────────────────────
    print("\n── Projects ──")

    projects = safe(results, "projects.list", lf.projects, NAMESPACE)
    if projects is not None:
        names = [p.name for p in projects]
        results.ok("projects.list", f"{len(projects)} projects: {names[:5]}")

    p = lf.project(NAMESPACE, PROJECT)
    info = safe(results, "project.info", p.info)
    if info:
        results.ok("project.info", f"name={info.name}")

    # ── 3. Chat ──────────────────────────────────────────────────────
    print("\n── Chat ──")

    resp = safe(results, "chat.sync", p.chat, "What is 2+2? One word.", stateless=True, max_tokens=20)
    if resp:
        results.ok("chat.sync", f"model={resp.model}, text={resp.text[:50]!r}")

    # Streaming
    chunks = []
    try:
        for chunk in p.chat_stream("Say hello.", stateless=True, max_tokens=10):
            if chunk.delta_content:
                chunks.append(chunk.delta_content)
        results.ok("chat.stream", f"chunks={len(chunks)}, text={''.join(chunks)[:50]!r}")
    except Exception as e:
        results.fail("chat.stream", str(e)[:120])

    # Async
    async def test_async_chat():
        resp = await p.achat("Say hi.", stateless=True, max_tokens=10)
        results.ok("chat.async", f"text={resp.text[:50]!r}")

        chunks = []
        async for chunk in p.achat_stream("Bye.", stateless=True, max_tokens=10):
            if chunk.delta_content:
                chunks.append(chunk.delta_content)
        results.ok("chat.async_stream", f"chunks={len(chunks)}")

    try:
        asyncio.run(test_async_chat())
    except Exception as e:
        results.fail("chat.async", str(e)[:120])

    # ChatMessage usage
    try:
        msgs = [
            ChatMessage(role="system", content="You are a pirate."),
            ChatMessage(role="user", content="What's your name? One word."),
        ]
        resp = p.chat(msgs, stateless=True, max_tokens=20)
        results.ok("chat.ChatMessage", f"text={resp.text[:50]!r}")
    except Exception as e:
        results.fail("chat.ChatMessage", str(e)[:120])

    # ── 4. Vision ────────────────────────────────────────────────────
    print("\n── Vision ──")

    vm = safe(results, "vision.models", lf.vision.models)
    if vm is not None:
        results.ok("vision.models", f"response={type(vm).__name__}")

    # ── 5. NLP ───────────────────────────────────────────────────────
    print("\n── NLP ──")

    # NOTE: NLP proxy has field name mismatch between server and runtime
    # (server sends "input", runtime expects "texts"). These will fail until
    # the server proxy is fixed. SDK sends correct fields per OpenAPI spec.
    emb = safe(results, "nlp.embeddings", lf.nlp.embeddings, "Hello world")
    if emb is not None:
        results.ok("nlp.embeddings", f"keys={list(emb.keys())[:5]}")

    cls = safe(results, "nlp.classify", lf.nlp.classify, "I love this!", labels=["positive", "negative"])
    if cls is not None:
        results.ok("nlp.classify", f"result={cls}")

    ner = safe(results, "nlp.ner", lf.nlp.ner, "John works at Google in NYC.")
    if ner is not None:
        results.ok("nlp.ner", f"entities={len(ner.get('entities', []))}")

    rr = safe(results, "nlp.rerank", lf.nlp.rerank, "AI", ["machine learning", "cooking", "deep learning"])
    if rr is not None:
        results.ok("nlp.rerank", f"results={len(rr.get('results', []))}")

    # ── 6. Anomaly Detection ─────────────────────────────────────────
    print("\n── Anomaly Detection ──")

    ab = safe(results, "anomaly.backends", lf.anomaly.backends)
    if ab is not None:
        results.ok("anomaly.backends", f"backends={list(ab.keys())[:5] if isinstance(ab, dict) else type(ab).__name__}")

    am = safe(results, "anomaly.models", lf.anomaly.models)
    if am is not None:
        results.ok("anomaly.models", f"response={type(am).__name__}")

    sample = [[1.0, 2.0], [1.1, 2.1], [1.0, 1.9], [10.0, 10.0]]
    af = safe(results, "anomaly.fit", lf.anomaly.fit, sample[:3])
    if af is not None:
        results.ok("anomaly.fit", f"result={af}")

    ad = safe(results, "anomaly.detect", lf.anomaly.detect, sample)
    if ad is not None:
        results.ok("anomaly.detect", f"anomalies={ad.get('anomalies', ad.get('predictions', '?'))}")

    # ── 7. Classifier ────────────────────────────────────────────────
    print("\n── Classifier ──")

    cm = safe(results, "classifier.models", lf.classifier.models)
    if cm is not None:
        results.ok("classifier.models", f"response={type(cm).__name__}")

    cf = safe(results, "classifier.fit", lf.classifier.fit,
              "sdk-test-intent",
              [
                  {"text": "cancel my order", "label": "cancel"},
                  {"text": "please cancel", "label": "cancel"},
                  {"text": "book a flight", "label": "book"},
                  {"text": "reserve a hotel", "label": "book"},
                  {"text": "check status", "label": "status"},
                  {"text": "where is my package", "label": "status"},
                  {"text": "I want to cancel", "label": "cancel"},
                  {"text": "make a reservation", "label": "book"},
              ])
    if cf is not None:
        results.ok("classifier.fit", f"result={cf}")

    cp = safe(results, "classifier.predict", lf.classifier.predict,
              "sdk-test-intent", ["cancel my trip", "book a room"])
    if cp is not None:
        results.ok("classifier.predict", f"predictions={cp}")

    # ── 8. Time Series ───────────────────────────────────────────────
    print("\n── Time Series ──")

    tb = safe(results, "timeseries.backends", lf.timeseries.backends)
    if tb is not None:
        results.ok("timeseries.backends", f"response={type(tb).__name__}")

    tm = safe(results, "timeseries.models", lf.timeseries.models)
    if tm is not None:
        results.ok("timeseries.models", f"response={type(tm).__name__}")

    # NOTE: timeseries proxy on :14345 has server-side 500 errors.
    # These work fine direct to runtime (:11540). SDK is correct per OpenAPI.
    ts_data = [{"timestamp": f"2024-01-{i+1:02d}", "value": float(i * 10 + 5)} for i in range(30)]
    tf = safe(results, "timeseries.fit", lf.timeseries.fit, ts_data, backend="exponential_smoothing")
    if tf is not None:
        model_name = tf.get("model", "?")
        results.ok("timeseries.fit", f"model={model_name}")
        # Only test predict if fit succeeded
        tp = safe(results, "timeseries.predict", lf.timeseries.predict, model_name, 5)
        if tp is not None:
            results.ok("timeseries.predict", f"forecasted={len(tp.get('predictions', []))} points")
    else:
        results.skip("timeseries.predict", "skipped — fit failed")

    # ── 9. ADTK ──────────────────────────────────────────────────────
    print("\n── ADTK ──")

    adtk_d = safe(results, "adtk.detectors", lf.adtk.detectors)
    if adtk_d is not None:
        results.ok("adtk.detectors", f"response={type(adtk_d).__name__}")

    adtk_m = safe(results, "adtk.models", lf.adtk.models)
    if adtk_m is not None:
        results.ok("adtk.models", f"response={type(adtk_m).__name__}")

    # ── 10. CatBoost ─────────────────────────────────────────────────
    print("\n── CatBoost ──")

    cb_i = safe(results, "catboost.info", lf.catboost.info)
    if cb_i is not None:
        results.ok("catboost.info", f"response={type(cb_i).__name__}")

    cb_m = safe(results, "catboost.models", lf.catboost.models)
    if cb_m is not None:
        results.ok("catboost.models", f"response={type(cb_m).__name__}")

    # ── 11. Drift ────────────────────────────────────────────────────
    print("\n── Drift ──")

    dr_d = safe(results, "drift.detectors", lf.drift.detectors)
    if dr_d is not None:
        results.ok("drift.detectors", f"response={type(dr_d).__name__}")

    dr_m = safe(results, "drift.models", lf.drift.models)
    if dr_m is not None:
        results.ok("drift.models", f"response={type(dr_m).__name__}")

    # ── 12. Explainability ───────────────────────────────────────────
    print("\n── Explainability ──")

    ex = safe(results, "explain.explainers", lf.explain.explainers)
    if ex is not None:
        results.ok("explain.explainers", f"response={type(ex).__name__}")

    # ── 13. Polars ───────────────────────────────────────────────────
    print("\n── Polars ──")

    pl = safe(results, "polars.list_buffers", lf.polars.list_buffers)
    if pl is not None:
        results.ok("polars.list_buffers", f"response={type(pl).__name__}")

    # ── 14. Config Generation ────────────────────────────────────────
    print("\n── Config Generation ──")

    config = build_project_config(
        "test-project",
        model="Qwen/Qwen3-8B",
        system_prompt="You are a helpful coding assistant.",
        temperature=0.7,
        max_tokens=2048,
    )
    assert config["version"] == "v1"
    assert config["runtime"]["models"][0]["model"] == "Qwen/Qwen3-8B"
    assert config["runtime"]["models"][0]["temperature"] == 0.7
    results.ok("build_project_config", f"keys={list(config.keys())}")

    # ── 15. OpenAI Compatibility ─────────────────────────────────────
    print("\n── OpenAI Compatibility ──")

    base_url = lf.openai_base_url
    assert base_url.endswith("/v1")
    results.ok("openai_base_url", f"url={base_url}")

    try:
        from openai import OpenAI
        oai = OpenAI(base_url=base_url, api_key="not-needed")
        oai_models = oai.models.list()
        results.ok("openai.models.list", f"{len(oai_models.data)} models")
    except ImportError:
        results.skip("openai.models.list", "openai package not installed")
    except Exception as e:
        results.fail("openai.models.list", str(e)[:120])

    # ── 16. Generated Types Import ───────────────────────────────────
    print("\n── Generated Types ──")

    try:
        from llamafarm import ADTKDetectRequest, CatBoostFitRequest, TimeseriesFitRequest
        results.ok("import.generated_types", "ADTKDetectRequest, CatBoostFitRequest, TimeseriesFitRequest")
    except ImportError as e:
        results.fail("import.generated_types", str(e))

    try:
        # Verify we can construct a generated type
        req = ADTKDetectRequest(data=[{"timestamp": "2024-01-01", "value": 1.0}])
        results.ok("construct.generated_type", f"ADTKDetectRequest(detector={req.detector})")
    except Exception as e:
        results.fail("construct.generated_type", str(e)[:120])

    # ── 17. Error Handling ───────────────────────────────────────────
    print("\n── Error Handling ──")

    from llamafarm import NotFoundError, ValidationError, LlamaFarmError
    try:
        lf.project("default", "nonexistent-project-xyz").info()
        results.fail("error.not_found", "Expected NotFoundError")
    except NotFoundError as e:
        results.ok("error.not_found", f"status={e.status_code}")
    except Exception as e:
        results.fail("error.not_found", f"Wrong exception: {type(e).__name__}: {e}")

    # ── Summary ──────────────────────────────────────────────────────
    lf.close()
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
