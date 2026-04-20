"""ROW-79: Error recovery tests for edge runtime.

Sends malformed and invalid requests to verify the edge runtime returns
proper HTTP error responses (4xx) instead of crashing (500) or hanging.

Usage:
    EDGE_URL=http://192.168.1.100:11540 python -m pytest test_errors.py -v
    EDGE_URL=http://192.168.1.100:11540 python test_errors.py
"""

from __future__ import annotations

import os
import sys

import httpx
import pytest

EDGE_URL = os.environ.get("EDGE_URL", "http://localhost:11540")
MODEL = os.environ.get("EDGE_MODEL", "mission-router-v3")

# Requests must not return 500 or cause the server to become unresponsive.
# Acceptable: 400, 404, 422 (validation error)
ACCEPTABLE_ERROR_CODES = {400, 404, 422}


@pytest.fixture(scope="module")
def client():
    with httpx.Client(base_url=EDGE_URL, timeout=15) as c:
        yield c


def _assert_clean_error(resp: httpx.Response, context: str) -> None:
    """Assert the response is a clean 4xx validation error.

    Callers only invoke this for malformed requests that must never
    legitimately return 200 — accepting them silently would mask
    regressions where the server starts happily processing invalid
    payloads. Use bare ``status_code != 500`` assertions for cases
    where 200 *is* a valid outcome (e.g. request-body clamping).

    503 is intentionally allowed through: the runtime returns 503 when
    an upstream model registry is unreachable, which is a legitimate
    operational state, not a crash.
    """
    assert resp.status_code != 500, f"{context}: got 500 Internal Server Error: {resp.text[:300]}"
    assert resp.status_code != 502, f"{context}: got 502 Bad Gateway (server may have crashed)"
    assert resp.status_code in ACCEPTABLE_ERROR_CODES, (
        f"{context}: expected one of {sorted(ACCEPTABLE_ERROR_CODES)}, "
        f"got {resp.status_code}: {resp.text[:300]}"
    )


def _verify_still_alive(client: httpx.Client) -> None:
    """Verify the server is still responding after a bad request."""
    resp = client.get("/health")
    assert resp.status_code == 200, f"Server unresponsive after bad request: /health returned {resp.status_code}"


# ---- Bad JSON ----

class TestBadJSON:
    def test_completely_invalid_json(self, client):
        resp = client.post(
            "/v1/chat/completions",
            content="this is not json at all",
            headers={"Content-Type": "application/json"},
        )
        _assert_clean_error(resp, "invalid JSON body")
        _verify_still_alive(client)

    def test_empty_body(self, client):
        resp = client.post(
            "/v1/chat/completions",
            content="",
            headers={"Content-Type": "application/json"},
        )
        _assert_clean_error(resp, "empty body")
        _verify_still_alive(client)

    def test_json_array_instead_of_object(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json=[{"model": MODEL}],
        )
        _assert_clean_error(resp, "JSON array instead of object")
        _verify_still_alive(client)

    def test_truncated_json(self, client):
        resp = client.post(
            "/v1/chat/completions",
            content='{"model": "functiongemma", "messages": [{"role": "user", "conte',
            headers={"Content-Type": "application/json"},
        )
        _assert_clean_error(resp, "truncated JSON")
        _verify_still_alive(client)


# ---- Missing Required Fields ----

class TestMissingFields:
    def test_missing_model(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        _assert_clean_error(resp, "missing model field")
        _verify_still_alive(client)

    def test_missing_messages(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
        })
        _assert_clean_error(resp, "missing messages field")
        _verify_still_alive(client)

    def test_empty_messages_list(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [],
        })
        _assert_clean_error(resp, "empty messages list")
        _verify_still_alive(client)

    def test_message_missing_role(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"content": "hello"}],
        })
        _assert_clean_error(resp, "message missing role")
        _verify_still_alive(client)

    def test_message_missing_content(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user"}],
        })
        # Missing content may be valid for some roles (tool), so allow 200
        assert resp.status_code != 500, "missing content caused 500"
        _verify_still_alive(client)


# ---- Invalid Values ----

class TestInvalidValues:
    def test_nonexistent_model(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "nonexistent-model-that-does-not-exist-12345",
            "messages": [{"role": "user", "content": "hello"}],
        })
        _assert_clean_error(resp, "nonexistent model")
        assert resp.status_code in {400, 404, 422}, f"Expected 4xx for missing model, got {resp.status_code}"
        _verify_still_alive(client)

    def test_invalid_role(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "banana", "content": "hello"}],
        })
        _assert_clean_error(resp, "invalid role")
        _verify_still_alive(client)

    def test_negative_max_tokens(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": -1,
        })
        _assert_clean_error(resp, "negative max_tokens")
        _verify_still_alive(client)

    def test_temperature_out_of_range(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 999.0,
        })
        # May be clamped or rejected; just shouldn't crash
        assert resp.status_code != 500, "temperature=999 caused 500"
        _verify_still_alive(client)

    def test_wrong_type_for_messages(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": "this should be a list",
        })
        _assert_clean_error(resp, "messages as string instead of list")
        _verify_still_alive(client)

    def test_model_as_number(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": 42,
            "messages": [{"role": "user", "content": "hello"}],
        })
        _assert_clean_error(resp, "model as number")
        _verify_still_alive(client)


# ---- Oversized / Edge Payloads ----

class TestEdgePayloads:
    def test_very_long_prompt(self, client):
        """Send a prompt that exceeds typical context. Should truncate or reject, not crash."""
        long_content = "hello " * 50_000  # ~300KB
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": long_content}],
            "max_tokens": 10,
        })
        # May succeed (auto_truncate) or 400/422; just not 500
        assert resp.status_code != 500, f"Very long prompt caused 500: {resp.text[:200]}"
        _verify_still_alive(client)

    def test_many_messages(self, client):
        """Send many messages. Should handle gracefully."""
        messages = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i}"}
                    for i in range(500)]
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 10,
        })
        assert resp.status_code != 500, f"500 messages caused 500: {resp.text[:200]}"
        _verify_still_alive(client)


# ---- Wrong Endpoints ----

class TestWrongEndpoints:
    def test_nonexistent_endpoint(self, client):
        resp = client.get("/v1/nonexistent")
        assert resp.status_code in {404, 405}
        _verify_still_alive(client)

    def test_get_on_post_endpoint(self, client):
        resp = client.get("/v1/chat/completions")
        assert resp.status_code in {405, 404, 422}
        _verify_still_alive(client)


# ---- Rapid Fire After Errors ----

class TestRecovery:
    def test_valid_request_after_errors(self, client):
        """After sending several bad requests, a valid request should still work."""
        # Fire off a few bad requests
        for _ in range(5):
            client.post("/v1/chat/completions", content="bad", headers={"Content-Type": "application/json"})

        # Now send a valid one
        resp = client.post("/v1/chat/completions", json={
            "model": MODEL,
            "messages": [{"role": "user", "content": "[STATE] altitude=100m [CMD] status check"}],
            "max_tokens": 64,
        })
        assert resp.status_code == 200, f"Valid request failed after error barrage: {resp.status_code}: {resp.text[:200]}"


# -- standalone entry point --

def _run_standalone() -> int:
    """Run all tests without pytest, print results."""
    client = httpx.Client(base_url=EDGE_URL, timeout=15)
    passed = 0
    failed = 0
    errors = []

    # Collect test methods from all test classes
    test_classes = [TestBadJSON, TestMissingFields, TestInvalidValues, TestEdgePayloads, TestWrongEndpoints, TestRecovery]

    for cls in test_classes:
        instance = cls()
        methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in sorted(methods):
            method = getattr(instance, method_name)
            name = f"{cls.__name__}.{method_name}"
            try:
                method(client)
                passed += 1
                print(f"  PASS  {name}")
            except AssertionError as e:
                failed += 1
                errors.append((name, str(e)))
                print(f"  FAIL  {name}: {e}")
            except Exception as e:
                failed += 1
                errors.append((name, str(e)))
                print(f"  ERROR {name}: {e}")

    client.close()
    print(f"\n{passed} passed, {failed} failed")
    if errors:
        print("\nFailures:")
        for name, msg in errors:
            print(f"  {name}: {msg}")
    return 1 if failed else 0


if __name__ == "__main__":
    print("Edge runtime error tests (ROW-79)")
    print(f"Target: {EDGE_URL}")
    print(f"Model: {MODEL}\n")
    sys.exit(_run_standalone())
