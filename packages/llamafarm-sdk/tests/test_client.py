"""Tests for LlamaFarm SDK client."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import httpx
import pytest
import respx

from llamafarm import LlamaFarm, build_project_config
from llamafarm.client import _normalize_messages, _parse_sse_line, ProjectHandle
from llamafarm.types import ChatMessage


# ── Message normalization ────────────────────────────────────────────────────


class TestNormalizeMessages:
    def test_string(self):
        assert _normalize_messages("hello") == [{"role": "user", "content": "hello"}]

    def test_list_of_strings(self):
        result = _normalize_messages(["hi", "there"])
        assert result == [{"role": "user", "content": "hi"}, {"role": "user", "content": "there"}]

    def test_list_of_dicts(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        assert _normalize_messages(msgs) == msgs

    def test_chat_message_objects(self):
        msgs = [ChatMessage(role="user", content="hello")]
        result = _normalize_messages(msgs)
        assert result == [{"role": "user", "content": "hello"}]

    def test_mixed(self):
        msgs = [{"role": "system", "content": "sys"}, "hello"]
        result = _normalize_messages(msgs)
        assert len(result) == 2
        assert result[1] == {"role": "user", "content": "hello"}


# ── SSE parsing ──────────────────────────────────────────────────────────────


class TestSSEParsing:
    def test_data_line(self):
        data = {"id": "1", "choices": [{"delta": {"content": "hi"}}]}
        result = _parse_sse_line(f"data: {json.dumps(data)}")
        assert result == data

    def test_done(self):
        assert _parse_sse_line("data: [DONE]") is None

    def test_empty(self):
        assert _parse_sse_line("") is None
        assert _parse_sse_line("   ") is None

    def test_non_data(self):
        assert _parse_sse_line("event: ping") is None

    def test_invalid_json(self):
        assert _parse_sse_line("data: {bad json") is None


# ── Auto-discovery ───────────────────────────────────────────────────────────


class TestAutoDiscovery:
    def test_env_var(self):
        with patch.dict(os.environ, {"LLAMAFARM_URL": "http://custom:9999"}):
            lf = LlamaFarm()
            assert lf.url == "http://custom:9999/v1"

    def test_env_var_strips_trailing_slash(self):
        with patch.dict(os.environ, {"LLAMAFARM_URL": "http://custom:9999/"}):
            lf = LlamaFarm()
            assert lf.url == "http://custom:9999/v1"

    def test_explicit_url_wins(self):
        with patch.dict(os.environ, {"LLAMAFARM_URL": "http://env:1111"}):
            lf = LlamaFarm(url="http://explicit:2222")
            assert lf.url == "http://explicit:2222/v1"

    def test_default_urls(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("LLAMAFARM_URL", None)
            os.environ.pop("LLAMAFARM_RUNTIME_URL", None)
            lf = LlamaFarm()
            assert lf.url == "http://localhost:14345/v1"
            assert lf.runtime_url == "http://localhost:11540"


# ── ProjectHandle path building ──────────────────────────────────────────────


class TestProjectHandle:
    def test_path(self):
        client = httpx.Client()
        aclient = httpx.AsyncClient()
        ph = ProjectHandle(client, aclient, "http://localhost:14345/v1", "default", "my-project")
        assert ph._path == "http://localhost:14345/v1/projects/default/my-project"
        assert ph.namespace == "default"
        assert ph.project_id == "my-project"
        client.close()

    def test_repr(self):
        client = httpx.Client()
        aclient = httpx.AsyncClient()
        ph = ProjectHandle(client, aclient, "http://localhost:14345/v1", "ns", "pid")
        assert repr(ph) == "ProjectHandle(ns/pid)"
        client.close()


# ── Config generation ────────────────────────────────────────────────────────


class TestConfigGeneration:
    def test_basic(self):
        cfg = build_project_config("test-app", "default")
        assert cfg["version"] == "v1"
        assert cfg["name"] == "test-app"
        assert cfg["namespace"] == "default"
        assert cfg["runtime"]["models"][0]["provider"] == "universal"
        assert cfg["prompts"][0]["messages"][0]["role"] == "system"

    def test_custom_model(self):
        cfg = build_project_config("app", model="Qwen/Qwen3-8B", temperature=0.3)
        assert cfg["runtime"]["models"][0]["model"] == "Qwen/Qwen3-8B"
        assert cfg["runtime"]["models"][0]["temperature"] == 0.3

    def test_no_rag_by_default(self):
        cfg = build_project_config("app")
        assert "rag" not in cfg

    def test_with_rag(self):
        cfg = build_project_config("app", rag_databases=[{"name": "db", "type": "ChromaStore"}])
        assert cfg["rag"]["databases"][0]["name"] == "db"


# ── API calls with respx mocks ──────────────────────────────────────────────


class TestAPICallsMocked:
    @respx.mock
    def test_health(self):
        respx.get("http://localhost:14345/health").respond(json={"status": "ok"})
        lf = LlamaFarm()
        assert lf.health() == {"status": "ok"}

    @respx.mock
    def test_projects_list(self):
        respx.get("http://localhost:14345/v1/projects/default").respond(
            json=[{"id": "p1", "name": "Project 1", "namespace": "default"}]
        )
        lf = LlamaFarm()
        projects = lf.projects("default")
        assert len(projects) == 1
        assert projects[0].id == "p1"

    @respx.mock
    def test_create_project(self):
        respx.post("http://localhost:14345/v1/projects/default").respond(
            json={"id": "new-proj", "name": "My App"}
        )
        lf = LlamaFarm()
        ph = lf.create_project("default", "My App", model="Qwen/Qwen3-8B")
        assert ph.project_id == "new-proj"
        assert ph.namespace == "default"

    @respx.mock
    def test_chat(self):
        respx.post("http://localhost:14345/v1/projects/default/myproj/chat/completions").respond(
            json={
                "id": "chat-1",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }
        )
        lf = LlamaFarm()
        p = lf.project("default", "myproj")
        resp = p.chat("Hi")
        assert resp.text == "Hello!"
        assert resp.usage.total_tokens == 7

    @respx.mock
    def test_chat_with_session(self):
        route = respx.post("http://localhost:14345/v1/projects/default/myproj/chat/completions").respond(
            json={"id": "c1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}]}
        )
        lf = LlamaFarm()
        p = lf.project("default", "myproj")
        p.chat("Hi", session_id="sess-123")
        assert route.calls[0].request.headers["X-Session-ID"] == "sess-123"

    @respx.mock
    def test_rag_query(self):
        respx.post("http://localhost:14345/v1/projects/ns/proj/rag/query").respond(
            json={"results": [{"content": "answer", "score": 0.95}], "query": "q"}
        )
        lf = LlamaFarm()
        p = lf.project("ns", "proj")
        result = p.rag.query("q")
        assert result.results[0].content == "answer"

    @respx.mock
    def test_finetune_sft(self):
        respx.post("http://localhost:11540/v1/finetune/sft").respond(
            json={"job_id": "ft-1", "status": "queued", "model": "qwen", "method": "sft"}
        )
        lf = LlamaFarm()
        job = lf.finetune.sft("qwen", "dataset.jsonl")
        assert job.job_id == "ft-1"
        assert job.status == "queued"

    @respx.mock
    def test_cache_stats(self):
        respx.get("http://localhost:11540/v1/cache/stats").respond(
            json={"total_entries": 10, "total_size_bytes": 4096, "hit_rate": 0.85, "entries": []}
        )
        lf = LlamaFarm()
        stats = lf.cache.stats()
        assert stats.total_entries == 10
        assert stats.hit_rate == 0.85

    @respx.mock
    def test_404_raises_not_found(self):
        respx.get("http://localhost:14345/v1/projects/ns/missing").respond(
            status_code=404, json={"detail": "not found"}
        )
        from llamafarm.errors import NotFoundError
        lf = LlamaFarm()
        p = lf.project("ns", "missing")
        with pytest.raises(NotFoundError):
            p.info()


# ── Async tests ──────────────────────────────────────────────────────────────


class TestAsync:
    @respx.mock
    @pytest.mark.asyncio
    async def test_async_health(self):
        respx.get("http://localhost:14345/health").respond(json={"status": "ok"})
        async with LlamaFarm() as lf:
            result = await lf.ahealth()
            assert result == {"status": "ok"}

    @respx.mock
    @pytest.mark.asyncio
    async def test_async_chat(self):
        respx.post("http://localhost:14345/v1/projects/default/p/chat/completions").respond(
            json={"id": "c1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}}]}
        )
        async with LlamaFarm() as lf:
            p = lf.project("default", "p")
            resp = await p.achat("hello")
            assert resp.text == "hi"
