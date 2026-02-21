"""LlamaFarm SDK — main client, project handles, and sub-API classes."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Iterator

import httpx

from .errors import ConnectionError, _raise_for_status
from .types import (
    CacheStats,
    ChatMessage,
    ChatResponse,
    ClassifyResponse,
    DatabaseInfo,
    DetectResponse,
    FineTuneJob,
    FineTuneTemplate,
    ProjectInfo,
    QueryResponse,
    StreamChunk,
    build_project_config,
)

__all__ = [
    "LlamaFarm", "ProjectHandle", "RAGAPI", "DatasetsAPI", "VisionAPI", "FineTuneAPI", "CacheAPI",
    "NLPAPI", "AnomalyAPI", "ClassifierAPI", "TimeSeriesAPI", "ADTKAPI", "CatBoostAPI",
    "DriftAPI", "ExplainAPI", "PolarsAPI",
]

_DEFAULT_URL = "http://localhost:14345"
_DEFAULT_RUNTIME_URL = "http://localhost:11540"
_DEFAULT_TIMEOUT = 120.0


def _normalize_messages(messages: str | list[str | dict | ChatMessage]) -> list[dict[str, str]]:
    """Normalize various message formats into OpenAI-compatible dicts."""
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    result = []
    for m in messages:
        if isinstance(m, str):
            result.append({"role": "user", "content": m})
        elif isinstance(m, ChatMessage):
            result.append(m.model_dump(exclude_none=True))
        elif isinstance(m, dict):
            result.append(m)
        else:
            raise TypeError(f"Unsupported message type: {type(m)}")
    return result


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Parse a single SSE data line into a dict, or None for non-data/done lines."""
    line = line.strip()
    if not line or not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if data == "[DONE]":
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


def _discover_url(env_var: str, default: str) -> str:
    """Discover server URL from env vars or config file."""
    val = os.environ.get(env_var)
    if val:
        return val.rstrip("/")
    config_path = Path.home() / ".llamafarm" / "config.json"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            key = "url" if env_var == "LLAMAFARM_URL" else "runtime_url"
            if key in cfg:
                return cfg[key].rstrip("/")
        except Exception:
            pass
    return default


# ── Sub-APIs ─────────────────────────────────────────────────────────────────


class RAGAPI:
    """RAG operations scoped to a project."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base: str):
        self._client = client
        self._aclient = aclient
        self._base = base  # e.g. /projects/{ns}/{pid}/rag

    def query(
        self,
        query: str,
        *,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filters: dict[str, Any] | None = None,
        hybrid_alpha: float | None = None,
        rerank_model: str | None = None,
        query_expansion: bool | None = None,
    ) -> QueryResponse:
        """Search project knowledge bases."""
        body: dict[str, Any] = {"query": query}
        for k, v in {
            "database": database, "retrieval_strategy": retrieval_strategy,
            "top_k": top_k, "score_threshold": score_threshold,
            "metadata_filters": metadata_filters, "hybrid_alpha": hybrid_alpha,
            "rerank_model": rerank_model, "query_expansion": query_expansion,
        }.items():
            if v is not None:
                body[k] = v
        r = self._client.post(f"{self._base}/query", json=body)
        _raise_for_status(r)
        return QueryResponse.model_validate(r.json())

    async def aquery(self, query: str, **kwargs: Any) -> QueryResponse:
        """Async variant of query."""
        body: dict[str, Any] = {"query": query}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        r = await self._aclient.post(f"{self._base}/query", json=body)
        _raise_for_status(r)
        return QueryResponse.model_validate(r.json())

    def health(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/health")
        _raise_for_status(r)
        return r.json()

    async def ahealth(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/health")
        _raise_for_status(r)
        return r.json()

    def stats(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/stats")
        _raise_for_status(r)
        return r.json()

    async def astats(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/stats")
        _raise_for_status(r)
        return r.json()

    def databases(self) -> list[DatabaseInfo]:
        r = self._client.get(f"{self._base}/databases")
        _raise_for_status(r)
        return [DatabaseInfo.model_validate(d) for d in r.json()]

    async def adatabases(self) -> list[DatabaseInfo]:
        r = await self._aclient.get(f"{self._base}/databases")
        _raise_for_status(r)
        return [DatabaseInfo.model_validate(d) for d in r.json()]

    def create_database(self, name: str, type: str = "ChromaStore", **kwargs: Any) -> dict[str, Any]:
        body = {"name": name, "type": type, **kwargs}
        r = self._client.post(f"{self._base}/databases", json=body)
        _raise_for_status(r)
        return r.json()

    async def acreate_database(self, name: str, type: str = "ChromaStore", **kwargs: Any) -> dict[str, Any]:
        body = {"name": name, "type": type, **kwargs}
        r = await self._aclient.post(f"{self._base}/databases", json=body)
        _raise_for_status(r)
        return r.json()

    def documents(self) -> list[dict[str, Any]]:
        r = self._client.get(f"{self._base}/documents")
        _raise_for_status(r)
        return r.json()

    async def adocuments(self) -> list[dict[str, Any]]:
        r = await self._aclient.get(f"{self._base}/documents")
        _raise_for_status(r)
        return r.json()


class DatasetsAPI:
    """Dataset operations scoped to a project."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base: str):
        self._client = client
        self._aclient = aclient
        self._base = base  # /projects/{ns}/{pid}/datasets

    def list(self) -> list[dict[str, Any]]:
        r = self._client.get(self._base)
        _raise_for_status(r)
        return r.json()

    async def alist(self) -> list[dict[str, Any]]:
        r = await self._aclient.get(self._base)
        _raise_for_status(r)
        return r.json()

    def upload(self, dataset_name: str, file_path: str | Path) -> dict[str, Any]:
        """Upload a file to a dataset."""
        fp = Path(file_path)
        with fp.open("rb") as f:
            r = self._client.post(
                f"{self._base}/{dataset_name}/upload",
                files={"file": (fp.name, f)},
            )
        _raise_for_status(r)
        return r.json()

    async def aupload(self, dataset_name: str, file_path: str | Path) -> dict[str, Any]:
        fp = Path(file_path)
        with fp.open("rb") as f:
            r = await self._aclient.post(
                f"{self._base}/{dataset_name}/upload",
                files={"file": (fp.name, f)},
            )
        _raise_for_status(r)
        return r.json()

    def delete_file(self, dataset_name: str, file_id: str) -> dict[str, Any]:
        r = self._client.delete(f"{self._base}/{dataset_name}/{file_id}")
        _raise_for_status(r)
        return r.json()

    async def adelete_file(self, dataset_name: str, file_id: str) -> dict[str, Any]:
        r = await self._aclient.delete(f"{self._base}/{dataset_name}/{file_id}")
        _raise_for_status(r)
        return r.json()


class VisionAPI:
    """Vision operations (server-level, proxied through main server)."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/vision"

    def _image_payload(self, image: str | Path, **extra: Any) -> tuple[dict | None, dict | None]:
        """Build either JSON or multipart payload depending on image type."""
        p = Path(image) if not str(image).startswith(("http://", "https://")) else None
        if p and p.exists():
            return None, None  # caller should use files=
        return {"image": str(image), **extra}, None

    def detect(self, image: str | Path, *, model: str | None = None, confidence: float | None = None) -> DetectResponse:
        body: dict[str, Any] = {}
        if model:
            body["model"] = model
        if confidence is not None:
            body["confidence"] = confidence
        p = Path(image) if not str(image).startswith(("http://", "https://")) else None
        if p and p.exists():
            with p.open("rb") as f:
                r = self._client.post(f"{self._base}/detect", data=body, files={"image": (p.name, f)})
        else:
            body["image"] = str(image)
            r = self._client.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return DetectResponse.model_validate(r.json())

    async def adetect(self, image: str | Path, **kwargs: Any) -> DetectResponse:
        body: dict[str, Any] = {"image": str(image)}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        r = await self._aclient.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return DetectResponse.model_validate(r.json())

    def classify(self, image: str | Path, classes: list[str], **kwargs: Any) -> ClassifyResponse:
        body: dict[str, Any] = {"image": str(image), "classes": classes}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        r = self._client.post(f"{self._base}/classify", json=body)
        _raise_for_status(r)
        return ClassifyResponse.model_validate(r.json())

    async def aclassify(self, image: str | Path, classes: list[str], **kwargs: Any) -> ClassifyResponse:
        body: dict[str, Any] = {"image": str(image), "classes": classes}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        r = await self._aclient.post(f"{self._base}/classify", json=body)
        _raise_for_status(r)
        return ClassifyResponse.model_validate(r.json())

    def detect_classify(self, image: str | Path, classes: list[str], **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"image": str(image), "classes": classes}
        body.update({k: v for k, v in kwargs.items() if v is not None})
        r = self._client.post(f"{self._base}/detect_classify", json=body)
        _raise_for_status(r)
        return r.json()

    def train(self, model: str, dataset: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model": model, "dataset": dataset, **kwargs}
        r = self._client.post(f"{self._base}/train", json=body)
        _raise_for_status(r)
        return r.json()


class FineTuneAPI:
    """Fine-tuning operations (runtime-level, port 11540)."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, runtime_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{runtime_url}/v1/finetune"

    def sft(self, model: str, dataset: str, **kwargs: Any) -> FineTuneJob:
        body: dict[str, Any] = {"model": model, "dataset": dataset, **kwargs}
        r = self._client.post(f"{self._base}/sft", json=body)
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    async def asft(self, model: str, dataset: str, **kwargs: Any) -> FineTuneJob:
        body: dict[str, Any] = {"model": model, "dataset": dataset, **kwargs}
        r = await self._aclient.post(f"{self._base}/sft", json=body)
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    def cpt(self, model: str, dataset: str, **kwargs: Any) -> FineTuneJob:
        body: dict[str, Any] = {"model": model, "dataset": dataset, **kwargs}
        r = self._client.post(f"{self._base}/cpt", json=body)
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    async def acpt(self, model: str, dataset: str, **kwargs: Any) -> FineTuneJob:
        body: dict[str, Any] = {"model": model, "dataset": dataset, **kwargs}
        r = await self._aclient.post(f"{self._base}/cpt", json=body)
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    def status(self, job_id: str) -> FineTuneJob:
        r = self._client.get(f"{self._base}/jobs/{job_id}")
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    async def astatus(self, job_id: str) -> FineTuneJob:
        r = await self._aclient.get(f"{self._base}/jobs/{job_id}")
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    def jobs(self) -> list[FineTuneJob]:
        r = self._client.get(f"{self._base}/jobs")
        _raise_for_status(r)
        return [FineTuneJob.model_validate(j) for j in r.json()]

    async def ajobs(self) -> list[FineTuneJob]:
        r = await self._aclient.get(f"{self._base}/jobs")
        _raise_for_status(r)
        return [FineTuneJob.model_validate(j) for j in r.json()]

    def cancel(self, job_id: str) -> FineTuneJob:
        r = self._client.delete(f"{self._base}/jobs/{job_id}")
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    async def acancel(self, job_id: str) -> FineTuneJob:
        r = await self._aclient.delete(f"{self._base}/jobs/{job_id}")
        _raise_for_status(r)
        return FineTuneJob.model_validate(r.json())

    def validate(self, dataset: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"dataset": dataset, **kwargs}
        r = self._client.post(f"{self._base}/validate", json=body)
        _raise_for_status(r)
        return r.json()

    async def avalidate(self, dataset: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"dataset": dataset, **kwargs}
        r = await self._aclient.post(f"{self._base}/validate", json=body)
        _raise_for_status(r)
        return r.json()

    def templates(self) -> list[FineTuneTemplate]:
        r = self._client.get(f"{self._base}/templates")
        _raise_for_status(r)
        return [FineTuneTemplate.model_validate(t) for t in r.json()]

    async def atemplates(self) -> list[FineTuneTemplate]:
        r = await self._aclient.get(f"{self._base}/templates")
        _raise_for_status(r)
        return [FineTuneTemplate.model_validate(t) for t in r.json()]


class NLPAPI:
    """NLP operations: embeddings, classification, NER, reranking."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str, runtime_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/nlp"
        self._runtime = runtime_url

    def embeddings(self, input: str | list[str], *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"input": input, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/embeddings", json=body)
        _raise_for_status(r)
        return r.json()

    async def aembeddings(self, input: str | list[str], *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"input": input, **kwargs}
        if model:
            body["model"] = model
        r = await self._aclient.post(f"{self._base}/embeddings", json=body)
        _raise_for_status(r)
        return r.json()

    def classify(self, text: str, *, labels: list[str] | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, **kwargs}
        if labels:
            body["labels"] = labels
        r = self._client.post(f"{self._base}/classify", json=body)
        _raise_for_status(r)
        return r.json()

    async def aclassify(self, text: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, **kwargs}
        r = await self._aclient.post(f"{self._base}/classify", json=body)
        _raise_for_status(r)
        return r.json()

    def ner(self, text: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, **kwargs}
        r = self._client.post(f"{self._base}/ner", json=body)
        _raise_for_status(r)
        return r.json()

    async def aner(self, text: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text, **kwargs}
        r = await self._aclient.post(f"{self._base}/ner", json=body)
        _raise_for_status(r)
        return r.json()

    def rerank(self, query: str, documents: list[str], *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "documents": documents, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/rerank", json=body)
        _raise_for_status(r)
        return r.json()

    async def arerank(self, query: str, documents: list[str], **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"query": query, "documents": documents, **kwargs}
        r = await self._aclient.post(f"{self._base}/rerank", json=body)
        _raise_for_status(r)
        return r.json()


class AnomalyAPI:
    """Anomaly detection operations."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/ml/anomaly"

    def fit(self, data: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    async def afit(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    def detect(self, data: list, *, model: str | None = None, explain: bool = False, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        if explain:
            body["explain"] = True
        r = self._client.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return r.json()

    async def adetect(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return r.json()

    def score(self, data: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/score", json=body)
        _raise_for_status(r)
        return r.json()

    async def ascore(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/score", json=body)
        _raise_for_status(r)
        return r.json()

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    async def aload(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    def save(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/save", json=body)
        _raise_for_status(r)
        return r.json()

    async def asave(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/save", json=body)
        _raise_for_status(r)
        return r.json()

    def models(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    async def amodels(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    def backends(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/backends")
        _raise_for_status(r)
        return r.json()

    async def abackends(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/backends")
        _raise_for_status(r)
        return r.json()

    def stream(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = self._client.post(f"{self._base}/stream", json=body)
        _raise_for_status(r)
        return r.json()

    async def astream(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/stream", json=body)
        _raise_for_status(r)
        return r.json()

    def stream_status(self, model_id: str) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/stream/{model_id}")
        _raise_for_status(r)
        return r.json()

    async def astream_status(self, model_id: str) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/stream/{model_id}")
        _raise_for_status(r)
        return r.json()


class ClassifierAPI:
    """ML classifier operations."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/ml/classifier"

    def fit(self, data: list, labels: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, "labels": labels, **kwargs}
        r = self._client.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    async def afit(self, data: list, labels: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, "labels": labels, **kwargs}
        r = await self._aclient.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    def predict(self, data: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/predict", json=body)
        _raise_for_status(r)
        return r.json()

    async def apredict(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/predict", json=body)
        _raise_for_status(r)
        return r.json()

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    async def aload(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    def save(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/save", json=body)
        _raise_for_status(r)
        return r.json()

    async def asave(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/save", json=body)
        _raise_for_status(r)
        return r.json()

    def models(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    async def amodels(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()


class TimeSeriesAPI:
    """Time series forecasting operations."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/timeseries"

    def fit(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = self._client.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    async def afit(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    def predict(self, *, model: str | None = None, horizon: int | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {**kwargs}
        if model:
            body["model"] = model
        if horizon is not None:
            body["horizon"] = horizon
        r = self._client.post(f"{self._base}/predict", json=body)
        _raise_for_status(r)
        return r.json()

    async def apredict(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {**kwargs}
        r = await self._aclient.post(f"{self._base}/predict", json=body)
        _raise_for_status(r)
        return r.json()

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    async def aload(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    def models(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    async def amodels(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    def backends(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/backends")
        _raise_for_status(r)
        return r.json()

    async def abackends(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/backends")
        _raise_for_status(r)
        return r.json()


class ADTKAPI:
    """ADTK anomaly detection toolkit operations."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/adtk"

    def fit(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = self._client.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    async def afit(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    def detect(self, data: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return r.json()

    async def adetect(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return r.json()

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    async def aload(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    def models(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    async def amodels(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    def detectors(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/detectors")
        _raise_for_status(r)
        return r.json()

    async def adetectors(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/detectors")
        _raise_for_status(r)
        return r.json()


class CatBoostAPI:
    """CatBoost ML operations."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/catboost"

    def fit(self, data: list, labels: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, "labels": labels, **kwargs}
        r = self._client.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    async def afit(self, data: list, labels: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, "labels": labels, **kwargs}
        r = await self._aclient.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    def predict(self, data: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/predict", json=body)
        _raise_for_status(r)
        return r.json()

    async def apredict(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/predict", json=body)
        _raise_for_status(r)
        return r.json()

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    async def aload(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    def update(self, data: list, labels: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, "labels": labels, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/update", json=body)
        _raise_for_status(r)
        return r.json()

    async def aupdate(self, data: list, labels: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, "labels": labels, **kwargs}
        r = await self._aclient.post(f"{self._base}/update", json=body)
        _raise_for_status(r)
        return r.json()

    def models(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    async def amodels(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    def info(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/info")
        _raise_for_status(r)
        return r.json()

    async def ainfo(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/info")
        _raise_for_status(r)
        return r.json()

    def importance(self, model_id: str) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/{model_id}/importance")
        _raise_for_status(r)
        return r.json()

    async def aimportance(self, model_id: str) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/{model_id}/importance")
        _raise_for_status(r)
        return r.json()


class DriftAPI:
    """Drift detection operations."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/drift"

    def fit(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = self._client.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    async def afit(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/fit", json=body)
        _raise_for_status(r)
        return r.json()

    def detect(self, data: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return r.json()

    async def adetect(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/detect", json=body)
        _raise_for_status(r)
        return r.json()

    def load(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = self._client.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    async def aload(self, model_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"model_id": model_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/load", json=body)
        _raise_for_status(r)
        return r.json()

    def models(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    async def amodels(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/models")
        _raise_for_status(r)
        return r.json()

    def detectors(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/detectors")
        _raise_for_status(r)
        return r.json()

    async def adetectors(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/detectors")
        _raise_for_status(r)
        return r.json()

    def status(self, model_name: str) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/status/{model_name}")
        _raise_for_status(r)
        return r.json()

    async def astatus(self, model_name: str) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/status/{model_name}")
        _raise_for_status(r)
        return r.json()


class ExplainAPI:
    """Explainability operations (SHAP, feature importance)."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/explain"

    def shap(self, data: list, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/shap", json=body)
        _raise_for_status(r)
        return r.json()

    async def ashap(self, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/shap", json=body)
        _raise_for_status(r)
        return r.json()

    def importance(self, *, model: str | None = None, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {**kwargs}
        if model:
            body["model"] = model
        r = self._client.post(f"{self._base}/importance", json=body)
        _raise_for_status(r)
        return r.json()

    async def aimportance(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {**kwargs}
        r = await self._aclient.post(f"{self._base}/importance", json=body)
        _raise_for_status(r)
        return r.json()

    def explainers(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/explainers")
        _raise_for_status(r)
        return r.json()

    async def aexplainers(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/explainers")
        _raise_for_status(r)
        return r.json()


class PolarsAPI:
    """Polars data processing buffer operations."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, base_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{base_url}/ml/polars"

    def create_buffer(self, **kwargs: Any) -> dict[str, Any]:
        r = self._client.post(f"{self._base}/buffers", json=kwargs)
        _raise_for_status(r)
        return r.json()

    async def acreate_buffer(self, **kwargs: Any) -> dict[str, Any]:
        r = await self._aclient.post(f"{self._base}/buffers", json=kwargs)
        _raise_for_status(r)
        return r.json()

    def append(self, buffer_id: str, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"buffer_id": buffer_id, "data": data, **kwargs}
        r = self._client.post(f"{self._base}/append", json=body)
        _raise_for_status(r)
        return r.json()

    async def aappend(self, buffer_id: str, data: list, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"buffer_id": buffer_id, "data": data, **kwargs}
        r = await self._aclient.post(f"{self._base}/append", json=body)
        _raise_for_status(r)
        return r.json()

    def features(self, buffer_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"buffer_id": buffer_id, **kwargs}
        r = self._client.post(f"{self._base}/features", json=body)
        _raise_for_status(r)
        return r.json()

    async def afeatures(self, buffer_id: str, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"buffer_id": buffer_id, **kwargs}
        r = await self._aclient.post(f"{self._base}/features", json=body)
        _raise_for_status(r)
        return r.json()

    def list_buffers(self) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/buffers")
        _raise_for_status(r)
        return r.json()

    async def alist_buffers(self) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/buffers")
        _raise_for_status(r)
        return r.json()

    def get_buffer(self, buffer_id: str) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/buffers/{buffer_id}")
        _raise_for_status(r)
        return r.json()

    async def aget_buffer(self, buffer_id: str) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/buffers/{buffer_id}")
        _raise_for_status(r)
        return r.json()

    def get_buffer_data(self, buffer_id: str) -> dict[str, Any]:
        r = self._client.get(f"{self._base}/buffers/{buffer_id}/data")
        _raise_for_status(r)
        return r.json()

    async def aget_buffer_data(self, buffer_id: str) -> dict[str, Any]:
        r = await self._aclient.get(f"{self._base}/buffers/{buffer_id}/data")
        _raise_for_status(r)
        return r.json()


class CacheAPI:
    """KV cache operations (runtime-level, port 11540)."""

    def __init__(self, client: httpx.Client, aclient: httpx.AsyncClient, runtime_url: str):
        self._client = client
        self._aclient = aclient
        self._base = f"{runtime_url}/v1/cache"

    def prepare(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, **kwargs}
        r = self._client.post(f"{self._base}/prepare", json=body)
        _raise_for_status(r)
        return r.json()

    async def aprepare(self, messages: list[dict[str, str]], **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, **kwargs}
        r = await self._aclient.post(f"{self._base}/prepare", json=body)
        _raise_for_status(r)
        return r.json()

    def stats(self) -> CacheStats:
        r = self._client.get(f"{self._base}/stats")
        _raise_for_status(r)
        return CacheStats.model_validate(r.json())

    async def astats(self) -> CacheStats:
        r = await self._aclient.get(f"{self._base}/stats")
        _raise_for_status(r)
        return CacheStats.model_validate(r.json())

    def gc(self) -> dict[str, Any]:
        r = self._client.post(f"{self._base}/gc")
        _raise_for_status(r)
        return r.json()

    async def agc(self) -> dict[str, Any]:
        r = await self._aclient.post(f"{self._base}/gc")
        _raise_for_status(r)
        return r.json()

    def clear(self, key: str) -> dict[str, Any]:
        r = self._client.delete(f"{self._base}/{key}")
        _raise_for_status(r)
        return r.json()

    async def aclear(self, key: str) -> dict[str, Any]:
        r = await self._aclient.delete(f"{self._base}/{key}")
        _raise_for_status(r)
        return r.json()


# ── ProjectHandle ────────────────────────────────────────────────────────────


class ProjectHandle:
    """Handle to a specific project, providing chat, RAG, and dataset operations."""

    def __init__(
        self,
        client: httpx.Client,
        aclient: httpx.AsyncClient,
        base_url: str,
        namespace: str,
        project_id: str,
    ):
        self._client = client
        self._aclient = aclient
        self._base_url = base_url
        self.namespace = namespace
        self.project_id = project_id
        self._path = f"{base_url}/projects/{namespace}/{project_id}"

    @property
    def rag(self) -> RAGAPI:
        return RAGAPI(self._client, self._aclient, f"{self._path}/rag")

    @property
    def datasets(self) -> DatasetsAPI:
        return DatasetsAPI(self._client, self._aclient, f"{self._path}/datasets")

    def info(self) -> ProjectInfo:
        """Get project info."""
        r = self._client.get(self._path)
        _raise_for_status(r)
        return ProjectInfo.model_validate(r.json())

    async def ainfo(self) -> ProjectInfo:
        r = await self._aclient.get(self._path)
        _raise_for_status(r)
        return ProjectInfo.model_validate(r.json())

    def update_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Update the project configuration."""
        r = self._client.put(self._path, json=config)
        _raise_for_status(r)
        return r.json()

    async def aupdate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        r = await self._aclient.put(self._path, json=config)
        _raise_for_status(r)
        return r.json()

    def delete(self) -> dict[str, Any]:
        """Delete this project."""
        r = self._client.delete(self._path)
        _raise_for_status(r)
        return r.json()

    async def adelete(self) -> dict[str, Any]:
        r = await self._aclient.delete(self._path)
        _raise_for_status(r)
        return r.json()

    def chat(
        self,
        messages: str | list[str | dict | ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        stateless: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            messages: A string, list of strings, or list of message dicts/ChatMessage objects.
            model: Override the project's default model.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            tools: Tool definitions for function calling.
            session_id: Session ID for conversation continuity.
            stateless: If True, don't create/use sessions.
        """
        body: dict[str, Any] = {"messages": _normalize_messages(messages), "stream": False}
        for k, v in {"model": model, "temperature": temperature, "max_tokens": max_tokens, "tools": tools}.items():
            if v is not None:
                body[k] = v
        body.update(kwargs)
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id
        if stateless:
            headers["X-No-Session"] = "true"
        r = self._client.post(f"{self._path}/chat/completions", json=body, headers=headers)
        _raise_for_status(r)
        return ChatResponse.model_validate(r.json())

    async def achat(
        self,
        messages: str | list[str | dict | ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        tools: list[dict[str, Any]] | None = None,
        session_id: str | None = None,
        stateless: bool = False,
        **kwargs: Any,
    ) -> ChatResponse:
        """Async variant of chat."""
        body: dict[str, Any] = {"messages": _normalize_messages(messages), "stream": False}
        for k, v in {"model": model, "temperature": temperature, "max_tokens": max_tokens, "tools": tools}.items():
            if v is not None:
                body[k] = v
        body.update(kwargs)
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id
        if stateless:
            headers["X-No-Session"] = "true"
        r = await self._aclient.post(f"{self._path}/chat/completions", json=body, headers=headers)
        _raise_for_status(r)
        return ChatResponse.model_validate(r.json())

    def chat_stream(
        self,
        messages: str | list[str | dict | ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        stateless: bool = False,
        **kwargs: Any,
    ) -> Iterator[StreamChunk]:
        """Stream a chat completion, yielding chunks."""
        body: dict[str, Any] = {"messages": _normalize_messages(messages), "stream": True}
        for k, v in {"model": model, "temperature": temperature, "max_tokens": max_tokens}.items():
            if v is not None:
                body[k] = v
        body.update(kwargs)
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id
        if stateless:
            headers["X-No-Session"] = "true"
        with self._client.stream("POST", f"{self._path}/chat/completions", json=body, headers=headers) as r:
            _raise_for_status(r)
            for line in r.iter_lines():
                parsed = _parse_sse_line(line)
                if parsed is not None:
                    yield StreamChunk.model_validate(parsed)

    async def achat_stream(
        self,
        messages: str | list[str | dict | ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        session_id: str | None = None,
        stateless: bool = False,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Async stream a chat completion."""
        body: dict[str, Any] = {"messages": _normalize_messages(messages), "stream": True}
        for k, v in {"model": model, "temperature": temperature, "max_tokens": max_tokens}.items():
            if v is not None:
                body[k] = v
        body.update(kwargs)
        headers: dict[str, str] = {}
        if session_id:
            headers["X-Session-ID"] = session_id
        if stateless:
            headers["X-No-Session"] = "true"
        async with self._aclient.stream("POST", f"{self._path}/chat/completions", json=body, headers=headers) as r:
            _raise_for_status(r)
            async for line in r.aiter_lines():
                parsed = _parse_sse_line(line)
                if parsed is not None:
                    yield StreamChunk.model_validate(parsed)

    def __repr__(self) -> str:
        return f"ProjectHandle({self.namespace}/{self.project_id})"


# ── Main Client ──────────────────────────────────────────────────────────────


class LlamaFarm:
    """LlamaFarm SDK client.

    Args:
        url: Main server URL (default: auto-discover or http://localhost:14345).
        api_key: API key for authentication (optional).
        runtime_url: Universal Runtime URL for fine-tuning/cache (default: http://localhost:11540).
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        api_key: str | None = None,
        runtime_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ):
        self._raw_url = (url or _discover_url("LLAMAFARM_URL", _DEFAULT_URL)).rstrip("/")
        self.url = f"{self._raw_url}/v1"
        self.runtime_url = (runtime_url or _discover_url("LLAMAFARM_RUNTIME_URL", _DEFAULT_RUNTIME_URL)).rstrip("/")

        headers: dict[str, str] = {}
        api_key = api_key or os.environ.get("LLAMAFARM_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.Client(base_url="", headers=headers, timeout=timeout)
        self._aclient = httpx.AsyncClient(base_url="", headers=headers, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        await self._aclient.aclose()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.aclose()

    # ── Top-level operations ─────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Check main server health."""
        try:
            r = self._client.get(f"{self._raw_url}/health")
            return r.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to LlamaFarm at {self._raw_url}") from e

    async def ahealth(self) -> dict[str, Any]:
        try:
            r = await self._aclient.get(f"{self._raw_url}/health")
            return r.json()
        except httpx.ConnectError as e:
            raise ConnectionError(f"Cannot connect to LlamaFarm at {self._raw_url}") from e

    def models(self) -> list[dict[str, Any]]:
        """List available models."""
        r = self._client.get(f"{self.url}/models")
        _raise_for_status(r)
        return r.json()

    async def amodels(self) -> list[dict[str, Any]]:
        r = await self._aclient.get(f"{self.url}/models")
        _raise_for_status(r)
        return r.json()

    # ── Project operations ───────────────────────────────────────────────

    def projects(self, namespace: str = "default") -> list[ProjectInfo]:
        """List projects in a namespace."""
        r = self._client.get(f"{self.url}/projects/{namespace}")
        _raise_for_status(r)
        data = r.json()
        items = data.get("projects", data) if isinstance(data, dict) else data
        return [ProjectInfo.model_validate(p) for p in items]

    async def aprojects(self, namespace: str = "default") -> list[ProjectInfo]:
        r = await self._aclient.get(f"{self.url}/projects/{namespace}")
        _raise_for_status(r)
        data = r.json()
        items = data.get("projects", data) if isinstance(data, dict) else data
        return [ProjectInfo.model_validate(p) for p in items]

    def project(self, namespace: str, project_id: str) -> ProjectHandle:
        """Get a handle to an existing project (no API call — lazy)."""
        return ProjectHandle(self._client, self._aclient, self.url, namespace, project_id)

    def create_project(
        self,
        namespace: str,
        name: str,
        *,
        model: str = "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
        provider: str = "universal",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        config_template: dict[str, Any] | None = None,
    ) -> ProjectHandle:
        """Create a new project and return a handle.

        If config_template is provided, it's used directly. Otherwise, a config
        is generated from the convenience parameters (model, system_prompt, etc.).
        """
        if config_template is None:
            config_template = build_project_config(
                name, namespace,
                model=model, provider=provider, system_prompt=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
        body = {"name": name, "config_template": config_template}
        r = self._client.post(f"{self.url}/projects/{namespace}", json=body)
        _raise_for_status(r)
        data = r.json()
        pid = data.get("id", data.get("project_id", name))
        return ProjectHandle(self._client, self._aclient, self.url, namespace, pid)

    async def acreate_project(
        self,
        namespace: str,
        name: str,
        *,
        model: str = "unsloth/Qwen3-1.7B-GGUF:Q4_K_M",
        provider: str = "universal",
        system_prompt: str = "You are a helpful assistant.",
        temperature: float | None = None,
        max_tokens: int | None = None,
        config_template: dict[str, Any] | None = None,
    ) -> ProjectHandle:
        if config_template is None:
            config_template = build_project_config(
                name, namespace,
                model=model, provider=provider, system_prompt=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
        body = {"name": name, "config_template": config_template}
        r = await self._aclient.post(f"{self.url}/projects/{namespace}", json=body)
        _raise_for_status(r)
        data = r.json()
        pid = data.get("id", data.get("project_id", name))
        return ProjectHandle(self._client, self._aclient, self.url, namespace, pid)

    # ── Sub-API accessors ────────────────────────────────────────────────

    @property
    def openai_base_url(self) -> str:
        """Return URL compatible with ``openai.OpenAI(base_url=...)``."""
        return f"{self.runtime_url}/v1"

    @property
    def vision(self) -> VisionAPI:
        return VisionAPI(self._client, self._aclient, self.url)

    @property
    def finetune(self) -> FineTuneAPI:
        return FineTuneAPI(self._client, self._aclient, self.runtime_url)

    @property
    def cache(self) -> CacheAPI:
        return CacheAPI(self._client, self._aclient, self.runtime_url)

    @property
    def nlp(self) -> NLPAPI:
        return NLPAPI(self._client, self._aclient, self.url, self.runtime_url)

    @property
    def anomaly(self) -> AnomalyAPI:
        return AnomalyAPI(self._client, self._aclient, self.url)

    @property
    def classifier(self) -> ClassifierAPI:
        return ClassifierAPI(self._client, self._aclient, self.url)

    @property
    def timeseries(self) -> TimeSeriesAPI:
        return TimeSeriesAPI(self._client, self._aclient, self.url)

    @property
    def adtk(self) -> ADTKAPI:
        return ADTKAPI(self._client, self._aclient, self.url)

    @property
    def catboost(self) -> CatBoostAPI:
        return CatBoostAPI(self._client, self._aclient, self.url)

    @property
    def drift(self) -> DriftAPI:
        return DriftAPI(self._client, self._aclient, self.url)

    @property
    def explain(self) -> ExplainAPI:
        return ExplainAPI(self._client, self._aclient, self.url)

    @property
    def polars(self) -> PolarsAPI:
        return PolarsAPI(self._client, self._aclient, self.url)

    def __repr__(self) -> str:
        return f"LlamaFarm(url={self.url!r})"
