from typing import Any

from config.datamodel import LlamaFarmConfig
from fastapi.testclient import TestClient

from api.main import llama_farm_api


def _client() -> TestClient:
    app = llama_farm_api()
    return TestClient(app)


def _base_config(
    components: dict[str, Any] | None = None,
    rag: dict[str, Any] | None = None,
) -> LlamaFarmConfig:
    cfg_dict: dict[str, Any] = {
        "version": "v1",
        "name": "proj",
        "namespace": "ns",
        "components": components or {},
        "rag": rag or {},
        "runtime": {
            "models": [
                {
                    "name": "default",
                    "provider": "openai",
                    "model": "gpt-4o-mini",
                }
            ]
        },
        "datasets": [],
    }
    return LlamaFarmConfig(**cfg_dict)


def _patch_project_service(mocker, state: dict[str, Any]):
    """Patch ProjectService.load_config/save_config to use in-memory state."""
    mock_ps = mocker.patch("services.database_service.ProjectService")
    mock_ps.load_config.side_effect = lambda ns, pr: state["config"]
    mock_ps.save_config.side_effect = lambda ns, pr, cfg: state.__setitem__(
        "config", cfg
    )
    return mock_ps


def _default_components():
    return {
        "embedding_strategies": [
            {
                "name": "default_embeddings",
                "type": "UniversalEmbedder",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
            }
        ],
        "retrieval_strategies": [
            {
                "name": "semantic_search",
                "type": "BasicSimilarityStrategy",
                "config": {"top_k": 5},
            }
        ],
        "defaults": {
            "embedding_strategy": "default_embeddings",
            "retrieval_strategy": "semantic_search",
        },
    }


def test_create_database_with_reference_success(mocker):
    state = {"config": _base_config(components=_default_components(), rag={})}
    _patch_project_service(mocker, state)
    client = _client()

    resp = client.post(
        "/v1/projects/ns/proj/rag/databases",
        json={
            "name": "main_db",
            "type": "ChromaStore",
            "config": {},
            "embedding_strategy": "default_embeddings",
            "retrieval_strategy": "semantic_search",
        },
    )
    assert resp.status_code == 201, resp.text

    # GET returns fully resolved inline config
    get_resp = client.get("/v1/projects/ns/proj/rag/databases/main_db")
    assert get_resp.status_code == 200, get_resp.text
    data = get_resp.json()
    assert data["embedding_strategies"][0]["name"] == "default_embeddings"
    assert "embedding_strategy" not in data  # no reference string returned
    assert data["retrieval_strategies"][0]["name"] == "semantic_search"
    assert "retrieval_strategy" not in data


def test_create_database_with_defaults_success(mocker):
    state = {"config": _base_config(components=_default_components(), rag={})}
    _patch_project_service(mocker, state)
    client = _client()

    resp = client.post(
        "/v1/projects/ns/proj/rag/databases",
        json={
            "name": "main_db",
            "type": "ChromaStore",
            "config": {},
            # No embedding/retrieval specified; should use defaults
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["database"]
    assert data["embedding_strategies"][0]["name"] == "default_embeddings"
    assert data["retrieval_strategies"][0]["name"] == "semantic_search"


def test_create_database_with_both_reference_and_inline_error(mocker):
    state = {"config": _base_config(components=_default_components(), rag={})}
    _patch_project_service(mocker, state)
    client = _client()

    resp = client.post(
        "/v1/projects/ns/proj/rag/databases",
        json={
            "name": "main_db",
            "type": "ChromaStore",
            "embedding_strategy": "default_embeddings",
            "embedding_strategies": [
                {
                    "name": "inline_embeddings",
                    "type": "UniversalEmbedder",
                    "config": {"model": "foo"},
                }
            ],
            "retrieval_strategy": "semantic_search",
        },
    )
    assert resp.status_code == 400
    assert (
        "either embedding_strategy reference or embedding_strategies inline"
        in resp.json()["detail"]
    )


def test_create_database_with_missing_reference_error(mocker):
    components = _default_components()
    components["embedding_strategies"] = []  # remove embeddings to force missing ref
    state = {"config": _base_config(components=components, rag={})}
    _patch_project_service(mocker, state)
    client = _client()

    resp = client.post(
        "/v1/projects/ns/proj/rag/databases",
        json={
            "name": "main_db",
            "type": "ChromaStore",
            "embedding_strategy": "nonexistent",
            "retrieval_strategy": "semantic_search",
        },
    )
    assert resp.status_code == 400
    assert (
        "embedding_strategy 'nonexistent' not found in components.embedding_strategies"
        in resp.json()["detail"]
    )


def test_create_database_with_no_strategies_and_no_defaults_error(mocker):
    # No components at all
    state = {"config": _base_config(components={}, rag={})}
    _patch_project_service(mocker, state)
    client = _client()

    resp = client.post(
        "/v1/projects/ns/proj/rag/databases",
        json={
            "name": "main_db",
            "type": "ChromaStore",
        },
    )
    assert resp.status_code == 400
    assert "No embedding strategy provided or resolved" in resp.json()["detail"]


def test_create_database_with_inline_strategies_success(mocker):
    state = {"config": _base_config(components=_default_components(), rag={})}
    _patch_project_service(mocker, state)
    client = _client()

    resp = client.post(
        "/v1/projects/ns/proj/rag/databases",
        json={
            "name": "main_db",
            "type": "ChromaStore",
            "embedding_strategies": [
                {
                    "name": "inline_embeddings",
                    "type": "UniversalEmbedder",
                    "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                }
            ],
            "retrieval_strategies": [
                {
                    "name": "inline_retrieval",
                    "type": "BasicSimilarityStrategy",
                    "config": {"top_k": 5},
                }
            ],
            "default_embedding_strategy": "inline_embeddings",
            "default_retrieval_strategy": "inline_retrieval",
        },
    )
    assert resp.status_code == 201, resp.text
    data = resp.json()["database"]
    assert data["embedding_strategies"][0]["name"] == "inline_embeddings"
    assert data["retrieval_strategies"][0]["name"] == "inline_retrieval"
