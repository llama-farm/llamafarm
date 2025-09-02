import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from core.logging import FastAPIStructLogger


logger = FastAPIStructLogger()
repo_root = Path(__file__).parent.parent.parent
rag_repo = repo_root / "rag"


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def build_v1_config_from_strategy(strategy) -> dict[str, Any]:
    """Build JSON-serializable v1-style RAG config from a LlamaFarm strategy.

    Structure:
    {
      "version": "v1",
      "rag": {
        "parsers": {"default": {...}},
        "embedders": {"default": {...}},
        "vector_stores": {"default": {...}},
        "retrieval_strategies": {"default": {...}},
        "defaults": {"parser": "default", "embedder": "default", "vector_store": "default", "strategy": "default"}
      }
    }
    """
    components = strategy.components

    # Parser
    parser_type = _enum_value(components.parser.type)
    parser_config = components.parser.config.model_dump(mode="json")

    # Embedder with fallback if sentence-transformer is not implemented in rag
    embedder_type = _enum_value(components.embedder.type)
    embedder_config = components.embedder.config.model_dump(mode="json")

    # Vector store
    vector_store_type = _enum_value(components.vector_store.type)
    vector_store_config = components.vector_store.config.model_dump(mode="json")

    # Retrieval strategy (Literal or Enum)
    retrieval_type_raw = components.retrieval_strategy.type
    retrieval_type = _enum_value(retrieval_type_raw)
    if not isinstance(retrieval_type, str | int | float | bool | type(None)):
        retrieval_type = str(retrieval_type)
    retrieval_config = components.retrieval_strategy.config.model_dump(mode="json")

    return {
        "version": "v1",
        "rag": {
            "parsers": {
                "default": {"type": parser_type, "config": parser_config},
            },
            "embedders": {
                "default": {"type": embedder_type, "config": embedder_config},
            },
            "vector_stores": {
                "default": {"type": vector_store_type, "config": vector_store_config},
            },
            "retrieval_strategies": {
                "default": {"type": retrieval_type, "config": retrieval_config},
            },
            "defaults": {
                "parser": "default",
                "embedder": "default",
                "vector_store": "default",
                "strategy": "default",
            },
        },
    }


def run_rag_cli_with_config(
    args: list[str], config_dict: dict[str, Any], *, cwd: Path | None = None
) -> tuple[int, str, str]:
    """Run rag CLI via uv run with a temp config file. Returns (code, out, err)."""
    cwd = cwd or rag_repo

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "rag_config.json"
        with open(cfg_path, "w") as f:
            json.dump(config_dict, f)

        code = [
            "uv",
            "run",
            "-q",
            "python",
            "cli.py",
            "--config",
            str(cfg_path),
            *args,
        ]

        try:
            completed = subprocess.run(
                code,
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
            )
            return completed.returncode, completed.stdout, completed.stderr
        except subprocess.CalledProcessError as e:
            return e.returncode, e.stdout or "", e.stderr or ""


def ingest_file_with_rag(strategy, source_path: str) -> bool:
    """Ingest a single file using rag CLI and provided strategy components."""
    logger.info(f"Ingesting file {source_path} with strategy {strategy.name}")
    cfg = build_v1_config_from_strategy(strategy)
    exit_code, stdout, stderr = run_rag_cli_with_config(["ingest", source_path], cfg)
    if exit_code != 0:
        logger.error(
            "RAG ingest failed", exit_code=exit_code, stderr=stderr, stdout=stdout
        )
        return False
    logger.info("RAG ingest succeeded", stdout=stdout)
    return True


def search_with_rag(strategy, query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Run a search via rag api in its own environment and return list of dict results."""
    # Build minimal config expected by rag/api.py
    v1 = build_v1_config_from_strategy(strategy)
    # Convert unified v1 rag config to the legacy fields expected by SearchAPI
    cfg = {
        "embedder": v1["rag"]["embedders"]["default"],
        "vector_store": v1["rag"]["vector_stores"]["default"],
        "retrieval_strategy": v1["rag"]["retrieval_strategies"]["default"],
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "rag_config.json"
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)

        code = (
            "from api import SearchAPI;"
            f"api=SearchAPI(config_path=r'{cfg_path}');"
            f"res=api.search(query={json.dumps(query)}, top_k={int(top_k)});"
            "import json; print(json.dumps([r.to_dict() for r in res]))"
        )
        cmd = [
            "uv",
            "run",
            "-q",
            "python",
            "-c",
            code,
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(rag_repo),
                check=True,
                capture_output=True,
                text=True,
            )
            stdout = completed.stdout.strip()
            return json.loads(stdout or "[]")
        except subprocess.CalledProcessError as e:
            logger.error(
                "RAG search subprocess failed",
                exit_code=e.returncode,
                stderr=e.stderr.strip(),
            )
            return []
        except json.JSONDecodeError:
            logger.error("Failed to decode RAG search output as JSON")
            return []
