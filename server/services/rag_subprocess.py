import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from core.logging import FastAPIStructLogger

from config.datamodel import LlamaFarmConfig


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


def ingest_file_with_rag(
    project_dir: str,
    project_config: LlamaFarmConfig,
    data_processing_strategy_name: str,
    database_name: str,
    source_path: str,
) -> bool:
    """
    Ingest a single file using the new RAG schema format.

    Args:
        project_dir: The directory of the project
        project_config: The full project configuration dictionary
        data_processing_strategy_name: Name of the data processing strategy to use
        database_name: Name of the database to use
        source_path: Path to the file to ingest

    Returns:
        True if ingestion succeeded, False otherwise
    """
    try:
        # Extract RAG configuration
        rag_config = project_config.rag
        if not rag_config:
            logger.error("No RAG configuration found in project config")
            return False

        # Find the specified data processing strategy
        data_processing_strategy = None
        for strategy in rag_config.data_processing_strategies or []:
            if strategy.name == data_processing_strategy_name:
                data_processing_strategy = strategy
                break

        if not data_processing_strategy:
            logger.error(
                f"Data processing strategy '{data_processing_strategy_name}' not found"
            )
            return False

        # Find the specified database
        database_config = None
        for db in rag_config.databases or []:
            if db.name == database_name:
                database_config = db
                break

        if not database_config:
            logger.error(f"Database '{database_name}' not found")
            return False

        # Run the RAG CLI with the new schema format
        exit_code, stdout, stderr = run_rag_cli_with_config_and_strategy(
            source_path, project_dir, database_name, data_processing_strategy_name
        )

        if exit_code != 0:
            logger.error(
                "RAG ingest failed",
                exit_code=exit_code,
                stderr=stderr,
                stdout=stdout,
                database=database_name,
                data_processing_strategy=data_processing_strategy_name,
            )
            return False

        logger.info(
            "RAG ingest succeeded",
            stdout=stdout,
            database=database_name,
            data_processing_strategy=data_processing_strategy_name,
        )
        return True

    except Exception as e:
        logger.error(f"Error during RAG ingestion: {e}")
        return False


def run_rag_cli_with_config_and_strategy(
    source_path: str,
    project_dir: str,
    database_name: str,
    data_processing_strategy_name: str,
    cwd: Path | None = None,
) -> tuple[int, str, str]:
    """
    Run RAG ingestion directly using the IngestHandler.

    Args:
        source_path: Path to the file to ingest
        project_dir: The directory of the project
        database_name: Name of the database to use
        data_processing_strategy_name: Name of the data processing strategy to use
        cwd: Working directory (not used anymore, kept for compatibility)

    Returns:
        Tuple of (exit_code, stdout, stderr)
    """
    try:
        # Import the IngestHandler directly
        import sys
        import traceback
        rag_path = str(repo_root)
        if rag_path not in sys.path:
            sys.path.insert(0, rag_path)
        
        try:
            from rag.core.ingest_handler import IngestHandler
        except ImportError as e:
            logger.error(f"Failed to import IngestHandler: {e}")
            logger.error(f"Python path: {sys.path}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1, "", f"Failed to import IngestHandler: {e}"
        
        # Configuration path
        config_path = f"{project_dir}/llamafarm.yaml"
        
        logger.info(
            "Using IngestHandler directly",
            config_path=config_path,
            data_processing_strategy=data_processing_strategy_name,
            database=database_name,
            source_path=source_path
        )
        
        # Initialize the ingest handler with separate fields
        try:
            handler = IngestHandler(
                config_path=config_path,
                data_processing_strategy=data_processing_strategy_name,
                database=database_name
            )
        except Exception as e:
            logger.error(f"Failed to initialize IngestHandler: {e}")
            logger.error(f"Config path: {config_path}")
            logger.error(f"Strategy: {data_processing_strategy_name}")
            logger.error(f"Database: {database_name}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1, "", f"Failed to initialize IngestHandler: {e}"
        
        # Read the file
        with open(source_path, 'rb') as f:
            file_data = f.read()
        
        # Create metadata
        from pathlib import Path as PathLib
        import json
        import os
        
        file_path = PathLib(source_path)
        
        # Check if this is a hash-based file in lf_data/raw
        if 'lf_data/raw' in str(source_path):
            # Extract file hash from the path
            file_hash = file_path.name
            # Try to load metadata file
            meta_dir = file_path.parent.parent / 'meta'
            meta_file = meta_dir / f"{file_hash}.json"
            
            if meta_file.exists():
                with open(meta_file, 'r') as mf:
                    meta_content = json.load(mf)
                    original_filename = meta_content.get('original_file_name', file_hash)
                    mime_type = meta_content.get('mime_type', 'application/octet-stream')
            else:
                original_filename = file_hash
                mime_type = 'application/octet-stream'
        else:
            # Regular file path
            original_filename = file_path.name
            mime_type = 'application/octet-stream'
        
        metadata = {
            'filename': original_filename,
            'filepath': str(file_path),
            'size': len(file_data),
            'content_type': mime_type
        }
        
        # Ingest the file
        result = handler.ingest_file(
            file_data=file_data,
            metadata=metadata
        )
        
        if result.get('status') == 'success':
            stdout = f"Successfully ingested {result.get('document_count', 0)} documents from {file_path.name}"
            return 0, stdout, ""
        else:
            stderr = f"Ingestion failed: {result.get('message', 'Unknown error')}"
            return 1, "", stderr
            
    except Exception as e:
        logger.error(f"Error during direct RAG ingestion: {e}")
        return 1, "", str(e)


def search_with_rag(
    project_dir: str,
    dataset: str,
    query: str,
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Run a search via rag api in its own environment and return list of dict results."""

    cfg_path = project_dir + "/llamafarm.yaml"

    # Add the repo root to sys.path to fix import issues
    code = (
        f"import sys; sys.path.insert(0, r'{str(repo_root)}');"
        "from rag.api import SearchAPI;"
        f"api=SearchAPI(config_path=r'{cfg_path}', dataset='{dataset}');"
        f"res=api.search(query={json.dumps(query)}, top_k={int(top_k)}, retrieval_strategy='{retrieval_strategy}');"
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


def search_with_rag_database(
    project_dir: str,
    database: str,
    query: str,
    top_k: int = 5,
    retrieval_strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Run a search directly against a database via rag api."""

    cfg_path = project_dir + "/llamafarm.yaml"

    # Add the repo root to sys.path to fix import issues
    # Use DatabaseSearchAPI instead of SearchAPI to search database directly
    code = (
        f"import sys; sys.path.insert(0, r'{str(repo_root)}');"
        "from rag.api import DatabaseSearchAPI;"
        f"api=DatabaseSearchAPI(config_path=r'{cfg_path}', database='{database}');"
        f"res=api.search(query={json.dumps(query)}, top_k={int(top_k)}, retrieval_strategy='{retrieval_strategy}');"
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
            "RAG database search subprocess failed",
            exit_code=e.returncode,
            stderr=e.stderr.strip(),
        )
        return []
    except json.JSONDecodeError:
        logger.error("Failed to decode RAG search output as JSON")
        return []
