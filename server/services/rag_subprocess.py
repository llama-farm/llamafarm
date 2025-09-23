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
    filename: str | None = None,
    dataset_name: str | None = None,
) -> tuple[bool, dict]:
    """
    Ingest a single file using the new RAG schema format.

    Args:
        project_dir: The directory of the project
        project_config: The full project configuration dictionary
        data_processing_strategy_name: Name of the data processing strategy to use
        database_name: Name of the database to use
        source_path: Path to the file to ingest

    Returns:
        Tuple of (success: bool, details: dict) with processing information
    """
    import os
    import re
    import json

    # Initialize details dict
    details = {
        "filename": filename or os.path.basename(source_path),
        "parser": None,
        "extractors": [],
        "chunks": None,
        "chunk_size": None,
        "embedder": None,
        "error": None,
        "reason": None,
        "result": None,  # Store the full result from IngestHandler
    }

    try:
        # Extract RAG configuration
        rag_config = project_config.rag
        if not rag_config:
            logger.error("No RAG configuration found in project config")
            details["error"] = "No RAG configuration found"
            return False, details

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
            details["error"] = f"Strategy '{data_processing_strategy_name}' not found"
            return False, details

        # Find the specified database
        database_config = None
        for db in rag_config.databases or []:
            if db.name == database_name:
                database_config = db
                break

        if not database_config:
            logger.error(f"Database '{database_name}' not found")
            details["error"] = f"Database '{database_name}' not found"
            return False, details

        # Extract strategy details for the response
        if data_processing_strategy:
            logger.info(f"Processing strategy: {data_processing_strategy_name}")
            logger.debug(f"Strategy object: {data_processing_strategy}")

            # Get parser info - use first parser if multiple
            if (
                data_processing_strategy.parsers
                and len(data_processing_strategy.parsers) > 0
            ):
                details["parser"] = data_processing_strategy.parsers[0].type
                logger.info(f"Parser type: {details['parser']}")
                # If there's chunking config in parser config
                if data_processing_strategy.parsers[0].config:
                    parser_config = data_processing_strategy.parsers[0].config
                    if (
                        isinstance(parser_config, dict)
                        and "chunk_size" in parser_config
                    ):
                        details["chunk_size"] = parser_config["chunk_size"]
                        logger.info(f"Chunk size from parser: {details['chunk_size']}")

            # Get extractors
            if data_processing_strategy.extractors:
                details["extractors"] = [
                    e.type for e in data_processing_strategy.extractors
                ]
                logger.info(f"Extractors: {details['extractors']}")

        # Get embedder info from database config
        if database_config:
            logger.info(f"Database: {database_name}")
            logger.debug(f"Database object: {database_config}")

            # Get first embedding strategy if available
            if (
                database_config.embedding_strategies
                and len(database_config.embedding_strategies) > 0
            ):
                details["embedder"] = database_config.embedding_strategies[0].type
                logger.info(f"Embedder type: {details['embedder']}")
            # Check for chunk size in database config
            if database_config.config:
                db_config = database_config.config
                if (
                    isinstance(db_config, dict)
                    and "chunk_size" in db_config
                    and details["chunk_size"] is None
                ):
                    details["chunk_size"] = db_config["chunk_size"]
                    logger.info(f"Chunk size from database: {details['chunk_size']}")

        # Run the RAG CLI with the new schema format
        exit_code, stdout, stderr = run_rag_cli_with_config_and_strategy(
            source_path,
            project_dir,
            database_name,
            data_processing_strategy_name,
            dataset_name,
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

            # Try to extract error details from stderr
            if "duplicate" in stderr.lower():
                details["reason"] = "duplicate"
                details["error"] = "File already exists in database"
            else:
                details["error"] = stderr[:200] if stderr else "Unknown error"

            return False, details

        # Try to extract processing details from stdout
        if stdout:
            # Look for JSON result - use non-greedy match with explicit end marker
            # Limit search to prevent ReDoS attacks
            max_json_size = 100000  # Limit JSON block to 100KB
            if len(stdout) > max_json_size:
                search_text = stdout[:max_json_size]
            else:
                search_text = stdout

            # Use more specific pattern that avoids catastrophic backtracking
            json_start = search_text.find("[RESULT_JSON]")
            json_end = (
                search_text.find("[/RESULT_JSON]", json_start)
                if json_start != -1
                else -1
            )

            if json_start != -1 and json_end != -1:
                json_match_text = search_text[json_start + 13 : json_end]
            else:
                json_match_text = None

            if json_match_text:
                try:
                    result = json.loads(json_match_text)
                    details["result"] = result

                    # Extract details from result
                    if result.get("parsers_used"):
                        details["parser"] = (
                            result["parsers_used"][0]
                            if len(result["parsers_used"]) == 1
                            else ", ".join(result["parsers_used"])
                        )

                    if result.get("extractors_applied"):
                        details["extractors"] = result["extractors_applied"]

                    if result.get("embedder"):
                        details["embedder"] = result["embedder"]

                    if result.get("document_count"):
                        details["chunks"] = result["document_count"]

                    if result.get("chunk_size"):
                        details["chunk_size"] = result["chunk_size"]

                    # Pass through the status and counts
                    if result.get("status"):
                        details["status"] = result["status"]
                    if "stored_count" in result:
                        details["stored_count"] = result["stored_count"]
                    if "skipped_count" in result:
                        details["skipped_count"] = result["skipped_count"]

                    # Set reason if it's a duplicate
                    if (
                        result.get("status") == "skipped"
                        or result.get("reason") == "duplicate"
                    ):
                        details["reason"] = "duplicate"
                        details["status"] = "skipped"
                    elif (
                        result.get("stored_count", 0) == 0
                        and result.get("skipped_count", 0) > 0
                    ):
                        details["reason"] = "duplicate"
                        details["status"] = "skipped"

                except json.JSONDecodeError:
                    logger.warning("Could not parse result JSON from stdout")

            # Fallback parsing for older format or if JSON parsing fails
            if not details.get("chunks"):
                # Try multiple patterns to extract chunk count - use limited search to prevent ReDoS
                # Limit to first 10KB of output for performance
                chunk_search_text = stdout[:10000] if len(stdout) > 10000 else stdout

                # Use more specific patterns with limited backtracking
                chunk_match = (
                    re.search(r"Chunks created:\s{0,10}(\d{1,6})", chunk_search_text)
                    or re.search(
                        r"(?:^|\s)(\d{1,6})\s+chunks?\b",
                        chunk_search_text,
                        re.IGNORECASE | re.MULTILINE,
                    )
                    or re.search(
                        r"\bcreated\s+(\d{1,6})\s+chunks?\b",
                        chunk_search_text,
                        re.IGNORECASE,
                    )
                    or re.search(
                        r"\btotal[^0-9]{0,20}(\d{1,6})\s+chunks?\b",
                        chunk_search_text,
                        re.IGNORECASE,
                    )
                )
                if chunk_match:
                    details["chunks"] = int(chunk_match.group(1))
                else:
                    logger.warning(
                        "Could not extract chunk count from stdout using fallback patterns. Output may be non-standard."
                    )

            # Look for stored/skipped counts - use bounded patterns to prevent ReDoS
            # Limit search to first 10KB for performance
            count_search_text = stdout[:10000] if len(stdout) > 10000 else stdout
            stored_match = re.search(r"Stored:\s{0,10}(\d{1,6})", count_search_text)
            # Use more specific pattern with limited characters between "Skipped" and the number
            skipped_match = re.search(
                r"Skipped[^:]{0,50}:\s{0,10}(\d{1,6})", count_search_text
            )
            if (
                stored_match
                and int(stored_match.group(1)) == 0
                and skipped_match
                and int(skipped_match.group(1)) > 0
            ):
                details["reason"] = "duplicate"

            # Also check for explicit duplicate message
            if (
                "FILE ALREADY PROCESSED" in stdout
                or "All chunks already exist" in stdout
            ):
                details["reason"] = "duplicate"

        logger.info(
            "RAG ingest succeeded",
            stdout=stdout,
            database=database_name,
            data_processing_strategy=data_processing_strategy_name,
            details=details,
        )
        return True, details

    except Exception as e:
        import traceback

        logger.error(
            "Error during RAG ingestion",
            error=str(e),
            source_path=source_path,
            database=database_name,
            strategy=data_processing_strategy_name,
            traceback=traceback.format_exc(),
        )
        details["error"] = str(e)
        return False, details


def run_rag_cli_with_config_and_strategy(
    source_path: str,
    project_dir: str,
    database_name: str,
    data_processing_strategy_name: str,
    dataset_name: str = None,
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
            source_path=source_path,
        )

        # Initialize the ingest handler with separate fields
        try:
            handler = IngestHandler(
                config_path=config_path,
                data_processing_strategy=data_processing_strategy_name,
                database=database_name,
                dataset_name=dataset_name,  # Pass dataset name for logging
            )
        except Exception as e:
            logger.error(f"Failed to initialize IngestHandler: {e}")
            logger.error(f"Config path: {config_path}")
            logger.error(f"Strategy: {data_processing_strategy_name}")
            logger.error(f"Database: {database_name}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return 1, "", f"Failed to initialize IngestHandler: {e}"

        # Read the file
        with open(source_path, "rb") as f:
            file_data = f.read()

        # Create metadata
        from pathlib import Path as PathLib
        import json
        import os

        file_path = PathLib(source_path)

        # Check if this is a hash-based file in lf_data/raw
        if "lf_data/raw" in str(source_path):
            # Extract file hash from the path
            file_hash = file_path.name
            # Try to load metadata file
            meta_dir = file_path.parent.parent / "meta"
            meta_file = meta_dir / f"{file_hash}.json"

            if meta_file.exists():
                with open(meta_file, "r") as mf:
                    meta_content = json.load(mf)
                    original_filename = meta_content.get(
                        "original_file_name", file_hash
                    )
                    mime_type = meta_content.get(
                        "mime_type", "application/octet-stream"
                    )
            else:
                original_filename = file_hash
                mime_type = "application/octet-stream"
        else:
            # Regular file path
            original_filename = file_path.name
            mime_type = "application/octet-stream"

        metadata = {
            "filename": original_filename,
            "filepath": str(file_path),
            "size": len(file_data),
            "content_type": mime_type,
        }

        # Ingest the file
        result = handler.ingest_file(file_data=file_data, metadata=metadata)

        # Handle different statuses
        status = result.get("status")

        # Convert result to JSON for parsing
        import json

        result_json = json.dumps(result)

        if status == "success":
            stdout = f"Successfully ingested {result.get('document_count', 0)} documents from {file_path.name}"
            stdout += f"\nStored: {result.get('stored_count', 0)}"
            stdout += f"\nSkipped: {result.get('skipped_count', 0)}"
            stdout += f"\n[RESULT_JSON]{result_json}[/RESULT_JSON]"
            return 0, stdout, ""
        elif status == "skipped":
            # File was skipped (duplicate)
            stdout = f"FILE ALREADY PROCESSED - All {result.get('skipped_count', 0)} chunks already exist in database"
            stdout += f"\nStored: 0"
            stdout += f"\nSkipped: {result.get('skipped_count', 0)}"
            stdout += f"\n[RESULT_JSON]{result_json}[/RESULT_JSON]"
            return 0, stdout, ""
        else:
            stderr = f"Ingestion failed: {result.get('message', 'Unknown error')}"
            return 1, "", stderr

    except Exception as e:
        logger.error(f"Error during direct RAG ingestion: {e}")
        return 1, "", str(e)


def search_with_rag(
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
        logger.error("Failed to decode RAG search output as JSON", exc_info=True)
        return []
