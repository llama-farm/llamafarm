"""Universal Runtime provider implementation with streaming support."""

import asyncio
import os
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

import httpx
import requests  # type: ignore
from huggingface_hub import (
    constants as hf_constants,
)
from huggingface_hub import (
    get_hf_file_metadata,
    get_token,
    hf_hub_url,
    list_repo_tree,
    scan_cache_dir,
)
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.file_download import repo_folder_name
from llamafarm_common import (
    list_gguf_files,
    parse_model_with_quantization,
    select_gguf_file_with_logging,
)

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI
from api.errors import NotFoundError
from core.logging import FastAPIStructLogger
from core.settings import settings

from .base import CachedModel, RuntimeProvider
from .health import HealthCheckResult

logger = FastAPIStructLogger(__name__)

# Chunk size for streaming downloads (1MB)
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


def get_hf_token() -> str | None:
    """Get HuggingFace token from settings or huggingface_hub.

    Checks settings.huggingface_token first (set via HF_TOKEN env var),
    then falls back to the token stored by `huggingface-cli login`.

    Returns:
        The token string if found, None otherwise.
    """
    # Check settings first (from HF_TOKEN env var)
    if settings.huggingface_token:
        return settings.huggingface_token

    # Fall back to get_token() (from `huggingface-cli login`)
    return get_token()


@dataclass
class ModelDownloadInfo:
    """Information about a model to be downloaded."""

    model_id: str
    quantization: str | None
    selected_file: str | None
    total_size: int
    is_gguf: bool
    files_to_download: list[str]  # List of files to download


@dataclass
class FileDownloadInfo:
    """Information about a specific file to download."""

    filename: str
    url: str
    size: int
    etag: str | None
    commit_hash: str | None


def get_hf_cache_path() -> Path:
    """Get the HuggingFace cache directory path."""
    return Path(hf_constants.HF_HUB_CACHE)


def get_repo_cache_path(model_id: str) -> Path:
    """Get the cache path for a specific repo."""
    cache_dir = get_hf_cache_path()
    folder_name = repo_folder_name(repo_id=model_id, repo_type="model")
    return cache_dir / folder_name


def get_file_download_info(
    model_id: str, filename: str, token: str | None = None
) -> FileDownloadInfo:
    """Get download info for a specific file in a repo.

    Args:
        model_id: HuggingFace repo ID
        filename: File path within the repo
        token: Optional HuggingFace token for private/gated models

    Returns:
        FileDownloadInfo with URL, size, etag, and commit hash
    """
    url = hf_hub_url(repo_id=model_id, filename=filename, revision="main")
    metadata = get_hf_file_metadata(url, token=token)

    return FileDownloadInfo(
        filename=filename,
        url=url,
        size=metadata.size or 0,
        etag=metadata.etag,
        commit_hash=metadata.commit_hash,
    )


async def stream_download_file(
    file_info: FileDownloadInfo,
    model_id: str,
    progress_queue: asyncio.Queue[dict],
    keepalive_interval: float = 10.0,
    token: str | None = None,
) -> Path:
    """Download a file with streaming progress reporting.

    Downloads to the HuggingFace cache in a compatible format.

    Args:
        file_info: Information about the file to download
        model_id: HuggingFace model ID
        progress_queue: Queue to send progress events to
        keepalive_interval: Seconds between keepalive messages if no progress
        token: Optional HuggingFace token for private/gated models

    Returns:
        Path to the downloaded file
    """
    repo_cache = get_repo_cache_path(model_id)
    blobs_dir = repo_cache / "blobs"
    snapshots_dir = repo_cache / "snapshots"
    refs_dir = repo_cache / "refs"

    # Create directories
    blobs_dir.mkdir(parents=True, exist_ok=True)
    refs_dir.mkdir(parents=True, exist_ok=True)

    # Determine blob filename from etag (HF uses etag without quotes as blob name)
    etag = file_info.etag
    if etag:
        # Remove quotes and any prefix
        blob_name = etag.strip('"').replace('"', "")
        # Handle etags with hashes like "abc123-5"
        if "-" in blob_name and not blob_name.startswith("oid:"):
            blob_name = blob_name.split("-")[0]
    else:
        # Fallback to commit hash if no etag
        blob_name = file_info.commit_hash or "unknown"

    blob_path = blobs_dir / blob_name

    # Check if already downloaded
    if blob_path.exists() and blob_path.stat().st_size == file_info.size:
        logger.info(f"File already cached: {file_info.filename}")
        await progress_queue.put(
            {
                "event": "cached",
                "file": file_info.filename,
                "size": file_info.size,
            }
        )
    else:
        # Download with streaming
        downloaded = 0
        start_time = time.time()

        # Use a temp file during download
        temp_path = blob_path.with_suffix(".tmp")

        # Build headers with auth token if provided (for private/gated models)
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:  # noqa: SIM117
            async with client.stream("GET", file_info.url, headers=headers) as response:
                response.raise_for_status()
                total = file_info.size or int(response.headers.get("content-length", 0))

                await progress_queue.put(
                    {
                        "event": "start",
                        "file": file_info.filename,
                        "total": total,
                        "downloaded": 0,
                    }
                )

                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(DOWNLOAD_CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Calculate transfer rate and ETA
                        elapsed = time.time() - start_time
                        bytes_per_sec = downloaded / elapsed if elapsed > 0 else 0
                        remaining_bytes = total - downloaded
                        eta_seconds = (
                            remaining_bytes / bytes_per_sec
                            if bytes_per_sec > 0
                            else None
                        )

                        # Emit progress update with rate and ETA
                        await progress_queue.put(
                            {
                                "event": "progress",
                                "file": file_info.filename,
                                "downloaded": downloaded,
                                "total": total,
                                "percent": (downloaded / total * 100) if total else 0,
                                "bytes_per_sec": int(bytes_per_sec),
                                "eta_seconds": (
                                    round(eta_seconds, 1) if eta_seconds else None
                                ),
                            }
                        )

        # Move temp file to final location
        temp_path.rename(blob_path)

        await progress_queue.put(
            {
                "event": "end",
                "file": file_info.filename,
                "downloaded": downloaded,
                "total": file_info.size,
            }
        )

    # Set up snapshot directory structure
    commit_hash = file_info.commit_hash or "main"
    snapshot_dir = snapshots_dir / commit_hash
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Handle nested paths in filename
    snapshot_file = snapshot_dir / file_info.filename
    snapshot_file.parent.mkdir(parents=True, exist_ok=True)

    # Create symlink from snapshot to blob
    if snapshot_file.exists() or snapshot_file.is_symlink():
        snapshot_file.unlink()

    # Use relative symlink
    rel_blob = os.path.relpath(blob_path, snapshot_file.parent)
    snapshot_file.symlink_to(rel_blob)

    # Update refs/main to point to commit hash
    refs_main = refs_dir / "main"
    refs_main.write_text(commit_hash)

    return snapshot_file


def get_model_download_info(
    model_name: str, token: str | None = None
) -> ModelDownloadInfo:
    """Get metadata about a model before downloading.

    Fetches file information from HuggingFace to determine:
    - For GGUF models: the selected quantization file and its size
    - For other models: the total size of all files

    Args:
        model_name: Model identifier, optionally with quantization suffix
        token: Optional HuggingFace token for private/gated models

    Returns:
        ModelDownloadInfo with model details and download size
    """
    model_id, quantization = parse_model_with_quantization(model_name)

    # Get all files with their sizes from the repo
    try:
        repo_files = list(list_repo_tree(model_id, recursive=True, token=token))
        file_sizes = {
            item.path: item.size
            for item in repo_files
            if hasattr(item, "size") and item.size is not None
        }
    except Exception as e:
        logger.warning(f"Could not fetch file sizes for {model_id}: {e}")
        repo_files = []
        file_sizes = {}

    # Check if this is a GGUF repository
    try:
        gguf_files = list_gguf_files(model_id, token=token)
    except Exception:
        gguf_files = []

    if gguf_files:
        # GGUF model - select the specific file to download
        selected_file = select_gguf_file_with_logging(
            gguf_files, preferred_quantization=quantization
        )
        file_size = file_sizes.get(selected_file, 0)
        return ModelDownloadInfo(
            model_id=model_id,
            quantization=quantization,
            selected_file=selected_file,
            total_size=file_size,
            is_gguf=True,
            files_to_download=[selected_file],
        )
    else:
        # Non-GGUF model - get all files to download
        all_files = [item.path for item in repo_files if hasattr(item, "size")]
        total_size = sum(file_sizes.values())
        return ModelDownloadInfo(
            model_id=model_id,
            quantization=quantization,
            selected_file=None,
            total_size=total_size,
            is_gguf=False,
            files_to_download=all_files,
        )


class UniversalProvider(RuntimeProvider):
    """Universal Runtime provider with SSE streaming support.

    This provider connects to the Universal Runtime (runtimes/universal) which:
    - Supports any HuggingFace transformer or diffusion model
    - Provides OpenAI-compatible endpoints
    - Supports SSE streaming for real-time token generation
    - Auto-detects hardware acceleration (MPS/CUDA/CPU)
    """

    name = "universal-runtime"

    @property
    def _base_url(self) -> str:
        """Get base URL for Universal Runtime API."""
        return (
            self._model_config.base_url
            or f"http://{settings.universal_host}:{settings.universal_port}/v1"
        )

    @property
    def _api_key(self) -> str:
        """Get API key for Universal Runtime."""
        return self._model_config.api_key or settings.universal_api_key

    def get_client(self) -> LFAgentClient:
        """Get Universal Runtime client with OpenAI compatibility.

        The client supports:
        - Standard chat completions
        - SSE streaming (set stream=True in requests)
        - Image generation (diffusion models)
        - Text embeddings (encoder models)
        - Audio transcription (speech-to-text)
        - Vision tasks (classification, VQA)

        Args:
            extra_body: Additional parameters to pass through

        Returns:
            LFAgentClient configured for Universal Runtime
        """
        if not self._model_config.base_url:
            self._model_config.base_url = self._base_url
        if not self._model_config.api_key:
            self._model_config.api_key = self._api_key

        client = LFAgentClientOpenAI(
            model_config=self._model_config,
        )
        return client

    def check_health(self) -> HealthCheckResult:
        """Check health of Universal Runtime.

        Verifies:
        - Runtime is reachable
        - Models are loaded
        - Device information (MPS/CUDA/CPU)
        - Response latency

        Returns:
            HealthCheckResult with status, loaded models, and device info
        """
        start = int(time.time() * 1000)
        base = self._base_url.replace("/v1", "").rstrip("/")

        # First check /health endpoint for device info
        health_url = f"{base}/health"
        models_url = f"{base}/v1/models"

        try:
            # Check health endpoint
            health_resp = requests.get(health_url, timeout=2.0)
            latency = int(time.time() * 1000) - start

            if 200 <= health_resp.status_code < 300:
                health_data = health_resp.json()
                device_info = health_data.get("device", {})
                device_type = (
                    device_info.get("type", "unknown")
                    if isinstance(device_info, dict)
                    else device_info
                )

                # Check models endpoint
                try:
                    models_resp = requests.get(models_url, timeout=1.0)
                    if 200 <= models_resp.status_code < 300:
                        models_data = models_resp.json()
                        models = models_data.get("data", [])
                        model_ids = [m.get("id") for m in models if m.get("id")]
                    else:
                        model_ids = []
                except Exception:
                    model_ids = []

                message = (
                    f"{base} reachable on {device_type}, "
                    f"{len(model_ids)} model(s) loaded"
                )
                return HealthCheckResult(
                    name=self.name,
                    status="healthy",
                    message=message,
                    latency_ms=latency,
                    details={
                        "host": base,
                        "device": device_type,
                        "model_count": len(model_ids),
                        "models": model_ids,
                        "streaming_supported": True,
                        "loaded_models": health_data.get("loaded_models", []),
                    },
                )
            else:
                return HealthCheckResult(
                    name=self.name,
                    status="unhealthy",
                    message=f"{base} returned HTTP {health_resp.status_code}",
                    latency_ms=latency,
                    details={"host": base, "status_code": health_resp.status_code},
                )
        except requests.exceptions.Timeout:
            port = settings.universal_port
            timeout_msg = (
                f"Timeout connecting to {base} - is Universal Runtime running? "
                f"(cd runtimes/universal && uv run uvicorn server:app --port {port})"
            )
            return HealthCheckResult(
                name=self.name,
                status="unhealthy",
                message=timeout_msg,
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base},
            )
        except Exception as e:
            return HealthCheckResult(
                name=self.name,
                status="unhealthy",
                message=f"Error: {str(e)}",
                latency_ms=int(time.time() * 1000) - start,
                details={"host": base, "error": str(e)},
            )

    @staticmethod
    def list_cached_models() -> list[CachedModel]:
        """List models that are available on this system"""
        cache_info = scan_cache_dir()
        return [
            CachedModel(
                id=repo.repo_id,
                name=repo.repo_id,
                size=repo.size_on_disk,
                path=str(repo.repo_path),
            )
            for repo in cache_info.repos
            if repo.repo_type == "model"
        ]

    @staticmethod
    async def download_model(model_name: str) -> AsyncIterator[dict]:
        """Download/cache a model for the given model name.

        Supports model names with quantization suffix (e.g., "model:Q4_K_M").
        For GGUF models, only downloads the specified quantization.

        Uses httpx streaming for real-time progress updates.

        Yields events:
            - init: Model metadata (model_id, quantization, selected_file, total_size)
            - start: Download of a file is starting
            - progress: Download progress update (with downloaded, total, percent)
            - cached: File was already in cache
            - end: Download of a file completed
            - done: All downloads complete
        """
        try:
            # Get HuggingFace token for private/gated model access
            hf_token = get_hf_token()

            # First, fetch model metadata and emit init event
            info = await asyncio.to_thread(
                get_model_download_info, model_name, hf_token
            )

            yield {
                "event": "init",
                "model_id": info.model_id,
                "quantization": info.quantization,
                "selected_file": info.selected_file,
                "total_size": info.total_size,
                "is_gguf": info.is_gguf,
                "file_count": len(info.files_to_download),
            }

            if not info.files_to_download:
                yield {
                    "event": "done",
                    "local_dir": str(get_repo_cache_path(info.model_id)),
                }
                return

            # Create progress queue for streaming updates
            progress_queue: asyncio.Queue[dict] = asyncio.Queue()
            keepalive_interval = 10  # seconds
            last_status: dict | None = None

            # Download each file with progress tracking
            for i, filename in enumerate(info.files_to_download):
                # Get download info for this file (URL, size, etag)
                file_info = await asyncio.to_thread(
                    get_file_download_info, info.model_id, filename, hf_token
                )

                logger.info(
                    f"Downloading file {i + 1}/{len(info.files_to_download)}: "
                    f"{filename} ({file_info.size} bytes)"
                )

                # Start the download task
                download_task = asyncio.create_task(
                    stream_download_file(
                        file_info=file_info,
                        model_id=info.model_id,
                        progress_queue=progress_queue,
                        keepalive_interval=keepalive_interval,
                        token=hf_token,
                    )
                )

                # Consume progress events until download is complete
                file_done = False
                while not file_done:
                    try:
                        evt = await asyncio.wait_for(
                            progress_queue.get(), timeout=keepalive_interval
                        )
                        last_status = evt
                        yield evt
                        # Check if this file is done
                        if evt.get("event") in ("end", "cached"):
                            file_done = True
                    except TimeoutError:
                        # Check if download task has failed
                        if download_task.done():
                            try:
                                await download_task
                            except Exception as task_error:
                                yield {
                                    "event": "error",
                                    "message": f"Download failed: {str(task_error)}",
                                    "file": filename,
                                }
                                raise
                            # Task done without end event - shouldn't happen
                            file_done = True
                        else:
                            # Emit keepalive
                            if last_status:
                                yield {**last_status, "keepalive": True}
                            else:
                                yield {
                                    "event": "progress",
                                    "file": filename,
                                    "downloaded": 0,
                                    "total": file_info.size,
                                    "percent": 0,
                                    "keepalive": True,
                                }

                # Ensure download task completed
                await download_task

            # All files downloaded
            repo_cache = get_repo_cache_path(info.model_id)
            yield {
                "event": "done",
                "local_dir": str(repo_cache),
            }

        except RepositoryNotFoundError as e:
            model_id, _ = parse_model_with_quantization(model_name)
            logger.error(f"Model {model_id} not found")
            raise NotFoundError(
                f"Model '{model_id}' not found. Check if the model exists on "
                f"HuggingFace and that you specified the correct repo path."
            ) from e
        except Exception as e:
            model_id, _ = parse_model_with_quantization(model_name)
            logger.exception(f"Error downloading model {model_id}: {e}")
            raise e

    @staticmethod
    def delete_model(model_name: str) -> dict:
        """Delete a cached model from the HuggingFace cache.

        Args:
            model_name: The model identifier (e.g., "meta-llama/Llama-2-7b-hf" or
                       "unsloth/Qwen3-4B-GGUF:Q4_K_M"). If quantization suffix is
                       present, it will be stripped since the cache stores by model_id.

        Returns:
            Dict with deleted model info including freed space

        Raises:
            NotFoundError: If the model is not found in the cache
        """
        # Parse model name to strip quantization suffix if present
        model_id, _ = parse_model_with_quantization(model_name)

        cache_info = scan_cache_dir()

        # Find the repo to delete
        target_repo = next(
            (
                repo
                for repo in cache_info.repos
                if repo.repo_id == model_id and repo.repo_type == "model"
            ),
            None,
        )

        if not target_repo:
            raise NotFoundError(
                f"Model '{model_id}' not found in cache. "
                "Use GET /v1/models to see available models."
            )

        # Store info before deletion
        size_on_disk = target_repo.size_on_disk
        repo_path = str(target_repo.repo_path)
        revision_count = len(target_repo.revisions)

        # Delete all revisions of the model
        # Need to pass the actual revision hashes to delete_revisions
        revisions_to_delete = [rev.commit_hash for rev in target_repo.revisions]
        delete_strategy = cache_info.delete_revisions(*revisions_to_delete)
        delete_strategy.execute()

        logger.info(
            f"Deleted model {model_id}",
            revisions=revision_count,
            size_freed=size_on_disk,
            path=repo_path,
        )

        return {
            "model_name": model_id,
            "revisions_deleted": revision_count,
            "size_freed": size_on_disk,
            "path": repo_path,
        }
