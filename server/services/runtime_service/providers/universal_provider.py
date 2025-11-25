"""Universal Runtime provider implementation with streaming support."""

import asyncio
import time
from collections.abc import AsyncIterator

import requests  # type: ignore
from huggingface_hub import scan_cache_dir, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError
from llamafarm_common import (
    parse_model_with_quantization,
    parse_quantization_from_filename,
    select_gguf_file,
)
from tqdm.asyncio import tqdm  # type: ignore

from agents.base.clients.client import LFAgentClient
from agents.base.clients.openai import LFAgentClientOpenAI
from api.errors import NotFoundError
from core.logging import FastAPIStructLogger
from core.settings import settings

from .base import CachedModel, RuntimeProvider
from .health import HealthCheckResult

logger = FastAPIStructLogger(__name__)


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
        """
        try:
            # Parse model name to extract quantization if present
            model_id, quantization = parse_model_with_quantization(model_name)

            queue: asyncio.Queue[dict] = asyncio.Queue()
            loop = asyncio.get_event_loop()

            def run_download():
                # Patch file_download module to capture actual download progress
                import huggingface_hub.file_download

                original = huggingface_hub.file_download.tqdm
                custom_class = make_reporting_tqdm(queue, loop)
                huggingface_hub.file_download.tqdm = custom_class

                try:
                    from huggingface_hub import HfApi

                    # Check if this is a GGUF model repository
                    api = HfApi()
                    try:
                        files = api.list_repo_files(repo_id=model_id, repo_type="model")
                        gguf_files = [f for f in files if f.endswith(".gguf")]

                        if gguf_files:
                            # This is a GGUF repo - use intelligent selection
                            # to download only one quantization variant
                            selected_file = None

                            if quantization:
                                # User specified a quantization - find matching file
                                matching_files = [
                                    f
                                    for f in gguf_files
                                    if quantization.upper() in f.upper()
                                ]

                                if matching_files:
                                    selected_file = matching_files[0]
                                    logger.info(
                                        f"Downloading GGUF with quantization {quantization}: {selected_file}"
                                    )
                                else:
                                    logger.warning(
                                        f"Quantization {quantization} not found in {model_id}. "
                                        f"Available: {gguf_files}"
                                    )

                            if not selected_file:
                                # No quantization specified or not found - use smart selection
                                # Use common selection logic from llamafarm_common
                                selected_file = select_gguf_file(
                                    gguf_files, preferred_quantization=None
                                )
                                if selected_file:
                                    quant = parse_quantization_from_filename(
                                        selected_file
                                    )
                                    if quant:
                                        logger.info(
                                            f"Auto-selected GGUF quantization {quant}: {selected_file}"
                                        )
                                    else:
                                        logger.info(
                                            f"No preferred quantization found, using first file: {selected_file}"
                                        )

                            # Download only the selected GGUF file
                            local_dir = snapshot_download(
                                repo_id=model_id,
                                revision="main",
                                allow_patterns=[selected_file],
                                tqdm_class=custom_class,
                            )
                        else:
                            # Not a GGUF repo - download normally
                            logger.info(
                                f"Not a GGUF model, downloading all files for {model_id}"
                            )
                            local_dir = snapshot_download(
                                repo_id=model_id,
                                revision="main",
                                tqdm_class=custom_class,
                            )
                    except Exception as e:
                        logger.warning(
                            f"Could not check files in {model_id}: {e}. "
                            "Falling back to downloading all files"
                        )
                        local_dir = snapshot_download(
                            repo_id=model_id, revision="main", tqdm_class=custom_class
                        )
                finally:
                    huggingface_hub.file_download.tqdm = original

                loop.call_soon_threadsafe(
                    queue.put_nowait, {"event": "done", "local_dir": local_dir}
                )

            worker = asyncio.to_thread(run_download)

            # consume events until "done"
            done = False

            # start the worker
            task = asyncio.create_task(worker)
            try:
                while not done:
                    # Use timeout to prevent infinite blocking if worker fails
                    try:
                        evt = await asyncio.wait_for(queue.get(), timeout=300)
                        yield evt
                        done = evt.get("event") == "done"
                    except TimeoutError:
                        # Check if worker task has failed
                        if task.done():
                            # Task finished but no "done" event - likely an error
                            try:
                                await (
                                    task
                                )  # This will raise the exception if there was one
                            except Exception as task_error:
                                yield {
                                    "event": "error",
                                    "message": f"Download failed: {str(task_error)}",
                                }
                                raise
                        # If task still running, continue waiting
                        continue

                # ensure thread finished (propagate any exception)
                await task
            finally:
                # Clean up task if we exit early
                if not task.done():
                    task.cancel()
        except RepositoryNotFoundError as e:
            logger.error(f"Model {model_id} not found")
            raise NotFoundError(
                f"Model '{model_id}' not found. Check if the model exists on "
                f"HuggingFace and that you specified the correct repo path."
            ) from e
        except Exception as e:
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


def make_reporting_tqdm(queue: asyncio.Queue[dict], loop: asyncio.AbstractEventLoop):
    """Create a tqdm class that reports progress to an asyncio queue.

    Args:
        queue: Asyncio queue to send progress events to
        loop: Event loop reference (must be passed from async context)
    """

    class ReportingTQDM(tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "event": "start",
                    "desc": self.desc,
                    "total": int(self.total) if self.total is not None else None,
                    "n": int(self.n),
                },
            )

        def update(self, n=1):
            r = super().update(n)
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "event": "progress",
                    "desc": self.desc,
                    "total": int(self.total) if self.total is not None else None,
                    "n": int(self.n),
                },
            )
            return r

        def close(self):
            loop.call_soon_threadsafe(
                queue.put_nowait,
                {
                    "event": "end",
                    "desc": self.desc,
                    "total": int(self.total) if self.total is not None else None,
                    "n": int(self.n),
                },
            )
            super().close()

    return ReportingTQDM
