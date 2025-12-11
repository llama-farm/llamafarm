import asyncio
import json
import logging
from dataclasses import asdict

from config.datamodel import Provider
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from llamafarm_common import (
    list_gguf_files,
    parse_model_with_quantization,
    parse_quantization_from_filename,
)
from pydantic import BaseModel
from server.services.disk_space_service import DiskSpaceService
from server.services.model_service import ModelService

logger = logging.getLogger(__name__)


class DownloadModelRequest(BaseModel):
    provider: Provider = Provider.universal
    model_name: str


class ValidateDownloadRequest(BaseModel):
    model_name: str


class GGUFOption(BaseModel):
    filename: str
    quantization: str | None
    size_bytes: int
    size_human: str


class GGUFOptionsResponse(BaseModel):
    options: list[GGUFOption]


router = APIRouter(prefix="/models", tags=["models"])


@router.get("")
def list_models(provider: Provider = Provider.universal):
    """List all models available on disk (HuggingFace cache).

    Returns cached models from the HuggingFace cache directory.
    This includes any models that have been downloaded and are
    available for use with the Universal Runtime.
    """

    try:
        cached_models = ModelService.list_cached_models(provider)
        # Convert CachedModel dataclasses to dicts for JSON serialization
        return {"data": [asdict(model) for model in cached_models]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/download")
async def download_model(request: DownloadModelRequest):
    """Download/cache a model for the given provider and model name."""

    # Check disk space before starting download
    # Run in thread pool to avoid blocking the async event loop (HuggingFace API calls are blocking)
    try:
        validation = await asyncio.to_thread(
            DiskSpaceService.validate_space_for_download, request.model_name
        )

        # If critical (can't download), return error immediately
        if not validation.can_download:
            raise HTTPException(
                status_code=400,
                detail=validation.message,
            )

        # If warning (low space), we'll emit warning event in stream
        warning_message = validation.message if validation.warning else None

    except HTTPException:
        # Re-raise HTTP exceptions (critical errors)
        raise
    except Exception as e:
        # If disk space check fails, log warning and proceed (graceful degradation)
        logger.warning(
            f"Disk space check failed for model '{request.model_name}': {e}. "
            "Proceeding with download.",
        )
        warning_message = None

    async def event_stream():
        # Emit warning event if space is low
        if warning_message:
            warning_event = {
                "event": "warning",
                "message": warning_message,
            }
            yield f"data: {json.dumps(warning_event)}\n\n"

        try:
            async for evt in ModelService.download_model(
                request.provider, request.model_name
            ):
                yield f"data: {json.dumps(evt)}\n\n"
        except Exception as e:
            logger.error(
                f"Error during model download for provider '{request.provider}', "
                f"model '{request.model_name}': {e}",
                exc_info=True,
            )
            error_event = {
                "event": "error",
                "message": "An internal error occurred while downloading the model.",
            }
            yield f"data: {json.dumps(error_event)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/validate-download")
def validate_download(request: ValidateDownloadRequest):
    """Validate if there's sufficient disk space for a model download.

    Returns validation result with can_download, warning, message, and space info.
    """
    logger.info(f"Validating disk space for model download: {request.model_name}")
    try:
        validation = DiskSpaceService.validate_space_for_download(request.model_name)

        logger.info(
            f"Validation result for {request.model_name}: "
            f"can_download={validation.can_download}, "
            f"warning={validation.warning}, "
            f"required={validation.required_bytes / (1024**3):.2f} GB, "
            f"available={validation.available_bytes / (1024**3):.2f} GB"
        )

        return {
            "can_download": validation.can_download,
            "warning": validation.warning,
            "message": validation.message,
            "available_bytes": validation.available_bytes,
            "required_bytes": validation.required_bytes,
            "cache_info": asdict(validation.cache_info),
            "system_info": asdict(validation.system_info),
        }
    except Exception as e:
        logger.warning(
            f"Disk space validation failed for model '{request.model_name}': {e}",
            exc_info=True,
        )
        # On error, allow download but with warning
        # Don't expose exception details to users for security
        return {
            "can_download": True,
            "warning": True,
            "message": "Could not validate disk space. Proceeding with download.",
            "available_bytes": 0,
            "required_bytes": 0,
            "cache_info": {
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "path": "",
                "percent_free": 0.0,
            },
            "system_info": {
                "total_bytes": 0,
                "used_bytes": 0,
                "free_bytes": 0,
                "path": "",
                "percent_free": 0.0,
            },
        }


@router.get("/{model_id:path}/quantizations", response_model=GGUFOptionsResponse)
def get_gguf_options(model_id: str) -> GGUFOptionsResponse:
    """Get all available GGUF quantization options for a model with file sizes.

    Args:
        model_id: HuggingFace model identifier (e.g., "unsloth/Qwen3-1.7B-GGUF")

    Returns:
        GGUFOptionsResponse with list of GGUF options, each containing:
        - filename: GGUF filename
        - quantization: Quantization type (e.g., "Q4_K_M")
        - size_bytes: File size in bytes
        - size_human: Human-readable size (e.g., "1.2 GB")
    """
    try:
        from huggingface_hub import HfApi

        # Parse model ID to remove quantization suffix if present
        base_model_id, _ = parse_model_with_quantization(model_id)
        api = HfApi()

        options = []

        # Primary method: use model_info with files_metadata to get sizes from siblings
        try:
            model_info = api.model_info(base_model_id, files_metadata=True)
            if hasattr(model_info, "siblings") and model_info.siblings:
                for sibling in model_info.siblings:
                    filename = getattr(sibling, "rfilename", None) or getattr(
                        sibling, "filename", None
                    )
                    if filename and filename.endswith(".gguf"):
                        quantization = parse_quantization_from_filename(filename)
                        size_bytes = getattr(sibling, "size", None) or 0

                        # Log files that don't match quantization pattern for debugging
                        if not quantization:
                            logger.debug(
                                f"GGUF file without quantization pattern: {filename} (size: {size_bytes} bytes)"
                            )

                        # Only include options with valid quantization and size
                        if size_bytes > 0 and quantization:
                            # Format size human-readable
                            size_human = _format_bytes(size_bytes)
                            options.append(
                                {
                                    "filename": filename,
                                    "quantization": quantization,
                                    "size_bytes": size_bytes,
                                    "size_human": size_human,
                                }
                            )

                if options:
                    logger.info(
                        f"Got {len(options)} GGUF options from siblings for {base_model_id}"
                    )
                    return GGUFOptionsResponse(options=options)
        except Exception as e:
            logger.debug(f"Could not get model info for {base_model_id}: {e}")

        # Fallback: list GGUF files and get individual sizes
        try:
            gguf_files = list_gguf_files(base_model_id)
            if gguf_files:
                for filename in gguf_files:
                    quantization = parse_quantization_from_filename(filename)
                    try:
                        file_info = api.get_path_info(
                            repo_id=base_model_id, path=filename, repo_type="model"
                        )
                        size_bytes = getattr(file_info, "size", None) or 0

                        # Only include options with valid quantization and size
                        if size_bytes > 0 and quantization:
                            size_human = _format_bytes(size_bytes)
                            options.append(
                                {
                                    "filename": filename,
                                    "quantization": quantization,
                                    "size_bytes": size_bytes,
                                    "size_human": size_human,
                                }
                            )
                    except Exception:
                        continue

                if options:
                    logger.info(
                        f"Got {len(options)} GGUF options via file listing for {base_model_id}"
                    )
                    return GGUFOptionsResponse(options=options)
        except Exception as e:
            logger.debug(f"Could not get GGUF options via file listing: {e}")

        # If no options found, return empty list
        if not options:
            logger.warning(f"No GGUF files found for {base_model_id}")
            return GGUFOptionsResponse(options=[])

        return GGUFOptionsResponse(options=options)

    except ImportError:
        logger.warning("huggingface_hub not available for GGUF options")
        raise HTTPException(
            status_code=500, detail="HuggingFace Hub API not available"
        ) from None
    except Exception as e:
        logger.error(f"Error getting GGUF options for {model_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get GGUF options: {str(e)}",
        ) from e


def _format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    unit_idx = 0
    while size >= 1024 and unit_idx < len(units) - 1:
        size /= 1024.0
        unit_idx += 1
    if unit_idx == 0:
        return f"{int(size)}{units[unit_idx]}"
    return f"{size:.2f}{units[unit_idx]}"


@router.delete("/{model_name:path}")
def delete_model(model_name: str, provider: Provider = Provider.universal):
    """Delete a cached model from disk.

    This will delete ALL revisions of the specified model from the HuggingFace cache.

    Args:
        model_name: The model identifier to delete (e.g., "meta-llama/Llama-2-7b-hf")
        provider: The model provider (defaults to universal)

    Returns:
        Dict with deleted model info including:
        - model_name: The deleted model identifier
        - revisions_deleted: Number of revisions deleted
        - size_freed: Total bytes freed from disk
        - path: Path to the deleted model cache directory

    Raises:
        404: If the model is not found in the cache
        400: If the provider is not supported
        500: For other errors
    """
    try:
        return ModelService.delete_model(provider, model_name)
    except ValueError as e:
        # Model not found or invalid provider
        if "not found" in str(e).lower():
            logger.warning(f"Model '{model_name}' not found for deletion")
            raise HTTPException(status_code=404, detail=str(e)) from e
        logger.error(f"Invalid request to delete model '{model_name}': {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(
            f"Failed to delete model '{model_name}' for provider '{provider}': {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "An error occurred while deleting the model. "
                "Please contact support if the issue persists."
            ),
        ) from e
