import json
import logging
from dataclasses import asdict

from config.datamodel import Provider
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from server.services.model_service import ModelService

logger = logging.getLogger(__name__)


class DownloadModelRequest(BaseModel):
    provider: Provider = Provider.universal
    model_name: str


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

    async def event_stream():
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
