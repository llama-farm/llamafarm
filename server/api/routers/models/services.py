import json
from dataclasses import asdict

from config.datamodel import Provider
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from server.services.model_service import ModelService


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
        async for evt in ModelService.download_model(
            request.provider, request.model_name
        ):
            yield f"data: {json.dumps(evt)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
