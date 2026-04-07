"""OpenAI-compatible text completions endpoint (/v1/completions).

Accepts a raw prompt string and generates a completion without applying
any chat template. Useful for models that require a specific prompt format
that doesn't align with the GGUF's embedded chat template.
"""

import logging
import time
import uuid

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()
logger = logging.getLogger(__name__)


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""

    model: str
    prompt: str
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    max_tokens: int | None = 512
    stop: str | list[str] | None = None
    # GGUF model parameters
    n_ctx: int | None = None
    n_gpu_layers: int | None = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""

    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[dict]
    usage: dict


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Raw text completions — no chat template applied."""
    from server import load_language

    model = await load_language(
        request.model,
        n_ctx=request.n_ctx,
        n_gpu_layers=request.n_gpu_layers,
    )

    max_tokens = request.max_tokens if request.max_tokens is not None else 512
    stop = request.stop if isinstance(request.stop, list) else ([request.stop] if request.stop else [])

    logger.info(f"[completions] model={request.model} prompt_len={len(request.prompt)} max_tokens={max_tokens}")

    result = await model._generate_from_prompt(
        prompt=request.prompt,
        max_tokens=max_tokens,
        temperature=request.temperature if request.temperature is not None else 1.0,
        top_p=request.top_p if request.top_p is not None else 1.0,
        stop=stop,
        thinking_budget=None,
    )

    return CompletionResponse(
        id=f"cmpl-{uuid.uuid4().hex[:8]}",
        created=int(time.time()),
        model=request.model,
        choices=[{
            "index": 0,
            "text": result,
            "finish_reason": "stop",
        }],
        usage={
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    )
