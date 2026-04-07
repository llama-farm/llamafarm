import logging

from fastapi import APIRouter

from .service import ChatCompletionsService
from .types import ChatCompletionRequest

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports any HuggingFace causal language model.
    """
    # Debug log the incoming request
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            f"Incoming chat completion request:\n"
            f"{chat_request.model_dump_json(indent=2)}"
        )

    return await ChatCompletionsService().chat_completions(chat_request)
