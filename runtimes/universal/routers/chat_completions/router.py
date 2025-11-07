from fastapi import APIRouter

from .types import ChatCompletionRequest
from .service import ChatCompletionsService

router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(chat_request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint.

    Supports any HuggingFace causal language model.
    """
    return await ChatCompletionsService().chat_completions(chat_request)
