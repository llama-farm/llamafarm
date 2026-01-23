"""Common types shared across all Universal Runtime endpoints."""

from typing import Any, Literal

from pydantic import BaseModel


class UsageInfo(BaseModel):
    """Usage information for API responses."""

    prompt_tokens: int = 0
    total_tokens: int = 0


class ListResponse(BaseModel):
    """Standard list response wrapper."""

    object: Literal["list"] = "list"
    data: list[Any]
    model: str | None = None
    usage: dict[str, Any] | None = None


class ErrorDetail(BaseModel):
    """Error detail for HTTP exceptions."""

    detail: str
    code: str | None = None
