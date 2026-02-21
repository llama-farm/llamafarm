"""LlamaFarm SDK errors."""

from __future__ import annotations


class LlamaFarmError(Exception):
    """Base error for all LlamaFarm SDK errors."""

    def __init__(self, message: str, status_code: int | None = None, detail: str | None = None):
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)


class ConnectionError(LlamaFarmError):
    """Cannot connect to LlamaFarm server."""


class NotFoundError(LlamaFarmError):
    """Resource not found (project, model, job, etc.)."""


class ValidationError(LlamaFarmError):
    """Request validation failed."""


class ServerError(LlamaFarmError):
    """Server-side error."""


def _raise_for_status(response) -> None:
    """Raise appropriate LlamaFarmError for non-2xx responses."""
    if response.status_code < 400:
        return

    try:
        body = response.json()
        detail = body.get("detail", response.text)
    except Exception:
        detail = response.text

    if response.status_code == 404:
        raise NotFoundError(f"Not found: {detail}", status_code=404, detail=detail)
    elif response.status_code == 422:
        raise ValidationError(f"Validation error: {detail}", status_code=422, detail=detail)
    elif response.status_code >= 500:
        raise ServerError(f"Server error: {detail}", status_code=response.status_code, detail=detail)
    else:
        raise LlamaFarmError(
            f"HTTP {response.status_code}: {detail}",
            status_code=response.status_code,
            detail=detail,
        )
