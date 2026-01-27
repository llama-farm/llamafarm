"""Unified error handling service for Universal Runtime.

Provides consistent error handling patterns that were previously
duplicated across all endpoint handlers.
"""

import functools
import logging
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import HTTPException

logger = logging.getLogger(__name__)

T = TypeVar("T")


class UniversalRuntimeError(Exception):
    """Base exception for Universal Runtime errors."""

    def __init__(self, message: str, status_code: int = 500, code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


class ModelNotFoundError(UniversalRuntimeError):
    """Raised when a requested model is not found."""

    def __init__(self, model_id: str, model_type: str = "model"):
        super().__init__(
            message=f"{model_type.capitalize()} not found: {model_id}",
            status_code=404,
            code="MODEL_NOT_FOUND",
        )
        self.model_id = model_id
        self.model_type = model_type


class ModelNotFittedError(UniversalRuntimeError):
    """Raised when attempting to use an unfitted model."""

    def __init__(self, model_id: str):
        super().__init__(
            message=f"Model '{model_id}' not fitted. Call fit() first or load a pre-trained model.",
            status_code=400,
            code="MODEL_NOT_FITTED",
        )
        self.model_id = model_id


class ValidationError(UniversalRuntimeError):
    """Raised for request validation errors."""

    def __init__(self, message: str):
        super().__init__(message=message, status_code=400, code="VALIDATION_ERROR")


class BackendNotInstalledError(UniversalRuntimeError):
    """Raised when a required backend is not installed."""

    def __init__(self, backend: str, install_hint: str | None = None):
        message = f"Backend '{backend}' not installed."
        if install_hint:
            message += f" {install_hint}"
        super().__init__(message=message, status_code=400, code="BACKEND_NOT_INSTALLED")
        self.backend = backend


def handle_endpoint_errors(
    endpoint_name: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for consistent endpoint error handling.

    Catches and formats errors in a consistent way across all endpoints.

    Args:
        endpoint_name: Name of the endpoint for logging

    Returns:
        Decorated function with error handling

    Usage:
        @app.post("/v1/embeddings")
        @handle_endpoint_errors("create_embeddings")
        async def create_embeddings(request: EmbeddingRequest):
            ...
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return await func(*args, **kwargs)
            except HTTPException:
                # Re-raise FastAPI HTTPExceptions as-is
                raise
            except UniversalRuntimeError as e:
                # Convert our custom errors to HTTPException
                logger.warning(f"Error in {endpoint_name}: {e.message}")
                raise HTTPException(status_code=e.status_code, detail=e.message) from e
            except ImportError as e:
                # Handle missing dependencies
                logger.error(f"Import error in {endpoint_name}: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Required dependency not installed: {str(e)}",
                ) from e
            except ValueError as e:
                # Handle validation errors
                logger.warning(f"Validation error in {endpoint_name}: {e}")
                raise HTTPException(status_code=400, detail=str(e)) from e
            except Exception as e:
                # Log and wrap unexpected errors
                logger.error(f"Error in {endpoint_name}: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=str(e)) from e

        return wrapper

    return decorator


def format_error_response(
    message: str, code: str | None = None, details: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Format an error response consistently.

    Args:
        message: Error message
        code: Optional error code
        details: Optional additional details

    Returns:
        Formatted error response dict
    """
    response: dict[str, Any] = {"error": {"message": message}}
    if code:
        response["error"]["code"] = code
    if details:
        response["error"]["details"] = details
    return response
