import logging

from config import ConfigError
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from api.errors import (
    NotFoundError,
    ProjectConfigError,
    ProjectNotFoundError,
    SchemaNotFoundError,
)
from api.models import ErrorResponse

logger = logging.getLogger(__name__)


async def _handle_project_not_found(
    request: Request, exc: Exception
) -> Response:
    logger.info(
        "Project not found",
        extra={
            "path": str(request.url.path),
            "namespace": getattr(exc, "namespace", None),
            "project_id": getattr(exc, "project_id", None),
        },
    )
    payload = ErrorResponse(error="ProjectNotFound", message=str(exc))
    return JSONResponse(status_code=404, content=payload.model_dump())


async def _handle_project_config_error(
    request: Request, exc: Exception
) -> Response:
    logger.warning(
        "Invalid project configuration",
        extra={
            "path": str(request.url.path),
            "namespace": getattr(exc, "namespace", None),
            "project_id": getattr(exc, "project_id", None),
            "message": str(exc),
        },
    )
    payload = ErrorResponse(error="ProjectConfigError", message=str(exc))
    return JSONResponse(status_code=422, content=payload.model_dump())


async def _handle_schema_not_found(
    request: Request, exc: Exception
) -> Response:
    logger.info(
        "Schema not found",
        extra={
            "path": str(request.url.path),
            "message": str(exc),
        },
    )
    payload = ErrorResponse(error="SchemaNotFound", message=str(exc))
    return JSONResponse(status_code=404, content=payload.model_dump())


async def _handle_config_error(request: Request, exc: Exception) -> Response:
    logger.warning(
        "Configuration error",
        extra={
            "path": str(request.url.path),
            "message": str(exc),
        },
    )
    payload = ErrorResponse(error="ProjectConfigError", message=str(exc))
    return JSONResponse(status_code=422, content=payload.model_dump())


async def _handle_generic_not_found(
    request: Request, exc: Exception
) -> Response:
    logger.info(
        "Resource not found",
        extra={
            "path": str(request.url.path),
            "message": str(exc),
        },
    )
    payload = ErrorResponse(error="NotFound", message=str(exc))
    return JSONResponse(status_code=404, content=payload.model_dump())


async def _handle_unexpected_error(
    request: Request, exc: Exception
) -> Response:
    # Log with full stack trace
    logger.exception(
        "Unhandled server error",
        extra={
            "path": str(request.url.path),
        },
    )
    payload = ErrorResponse(
        error="InternalServerError",
        message="An unexpected error occurred",
    )
    return JSONResponse(status_code=500, content=payload.model_dump())


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent API errors."""

    app.add_exception_handler(ProjectNotFoundError, _handle_project_not_found)
    app.add_exception_handler(ProjectConfigError, _handle_project_config_error)
    app.add_exception_handler(SchemaNotFoundError, _handle_schema_not_found)
    app.add_exception_handler(ConfigError, _handle_config_error)
    app.add_exception_handler(NotFoundError, _handle_generic_not_found)
    app.add_exception_handler(Exception, _handle_unexpected_error)


