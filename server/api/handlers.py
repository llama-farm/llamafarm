import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.errors import NotFoundError, ProjectConfigError, ProjectNotFoundError
from api.models import ErrorResponse
from config import ConfigError


logger = logging.getLogger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers for consistent API errors."""

    @app.exception_handler(ProjectNotFoundError)
    async def handle_project_not_found(
        request: Request, exc: ProjectNotFoundError
    ) -> JSONResponse:
        # Attach request context and exception info
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

    @app.exception_handler(ProjectConfigError)
    async def handle_project_config_error(
        request: Request, exc: ProjectConfigError
    ) -> JSONResponse:
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

    @app.exception_handler(ConfigError)
    async def handle_config_error(request: Request, exc: ConfigError) -> JSONResponse:
        logger.warning(
            "Configuration error",
            extra={
                "path": str(request.url.path),
                "message": str(exc),
            },
        )
        payload = ErrorResponse(error="ProjectConfigError", message=str(exc))
        return JSONResponse(status_code=422, content=payload.model_dump())

    @app.exception_handler(NotFoundError)
    async def handle_generic_not_found(request: Request, exc: NotFoundError) -> JSONResponse:
        logger.info(
            "Resource not found",
            extra={
                "path": str(request.url.path),
                "message": str(exc),
            },
        )
        payload = ErrorResponse(error="NotFound", message=str(exc))
        return JSONResponse(status_code=404, content=payload.model_dump())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
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


