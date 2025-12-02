from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger()


class NotFoundError(Exception):
    def __init__(self, message: str | None = None):
        super().__init__(message or "Not found")


class NamespaceNotFoundError(NotFoundError):
    def __init__(self, namespace: str):
        self.namespace = namespace
        super().__init__(f"Namespace {namespace} not found")


class DatasetNotFoundError(NotFoundError):
    def __init__(self, dataset: str):
        self.dataset = dataset
        super().__init__(f"Dataset {dataset} not found")


class DatabaseNotFoundError(NotFoundError):
    def __init__(self, database: str):
        self.database = database
        super().__init__(f"Database '{database}' not found")


class ProjectNotFoundError(NotFoundError):
    """Raised when a project doesn't exist."""

    def __init__(self, namespace: str, project_id: str):
        self.namespace = namespace
        self.project_id = project_id
        super().__init__(f"Project {namespace}/{project_id} not found")


class ProjectConfigError(Exception):
    """Raised when project exists but config is invalid or missing."""

    def __init__(self, namespace: str, project_id: str, message: str | None = None):
        self.namespace = namespace
        self.project_id = project_id
        msg = (
            message or f"Invalid or missing config for project {namespace}/{project_id}"
        )
        super().__init__(msg)


class ConfigTemplateNotFoundError(NotFoundError):
    """Raised when the specified config template cannot be located on disk."""

    def __init__(self, template: str, searched_paths: list[str] | None = None):
        self.template = template
        self.searched_paths = searched_paths or []
        message = f"No config template file found for template '{template}'. " + (
            f"Searched: {', '.join(self.searched_paths)}" if self.searched_paths else ""
        )
        super().__init__(message)


class ReservedNamespaceError(Exception):
    """Raised when a namespace is reserved."""

    def __init__(self, namespace: str):
        self.namespace = namespace
        super().__init__(f"Namespace {namespace} is reserved")


# FastAPI exception handlers kept alongside their corresponding error types for cohesion
class ErrorResponse(BaseModel):
    error: str
    message: str
    details: str | None = None


async def _handle_project_not_found(request: Request, exc: Exception) -> Response:
    payload = ErrorResponse(error="ProjectNotFound", message=str(exc))
    return JSONResponse(status_code=404, content=payload.model_dump())


async def _handle_project_config_error(request: Request, exc: Exception) -> Response:
    payload = ErrorResponse(error="ProjectConfigError", message=str(exc))
    return JSONResponse(status_code=422, content=payload.model_dump())


async def _handle_schema_not_found(request: Request, exc: Exception) -> Response:
    payload = ErrorResponse(error="ConfigTemplateNotFound", message=str(exc))
    return JSONResponse(status_code=404, content=payload.model_dump())


async def _handle_config_error(request: Request, exc: Exception) -> Response:
    payload = ErrorResponse(error="ProjectConfigError", message=str(exc))
    return JSONResponse(status_code=422, content=payload.model_dump())


async def _handle_generic_not_found(request: Request, exc: Exception) -> Response:
    payload = ErrorResponse(error="NotFound", message=str(exc))
    return JSONResponse(status_code=404, content=payload.model_dump())


async def _handle_reserved_namespace_error(
    request: Request, exc: ReservedNamespaceError
) -> Response:
    payload = ErrorResponse(error="ReservedNamespace", message=str(exc))
    return JSONResponse(status_code=400, content=payload.model_dump())


async def _handle_unexpected_error(request: Request, exc: Exception) -> Response:
    # Log the exception with full stack trace
    logger.exception(
        "Unhandled exception in request handler",
        exception_class=exc.__class__.__name__,
        path=request.url.path,
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
    app.add_exception_handler(ConfigTemplateNotFoundError, _handle_schema_not_found)
    app.add_exception_handler(ReservedNamespaceError, _handle_reserved_namespace_error)
    # ConfigError comes from config package; import locally to avoid
    # hard dependency at import time
    from config import ConfigError  # type: ignore

    app.add_exception_handler(ConfigError, _handle_config_error)
    app.add_exception_handler(NotFoundError, _handle_generic_not_found)
    app.add_exception_handler(Exception, _handle_unexpected_error)
