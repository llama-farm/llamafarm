from __future__ import annotations  # noqa: I001

import contextvars
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware  # type: ignore[import-not-found]

from core.logging import FastAPIStructLogger
from core.settings import settings


logger = FastAPIStructLogger()

# Context variable to carry the client-provided working directory per-request
client_cwd: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "client_cwd", default=None
)


class ClientCWDHeaderMiddleware(BaseHTTPMiddleware):
    """Middleware that propagates the X-LF-Client-CWD header into a contextvar.

    When settings.lf_use_data_dir is False, the server should interpret client
    paths relative to the client's current working directory. The CLI sends the
    path in the X-LF-Client-CWD header for localhost calls. This middleware
    captures that header and makes it available to downstream services via
    the `client_cwd` context variable.
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)

    async def dispatch(self, request, call_next):
        # Only relevant when not using the shared data dir
        if not settings.lf_use_data_dir and (
            header_value := request.headers.get("x-lf-client-cwd")
        ):
            token = client_cwd.set(header_value)
            try:
                response = await call_next(request)
            finally:
                # Restore previous value to avoid leaking context between requests
                client_cwd.reset(token)
            return response
        # Default path
        return await call_next(request)
