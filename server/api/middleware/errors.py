import fastapi
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from api.errors import NamespaceNotFoundError, NotFoundError


class ErrorHandlerMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app
        pass

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # If the request is not an HTTP request, we don't need to do anything special
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        try:
            return await self.app(scope, receive, send)
        except NotFoundError as e:
            response = JSONResponse(status_code=404, content={"message": str(e)})
            await response(scope, receive, send)
