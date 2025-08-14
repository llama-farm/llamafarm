from starlette.types import ASGIApp, Receive, Scope, Send


class ErrorHandlerMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        # Pass-through; rely on FastAPI exception handlers for consistent responses
        await self.app(scope, receive, send)
