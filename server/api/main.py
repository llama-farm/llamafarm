import fastapi
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

import api.routers as routers
from api.errors import register_exception_handlers
from api.middleware.errors import ErrorHandlerMiddleware
from api.middleware.structlog import StructLogMiddleware
from core.logging import FastAPIStructLogger
from core.settings import settings
from core.version import version

logger = FastAPIStructLogger()

API_PREFIX = "/v1"


class NoNoneJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        # `jsonable_encoder` lets you apply exclude_none globally
        return super().render(jsonable_encoder(content, exclude_none=True))


def llama_farm_api() -> fastapi.FastAPI:
    app = fastapi.FastAPI(default_response_class=NoNoneJSONResponse)

    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(StructLogMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # Register global exception handlers
    register_exception_handlers(app)

    app.include_router(routers.projects_router, prefix=API_PREFIX)
    app.include_router(routers.datasets_router, prefix=API_PREFIX)
    app.include_router(routers.inference_router, prefix=API_PREFIX)
    app.include_router(routers.rag_router, prefix=API_PREFIX)
    # Health endpoints are exposed at the root (no version prefix)
    app.include_router(routers.health_router)

    app.add_api_route(
        path="/", methods=["GET"], endpoint=lambda: {"message": "Hello, World!"}
    )
    app.add_api_route(
        path="/info",
        methods=["GET"],
        endpoint=lambda: {
            "version": version,
            "data_directory": settings.lf_data_dir,
        },
    )

    return app


app = llama_farm_api()
