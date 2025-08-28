import fastapi
from asgi_correlation_id import CorrelationIdMiddleware

import api.routers as routers
from api.errors import register_exception_handlers
from api.middleware.errors import ErrorHandlerMiddleware
from api.middleware.structlog import StructLogMiddleware
from core.logging import FastAPIStructLogger
from core.settings import settings
from core.version import version

logger = FastAPIStructLogger()

API_PREFIX = "/v1"


def llama_farm_api() -> fastapi.FastAPI:
    app = fastapi.FastAPI()

    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(StructLogMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    # Register global exception handlers
    register_exception_handlers(app)

    app.include_router(routers.projects_router, prefix=API_PREFIX)
    app.include_router(routers.datasets_router, prefix=API_PREFIX)
    app.include_router(routers.inference_router, prefix=API_PREFIX)

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
