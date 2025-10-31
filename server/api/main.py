import os
from contextlib import asynccontextmanager

import fastapi
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import api.routers as routers
from api.errors import register_exception_handlers
from api.middleware.errors import ErrorHandlerMiddleware
from api.middleware.structlog import StructLogMiddleware
from core.designer import get_designer_dist_path
from core.logging import FastAPIStructLogger
from core.mcp_registry import cleanup_all_mcp_services
from core.settings import settings
from core.version import version

logger = FastAPIStructLogger()

API_PREFIX = "/v1"


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    """Manage application lifecycle (startup and shutdown)."""

    # Startup
    logger.info("Starting LlamaFarm API")
    yield
    # Shutdown
    logger.info("Shutting down LlamaFarm API")
    await cleanup_all_mcp_services()
    logger.info("Shutdown complete")


class NoNoneJSONResponse(JSONResponse):
    def render(self, content) -> bytes:
        # `jsonable_encoder` lets you apply exclude_none globally
        return super().render(jsonable_encoder(content, exclude_none=True))


def _inject_env_vars(html_content: str) -> str:
    """Inject VITE_APP_* environment variables into index.html."""
    import json
    import os

    # Collect VITE_APP_* environment variables
    env_vars = {
        key: value for key, value in os.environ.items() if key.startswith("VITE_APP_")
    }

    # Create JSON string
    env_json = json.dumps(env_vars)

    # Replace the placeholder using literal string replacement
    # (the placeholder is a fixed literal, so no regex needed)
    placeholder = '<noscript id="env-insertion-point"></noscript>'
    replacement = f"<script>var ENV={env_json}</script>"

    return html_content.replace(placeholder, replacement)


def llama_farm_api() -> fastapi.FastAPI:
    app = fastapi.FastAPI(default_response_class=NoNoneJSONResponse, lifespan=lifespan)

    app.add_middleware(ErrorHandlerMiddleware)
    app.add_middleware(StructLogMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    # Enable CORS for local designer/dev environments
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    # Register global exception handlers
    register_exception_handlers(app)

    app.include_router(routers.projects_router, prefix=API_PREFIX)
    app.include_router(routers.datasets_router, prefix=API_PREFIX)
    app.include_router(routers.rag_router, prefix=API_PREFIX)
    app.include_router(routers.upgrades_router, prefix=API_PREFIX)
    app.include_router(routers.examples_router, prefix=API_PREFIX)
    app.include_router(routers.models_router, prefix=API_PREFIX)
    # Health endpoints are exposed at the root (no version prefix)
    app.include_router(routers.health_router)

    app.add_api_route(
        path="/info",
        methods=["GET"],
        endpoint=lambda: {
            "version": version,
            "data_directory": settings.lf_data_dir,
        },
    )

    # Serve designer static files
    # This must be registered AFTER all API routes so API routes take precedence
    designer_dist_path = get_designer_dist_path()
    if designer_dist_path:
        # Mount static assets (JS, CSS, etc.)
        # These have specific paths so won't conflict
        static_dir = designer_dist_path / "assets"
        if static_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(static_dir)), name="assets")

        # Catch-all route for designer SPA
        # Registered last so API routes take precedence
        @app.get("/{path:path}", include_in_schema=False)
        async def serve_designer_spa(path: str):
            # API routes are already registered and will match first
            # This handler only runs if no API route matched

            # Try to serve static file first (favicon, etc.)
            # Normalize path to prevent directory traversal attacks
            try:
                # Resolve the path relative to designer_dist_path
                file_path = (designer_dist_path / path).resolve()
                # Ensure the resolved path is still within designer_dist_path
                # by checking that it's a subpath using robust path containment check
                designer_dist_path_resolved = designer_dist_path.resolve()
                # Use os.path.commonpath for robust containment check (safe even with symlinks)
                if os.path.commonpath(
                    [str(designer_dist_path_resolved), str(file_path)]
                ) != str(designer_dist_path_resolved):
                    raise fastapi.HTTPException(status_code=403, detail="Access denied")
            except (ValueError, RuntimeError):
                # Path resolution failed (e.g., contains invalid components)
                raise fastapi.HTTPException(
                    status_code=400, detail="Invalid path"
                ) from None

            if (
                file_path.exists()
                and file_path.is_file()
                and file_path.name != "index.html"
            ):
                return FileResponse(str(file_path))

            # For SPA routing, serve index.html for all other routes
            index_path = designer_dist_path / "index.html"
            if index_path.exists():
                html_content = index_path.read_text(encoding="utf-8")
                html_content = _inject_env_vars(html_content)
                return fastapi.Response(content=html_content, media_type="text/html")

            raise fastapi.HTTPException(status_code=404, detail="Designer not found")
    else:
        # No designer build found - serve minimal root endpoint
        @app.get("/", include_in_schema=False)
        async def root():
            return {"message": "LlamaFarm API", "designer": "not available"}

    return app


app = llama_farm_api()
