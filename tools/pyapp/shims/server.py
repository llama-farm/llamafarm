"""PyApp entry point for llamafarm-server.

Adds the server package directory to sys.path so that bare imports
(from api.main import ...) work alongside prefixed imports
(from server.services.xxx import ...).
"""
import os
import sys

# Allow bare imports (from api.xxx, from core.xxx, etc.)
sys.path.insert(0, os.path.dirname(__file__))

# Signal PyApp mode for runtime detection
os.environ["LLAMAFARM_PYAPP"] = "1"

# Import main module — this executes module-level setup
# (logging, PID file, seed copying, FastAPI app creation)
from server import main  # noqa: F401

if __name__ == '__main__':
    import uvicorn
    from server.core.settings import settings

    uvicorn.run(
        main.app,
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        log_config=None,
        access_log=False,
    )
