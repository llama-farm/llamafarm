"""PyApp entry point for llamafarm-runtime.

Adds the runtime package directory to sys.path so that bare imports
(from models import ..., from state import ...) work alongside prefixed
imports (from runtime.models import ...).
"""
import os
import sys

# Allow bare imports (from models.xxx, from routers.xxx, etc.)
sys.path.insert(0, os.path.dirname(__file__))

# Signal PyApp mode for runtime detection
os.environ["LLAMAFARM_PYAPP"] = "1"

# Import server module — this executes module-level setup
# (logging, device detection, model loaders, FastAPI app creation)
from runtime import server  # noqa: F401

if __name__ == '__main__':
    import uvicorn
    from llamafarm_common.pidfile import write_pid

    write_pid("universal-runtime")

    port = int(os.getenv("LF_RUNTIME_PORT", os.getenv("PORT", "11540")))
    host = os.getenv("LF_RUNTIME_HOST", os.getenv("HOST", "127.0.0.1"))

    uvicorn.run(
        server.app,
        host=host,
        port=port,
        log_config=None,
        access_log=False,
        ws_ping_interval=30.0,
        ws_ping_timeout=60.0,
    )
