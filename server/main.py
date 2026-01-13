import os
import shutil
from pathlib import Path

import uvicorn
from fastapi_mcp import FastApiMCP
from llamafarm_common.pidfile import write_pid

from api.main import llama_farm_api
from core.logging import setup_logging
from core.settings import settings

# Configure logging FIRST, before anything else
setup_logging(settings.LOG_JSON_FORMAT, settings.LOG_LEVEL, settings.LOG_FILE)

# Write PID file for service discovery
write_pid("server")

# Create the data directory if it doesn't exist
os.makedirs(settings.lf_data_dir, exist_ok=True)
os.makedirs(os.path.join(settings.lf_data_dir, "projects"), exist_ok=True)

# Copy seed projects to projects directory
seed_source = Path(__file__).parent / "seeds"
seed_dest = Path(settings.lf_data_dir) / "projects" / "llamafarm"
shutil.copytree(seed_source, seed_dest, dirs_exist_ok=True)

app = llama_farm_api()

mcp = FastApiMCP(
    app,
    include_tags=["mcp"],
)

mcp.mount_http(
    mount_path="/mcp",
)


if __name__ == "__main__":
    # Nuitka-compiled binaries can't use string imports for uvicorn
    # Check if running compiled (Nuitka sets __compiled__)
    is_compiled = "__compiled__" in dir()

    uvicorn.run(
        app if is_compiled else "server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD if not is_compiled else False,
        # Limit reload scanning to the server app directory only
        reload_dirs=[str(Path(__file__).parent.resolve())] if not is_compiled else None,
        log_config=None,  # Disable uvicorn's log config (handled in setup_logging)
        access_log=False,  # Disable uvicorn access logs (handled by StructLogMiddleware)
    )
