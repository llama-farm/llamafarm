import os
import shutil
from pathlib import Path

import uvicorn

from api.main import llama_farm_api
from core.logging import setup_logging
from core.settings import settings

# Configure logging FIRST, before anything else
setup_logging(settings.LOG_JSON_FORMAT, settings.LOG_LEVEL)

# Create the data directory if it doesn't exist
os.makedirs(settings.lf_data_dir, exist_ok=True)
os.makedirs(os.path.join(settings.lf_data_dir, "projects"), exist_ok=True)

# Copy seed projects to projects directory
seed_source = Path(__file__).parent / "seeds"
seed_dest = Path(settings.lf_data_dir) / "projects" / "llamafarm"
shutil.copytree(seed_source, seed_dest, dirs_exist_ok=True)

app = llama_farm_api()

if __name__ == "__main__":
    uvicorn.run(
        "server.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        reload_dirs=["../"],
        log_config=None,  # Disable uvicorn's log config (handled in setup_logging)
        access_log=False,  # Disable uvicorn access logs (handled by StructLogMiddleware)
    )
