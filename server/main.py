import os

import uvicorn

from api.main import llama_farm_api
from core.logging import setup_logging
from core.settings import settings

# Configure logging FIRST, before anything else
setup_logging(settings.LOG_JSON_FORMAT, settings.LOG_LEVEL)

# Create the data directory if it doesn't exist
os.makedirs(settings.lf_data_dir, exist_ok=True)
os.makedirs(os.path.join(settings.lf_data_dir, "projects"), exist_ok=True)

app = llama_farm_api()

if __name__ == "__main__":
  uvicorn.run(
    "server.main:app",
    host="0.0.0.0",
    port=8000,
    reload=True,
    reload_dirs=["../"],
    log_config=None,  # Disable uvicorn's log config (handled in setup_logging)
    access_log=False, # Disable uvicorn access logs (handled by StructLogMiddleware)
  )
