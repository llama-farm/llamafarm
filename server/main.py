import os
import uvicorn
from fastapi import FastAPI
from api.routers.projects import router as projects_router
from core.config import settings
from core.version import version

# Create the data directory if it doesn't exist
os.makedirs(settings.lf_data_dir, exist_ok=True)
os.makedirs(os.path.join(settings.lf_data_dir, "projects"), exist_ok=True)
from api.routers.inference import router as inference_router

app = FastAPI()

app.include_router(projects_router, prefix="/v1")
app.include_router(inference_router, prefix="/v1")

@app.get("/")
def read_root():
  return {"message": "Hello, World!"}

@app.get("/info")
def get_info():
  return {
    "version": version,
    "data_directory": settings.lf_data_dir,
  }

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
