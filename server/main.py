import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.exception_handlers import http_exception_handler
from api.routers.projects import router as projects_router
from api.routers.datasets import router as datasets_router
from core.config import settings
from core.version import version
import traceback
# Create the data directory if it doesn't exist
os.makedirs(settings.lf_data_dir, exist_ok=True)
os.makedirs(os.path.join(settings.lf_data_dir, "projects"), exist_ok=True)
from api.routers.inference import router as inference_router

app = FastAPI()

@app.exception_handler(HTTPException)
async def global_exception_handler(request, exc: HTTPException):
    if exc.status_code >= 500:
        print(f"Unhandled exception for request {request.url.path}")
        print(traceback.format_exc())

    return await http_exception_handler(request, exc)

app.include_router(projects_router, prefix="/v1")
app.include_router(inference_router, prefix="/v1")
app.include_router(datasets_router, prefix="/v1")

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
  uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["../"])
