import uvicorn
from fastapi import FastAPI
from api.routers.projects import router as projects_router

app = FastAPI()

app.include_router(projects_router, prefix="/v1")

@app.get("/")
def read_root():
  return {"message": "Hello, World!"}

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8000)
