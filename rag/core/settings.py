import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

try:
    default_data_dir = str(Path.home() / ".llamafarm")
except RuntimeError:
    _fb = os.environ.get("USERPROFILE") or os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
    try:
        _fallback = Path(_fb) if _fb else Path.cwd()
    except OSError:
        _fallback = Path(".")
    default_data_dir = str(_fallback / ".llamafarm")


class Settings(BaseSettings, env_file=".env"):
    LF_DATA_DIR: str = default_data_dir

    # Logging Configuration
    LOG_JSON_FORMAT: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_NAME: str = "rag"
    LOG_ACCESS_NAME: str = "rag.access"

    CELERY_LOG_LEVEL: str = "INFO"

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"

    # Celery Broker Override Configuration
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""


settings = Settings()
