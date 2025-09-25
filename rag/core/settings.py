from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

default_data_dir = str(Path.home() / ".llamafarm")


class Settings(BaseSettings, env_file=".env"):
    LF_DATA_DIR: str = default_data_dir

    # Ollama Configuration
    OLLAMA_HOST: str = "http://localhost:11434"

    # Celery Broker Override Configuration
    CELERY_BROKER_URL: str = ""
    CELERY_RESULT_BACKEND: str = ""


settings = Settings()
