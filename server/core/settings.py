from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

default_data_dir = str(Path.home() / ".llamafarm")


class Settings(BaseSettings, env_file=".env"):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = False  # if true, the server will reload on code changes

    LOG_JSON_FORMAT: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_NAME: str = "server"
    LOG_ACCESS_NAME: str = "server.access"

    lf_data_dir: str = default_data_dir
    lf_config_template: str = "default"

    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_api_key: str = "ollama"


settings = Settings()
