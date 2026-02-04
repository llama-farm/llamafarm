import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

try:
    default_data_dir = str(Path.home() / ".llamafarm")
except RuntimeError:
    # Path.home() fails in PyApp-embedded Python on Windows when
    # HOME/USERPROFILE env vars are absent during bootstrap.
    _fb = os.environ.get("USERPROFILE") or os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
    try:
        _fallback = Path(_fb) if _fb else Path.cwd()
    except OSError:
        _fallback = Path(".")
    default_data_dir = str(_fallback / ".llamafarm")


class Settings(BaseSettings):
    # Allow extra fields in .env file without validation errors
    # This is important for deployments where users may have various env vars
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",  # Ignore unknown env vars instead of raising errors
    )
    HOST: str = "0.0.0.0"
    PORT: int = 14345
    RELOAD: bool = False  # if true, the server will reload on code changes

    LOG_JSON_FORMAT: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_NAME: str = "server"
    LOG_ACCESS_NAME: str = "server.access"
    # If set, logs will be written to this file in addition to stdout
    LOG_FILE: str = ""

    CELERY_LOG_LEVEL: str = "INFO"

    lf_data_dir: str = default_data_dir
    lf_config_template: str = "default"

    # Ollama Configuration
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3:8b"
    ollama_api_key: str = "ollama"

    # HuggingFace Configuration
    huggingface_token: str = ""  # HF_TOKEN environment variable

    # Celery Broker Override Configuration
    # If set, these will override the default filesystem broker
    celery_broker_url: str = (
        ""  # e.g., "redis://localhost:6379/0" or "amqp://guest@localhost//"
    )
    celery_result_backend: str = ""  # e.g., "redis://localhost:6379/0"

    # Dev mode settings
    lf_dev_mode_docs_enabled: bool = True
    lf_dev_mode_greeting_enabled: bool = True

    # Lemonade Configuration
    lemonade_port: int = 11534
    lemonade_host: str = "127.0.0.1"
    lemonade_api_key: str = "lemonade"

    # Universal Runtime Configuration
    universal_port: int = 11540
    universal_host: str = "127.0.0.1"
    universal_api_key: str = "universal"


settings = Settings()
