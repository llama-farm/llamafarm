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


settings = Settings()
