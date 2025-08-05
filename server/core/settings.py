from appdirs import user_data_dir
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

default_data_dir = user_data_dir("LlamaFarm", "LlamaFarm")

class Settings(BaseSettings, env_file=".env"):
    LOG_JSON_FORMAT: bool = False
    LOG_LEVEL: str = "INFO"
    LOG_NAME: str = "default"
    LOG_ACCESS_NAME: str = "access"

    lf_project_dir: str | None = None
    lf_data_dir: str = default_data_dir
    # Ollama Configuration
    ollama_host: str = "http://localhost:11434/v1"
    ollama_model: str = "llama3.1:8b"
    ollama_api_key: str = "ollama"

settings = Settings()
