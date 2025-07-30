import os
from appdirs import user_data_dir
from pydantic_settings import BaseSettings
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

default_data_dir = user_data_dir("LlamaFarm", "LlamaFarm")

class Settings(BaseSettings, env_file=".env"):
    lf_project_dir: Optional[str] = None
    lf_data_dir: str = default_data_dir

settings = Settings()
