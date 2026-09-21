import json
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]  # فولدر backend


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    vector_store_dir: Path = BASE_DIR / "data" / "vector_store"
    config_path: Path = BASE_DIR / "data" / "rag_config.json"
    titles_path: Path = BASE_DIR / "data" / "titles.json"
    ollama_host: str = "http://localhost:11434"
    cors_origins: str = "http://localhost:8501,http://localhost:7860"
    log_level: str = "INFO"


settings = Settings()


def load_rag_config() -> dict:
    with open(settings.config_path, encoding="utf-8") as f:
        return json.load(f)