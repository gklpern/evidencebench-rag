from functools import lru_cache

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EB_", env_file=".env", extra="ignore")

    database_url: str = (
        "postgresql+asyncpg://evidencebench:evidencebench@127.0.0.1:5432/evidencebench"
    )
    api_keys_json: SecretStr = SecretStr("{}")
    embedding_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8081/v1")
    embedding_api_key: SecretStr = SecretStr("")
    embedding_model: str = "BAAI/bge-m3"
    embedding_dimensions: int = 1024
    generation_base_url: AnyHttpUrl = AnyHttpUrl("http://127.0.0.1:8082/v1")
    generation_api_key: SecretStr = SecretStr("")
    generation_model: str = "Qwen/Qwen2.5-7B-Instruct"
    retrieval_limit_per_channel: int = 20
    fused_limit: int = 8
    minimum_evidence: int = 2
    minimum_rrf_score: float = 0.015


@lru_cache
def get_settings() -> Settings:
    return Settings()
