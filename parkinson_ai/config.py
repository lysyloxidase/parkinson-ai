"""Application configuration."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PDAI_", extra="ignore")

    OLLAMA_HOST: str = "http://localhost:11434"
    MODEL_ROUTER: str = "llama3.2:3b"
    MODEL_REASONER: str = "qwen2.5:14b"
    MODEL_EXTRACTOR: str = "llama3.1:8b"
    MODEL_EMBEDDING: str = "nomic-embed-text"

    CHROMA_PERSIST_DIR: str = "data/chroma"
    GRAPH_PERSIST_PATH: str = "data/pd_knowledge_graph.gpickle"

    DEVICE: str = "cpu"
    RANDOM_SEED: int = 42
    RAG_TOP_K: int = 10
    KG_CONTEXT_DEPTH: int = 2
    KG_CONTEXT_MAX_TRIPLES: int = 50

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached settings instance."""

    return Settings()
