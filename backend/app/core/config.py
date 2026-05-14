from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "ActionPilot"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"
    embedding_provider: str = "ollama"
    embedding_model: str = "nomic-embed-text"
    fallback_embedding_model: str = "nomic-embed-text"
    embedding_local_files_only: bool = True
    chroma_dir: str = "./data/chroma"
    docs_dir: str = "./data/docs"
    runtime_dir: str = "./data/runtime"
    max_agent_steps: int = 5
    ollama_num_ctx: int = 128000
    memory_context_tokens: int = 128000
    memory_recent_tokens: int = 8000
    neo4j_enabled: bool = False
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "actionpilot"
    neo4j_timeout_seconds: float = 1.5
    reranker_enabled: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    fallback_reranker_model: str = ""
    reranker_local_files_only: bool = True
    retrieval_candidates: int = 8
    retrieval_top_k: int = 4
    reranker_min_candidates: int = 8
    answer_llm_enabled: bool = True
    answer_llm_min_source_chars: int = 0
    ticketing_backend: str = "local"
    github_token: str = ""
    github_repo: str = ""
    github_api_url: str = "https://api.github.com"
    clear_runtime_on_startup: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def docs_path(self) -> Path:
        return Path(self.docs_dir)

    @property
    def runtime_path(self) -> Path:
        return Path(self.runtime_dir)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    backend_root = Path(__file__).resolve().parents[2]
    for field_name in ("chroma_dir", "docs_dir", "runtime_dir"):
        value = Path(getattr(settings, field_name))
        if not value.is_absolute():
            setattr(settings, field_name, str(backend_root / value))
    settings.runtime_path.mkdir(parents=True, exist_ok=True)
    settings.docs_path.mkdir(parents=True, exist_ok=True)
    return settings
