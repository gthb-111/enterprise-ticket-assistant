from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    db_path: Path
    knowledge_dir: Path
    web_dir: Path
    auto_resolve_threshold: float
    min_relevance_threshold: float
    llm_enabled: bool
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    vector_weight: float
    vector_dimensions: int
    llm_timeout_seconds: float
    llm_max_retries: int
    llm_max_concurrency: int
    llm_max_tokens: int

    @classmethod
    def from_env(cls) -> "Settings":
        data_dir = Path(os.getenv("SERVICEFLOW_DATA_DIR", PROJECT_ROOT / ".data"))
        return cls(
            host=os.getenv("SERVICEFLOW_HOST", "127.0.0.1"),
            port=int(os.getenv("SERVICEFLOW_PORT", "8877")),
            db_path=data_dir / "serviceflow.db",
            knowledge_dir=PROJECT_ROOT / "knowledge",
            web_dir=PROJECT_ROOT / "web",
            auto_resolve_threshold=float(
                os.getenv("SERVICEFLOW_AUTO_RESOLVE_THRESHOLD", "0.66")
            ),
            min_relevance_threshold=float(
                os.getenv("SERVICEFLOW_MIN_RELEVANCE_THRESHOLD", "0.44")
            ),
            llm_enabled=os.getenv("SERVICEFLOW_LLM_ENABLED", "0").lower()
            in {"1", "true", "yes"},
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ).rstrip("/"),
            openai_model=os.getenv("SERVICEFLOW_MODEL", "gpt-4.1-mini"),
            vector_weight=float(os.getenv("SERVICEFLOW_VECTOR_WEIGHT", "0.35")),
            vector_dimensions=int(os.getenv("SERVICEFLOW_VECTOR_DIMENSIONS", "512")),
            llm_timeout_seconds=float(
                os.getenv("SERVICEFLOW_LLM_TIMEOUT_SECONDS", "30")
            ),
            llm_max_retries=int(os.getenv("SERVICEFLOW_LLM_MAX_RETRIES", "2")),
            llm_max_concurrency=int(
                os.getenv("SERVICEFLOW_LLM_MAX_CONCURRENCY", "2")
            ),
            llm_max_tokens=int(os.getenv("SERVICEFLOW_LLM_MAX_TOKENS", "900")),
        )

