import os
from typing import List, Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings



class Settings(BaseSettings):
    SERVICE_NAME: str = "rag-chat-service"
    PORT: int = 8004
    DEBUG: bool = False

    # External Microservice URLs
    IDENTITY_SERVICE_URL: str = os.getenv("IDENTITY_SERVICE_URL", "http://127.0.0.1:8001")
    INGESTION_SERVICE_URL: str = os.getenv("INGESTION_SERVICE_URL", "http://127.0.0.1:8003")

    # RAG Settings
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.30"))
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "groq/openai/gpt-oss-120b")
    FALLBACK_MODEL: str = os.getenv("FALLBACK_MODEL", "groq/openai/gpt-oss-20b")
    FALLBACK_MESSAGE: str = "This isn't covered in company documents. Would you like this flagged to HR?"

    # Semantic Cache Settings (Part A)
    SEMANTIC_CACHE_ENABLED: bool = os.getenv("SEMANTIC_CACHE_ENABLED", "True").lower() in ("true", "1", "yes")
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.90"))
    SEMANTIC_CACHE_TTL_SECONDS: int = int(os.getenv("SEMANTIC_CACHE_TTL_SECONDS", "86400"))
    SEMANTIC_CACHE_DIR: str = os.getenv("SEMANTIC_CACHE_DIR", "./data/semantic_cache_db")
    SEMANTIC_CACHE_COLLECTION: str = os.getenv("SEMANTIC_CACHE_COLLECTION", "hr_semantic_cache")
    EMBEDDING_MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    EMBEDDING_DIMENSION: int = 384

    # Reranker Settings (Part B)
    RERANKER_ENABLED: bool = os.getenv("RERANKER_ENABLED", "True").lower() in ("true", "1", "yes")
    RERANKER_MODEL_NAME: str = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    INITIAL_RETRIEVAL_TOP_K: int = int(os.getenv("INITIAL_RETRIEVAL_TOP_K", "4"))
    RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "3"))
    RERANK_CONFIDENCE_THRESHOLD: float = float(os.getenv("RERANK_CONFIDENCE_THRESHOLD", "0.25"))

    # LangSmith Tracing & Privacy Settings (Part C)
    LANGSMITH_TRACING: bool = os.getenv("LANGSMITH_TRACING", "false").lower() in ("true", "1", "yes")
    LANGSMITH_API_KEY: Optional[str] = os.getenv("LANGSMITH_API_KEY", None)
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "hr-rag-assistant")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    TRACING_MASK_PII: bool = os.getenv("TRACING_MASK_PII", "true").lower() in ("true", "1", "yes")
    TRACING_INCLUDE_RAW_ATTACHMENTS: bool = os.getenv("TRACING_INCLUDE_RAW_ATTACHMENTS", "false").lower() in ("true", "1", "yes")





    # Groq API Keys list (Loaded securely from .env / environment)
    GROQ_API_KEYS: Union[List[str], str] = [
        "gsk_dummy_groq_api_key_placeholder_for_ci_1",
        "gsk_dummy_groq_api_key_placeholder_for_ci_2",
    ]

    @field_validator("GROQ_API_KEYS", mode="before")
    @classmethod
    def parse_groq_keys(cls, v):
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    # Guardrail Limits
    MAX_MESSAGE_LENGTH: int = 2000
    MAX_OUTPUT_LENGTH: int = 4000
    OCR_MAX_IMAGE_SIZE_MB: int = 10

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
