from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PORT: int = 8003
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    COLLECTION_NAME: str = "hr_policy_documents"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 100

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
