from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    SMTP_HOST: str = "sandbox.smtp.mailtrap.io"
    SMTP_PORT: int = 2525
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = "no-reply@hr-rag-assistant.local"
    SMTP_FROM_NAME: str = "HR Support & Security Team"
    SMTP_USE_TLS: bool = True
    SMTP_USE_SSL: bool = False

    SMTP_MAX_RETRIES: int = 2
    SMTP_RETRY_BACKOFF_SECONDS: float = 0.5

    PORT: int = 8002

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
