from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./identity.db"
    JWT_SECRET_KEY: str = "test_super_secret_jwt_key_identity_service_2026"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours for smooth development/testing session
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    INITIAL_ADMIN_EMAIL: str = "admin@example.com"
    INITIAL_ADMIN_PASSWORD: str = "Admin@123456"
    INITIAL_ADMIN_NAME: str = "Super Admin"
    INITIAL_COMPANY_NAME: str = "Default Organization"

    PORT: int = 8001

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


settings = Settings()
