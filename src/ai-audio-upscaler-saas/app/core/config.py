"""
Application Configuration
Centralized configuration management using Pydantic Settings
"""

import os
from typing import Optional, List
from pydantic import BaseSettings, validator
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application
    APP_NAME: str = "AI Audio Upscaler Pro"
    VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"

    # Database configuration
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/aiupscaler"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_POOL_OVERFLOW: int = 30
    DATABASE_POOL_TIMEOUT: int = 30

    # Redis configuration
    REDIS_URL: str = "redis://localhost:6379/0"

    # Azure B2C Authentication
    AZURE_B2C_TENANT_NAME: str = ""
    AZURE_B2C_POLICY_NAME: str = "B2C_1_signupsignin"
    AZURE_B2C_CLIENT_ID: str = ""
    AZURE_B2C_CLIENT_SECRET: str = ""

    # JWT Configuration
    JWT_SECRET_KEY: str = ""
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Azure Storage
    AZURE_STORAGE_ACCOUNT_NAME: str = ""
    AZURE_STORAGE_ACCOUNT_KEY: str = ""
    AZURE_STORAGE_CONNECTION_STRING: str = ""
    AZURE_STORAGE_CONTAINER_INPUTS: str = "audio-inputs"
    AZURE_STORAGE_CONTAINER_OUTPUTS: str = "audio-outputs"
    AZURE_STORAGE_CONTAINER_MODELS: str = "models"

    # Azure Key Vault (if used)
    AZURE_KEY_VAULT_URL: str = ""
    AZURE_CLIENT_ID: str = ""
    AZURE_CLIENT_SECRET: str = ""
    AZURE_TENANT_ID: str = ""

    # Processing Configuration
    MAX_FILE_SIZE_MB: int = 500
    MAX_CONCURRENT_JOBS: int = 10
    JOB_TIMEOUT_MINUTES: int = 60
    CLEANUP_TEMP_FILES_HOURS: int = 24

    # GPU Configuration
    CUDA_VISIBLE_DEVICES: Optional[str] = None
    GPU_MEMORY_FRACTION: float = 0.8

    # Monitoring and Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090

    # Security
    CORS_ORIGINS: List[str] = ["*"]
    ALLOWED_HOSTS: List[str] = ["*"]
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW_SECONDS: int = 3600

    # Celery Configuration
    CELERY_BROKER_URL: Optional[str] = None
    CELERY_RESULT_BACKEND: Optional[str] = None
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True

    @validator("CELERY_BROKER_URL", pre=True)
    def set_celery_broker_url(cls, v, values):
        if v:
            return v
        return values.get("REDIS_URL", "redis://localhost:6379/0")

    @validator("CELERY_RESULT_BACKEND", pre=True)
    def set_celery_result_backend(cls, v, values):
        if v:
            return v
        return values.get("REDIS_URL", "redis://localhost:6379/0")

    @validator("CORS_ORIGINS", pre=True)
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("JWT_SECRET_KEY")
    def validate_jwt_secret_key(cls, v):
        if not v and os.getenv("ENVIRONMENT", "development") == "production":
            raise ValueError("JWT_SECRET_KEY is required in production")
        return v

    @validator("AZURE_B2C_TENANT_NAME")
    def validate_azure_b2c_config(cls, v, values):
        if values.get("ENVIRONMENT") == "production":
            required_fields = [
                "AZURE_B2C_TENANT_NAME",
                "AZURE_B2C_CLIENT_ID",
                "AZURE_B2C_CLIENT_SECRET"
            ]
            for field in required_fields:
                if not values.get(field):
                    raise ValueError(f"{field} is required in production")
        return v

    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()