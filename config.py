from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    GROQ_API_KEY: Optional[str] = None
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
