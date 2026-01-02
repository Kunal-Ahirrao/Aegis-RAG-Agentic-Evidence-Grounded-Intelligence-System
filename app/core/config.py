from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """Application configuration loaded from environment variables (.env).

    This version supports multiple LLM providers and a local vector store (FAISS),
    so it no longer requires any hosted vector DB keys.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM configuration ---
    # Provider: 'gemini' or 'openai'
    LLM_PROVIDER: str = "gemini"

    # Gemini (Google) API key
    GOOGLE_API_KEY: Optional[str] = None

    # OpenAI API key (for GPT-style models)
    OPENAI_API_KEY: Optional[str] = None

    # Optional bearer token for securing the public API
    HACKATHON_BEARER_TOKEN: Optional[str] = None

settings = Settings()
