# app/core/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Manages application configuration using environment variables."""
    APP_NAME: str = "Bajaj HackRx 6.0 - Intelligent Query-Retrieval System"
    API_V1_STR: str = "/api/v1"
    
    # Security Token provided in the problem statement
    BEARER_TOKEN: str

    # Model Configuration - can be easily swapped
    EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
    LLM_MODEL_NAME: str = 'gemini-1.5-flash-latest'
    GOOGLE_API_KEY: str

    class Config:
        # Pydantic will automatically look for a .env file
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create a single settings instance to be used across the application
settings = Settings()