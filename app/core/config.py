# app/core/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Manages application configuration using environment variables."""
    APP_NAME: str = "Bajaj HackRx 6.0 - Optimized RAG System"
    API_V1_STR: str = "/api/v1"
    
    # Security Token provided in the problem statement
    BEARER_TOKEN: str = ""  # Will be loaded from .env
    
    # Model Configuration - optimized for speed and accuracy
    EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'  # Fast and efficient
    LLM_MODEL_NAME: str = 'gemini-1.5-flash'  # Faster model by default
    LLM_MODEL_NAME_PRECISE: str = 'gemini-1.5-pro-latest'  # For complex questions
    GOOGLE_API_KEY: str = ""  # Will be loaded from .env
    
    # Performance settings
    MAX_CHUNKS_PER_QUERY: int = 10  # Reduced for speed
    MAX_CONCURRENT_QUESTIONS: int = 5  # Parallel processing limit
    ANSWER_TIMEOUT_SECONDS: int = 10  # Per question timeout
    TOTAL_TIMEOUT_SECONDS: int = 25  # Total processing timeout
    
    # Cache settings
    CACHE_SIZE_MB: int = 2000  # Increased cache size
    CACHE_TTL_SECONDS: int = 7200  # 2 hours
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create singleton
settings = Settings()