# app/core/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Manages application configuration using environment variables."""
    APP_NAME: str = "Bajaj HackRx 6.0 - Hybrid Optimized RAG System"
    API_V1_STR: str = "/api/v1"
    
    # Security Token
    BEARER_TOKEN: str = ""  # Will be loaded from .env
    
    # Model Configuration - balanced for speed and accuracy
    EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'  # Fast embeddings
    LLM_MODEL_NAME: str = 'gemini-1.5-flash'  # Fast model
    LLM_MODEL_NAME_PRECISE: str = 'gemini-1.5-pro-latest'  # Accurate model
    GOOGLE_API_KEY: str = ""  # Will be loaded from .env
    
    # Performance settings - balanced
    MAX_CHUNKS_PER_QUERY: int = 15  # Increased for better context
    MAX_CONCURRENT_QUESTIONS: int = 2  # More parallel processing
    ANSWER_TIMEOUT_SECONDS: int = 30  # Balanced timeout
    TOTAL_TIMEOUT_SECONDS: int = 1800  # Slightly increased
    
    # Cache settings - less aggressive
    CACHE_SIZE_MB: int = 1000  # Reduced from 2000
    CACHE_TTL_SECONDS: int = 7200  # 1 hour instead of 2
    USE_DISK_CACHE: bool = True  # Enable disk cache for large docs
    
    # Document processing
    MAX_DOCUMENT_SIZE_MB: int = 200
    CHUNK_SIZE_CHARS: int = 1000  # Optimal chunk size
    CHUNK_OVERLAP_CHARS: int = 200  # Good overlap
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create singleton
settings = Settings()