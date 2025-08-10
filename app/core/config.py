# app/core/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Manages application configuration using environment variables."""
    APP_NAME: str = "Bajaj HackRx 6.0 - Hybrid Optimized RAG System"
    API_V1_STR: str = "/api/v1"
    
    # Security Token
    BEARER_TOKEN: str = ""  # Will be loaded from .env
    
    # Model Configuration - optimized for performance and accuracy
    EMBEDDING_MODEL_NAME: str = 'paraphrase-multilingual-MiniLM-L12-v2'  # Good balance
    LLM_MODEL_NAME: str = 'gemini-1.5-flash'  # Fast model
    LLM_MODEL_NAME_PRECISE: str = 'gemini-1.5-pro-latest'  # Accurate model
    GOOGLE_API_KEY: str = ""  # Will be loaded from .env
    
    # Performance settings - optimized for speed and accuracy
    MAX_CHUNKS_PER_QUERY: int = 20  # Increased for better context
    MAX_CONCURRENT_QUESTIONS: int = 5  # More parallel processing
    ANSWER_TIMEOUT_SECONDS: int = 25  # Slightly reduced for speed
    TOTAL_TIMEOUT_SECONDS: int = 300  # 5 minutes - more reasonable
    
    # Cache settings - optimized for performance
    CACHE_SIZE_MB: int = 1500  # Balanced cache size
    CACHE_TTL_SECONDS: int = 7200  # 2 hours
    USE_DISK_CACHE: bool = True  # Enable disk cache for large docs
    
    # Document processing - optimized chunk sizes
    MAX_DOCUMENT_SIZE_MB: int = 100  # Reduced for better performance
    CHUNK_SIZE_CHARS: int = 800  # Optimal chunk size for balance
    CHUNK_OVERLAP_CHARS: int = 150  # Good overlap

    ENABLE_UTF8_SUPPORT: bool = True
    DEFAULT_ENCODING: str = "utf-8"
    
    # New performance settings
    MAX_QUESTION_LENGTH: int = 500  # Limit question length
    MIN_CHUNK_QUALITY_SCORE: float = 0.3  # Minimum relevance score
    ENABLE_AGGRESSIVE_CACHING: bool = True  # Enable all caching optimizations
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Create singleton
settings = Settings()