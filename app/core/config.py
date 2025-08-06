# app/core/config.py
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    """Enhanced configuration with new features"""
    APP_NAME: str = "Bajaj HackRx 6.0 - Advanced RAG System v2"
    API_V1_STR: str = "/api/v1"
    
    # Security Token
    BEARER_TOKEN: str = ""
    
    # Model Configuration - optimized for accuracy
    EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
    RERANKER_MODEL_NAME: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    LLM_MODEL_NAME: str = 'gemini-1.5-flash'
    LLM_MODEL_NAME_PRECISE: str = 'gemini-1.5-pro-latest'
    GOOGLE_API_KEY: str = ""
    
    # Performance settings - balanced for accuracy
    MAX_CHUNKS_PER_QUERY: int = 25  # Increased for better context
    MAX_RERANK_CHUNKS: int = 50  # Initial retrieval before reranking
    MAX_CONCURRENT_QUESTIONS: int = 3
    ANSWER_TIMEOUT_SECONDS: int = 45
    TOTAL_TIMEOUT_SECONDS: int = 1200
    
    # Cache settings
    CACHE_SIZE_MB: int = 1500
    CACHE_TTL_SECONDS: int = 10800  # 3 hours
    USE_DISK_CACHE: bool = True
    
    # Document processing
    MAX_DOCUMENT_SIZE_MB: int = 500
    CHUNK_SIZE_CHARS: int = 800  # Smaller for precision
    CHUNK_OVERLAP_CHARS: int = 200
    LARGE_CHUNK_SIZE_CHARS: int = 2000  # For hierarchical chunking
    LARGE_CHUNK_OVERLAP_CHARS: int = 400
    
    # Universal parser settings
    USE_TIKA: bool = True
    # In app/core/config.py, ensure:
    TIKA_SERVER_URL: str = "http://localhost:9998"  # Not https
    
    # Question handling
    ENABLE_QUERY_EXPANSION: bool = True
    ENABLE_MULTI_STEP_REASONING: bool = True
    CONFIDENCE_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()