# app/core/config.py
from pydantic_settings import BaseSettings
import os
from typing import Optional

class Settings(BaseSettings):
    """Enhanced configuration with new features"""
    APP_NAME: str = "Bajaj HackRx 6.0 - Advanced RAG System v2"
    API_V1_STR: str = "/api/v1"
    
    # Security Token
    BEARER_TOKEN: str = ""
    
    # Model Configuration - optimized for accuracy
    # EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
    # EMBEDDING_MODEL_NAME: str = 'BAAI/bge-fast-en'
    EMBEDDING_MODEL_NAME: str = 'BAAI/bge-fast-en-v1.5'
    RERANKER_MODEL_NAME: str = 'cross-encoder/ms-marco-MiniLM-L-12-v2'
    LLM_MODEL_NAME: str = 'gemini-1.5-flash'
    LLM_MODEL_NAME_PRECISE: str = 'gemini-1.5-pro-latest'
    GOOGLE_API_KEY: str = ""
    
    # Performance settings - balanced for accuracy
    MAX_CHUNKS_PER_QUERY: int = 15  # Increased for better context
    MAX_RERANK_CHUNKS: int = 30  # Initial retrieval before reranking
    MAX_CONCURRENT_QUESTIONS: int = 2
    ANSWER_TIMEOUT_SECONDS: int = 1800
    TOTAL_TIMEOUT_SECONDS: int = 3600
    MAX_TOTAL_CHUNKS: int = 1000  # New limit for total chunks

    # Parallel Processing
    # MAX_CONCURRENT_QUESTIONS: int = 1  # Process 1 at a time for stability
    QUESTION_BATCH_SIZE: int = 3  # Batch size for parallel processing
    USE_PARALLEL_RETRIEVAL: bool = True  # Parallel chunk retrieval
    PARALLEL_EMBEDDING_GENERATION: bool = True  # Parallel embedding generation

    
    # Cache settings
    CACHE_SIZE_MB: int = 1500
    CACHE_TTL_SECONDS: int = 10800  # 3 hours
    USE_DISK_CACHE: bool = True
    
    # Document processing
    # MAX_DOCUMENT_SIZE_MB: int = None
    MAX_DOCUMENT_SIZE_MB: Optional[int] = None
    CHUNK_SIZE_CHARS: int = 1000  # Smaller for precision
    CHUNK_OVERLAP_CHARS: int = 200
    LARGE_CHUNK_SIZE_CHARS: int = 2000  # For hierarchical chunking
    LARGE_CHUNK_OVERLAP_CHARS: int = 400

    # Caching Configuration
    USE_ANSWER_CACHE: bool = True  # Cache individual answers
    ANSWER_CACHE_TTL: int = 10800  # 2 hours for answer cache
    USE_EMBEDDING_CACHE: bool = True  # Cache embeddings
    EMBEDDING_CACHE_TTL: int = 86400  # 24 hours for embeddings
    USE_VECTOR_STORE_CACHE: bool = True  # Cache vector stores
    VECTOR_STORE_CACHE_TTL: int = 10800  # 3 hours for vector stores

    #  Performance Tuning
    CACHE_COMPRESSION: bool = True  # Compress cached items
    CACHE_BATCH_SIZE: int = 10  # Batch size for cache operations

    # Cache Size Limits
    MAX_CACHE_MEMORY_MB: int = 500  # Max memory cache size
    MAX_CACHE_DISK_MB: int = 2000  # Max disk cache size
    
    # Universal parser settings
    USE_TIKA: bool = True
    # In app/core/config.py, ensure:
    TIKA_SERVER_URL: str = "http://localhost:9998"  # Not https
    
    # Question handling
    ENABLE_QUERY_EXPANSION: bool = True
    ENABLE_MULTI_STEP_REASONING: bool = True
    CONFIDENCE_THRESHOLD: float = 0.7


    # EMBEDDING_CACHE_TTL: int = 86400  # 24 hours
    EMBEDDING_BATCH_SIZE: int = 16  # Optimized batch size
    # USE_EMBEDDING_CACHE: bool = True
    # In config.py

# ... your other settings ...

# --- NEW: Parameters for Hierarchical RAG ---
    # PARENT_CHUNK_SIZE = 10000
    # PARENT_CHUNK_OVERLAP = 500
    # CHILD_CHUNK_SIZE = 1000
    # CHILD_CHUNK_OVERLAP = 100
    PARENT_CHUNK_SIZE: int = 10000
    PARENT_CHUNK_OVERLAP: int = 500
    CHILD_CHUNK_SIZE: int = 1000
    CHILD_CHUNK_OVERLAP: int = 100
        
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()