# app/main.py - Main FastAPI application
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import time

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.rag_pipeline import HybridRAGPipeline
from app.api.endpoints import query
from app.core.cache import cache

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    logger.info("Starting Hybrid Optimized RAG Pipeline...")
    
    try:
        # Load embedding model
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL_NAME}")
        
        # Initialize pipeline
        app.state.rag_pipeline = HybridRAGPipeline(embedding_model)
        logger.info("RAG Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")
    cache.clear()
    logger.info(f"Final cache stats: {cache.get_stats()}")
    logger.info("Shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="High-performance document query system with balanced speed and accuracy",
    version="6.0.0",
    lifespan=lifespan
)

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    if request.url.path.startswith("/api"):
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    
    return response

# Include routers
app.include_router(query.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint with system information"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": "6.0.0",
        "features": [
            "Balanced speed and accuracy optimization",
            "Hybrid memory and disk caching",
            "Comprehensive format support (PDF, DOCX, Excel, PPT, Images, ZIP)",
            "Smart context-aware chunking",
            "Multi-strategy retrieval (BM25, TF-IDF, Semantic)",
            "Adaptive answer generation",
            "Parallel question processing"
        ],
        "performance": {
            "target_accuracy": "85%+",
            "target_speed": "<20s for 10 questions",
            "max_document_size": f"{settings.MAX_DOCUMENT_SIZE_MB}MB"
        },
        "models": {
            "embedding": settings.EMBEDDING_MODEL_NAME,
            "llm_fast": settings.LLM_MODEL_NAME,
            "llm_accurate": settings.LLM_MODEL_NAME_PRECISE
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    cache_stats = cache.get_stats()
    
    return {
        "status": "healthy",
        "pipeline": "hybrid-optimized-v6",
        "cache": cache_stats
    }

@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Get detailed cache statistics"""
    return cache.get_stats()

@app.post("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear all caches"""
    cache.clear()
    return {"message": "Cache cleared successfully"}