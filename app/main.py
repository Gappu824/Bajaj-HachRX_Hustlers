# app/main.py - Optimized version
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer
import time

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.rag_pipeline import HybridFastTrackRAGPipeline
from app.api.endpoints import query
from app.core.cache import cache

# Set up logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # --- Startup ---
    logger.info("Starting Optimized RAG Pipeline...")
    
    try:
        # Load embedding model
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError("Failed to initialize")
    
    # Initialize pipeline
    app.state.rag_pipeline = HybridFastTrackRAGPipeline(embedding_model)
    logger.info("Optimized RAG Pipeline ready")
    
    yield
    
    # --- Shutdown ---
    logger.info("Shutting down...")
    cache.clear()
    logger.info(f"Cache stats at shutdown: {cache.get_stats()}")
    logger.info("Shutdown complete")

# Middleware for logging
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    if request.url.path.startswith("/api/v1/hackrx"):
        logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    
    return response

app = FastAPI(
    title=settings.APP_NAME,
    description="High-performance document query system optimized for speed and accuracy",
    version="5.0.0",
    lifespan=lifespan
)

app.middleware("http")(log_requests)

# Include API router
app.include_router(query.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}", 
        "version": "5.0.0",
        "features": [
            "Optimized for <20s processing time",
            "Handles binary and non-text files gracefully",
            "Smart adaptive chunking",
            "Enhanced caching with compression",
            "Parallel question processing",
            "Support for PDF, DOCX, Excel, PowerPoint",
            "Image processing with OCR (PNG, JPG, etc.)",
            "ZIP archive processing (extracts and handles multiple files)"
        ],
        "performance": {
            "target_accuracy": "80%+",
            "target_speed": "<20s for 10 questions",
            "max_document_size": "200MB"
        }
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check with cache stats"""
    cache_stats = cache.get_stats()
    
    return {
        "status": "healthy",
        "pipeline": "optimized-hybrid-v5",
        "cache": cache_stats,
        "models": {
            "embedding": settings.EMBEDDING_MODEL_NAME,
            "llm": settings.LLM_MODEL_NAME,
            "llm_precise": settings.LLM_MODEL_NAME_PRECISE
        }
    }

@app.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """Detailed cache statistics"""
    return cache.get_stats()

@app.post("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear the cache"""
    cache.clear()
    return {"message": "Cache cleared successfully"}