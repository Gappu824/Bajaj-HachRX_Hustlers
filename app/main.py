# app/main.py - Enhanced version
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
    logger.info("Starting Enhanced RAG Pipeline...")
    
    try:
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        raise RuntimeError("Failed to initialize")
    
    app.state.rag_pipeline = HybridFastTrackRAGPipeline(embedding_model)
    logger.info("Enhanced RAG Pipeline ready")
    
    yield
    
    # --- Shutdown ---
    logger.info("Shutting down...")
    cache.clear()
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
    title=settings.APP_NAME + " - Enhanced",
    description="High-accuracy document query system with hybrid retrieval",
    version="4.0.0",
    lifespan=lifespan
)

app.middleware("http")(log_requests)

# Include API router
app.include_router(query.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME} - Enhanced", 
        "version": "4.0.0",
        "features": [
            "Hybrid BM25 + Semantic search",
            "Multi-stage answer generation",
            "Smart semantic chunking",
            "Enhanced caching with compression",
            "Cross-encoder reranking",
            "Support for large documents"
        ],
        "optimization": "Optimized for 80%+ accuracy with <30s processing"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check with cache stats"""
    cache_stats = cache.get_stats()
    
    return {
        "status": "healthy",
        "pipeline": "enhanced-hybrid",
        "cache": cache_stats
    }

@app.get("/cache/stats", tags=["Cache"])
async def cache_stats():
    """Detailed cache statistics"""
    return cache.get_stats()