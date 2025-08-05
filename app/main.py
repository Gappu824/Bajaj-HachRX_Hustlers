# app/main.py - Optimized for Hybrid Fast-Track Pipeline
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

# Set up logging as the very first step
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events with optimized setup.
    """
    # --- Startup ---
    logger.info("Application startup: Initializing Hybrid Fast-Track Pipeline...")
    
    # Load embedding model
    try:
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Could not load embedding model: {e}")
        raise RuntimeError("Failed to load embedding model")
    
    # Instantiate the Hybrid Fast-Track RAG Pipeline
    app.state.rag_pipeline = HybridFastTrackRAGPipeline(embedding_model)
    
    logger.info("Hybrid Fast-Track Pipeline initialized. Ready for high-speed, accurate queries.")
    yield
    
    # --- Shutdown ---
    logger.info("Application shutdown...")
    # Clear cache on shutdown
    if hasattr(cache, '_cache'):
        cache._cache.clear()
    logger.info("Cache cleared. Shutdown complete.")

# Add middleware for request logging
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.2f}s")
    return response

app = FastAPI(
    title=settings.APP_NAME + " - Hybrid Fast-Track",
    description="Ultra-fast document query system with intelligent routing for 90%+ accuracy",
    version="3.0.0",
    lifespan=lifespan
)

# Add middleware
app.middleware("http")(log_requests)

# Include the API router
app.include_router(query.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint with system information"""
    return {
        "message": f"Welcome to {settings.APP_NAME} - Hybrid Fast-Track", 
        "version": "3.0.0",
        "features": [
            "Intelligent question complexity routing",
            "Optimized PDF parsing with page limits",
            "Hybrid search with keyword and vector retrieval",
            "Parallel processing with larger batches",
            "Multi-level caching for maximum speed",
            "Support for PDF, DOCX, ODT formats"
        ],
        "optimization": "Designed for <20s processing of complex documents with 90%+ accuracy"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "pipeline": "hybrid-fast-track",
        "cache_enabled": True,
        "parallel_processing": True
    }