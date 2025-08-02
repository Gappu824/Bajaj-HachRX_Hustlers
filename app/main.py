# app/main.py
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.rag_pipeline import RAGPipeline
from app.api.endpoints import query

# Set up logging as the very first step
setup_logging()
logger = logging.getLogger(__name__)

# Global variable to cache the model
_cached_model = None

def get_cached_embedding_model():
    """Load embedding model with caching and optimizations"""
    global _cached_model
    if _cached_model is None:
        logger.info("Loading embedding model...")
        try:
            # Use cached model location if available
            cache_folder = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/code/.cache/sentence_transformers')
            
            _cached_model = SentenceTransformer(
                settings.EMBEDDING_MODEL_NAME,
                cache_folder=cache_folder,
                device='cpu'  # Force CPU to avoid GPU detection overhead
            )
            logger.info(f"Embedding model loaded successfully from {cache_folder}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Try without cache folder as fallback
            try:
                _cached_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME, device='cpu')
                logger.info("Embedding model loaded (fallback)")
            except Exception as fallback_error:
                logger.critical(f"Failed to load embedding model: {fallback_error}")
                raise RuntimeError("Could not load embedding model")
    
    return _cached_model

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    Optimized for faster startup in Cloud Run.
    """
    # --- Startup ---
    logger.info("Application startup: Initializing RAG pipeline...")
    
    try:
        # Load embedding model (should be fast due to pre-download in Docker)
        embedding_model = get_cached_embedding_model()
        
        # Instantiate the RAGPipeline
        app.state.rag_pipeline = RAGPipeline(embedding_model)
        
        # Mark as ready
        app.state.ready = True
        
        logger.info("RAG pipeline ready. Application startup complete.")
        
    except Exception as e:
        logger.critical(f"Failed to initialize application: {e}")
        app.state.ready = False
        raise
    
    yield
    
    # --- Shutdown ---
    logger.info("Application shutdown...")
    try:
        from app.core.cache import cache
        cache._cache.clear()
        app.state.ready = False
        logger.info("Cache cleared. Shutdown complete.")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan
)

# Include the API router from the endpoints file
app.include_router(query.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    """A simple root endpoint to confirm the API is running."""
    ready = getattr(app.state, 'ready', False)
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "status": "ready" if ready else "initializing"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Cloud Run"""
    ready = getattr(app.state, 'ready', False)
    return {"status": "healthy" if ready else "initializing", "ready": ready}

@app.get("/readiness", tags=["Health"])
async def readiness_check():
    """Readiness probe for Cloud Run"""
    ready = getattr(app.state, 'ready', False)
    if ready:
        return {"status": "ready"}
    else:
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Service not ready")