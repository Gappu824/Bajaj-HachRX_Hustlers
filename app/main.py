# app/main.py - Ultra-fast startup with lazy loading
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.endpoints import query

# Set up logging as the very first step
setup_logging()
logger = logging.getLogger(__name__)

# Global variables for lazy loading
_rag_pipeline = None
_is_initializing = False

async def get_rag_pipeline():
    """Lazy load RAG pipeline only when needed"""
    global _rag_pipeline, _is_initializing
    
    if _rag_pipeline is not None:
        return _rag_pipeline
    
    if _is_initializing:
        # Wait for initialization to complete
        import asyncio
        while _is_initializing:
            await asyncio.sleep(0.1)
        return _rag_pipeline
    
    _is_initializing = True
    try:
        logger.info("Lazy loading RAG pipeline...")
        
        # Import here to avoid slow imports during startup
        from sentence_transformers import SentenceTransformer
        from app.core.rag_pipeline import RAGPipeline
        
        # Load model
        cache_folder = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/code/.cache/sentence_transformers')
        embedding_model = SentenceTransformer(
            settings.EMBEDDING_MODEL_NAME,
            cache_folder=cache_folder,
            device='cpu'
        )
        
        # Create pipeline
        _rag_pipeline = RAGPipeline(embedding_model)
        logger.info("RAG pipeline loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load RAG pipeline: {e}")
        raise
    finally:
        _is_initializing = False
    
    return _rag_pipeline

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Minimal lifespan - just start the app quickly
    """
    # --- Startup ---
    logger.info("Application starting up quickly...")
    app.state.ready = True
    logger.info("Application ready (RAG pipeline will load on first request)")
    
    yield
    
    # --- Shutdown ---
    logger.info("Application shutdown...")
    try:
        from app.core.cache import cache
        cache._cache.clear()
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
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "status": "ready",
        "note": "RAG pipeline loads on first API call"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for Cloud Run"""
    return {"status": "healthy", "ready": True}

@app.get("/warmup", tags=["Health"])
async def warmup():
    """Warmup endpoint to pre-load the RAG pipeline"""
    try:
        await get_rag_pipeline()
        return {"status": "warmed_up", "message": "RAG pipeline is ready"}
    except Exception as e:
        logger.error(f"Warmup failed: {e}")
        return {"status": "error", "message": str(e)}

# Monkey patch the query endpoint to use lazy loading
import app.api.endpoints.query as query_module
original_run_query = query_module.run_query

async def lazy_run_query(fastapi_req, request_data, token):
    """Modified query handler with lazy loading"""
    # Get RAG pipeline lazily
    rag_pipeline = await get_rag_pipeline()
    
    # Set it in app state for the original function
    fastapi_req.app.state.rag_pipeline = rag_pipeline
    
    # Call original function
    return await original_run_query(fastapi_req, request_data, token)

# Replace the original function
query_module.run_query = lazy_run_query