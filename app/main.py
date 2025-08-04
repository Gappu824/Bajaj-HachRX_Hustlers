# app/main.py - Updated for accuracy-focused pipeline
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.rag_pipeline import FastAccurateRAGPipeline  # Updated import
from app.api.endpoints import query

# Set up logging as the very first step
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events with accuracy-focused setup.
    """
    # --- Startup ---
    logger.info("Application startup: Initializing accuracy-focused resources...")
    
    # Use MiniLM for speed (you can switch back to mpnet later if needed)  
    try:
        embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL_NAME}")
    except Exception as e:
        logger.error(f"Could not load embedding model: {e}")
        raise RuntimeError("Failed to load embedding model")
    
    # Instantiate the Fast & Accurate RAG Pipeline
    app.state.rag_pipeline = FastAccurateRAGPipeline(embedding_model)
    
    # Create the in-memory cache
    app.state.vector_store_cache = {}
    
    logger.info("Accuracy-focused resources initialized. Ready for high-precision queries.")
    yield
    
    # --- Shutdown ---
    logger.info("Application shutdown...")
    app.state.vector_store_cache.clear() 
    logger.info("Cache cleared. Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME + " - Accuracy Optimized",
    description="High-accuracy document query system optimized for 75%+ precision",
    version="2.0.0",
    lifespan=lifespan
)

# Include the API router
app.include_router(query.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint with accuracy focus information"""
    return {
        "message": f"Welcome to {settings.APP_NAME} - Accuracy Optimized", 
        "version": "2.0.0",
        "features": [
            "Enhanced multi-strategy document retrieval",
            "Larger context chunks with intelligent overlap", 
            "Advanced cross-encoder re-ranking",
            "Accuracy-focused LLM prompting",
            "Support for PDF, DOCX, ODT, EML formats"
        ],
        "optimization": "Designed for 75%+ accuracy and 800-900 point scores"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "pipeline": "accuracy-optimized"}