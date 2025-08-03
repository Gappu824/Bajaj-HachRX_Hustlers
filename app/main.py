# app/main.py
import logging
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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages application startup and shutdown events.
    The RAG pipeline and vector store cache are initialized here once
    to be reused across all requests.
    """
    # --- Startup ---
    logger.info("Application startup: Initializing resources...")
    
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    
    # Instantiate the RAGPipeline and attach it to the app state
    app.state.rag_pipeline = RAGPipeline(embedding_model)
    
    # --- CACHING IMPROVEMENT ---
    # Create the in-memory cache and attach it to the app state
    app.state.vector_store_cache = {}
    # --- END CACHING IMPROVEMENT ---
    
    logger.info("Resources initialized. RAG pipeline and cache are ready.")
    yield
    # --- Shutdown ---
    logger.info("Application shutdown...")
    # Clear the cache on shutdown
    app.state.vector_store_cache.clear() 
    logger.info("Cache cleared. Shutdown complete.")


app = FastAPI(
    title=settings.APP_NAME,
    lifespan=lifespan
)

# Include the API router from the endpoints file
app.include_router(query.router, prefix=settings.API_V1_STR)

@app.get("/", tags=["Root"])
async def read_root():
    """A simple root endpoint to confirm the API is running."""
    return {"message": f"Welcome to {settings.APP_NAME}"}