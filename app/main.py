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
    The embedding model is loaded here once to be reused across all requests.
    """
    # --- Startup ---
    logger.info("Application startup: Loading embedding model...")
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    
    # Instantiate the RAGPipeline. 
    # The pipeline now configures the Google Gemini client internally.
    app.state.rag_pipeline = RAGPipeline(embedding_model)
    
    logger.info("Embedding model loaded. RAG pipeline is ready.")
    yield
    # --- Shutdown ---
    logger.info("Application shutdown...")
    # This is a good practice to clear resources if needed.
    from app.core.cache import cache
    cache._cache.clear() 
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