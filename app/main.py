# app/main.py - Updated with enhanced pipeline
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from sentence_transformers import SentenceTransformer, CrossEncoder
import time
import tempfile
import os
import shutil

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.core.enhanced_rag_pipeline import EnhancedRAGPipeline
from app.api.endpoints import query
from app.core.cache import cache
from app.core.performance_monitor import PerformanceMonitor

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management"""
    # Startup
    # logger.info("Starting Enhanced RAG Pipeline v2...")
    
    # try:
    #     # Load models
    #     logger.info("Loading embedding model...")
    #     embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        
    #     logger.info("Loading re-ranker model...")
    #     reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
        
    #     logger.info("Models loaded successfully")
        
    #     # Initialize pipeline
    #     app.state.rag_pipeline = EnhancedRAGPipeline(embedding_model, reranker_model)
    #     logger.info("Enhanced RAG Pipeline initialized successfully")
        
    #     # Download NLTK data
    #     import nltk
    #     nltk.download('punkt', quiet=True)
    #     nltk.download('stopwords', quiet=True)
    #     nltk.download('wordnet', quiet=True)
        
    # except Exception as e:
    #     logger.error(f"Failed to initialize: {e}")
    #     raise
    
    # yield
    
    # # Shutdown
    # logger.info("Shutting down...")
    # cache.clear()
    # logger.info(f"Final cache stats: {cache.get_stats()}")
    # logger.info("Shutdown complete")
    # Startup
    logger.info("Starting Enhanced RAG Pipeline v2...")
    app.state.performance_monitor = PerformanceMonitor()
    
    try:
        # Load models with error handling
        logger.info("Loading embedding model...")
        try:
            embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a smaller model
            logger.info("Trying fallback embedding model...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logger.info("Loading re-ranker model...")
        try:
            reranker_model = CrossEncoder(settings.RERANKER_MODEL_NAME)
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            # Continue without reranker
            reranker_model = None
            logger.warning("Running without reranker model")
        
        # Initialize pipeline
        app.state.rag_pipeline = EnhancedRAGPipeline(embedding_model, reranker_model)
        logger.info("Enhanced RAG Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        raise
    
    yield
    
    # Shutdown with cleanup
    logger.info("Shutting down...")
    
    # Clean up pipeline resources
    if hasattr(app.state, 'rag_pipeline'):
        await app.state.rag_pipeline.cleanup()
    if hasattr(app.state, 'performance_monitor'):
        final_stats = app.state.performance_monitor.get_stats()
        logger.info(f"Final performance stats: {final_stats}")
    
    # Final cache cleanup
    cache.clear()
    logger.info(f"Final cache stats: {cache.get_stats()}")
    
    # Clean any remaining temp files
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.startswith('vecstore_') or filename.startswith('tmp'):
            try:
                filepath = os.path.join(temp_dir, filename)
                if os.path.isdir(filepath):
                    shutil.rmtree(filepath)
                else:
                    os.unlink(filepath)
            except:
                pass
    
    logger.info("Shutdown complete")
@app.get("/performance/stats", tags=["Monitoring"])
async def get_performance_stats(request: Request):
    """Get current performance statistics"""
    if hasattr(request.app.state, 'performance_monitor'):
        return request.app.state.performance_monitor.get_stats()
    return {"error": "Performance monitor not initialized"}
# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Advanced document query system with 80%+ accuracy target",
    version="2.0.0",
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
        "version": "2.0.0",
        "features": [
            "Universal document parsing with Tika",
            "Hierarchical chunking for better context",
            "Cross-encoder re-ranking for accuracy",
            "Multi-step reasoning for complex questions",
            "Query expansion for better retrieval",
            "Answer validation and correction",
            "Question type detection and routing",
            "Support for 100+ file formats",
            "Binary file handling",
            "Advanced table extraction"
        ],
        "performance": {
            "target_accuracy": "80%+",
            "target_speed": "<30s for 10 questions",
            "max_document_size": f"{settings.MAX_DOCUMENT_SIZE_MB}MB"
        },
        "models": {
            "embedding": settings.EMBEDDING_MODEL_NAME,
            "reranker": settings.RERANKER_MODEL_NAME,
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
        "pipeline": "enhanced-v2",
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