# app/main.py - Main FastAPI application
import logging
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
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
        # Verify required environment variables
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY environment variable is required")
        
        # Load embedding model with error handling
        try:
            embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            logger.info(f"✅ Loaded embedding model: {settings.EMBEDDING_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to a smaller model if primary fails
            try:
                embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.warning("Using fallback embedding model: all-MiniLM-L6-v2")
            except Exception as e2:
                logger.error(f"Fallback model also failed: {e2}")
                raise
        
        # Initialize pipeline
        app.state.rag_pipeline = HybridRAGPipeline(embedding_model)
        logger.info("✅ RAG Pipeline initialized successfully")
        
        # Initialize cache directories
        os.makedirs('.cache', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        logger.info("🚀 Application startup complete")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize application: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down application...")
    try:
        cache_stats = cache.get_stats()
        cache.clear()
        logger.info(f"📊 Final cache stats: {cache_stats}")
        logger.info("✅ Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="High-performance document query system with balanced speed and accuracy",
    version="6.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def fix_encoding(request: Request, call_next):
    # Fix UTF-8 encoding for request body
    if request.headers.get("content-type") == "application/json":
        body = await request.body()
        if body:
            try:
                # Ensure proper UTF-8 decoding
                decoded = body.decode('utf-8')
                # Re-encode to fix any encoding issues
                request._body = decoded.encode('utf-8')
            except:
                pass
    
    response = await call_next(request)
    return response

# Middleware for request logging and error handling
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        if request.url.path.startswith("/api"):
            logger.info(
                f"{request.method} {request.url.path} - "
                f"Status: {response.status_code} - "
                f"Time: {process_time:.2f}s"
            )
        
        # Add performance headers
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            f"{request.method} {request.url.path} - "
            f"Error: {str(e)} - "
            f"Time: {process_time:.2f}s"
        )
        raise

# Include routers
app.include_router(query.router, prefix=settings.API_V1_STR, tags=["Query"])

@app.get("/", tags=["Root"])
async def read_root():
    """Root endpoint with system information"""
    try:
        cache_stats = cache.get_stats()
        return {
            "message": f"Welcome to {settings.APP_NAME}",
            "version": "6.0.0",
            "status": "operational",
            "features": [
                "✅ Balanced speed and accuracy optimization",
                "✅ Hybrid memory and disk caching", 
                "✅ Comprehensive format support (PDF, DOCX, Excel, PPT, Images, ZIP)",
                "✅ Smart context-aware chunking",
                "✅ Multi-strategy retrieval (BM25, TF-IDF, Semantic)",
                "✅ Adaptive answer generation",
                "✅ Parallel question processing",
                "✅ Dynamic document intelligence extraction",
                "✅ Human-like response generation"
            ],
            "performance": {
                "target_accuracy": "90%+",
                "target_speed": "<10s for 10 questions",
                "max_document_size": f"{settings.MAX_DOCUMENT_SIZE_MB}MB"
            },
            "models": {
                "embedding": settings.EMBEDDING_MODEL_NAME,
                "llm_fast": settings.LLM_MODEL_NAME,
                "llm_accurate": settings.LLM_MODEL_NAME_PRECISE
            },
            "cache": {
                "status": "active",
                "stats": cache_stats
            }
        }
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        return {"message": "Service running but with limited information", "status": "degraded"}

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    try:
        # Basic health checks
        pipeline_status = "healthy" if hasattr(app.state, 'rag_pipeline') else "unhealthy"
        cache_stats = cache.get_stats()
        
        # Check if required models are accessible
        model_status = "healthy"
        try:
            if hasattr(app.state, 'rag_pipeline'):
                # Quick model check
                test_embedding = app.state.rag_pipeline.embedding_model.encode(["test"])
                if test_embedding is None:
                    model_status = "unhealthy"
        except Exception:
            model_status = "unhealthy"
        
        health_status = "healthy" if (pipeline_status == "healthy" and model_status == "healthy") else "degraded"
        
        return {
            "status": health_status,
            "pipeline": "hybrid-optimized-v6",
            "components": {
                "rag_pipeline": pipeline_status,
                "embedding_model": model_status,
                "cache": "healthy"
            },
            "cache": cache_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/cache/stats", tags=["Cache"])
async def get_cache_stats():
    """Get detailed cache statistics"""
    try:
        return {
            "cache_stats": cache.get_stats(),
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to get cache statistics")

@app.post("/cache/clear", tags=["Cache"])
async def clear_cache():
    """Clear all caches"""
    try:
        old_stats = cache.get_stats()
        cache.clear()
        new_stats = cache.get_stats()
        
        logger.info("Cache cleared manually")
        return {
            "message": "Cache cleared successfully",
            "old_stats": old_stats,
            "new_stats": new_stats,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear cache")

@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get system metrics for monitoring"""
    try:
        cache_stats = cache.get_stats()
        
        return {
            "cache": cache_stats,
            "models": {
                "embedding_model": settings.EMBEDDING_MODEL_NAME,
                "llm_model": settings.LLM_MODEL_NAME,
                "llm_precise": settings.LLM_MODEL_NAME_PRECISE
            },
            "config": {
                "max_document_size_mb": settings.MAX_DOCUMENT_SIZE_MB,
                "app_name": settings.APP_NAME
            },
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get metrics")

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}", exc_info=True)
    return HTTPException(
        status_code=500,
        detail="An internal server error occurred"
    )