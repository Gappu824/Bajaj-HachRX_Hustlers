# app/utils/cache_warmup.py
import asyncio
import logging
from app.core.rag_pipeline import HybridFastTrackRAGPipeline
from sentence_transformers import SentenceTransformer
from app.core.config import settings

logger = logging.getLogger(__name__)

# Large documents to pre-cache
LARGE_DOCUMENTS = [
    "https://hackrx.blob.core.windows.net/assets/principia_newton.pdf?sv=2023-01-03&st=2025-07-28T07%3A20%3A32Z&se=2026-07-29T07%3A20%3A00Z&sr=b&sp=r&sig=V5I1QYyigoxeUMbnUKsdEaST99F5%2FDfo7wpKg9XXF5w%3D"
]

async def warmup_cache():
    """Pre-cache large documents"""
    logger.info("Starting cache warmup...")
    
    # Initialize pipeline
    embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
    pipeline = HybridFastTrackRAGPipeline(embedding_model)
    
    for url in LARGE_DOCUMENTS:
        try:
            logger.info(f"Pre-caching: {url}")
            await pipeline.get_or_create_optimized_vector_store(url)
            logger.info(f"Successfully cached: {url}")
        except Exception as e:
            logger.error(f"Failed to cache {url}: {e}")
    
    logger.info("Cache warmup completed")

if __name__ == "__main__":
    asyncio.run(warmup_cache())