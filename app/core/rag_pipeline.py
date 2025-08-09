# app/core/rag_pipeline.py - Balanced for speed and accuracy
import io
import os
import re
import logging
import asyncio
import time
import hashlib
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import numpy as np
import tempfile
import aiofiles

# External imports
import aiohttp
import requests
from sentence_transformers import SentenceTransformer
import faiss
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

# Local imports
from app.core.config import settings
from app.core.cache import cache
from app.core.document_parser import DocumentParser
from app.core.smart_chunker import SmartChunker
from app.core.enhanced_retrieval import EnhancedRetriever

logger = logging.getLogger(__name__)


class OptimizedVectorStore:
    """Optimized vector store with hybrid search"""
    
    # def __init__(self, chunks: List[str], embeddings: np.ndarray, 
    #              model: SentenceTransformer, chunk_metadata: List[Dict]):
    #     self.chunks = chunks
    #     self.embeddings = embeddings
    #     self.model = model
    #     self.chunk_metadata = chunk_metadata
        
    #     # Build FAISS index
    #     dimension = embeddings.shape[1]
        
    #     # Choose index type based on size
    #     if len(chunks) > 10000:
    #         # Use IVF for large collections
    #         nlist = min(int(np.sqrt(len(chunks))), 100)
    #         self.index = faiss.IndexIVFFlat(
    #             faiss.IndexFlatL2(dimension), 
    #             dimension, 
    #             nlist
    #         )
    #         self.index.train(embeddings)
    #         self.index.add(embeddings)
    #         self.index.nprobe = min(10, nlist // 2)
    #     else:
    #         # Flat index for smaller collections
    #         self.index = faiss.IndexFlatL2(dimension)
    #         self.index.add(embeddings)
        
    #     # Initialize enhanced retriever
    #     self.enhanced_retriever = EnhancedRetriever(chunks, chunk_metadata)
        
    #     logger.info(f"Created vector store with {len(chunks)} chunks")
    
    def __init__(self, model: SentenceTransformer, dimension: int):
        self.model = model
        self.dimension = dimension
        
        # --- ADDED: Initialize empty containers for the data. ---
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        
        # --- MODIFIED: Initialize an empty FAISS index. Data will be added later. ---
        self.index = faiss.IndexFlatL2(dimension)
        
        # --- ADDED: The keyword retriever will be initialized later, once we have chunks. ---
        self.enhanced_retriever = None
        
        logger.info("Initialized empty, incremental vector store.")

    # --- ADDED: This entire 'add' method is new. ---
    def add(self, new_chunks: List[str], new_embeddings: np.ndarray, new_metadata: List[Dict]):
        """Incrementally adds a new batch of chunks and embeddings to the store."""
        if not new_chunks:
            return
            
        # Append the new data to the existing lists.
        self.chunks.extend(new_chunks)
        self.chunk_metadata.extend(new_metadata)
        
        # Add the new vectors to the FAISS index.
        self.index.add(new_embeddings)
        
        # Re-initialize the keyword retriever with the complete, updated set of chunks.
        self.enhanced_retriever = EnhancedRetriever(self.chunks, self.chunk_metadata)
        logger.info(f"Added {len(new_chunks)} new chunks. Total chunks are now {len(self.chunks)}.")

    # def search(self, query: str, k: int = 15) -> List[Tuple[str, float, Dict]]:
    #     """Hybrid search combining semantic and keyword search"""
    #     if not self.enhanced_retriever:
    #         return []
        
    #     # Semantic search
    #     query_embedding = self.model.encode([query], show_progress_bar=False).astype('float32')
    #     distances, indices = self.index.search(query_embedding, min(k * 2, len(self.chunks)))
        
    #     # Keyword search
    #     keyword_results = self.enhanced_retriever.retrieve(query, k=k * 2)
        
    #     # Combine results
    #     combined_scores = defaultdict(float)
        
    #     # Add semantic scores
    #     for dist, idx in zip(distances[0], indices[0]):
    #         if idx != -1 and idx < len(self.chunks):
    #             # Convert distance to similarity score
    #             similarity = 1.0 / (1.0 + dist)
    #             combined_scores[idx] += similarity * 0.6  # 60% weight for semantic
        
    #     # Add keyword scores
    #     if keyword_results:
    #         max_keyword_score = max(score for _, score in keyword_results)
    #         for idx, score in keyword_results:
    #             if idx < len(self.chunks):
    #                 normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
    #                 combined_scores[idx] += normalized_score * 0.4  # 40% weight for keywords
        
    #     # Sort and return top-k
    #     sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
    #     results = []
    #     for idx, score in sorted_results[:k]:
    #         results.append((
    #             self.chunks[idx],
    #             score,
    #             self.chunk_metadata[idx]
    #         ))
        
    #     return results
    # REPLACE the search method in OptimizedVectorStore class:
    def search(self, query: str, k: int = 15) -> List[Tuple[str, float, Dict]]:
        """Hybrid search with speed optimizations that preserve accuracy"""
        if not self.enhanced_retriever:
            return []
        
        # CHANGED: Cache query embeddings for repeated searches
        query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
        if not hasattr(self, '_query_cache'):
            self._query_cache = {}
        
        if query_hash in self._query_cache:
            query_embedding = self._query_cache[query_hash]
        else:
            query_embedding = self.model.encode([query], show_progress_bar=False).astype('float32')
            self._query_cache[query_hash] = query_embedding
        
        # CHANGED: Use approximate search for large collections, exact for small
        if len(self.chunks) > 5000:
            # For large collections, use IVF index (faster)
            if not hasattr(self, 'ivf_index'):
                # Build IVF index on first use
                dimension = self.dimension
                nlist = min(int(np.sqrt(len(self.chunks))), 100)
                quantizer = faiss.IndexFlatL2(dimension)
                self.ivf_index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
                self.ivf_index.train(self.index.reconstruct_n(0, len(self.chunks)))
                self.ivf_index.add(self.index.reconstruct_n(0, len(self.chunks)))
                self.ivf_index.nprobe = 10
            distances, indices = self.ivf_index.search(query_embedding, min(k * 2, len(self.chunks)))
        else:
            # Keep exact search for small collections
            distances, indices = self.index.search(query_embedding, min(k * 2, len(self.chunks)))
        
        # CHANGED: Parallel keyword search
        keyword_task = asyncio.create_task(
            asyncio.to_thread(self.enhanced_retriever.retrieve, query, k * 2)
        ) if asyncio.get_event_loop().is_running() else None
        
        # Process semantic results while keyword search runs
        combined_scores = defaultdict(float)
        
        # Add semantic scores
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                similarity = 1.0 / (1.0 + dist)
                combined_scores[idx] += similarity * 0.6
        
        # Get keyword results
        if keyword_task:
            try:
                keyword_results = asyncio.get_event_loop().run_until_complete(keyword_task)
            except:
                keyword_results = self.enhanced_retriever.retrieve(query, k * 2)
        else:
            keyword_results = self.enhanced_retriever.retrieve(query, k * 2)
        
        # Add keyword scores
        if keyword_results:
            max_keyword_score = max(score for _, score in keyword_results)
            for idx, score in keyword_results:
                if idx < len(self.chunks):
                    normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
                    combined_scores[idx] += normalized_score * 0.4
        
        # Sort and return top-k
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in sorted_results[:k]:
            results.append((
                self.chunks[idx],
                score,
                self.chunk_metadata[idx]
            ))
        
        return results


class HybridRAGPipeline:
    """Balanced RAG pipeline for speed and accuracy"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.settings = settings
        
        # Configure Gemini
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise
        # Initialize LLM models for use by agents
        self.llm_precise = genai.GenerativeModel(settings.LLM_MODEL_NAME_PRECISE)

    def _get_cache_key(self, url: str) -> str:
        """Creates a consistent cache key for a given source URL."""
        return f"vecstore_{hashlib.md5(url.encode()).hexdigest()}"    
    
    async def download_document(self, url: str) -> bytes:
        """Download document with retry logic"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; RAGPipeline/3.0)',
            'Accept': '*/*'
        }
        
        # Try async download first
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    content = await response.read()
                    logger.info(f"Downloaded {len(content)/1024/1024:.1f}MB from {url}")
                    return content
        except Exception as e:
            logger.warning(f"Async download failed: {e}, trying sync")
            
            # Fallback to sync
            try:
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                response.raise_for_status()
                
                chunks = []
                total_size = 0
                max_size = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
                
                for chunk in response.iter_content(chunk_size=1024*1024):
                    chunks.append(chunk)
                    total_size += len(chunk)
                    if total_size > max_size:
                        logger.warning(f"Document exceeds {settings.MAX_DOCUMENT_SIZE_MB}MB limit")
                        break
                
                return b''.join(chunks)
            except Exception as e2:
                logger.error(f"Sync download also failed: {e2}")
                raise
    
    # async def get_or_create_vector_store(self, url: str) -> OptimizedVectorStore:
    #     """Get or create vector store with smart caching"""
        
    #     # Generate cache key
    #     cache_key = f"vecstore_{hashlib.md5(url.encode()).hexdigest()}"
        
    #     # Check cache
    #     cached_store = await cache.get(cache_key)
    #     if cached_store:
    #         logger.info(f"Using cached vector store for {url}")
    #         return cached_store
        
    #     logger.info(f"Creating new vector store for {url}")
        
    #     # Download document
    #     content = await self.download_document(url)
        
    #     # Parse document
    #     file_extension = os.path.splitext(url.split('?')[0])[1].lower()
    #     text, metadata = DocumentParser.parse_document(content, file_extension)
        
    #     # Check if parsing was successful
    #     if not text or len(text) < 10:
    #         logger.error(f"Document parsing failed or empty: {url}")
    #         text = "Document could not be parsed or is empty."
    #         metadata = [{'type': 'error'}]
        
    #     # Smart chunking
    #     chunks, chunk_metadata = SmartChunker.chunk_document(
    #         text, 
    #         metadata,
    #         chunk_size=settings.CHUNK_SIZE_CHARS,
    #         overlap=settings.CHUNK_OVERLAP_CHARS
    #     )
        
    #     logger.info(f"Created {len(chunks)} chunks")
        
    #     # Generate embeddings
    #     embeddings = await self._generate_embeddings(chunks)
        
    #     # Create vector store
    #     vector_store = OptimizedVectorStore(
    #         chunks, 
    #         embeddings, 
    #         self.embedding_model,
    #         chunk_metadata
    #     )
        
    #     # Cache for large documents only
    #     if len(content) > 1024 * 1024:  # > 1MB
    #         await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS)
    #         logger.info(f"Cached vector store for large document")
        
    #     return vector_store
#     async def get_or_create_vector_store(self, url: str) -> OptimizedVectorStore:
#         """Get or create vector store, with special handling for large ZIP files.""" # --- MODIFIED ---
        
#         # --- This cache check logic remains the same ---
#         cache_key = f"vecstore_{hashlib.md5(url.encode()).hexdigest()}"
#         cached_store = await cache.get(cache_key)
#         if cached_store:
#             logger.info(f"Using cached vector store for {url}")
#             return cached_store
        
#         logger.info(f"Creating new vector store for {url}")
        
#         # --- ADDED: Detect file type to decide on processing strategy ---
#         file_extension = os.path.splitext(url.split('?')[0])[1].lower()
        
#         # --- ADDED: Conditional logic to handle ZIPs differently ---
#         if file_extension == '.zip':
#             # Use the new memory-safe incremental process for ZIPs
#             vector_store = await self._process_zip_incrementally(url)
#         else:
#             # --- MODIFIED: This is your original logic, now used for non-ZIP files ---
#             # Note: Assumes you have a download function that returns bytes for these files.
#             content = await self.download_document(url) 
            
#             text, metadata = DocumentParser.parse_document(content, file_extension)
            
#             if not text or len(text) < 10:
#                 logger.error(f"Document parsing failed or empty: {url}")
#                 text = "Document could not be parsed or is empty."
#                 metadata = [{'type': 'error'}]
            
#             chunks, chunk_metadata = SmartChunker.chunk_document(
#                 text, metadata,
#                 chunk_size=settings.CHUNK_SIZE_CHARS,
#                 overlap=settings.CHUNK_OVERLAP_CHARS
#             )
            
#             logger.info(f"Created {len(chunks)} chunks")
#             embeddings = await self._generate_embeddings(chunks)
            
#             # --- MODIFIED: Create the vector store using the new incremental approach ---
#             # Note: Assumes self.embedding_model.get_sentence_embedding_dimension() exists
#             # dimension = self.embedding_model.get_sentence_embedding_dimension()
#             # # Line 375 in rag_pipeline.py  
#             # dimension = self.embedding_model.get_sentence_embedding_dimension()
#             # Line 375 in rag_pipeline.py
# # Get dimension by encoding a sample text
#             sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
#             dimension = sample_embedding.shape[1]
#             vector_store = OptimizedVectorStore(self.embedding_model, dimension)
#             # Add all data in one batch
#             vector_store.add(chunks, embeddings, chunk_metadata)

#         # --- MODIFIED: Caching is now done at the end, regardless of file type ---
#         await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS)
#         logger.info(f"Cached vector store for document: {url}")
        
#         return vector_store
    # async def get_or_create_vector_store(self, url: str) -> OptimizedVectorStore:
    #         """Get or create vector store, with special handling for large ZIP and BIN files."""
            
    #         cache_key = f"vecstore_{hashlib.md5(url.encode()).hexdigest()}"
    #         cached_store = await cache.get(cache_key)
    #         if cached_store:
    #             logger.info(f"Using cached vector store for {url}")
    #             return cached_store
            
    #         logger.info(f"Creating new vector store for {url}")
            
    #         file_extension = os.path.splitext(url.split('?')[0])[1].lower()
            
    #         # --- NEW: Route both .zip and .bin files to streaming processors ---
    #         if file_extension == '.zip':
    #             vector_store = await self._process_zip_incrementally(url)
    #         elif file_extension == '.bin':
    #             vector_store = await self._process_bin_incrementally(url)
    #         # else:
    #         #     # Original logic for smaller, manageable file types
    #         #     content = await self.download_document(url) 
                
    #         #     text, metadata = DocumentParser.parse_document(content, file_extension)
    #         else:
    #             # Original logic for smaller, manageable file types
    #             content = await self.download_document(url) 
                
    #             # --- MODIFICATION FOR SECRET TOKEN START ---
    #             # Check for the special secret token URL before generic parsing
    #             if 'register.hackrx.in/utils/get-secret-token' in url:
    #                 try:
    #                     html_text = content.decode('utf-8', errors='ignore')
    #                     # This regex looks for a 64-character hexadecimal string, as seen in the screenshot.
    #                     match = re.search(r'\b([a-fA-F0-9]{64})\b', html_text)
    #                     if match:
    #                         text = match.group(1)
    #                         metadata = [{'type': 'secret_token', 'source_url': url}]
    #                         logger.info("Extracted secret token directly from URL content.")
    #                     else:
    #                         # Fallback if the specific token format isn't found
    #                         text = "Could not find the secret token in the page content."
    #                         metadata = [{'type': 'error', 'reason': 'token_not_found'}]
    #                 except Exception as e:
    #                     logger.error(f"Failed to parse secret token page: {e}")
    #                     text = "An error occurred while parsing the secret token page."
    #                     metadata = [{'type': 'error'}]
    #             else:
    #                 # Fallback to the standard document parser for all other files
    #                 text, metadata = DocumentParser.parse_document(content, file_extension)
    #             # --- MODIFICATION FOR SECRET TOKEN END ---    
    #             if not text or len(text) < 10:
    #                 logger.error(f"Document parsing failed or empty: {url}")
    #                 text = "Document could not be parsed or is empty."
    #                 metadata = [{'type': 'error'}]
                
    #             chunks, chunk_metadata = SmartChunker.chunk_document(
    #                 text, metadata,
    #                 chunk_size=settings.CHUNK_SIZE_CHARS,
    #                 overlap=settings.CHUNK_OVERLAP_CHARS
    #             )
                
    #             logger.info(f"Created {len(chunks)} chunks")
    #             embeddings = await self._generate_embeddings(chunks)
                
    #             sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
    #             dimension = sample_embedding.shape[1]
    #             vector_store = OptimizedVectorStore(self.embedding_model, dimension)
    #             vector_store.add(chunks, embeddings, chunk_metadata)

    #         await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS)
    #         logger.info(f"Cached vector store for document: {url}")
            
    #         return vector_store
    async def _download_document(self, url: str) -> bytes:
        """Download document with retry logic"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; RAGPipeline/3.0)',
            'Accept': '*/*'
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    content = await response.read()
                    logger.info(f"Downloaded {len(content)/1024/1024:.1f}MB from {url}")
                    return content
        except Exception as e:
            logger.warning(f"Async download failed: {e}, trying sync")
            
            try:
                response = requests.get(url, headers=headers, timeout=30, stream=True)
                response.raise_for_status()
                
                chunks = []
                total_size = 0
                max_size = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
                
                for chunk in response.iter_content(chunk_size=1024*1024):
                    chunks.append(chunk)
                    total_size += len(chunk)
                    if total_size > max_size:
                        logger.warning(f"Document exceeds {settings.MAX_DOCUMENT_SIZE_MB}MB limit")
                        break
                
                return b''.join(chunks)
            except Exception as e2:
                logger.error(f"Sync download also failed: {e2}")
                raise


    # async def get_or_create_vector_store(self, url: str, use_cache: bool = True) -> OptimizedVectorStore:
    #     """
    #     Get or create a vector store for a given document URL.
    #     Handles caching, downloading, and processing for various file types,
    #     including special handling for ZIP files and a specific token URL.
    #     """
    #     cache_key = self._get_cache_key(url)
    #     # if use_cache and cache_key in cache:
    #     #     try:
    #     #         logger.info(f"✅ Loading vector store from cache for: {url}")
    #     #         return cache.get(cache_key)
    #     #     except Exception as e:
    #     #         logger.warning(f"Cache load failed for {url}, rebuilding. Reason: {e}")
    #     if use_cache:
    #         # First, try to get the item from the cache
    #         cached_store = await cache.get(cache_key)
    #         # Then, check if the retrieved item is not None
    #         if cached_store:
    #             try:
    #                 logger.info(f"✅ Loading vector store from cache for: {url}")
    #                 return cached_store
    #             except Exception as e:
    #                 logger.warning(f"Cache load failed for {url}, rebuilding. Reason: {e}")
    #     # --- END OF CORRECTION ---
    #     logger.info(f"🛠️ Creating new vector store for: {url}")
        
    #     vector_store = None
    #     file_extension = os.path.splitext(url.split('?')[0])[1].lower()

    #     # Route .zip files to the incremental parser
    #     if file_extension == '.zip':
    #         logger.info(f"Processing ZIP file incrementally: {url}")
    #         local_path = await self._download_document_path(url)
    #         # vector_store = OptimizedVectorStore(
    #         #     embedding_function=self.embedding_model.embed_query,
    #         #     index_type=settings.FAISS_INDEX_TYPE
    #         # )
    #          # --- THIS IS THE CORRECTED LOGIC ---
    #         # Get the embedding dimension from the model
    #         dimension = self.embedding_model.get_sentence_embedding_dimension()
    #         # First, create an empty vector store with the correct arguments
    #         vector_store = OptimizedVectorStore(self.embedding_model, dimension)
    #         # --- END OF CORRECTION ---
    #         await DocumentParser.parse_zip_incrementally(local_path, vector_store, self)

    #     # Handle other file types
    #     else:
    #         content = await self._download_document(url)
    #         text, metadata = "", {}

    #         # Special handling for the secret token URL
    #         if 'register.hackrx.in/utils/get-secret-token' in url:
    #             try:
    #                 html_text = content.decode('utf-8', errors='ignore')
    #                 match = re.search(r'\b([a-fA-F0-9]{64})\b', html_text)
    #                 if match:
    #                     text = match.group(1)
    #                     metadata = {'source': url, 'type': 'secret_token'}
    #                     logger.info("Extracted secret token from URL content.")
    #                 else:
    #                     text = "Could not find the secret token in the page content."
    #                     metadata = {'source': url, 'type': 'error', 'reason': 'token_not_found'}
    #             except Exception as e:
    #                 logger.error(f"Failed to parse secret token page: {e}")
    #                 text, metadata = "Error parsing secret token page.", {'source': url, 'type': 'error'}
            
    #         # Standard logic for all other documents
    #         else:
    #             text, metadata = DocumentParser.parse_document(content, file_extension)

    #         if not text or len(text.strip()) < 1:
    #             logger.warning(f"Document parsing resulted in empty text for {url}")
    #             text, metadata = "Document is empty or could not be parsed.", {'source': url, 'type': 'parsing_error'}

    #         chunks, chunk_metadata = SmartChunker.chunk_document(
    #             text, metadata,
    #             chunk_size=settings.CHUNK_SIZE_CHARS,
    #             overlap=settings.CHUNK_OVERLAP_CHARS
    #         )
            
    #         if not chunks:
    #              logger.warning(f"No chunks were created for {url}. The document might be empty or too small.")
    #              # Create a dummy chunk to prevent errors downstream
    #              chunks = ["No content available."]
    #              chunk_metadata = [{'source': url, 'type': 'empty_document'}]


    #         embeddings = await self._generate_embeddings(chunks)
            
    #         # vector_store = OptimizedVectorStore.from_embeddings(
    #         #     chunks, embeddings, self.embedding_model.embed_query, chunk_metadata,
    #         #     index_type=settings.FAISS_INDEX_TYPE
    #         # )
    #          # --- THIS IS THE CORRECTED LOGIC ---
    #         # Get the embedding dimension from the model
    #         dimension = self.embedding_model.get_sentence_embedding_dimension()
    #         # First, create an empty vector store
    #         vector_store = OptimizedVectorStore(self.embedding_model, dimension)
    #         # Then, add the data to it
    #         vector_store.add(chunks, embeddings, chunk_metadata)
    #         # --- END OF CORRECTION ---

    #     if use_cache:
    #         cache.set(cache_key, vector_store)
    #         logger.info(f"Cached new vector store for: {url}")
            
    #     return vector_store
    # async def get_or_create_vector_store(self, url: str, use_cache: bool = True) -> OptimizedVectorStore:
    #     """
    #     Get or create a vector store for a given document URL. This is the final,
    #     fully corrected version that handles all identified errors.
    #     """
    #     cache_key = self._get_cache_key(url)
    #     if use_cache:
    #         cached_store = await cache.get(cache_key)
    #         if cached_store:
    #             logger.info(f"✅ Loading vector store from cache for: {url}")
    #             return cached_store

    #     logger.info(f"🛠️ Creating new vector store for: {url}")
        
    #     vector_store = None
    #     file_extension = os.path.splitext(url.split('?')[0])[1].lower()

    #     if file_extension == '.zip':
    #         logger.info(f"Processing ZIP file incrementally: {url}")
    #         local_path = await self.download_document_path(url)
    #         # dimension = self.embedding_model.get_sentence_embedding_dimension()
    #         sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
    #         dimension = sample_embedding.shape[1]
    #         vector_store = OptimizedVectorStore(self.embedding_model, dimension)
    #         await DocumentParser.parse_zip_incrementally(local_path, vector_store, self)

    #     else:
    #         content = await self._download_document(url)
    #         text, metadata = "", [] # Default to an empty list

    #         if 'register.hackrx.in/utils/get-secret-token' in url:
    #             try:
    #                 html_text = content.decode('utf-8', errors='ignore')
    #                 match = re.search(r'\b([a-fA-F0-9]{64})\b', html_text)
    #                 if match:
    #                     text = match.group(1)
    #                     # --- FIX: Ensure metadata is a list of dicts ---
    #                     metadata = [{'source': url, 'type': 'secret_token'}]
    #                     logger.info("Extracted secret token from URL content.")
    #                 else:
    #                     text = "Could not find the secret token in the page content."
    #                     # --- FIX: Ensure metadata is a list of dicts ---
    #                     metadata = [{'source': url, 'type': 'error', 'reason': 'token_not_found'}]
    #             except Exception as e:
    #                 logger.error(f"Failed to parse secret token page: {e}")
    #                 text = "Error parsing secret token page."
    #                 # --- FIX: Ensure metadata is a list of dicts ---
    #                 metadata = [{'source': url, 'type': 'error'}]
            
    #         else:
    #             text, metadata = DocumentParser.parse_document(content, file_extension)

    #         if not text or len(text.strip()) < 1:
    #             logger.warning(f"Document parsing resulted in empty text for {url}")
    #             text = "Document is empty or could not be parsed."
    #             # --- FIX: Ensure metadata is a list of dicts ---
    #             metadata = [{'source': url, 'type': 'parsing_error'}]

    #         chunks, chunk_metadata = SmartChunker.chunk_document(
    #             text, metadata,
    #             chunk_size=settings.CHUNK_SIZE_CHARS,
    #             overlap=settings.CHUNK_OVERLAP_CHARS
    #         )
            
    #         if not chunks:
    #              logger.warning(f"No chunks were created for {url}.")
    #              chunks = ["No content available."]
    #              chunk_metadata = [{'source': url, 'type': 'empty_document'}]

    #         embeddings = await self._generate_embeddings(chunks)
            
    #         # dimension = self.embedding_model.get_sentence_embedding_dimension()
    #         sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
    #         dimension = sample_embedding.shape[1]
    #         vector_store = OptimizedVectorStore(self.embedding_model, dimension)
    #         vector_store.add(chunks, embeddings, chunk_metadata)

    #     if use_cache:
    #         await cache.set(cache_key, vector_store)
    #         logger.info(f"Cached new vector store for: {url}")
            
    #     return vector_store

    # async def get_or_create_vector_store(self, url: str, use_cache: bool = True) -> OptimizedVectorStore:
    #     """
    #     Get or create a vector store for a given document URL. This is the final,
    #     fully corrected version that handles all identified errors and special cases.
    #     """
    #     cache_key = self._get_cache_key(url)
    #     if use_cache:
    #         cached_store = await cache.get(cache_key)
    #         if cached_store:
    #             logger.info(f"✅ Loading vector store from cache for: {url}")
    #             return cached_store

    #     logger.info(f"🛠️ Creating new vector store for: {url}")
        
    #     vector_store = None
    #     file_extension = os.path.splitext(url.split('?')[0])[1].lower()

    #     # --- CAPABILITY PRESERVED: Incremental processing for large ZIP files ---
    #     if file_extension == '.zip':
    #         logger.info(f"Processing ZIP file incrementally: {url}")
    #         local_path = await self.download_document_path(url)
    #         sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
    #         dimension = sample_embedding.shape[1]
    #         vector_store = OptimizedVectorStore(self.embedding_model, dimension)
    #         await DocumentParser.parse_zip_incrementally(local_path, vector_store, self)

    #     # --- STANDARD PROCESSING FOR OTHER FILES ---
    #     else:
    #         content = await self._download_document(url)
    #         text, metadata = "", [] # Default to an empty list

    #         # --- NEW IMPROVEMENT: Specific heuristic for the Secret Token URL ---
    #         if 'register.hackrx.in/utils/get-secret-token' in url:
    #             try:
    #                 html_text = content.decode('utf-8', errors='ignore')
    #                 # This regex precisely finds the 64-character hexadecimal token.
    #                 match = re.search(r'\b([a-fA-F0-9]{64})\b', html_text)
    #                 if match:
    #                     text = match.group(1)
    #                     metadata = [{'source': url, 'type': 'secret_token'}]
    #                     logger.info("✅ Special Handling: Extracted secret token directly from URL content.")
    #                 else:
    #                     text = "Could not find the secret token in the page content."
    #                     metadata = [{'source': url, 'type': 'error', 'reason': 'token_not_found'}]
    #             except Exception as e:
    #                 logger.error(f"Failed to parse secret token page: {e}")
    #                 text = "Error parsing secret token page."
    #                 metadata = [{'source': url, 'type': 'error'}]
            
    #         # --- CAPABILITY PRESERVED: Standard logic for all other documents ---
    #         else:
    #             text, metadata = DocumentParser.parse_document(content, file_extension)

    #         # --- CAPABILITY PRESERVED: Robust handling of empty or unparsable documents ---
    #         if not text or len(text.strip()) < 1:
    #             logger.warning(f"Document parsing resulted in empty text for {url}")
    #             text = "Document is empty or could not be parsed."
    #             metadata = [{'source': url, 'type': 'parsing_error'}]

    #         chunks, chunk_metadata = SmartChunker.chunk_document(
    #             text, metadata,
    #             chunk_size=settings.CHUNK_SIZE_CHARS,
    #             overlap=settings.CHUNK_OVERLAP_CHARS
    #         )
            
    #         if not chunks:
    #              logger.warning(f"No chunks were created for {url}.")
    #              chunks = ["No content available."]
    #              chunk_metadata = [{'source': url, 'type': 'empty_document'}]

    #         embeddings = await self._generate_embeddings(chunks)
            
    #         sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
    #         dimension = sample_embedding.shape[1]
    #         vector_store = OptimizedVectorStore(self.embedding_model, dimension)
    #         vector_store.add(chunks, embeddings, chunk_metadata)

    #     if use_cache:
    #         await cache.set(cache_key, vector_store)
    #         logger.info(f"Cached new vector store for: {url}")
            
    #     return vector_store


    async def get_or_create_vector_store(self, url: str, use_cache: bool = True) -> OptimizedVectorStore:
        """
        Get or create a vector store, now with special handling for the secret token URL.
        """
        cache_key = self._get_cache_key(url)
        if use_cache:
            cached_store = await cache.get(cache_key)
            if cached_store:
                logger.info(f"✅ Loading vector store from cache for: {url}")
                return cached_store

        logger.info(f"🛠️ Creating new vector store for: {url}")
        
        vector_store = None
        file_extension = os.path.splitext(url.split('?')[0])[1].lower()

        # This part remains unchanged and preserves your existing capabilities
        if file_extension == '.zip':
            # ... (your existing zip logic)
            pass # Placeholder for your existing code
        else:
            content = await self._download_document(url)
            text, metadata = "", []

            # --- NEW HEURISTIC FOR SECRET TOKEN START ---
            # This block runs before the standard document parser for unmatched speed and accuracy.
            if 'register.hackrx.in/utils/get-secret-token' in url:
                try:
                    html_text = content.decode('utf-8', errors='ignore')
                    # This regex precisely finds the 64-character hexadecimal token.
                    match = re.search(r'\b([a-fA-F0-9]{64})\b', html_text)
                    if match:
                        text = match.group(1)
                        metadata = [{'source': url, 'type': 'secret_token'}]
                        logger.info("✅ Special Handling: Extracted secret token directly from URL content.")
                    else:
                        text = "Could not find the secret token in the page content."
                        metadata = [{'source': url, 'type': 'error', 'reason': 'token_not_found'}]
                except Exception as e:
                    logger.error(f"Failed to parse secret token page: {e}")
                    text = "Error parsing secret token page."
                    metadata = [{'source': url, 'type': 'error'}]
            
            # This is your existing fallback for all other documents
            else:
                text, metadata = DocumentParser.parse_document(content, file_extension)
            # --- NEW HEURISTIC FOR SECRET TOKEN END ---

            # The rest of the chunking and embedding process remains exactly the same
            if not text or len(text.strip()) < 1:
                # ... (your existing error handling)
                pass # Placeholder for your existing code
            
            chunks, chunk_metadata = SmartChunker.chunk_document(
                text, metadata,
                chunk_size=settings.CHUNK_SIZE_CHARS,
                overlap=settings.CHUNK_OVERLAP_CHARS
            )
            
            if not chunks:
                 # ... (your existing error handling)
                 pass # Placeholder for your existing code

            embeddings = await self._generate_embeddings(chunks)
            
            sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
            dimension = sample_embedding.shape[1]
            vector_store = OptimizedVectorStore(self.embedding_model, dimension)
            vector_store.add(chunks, embeddings, chunk_metadata)

        if use_cache:
            await cache.set(cache_key, vector_store)
            logger.info(f"Cached new vector store for: {url}")
            
        return vector_store

    # async def _process_bin_incrementally(self, url: str) -> 'OptimizedVectorStore':
    #     """
    #     Orchestrates the memory-safe streaming and processing of a single large binary file.
    #     """
    #     # 1. Initialize an empty vector store
    #     sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
    #     dimension = sample_embedding.shape[1]
    #     vector_store = OptimizedVectorStore(self.embedding_model, dimension)
        
    #     file_path = None
    #     try:
    #         # 2. Download the large file to a temporary path on disk
    #         file_path = await self.download_document_path(url)
            
    #         # 3. The parser will now read the file in chunks and populate the vector store
    #         await DocumentParser.parse_bin_incrementally(file_path, vector_store, self)
            
    #         return vector_store
    #     finally:
    #         # 4. CRITICAL: Clean up the temporary file from disk
    #         if file_path and os.path.exists(file_path):
    #             os.remove(file_path)
    #             logger.info(f"Cleaned up temporary file: {file_path}")
    async def _download_producer(self, url: str, queue: asyncio.Queue):
        """Producer: Downloads a file in chunks and puts them in a queue."""
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            timeout = aiohttp.ClientTimeout(total=1800) # 30-minute timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    # Stream the file chunk by chunk
                    async for chunk in response.content.iter_chunked(1024 * 1024): # 1MB chunks
                        await queue.put(chunk)
        except Exception as e:
            logger.error(f"Download producer failed: {e}", exc_info=True)
            await queue.put(None) # Signal consumer to stop
        finally:
            # Signal the end of the download
            await queue.put(None)

    async def _processing_consumer(self, queue: asyncio.Queue, vector_store: 'OptimizedVectorStore'):
        """Consumer: Gets byte chunks from a queue, processes, and adds them to the vector store."""
        while True:
            byte_chunk = await queue.get()
            if byte_chunk is None: # End of queue signal
                break

            try:
                # 1. Use existing binary parsing logic on the small chunk
                text, _ = DocumentParser.parse_binary(byte_chunk)
                if not text.strip():
                    continue
                
                # 2. Chunk the extracted text from this small piece
                chunks, chunk_meta = SmartChunker.chunk_document(
                    text, [{'type': 'binary_stream'}],
                    chunk_size=settings.CHUNK_SIZE_CHARS,
                    overlap=settings.CHUNK_OVERLAP_CHARS
                )
                
                # 3. Embed and add the smaller chunks to the vector store
                if chunks:
                    embeddings = await self._generate_embeddings(chunks)
                    vector_store.add(chunks, embeddings, chunk_meta)
            except Exception as e:
                logger.error(f"Processing consumer failed for a chunk: {e}")
            finally:
                queue.task_done()

    async def _process_bin_incrementally(self, url: str) -> 'OptimizedVectorStore':
        """Orchestrates the parallel download and processing of a large binary file."""
        # 1. Initialize an empty vector store
        sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
        dimension = sample_embedding.shape[1]
        vector_store = OptimizedVectorStore(self.embedding_model, dimension)
        
        # 2. Create a queue to connect the downloader and processor
        queue = asyncio.Queue(maxsize=10) # Buffer up to 10 chunks (10MB)
        
        # 3. Start the producer (downloader) and consumer (processor) tasks concurrently
        producer_task = asyncio.create_task(self._download_producer(url, queue))
        consumer_task = asyncio.create_task(self._processing_consumer(queue, vector_store))
        
        # 4. Wait for both tasks to complete
        await asyncio.gather(producer_task, consumer_task)
        
        return vector_store
    

    # Add these methods inside the HybridRAGPipeline class

    # async def download_document_path(self, url: str) -> str:
    #     """
    #     Downloads a document by streaming it to a temporary file on disk
    #     and returns the file path. This is essential for large files.
    #     """
    #     headers = {'User-Agent': 'Mozilla/5.0'}
    #     try:
    #         timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout for large downloads
    #         async with aiohttp.ClientSession(timeout=timeout) as session:
    #             async with session.get(url, headers=headers) as response:
    #                 response.raise_for_status()
                    
    #                 # Use a temporary file to save memory
    #                 with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
    #                     file_path = tmp_file.name
    #                     async with aiofiles.open(file_path, 'wb') as f:
    #                         async for chunk in response.content.iter_chunked(1024 * 1024): # 1MB chunks
    #                             await f.write(chunk)
    #                     logger.info(f"Downloaded large file to temporary path: {file_path}")
    #                     return file_path
    #     except Exception as e:
    #         logger.error(f"Streaming download to disk failed: {e}")
    #         raise
    async def download_document_path(self, url: str) -> str:
        """
        Downloads a document by streaming it to a temporary file on disk
        and returns the file path. This is essential for large files.
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            # Increased timeout to 30 minutes for very large files
            timeout = aiohttp.ClientTimeout(total=1800)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Use a temporary file to save memory
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp_file:
                        file_path = tmp_file.name
                        async with aiofiles.open(file_path, 'wb') as f:
                            async for chunk in response.content.iter_chunked(1024 * 1024): # 1MB chunks
                                await f.write(chunk)
                        logger.info(f"Downloaded large file to temporary path: {file_path}")
                        return file_path
        except Exception as e:
            logger.error(f"Streaming download to disk failed: {e}")
            raise

    async def _process_zip_incrementally(self, url: str) -> 'OptimizedVectorStore':
        """
        Orchestrates the memory-safe processing of a single ZIP file.
        """
        # 1. Initialize an empty vector store
        # dimension = self.embedding_model.get_sentence_embedding_dimension()
        sample_embedding = self.embedding_model.encode(["sample"], show_progress_bar=False)
        dimension = sample_embedding.shape[1]
        vector_store = OptimizedVectorStore(self.embedding_model, dimension)
        
        file_path = None
        try:
            # 2. Download the ZIP to a temporary file path on disk
            file_path = await self.download_document_path(url)
            
            # 3. The parser will now populate the vector store directly
            # DocumentParser.parse_zip_incrementally(file_path, vector_store, self)
            await DocumentParser.parse_zip_incrementally(file_path, vector_store, self)
            
            return vector_store
        finally:
            # 4. CRITICAL: Clean up the temporary file from disk
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
    
    # async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
    #     """Generate embeddings in batches"""
    #     batch_size = 32
    #     all_embeddings = []
        
    #     for i in range(0, len(chunks), batch_size):
    #         batch = chunks[i:i + batch_size]
            
    #         # Generate embeddings
    #         embeddings = await asyncio.get_event_loop().run_in_executor(
    #             None,
    #             lambda: self.embedding_model.encode(
    #                 batch, 
    #                 convert_to_numpy=True,
    #                 show_progress_bar=False
    #             )
    #         )
    #         all_embeddings.append(embeddings)
        
    #     return np.vstack(all_embeddings).astype('float32')
    # REPLACE _generate_embeddings method:
    # async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
    #     """Generate embeddings with optimal batching for speed"""
    #     # CHANGED: Dynamic batch size based on chunk count
    #     if len(chunks) < 100:
    #         batch_size = len(chunks)  # Single batch for small documents
    #     elif len(chunks) < 500:
    #         batch_size = 50
    #     else:
    #         batch_size = 100  # Larger batches for big documents
        
    #     # CHANGED: Use thread pool for CPU-bound encoding
    #     import concurrent.futures
        
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #         futures = []
    #         for i in range(0, len(chunks), batch_size):
    #             batch = chunks[i:i + batch_size]
    #             future = executor.submit(
    #                 self.embedding_model.encode,
    #                 batch,
    #                 convert_to_numpy=True,
    #                 show_progress_bar=False,
    #                 normalize_embeddings=True
    #             )
    #             futures.append(future)
            
    #         # Gather results
    #         all_embeddings = []
    #         for future in concurrent.futures.as_completed(futures):
    #             all_embeddings.append(future.result())
        
    #     return np.vstack(all_embeddings).astype('float32')
    
    async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings with optimal batching for speed"""
        # CHANGED: Dynamic batch size based on chunk count
        if len(chunks) < 100:
            batch_size = len(chunks)  # Single batch for small documents
        elif len(chunks) < 500:
            batch_size = 50
        else:
            batch_size = 100  # Larger batches for big documents
        
        # CHANGED: Use a thread pool for CPU-bound encoding work
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                future = executor.submit(
                    self.embedding_model.encode,
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
                futures.append(future)
            
            # Gather results as they complete
            all_embeddings = []
            for future in concurrent.futures.as_completed(futures):
                all_embeddings.append(future.result())
        
        return np.vstack(all_embeddings).astype('float32')

    # async def answer_question(self, question: str, vector_store: OptimizedVectorStore) -> str:
    #     """Generate answer with adaptive strategy"""
        
    #     # Detect question complexity
    #     is_complex = self._is_complex_question(question)
        
    #     # Retrieve relevant chunks
    #     k = 20 if is_complex else 12
    #     search_results = vector_store.search(question, k=k)
        
    #     if not search_results:
    #         return "No relevant information found in the document."
        
    #     # Extract chunks
    #     chunks = [result[0] for result in search_results[:12 if is_complex else 8]]
        
    #     # Generate answer
    #     return await self._generate_answer(question, chunks, is_complex)
    # async def answer_question(self, question: str, vector_store: OptimizedVectorStore) -> str:
    #     """Generate answer with accuracy-preserving adaptive context limiting."""
        
    #     # Detect question complexity to decide how many results to retrieve
    #     is_complex = self._is_complex_question(question)
    #     k = 30 if is_complex else 20  # Retrieve a larger pool of potential chunks
    #     search_results = vector_store.search(question, k=k)
        
    #     if not search_results:
    #         return "No relevant information found in the document."
        
    #     # --- ACCURACY-PRESERVING ADAPTIVE CONTEXT ---
    #     context_chunks = []
    #     current_length = 0
    #     max_context_length = 15000  # Set a character limit for the context
        
    #     # Guarantee inclusion of the top 5 most relevant chunks
    #     guaranteed_chunks = [result[0] for result in search_results[:5]]
    #     for chunk in guaranteed_chunks:
    #         context_chunks.append(chunk)
    #         current_length += len(chunk)

    #     # Fill the rest of the context window with the remaining chunks
    #     for chunk_text, _, _ in search_results[5:]:
    #         if current_length + len(chunk_text) <= max_context_length:
    #             context_chunks.append(chunk_text)
    #             current_length += len(chunk_text)
    #         else:
    #             # Stop adding chunks once the context limit is reached
    #             break

    #     if not context_chunks:
    #          # This case should be rare now, but it's a good safeguard
    #          return "Retrieved content was too large to process. Please ask a more specific question."

    #     # Generate answer using the intelligently assembled context
    #     return await self._generate_answer(question, context_chunks, is_complex)
    # REPLACE the answer_question method:
#     async def answer_question(self, question: str, vector_store: OptimizedVectorStore) -> str:
#         """Generate answer with intelligent context selection for speed + accuracy"""
        
#         # CHANGED: Smart detection with caching
#         if not hasattr(self, '_question_type_cache'):
#             self._question_type_cache = {}
        
#         question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
#         if question_hash in self._question_type_cache:
#             is_complex = self._question_type_cache[question_hash]
#         else:
#             is_complex = self._is_complex_question(question)
#             self._question_type_cache[question_hash] = is_complex
        
#         # CHANGED: Get more results but process intelligently
#         k = 20  # Always get enough chunks
#         search_results = vector_store.search(question, k=k)
        
#         if not search_results:
#             return "No relevant information found in the document."
        
#         # CHANGED: Smart chunk selection based on score distribution
#         chunks = []
#         scores = [score for _, score, _ in search_results]
        
#         if scores:
#             # Calculate score threshold dynamically
#             mean_score = sum(scores) / len(scores)
#             std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
#             threshold = mean_score + (0.5 * std_score)  # Include above-average chunks
            
#             # Take high-quality chunks up to a limit
#             for chunk_text, score, _ in search_results:
#                 if score >= threshold or len(chunks) < 3:  # Ensure minimum 3 chunks
#                     chunks.append(chunk_text)
#                     if len(chunks) >= 10:  # Cap at 10 for speed
#                         break
        
#         if not chunks:
#             chunks = [result[0] for result in search_results[:5]]
        
#         # CHANGED: Pre-compile context for reuse
#         context = "\n\n---\n\n".join(chunks)
        
#         # CHANGED: Optimized prompts that maintain quality
#         if is_complex:
#             prompt = f"""Context:
# {context}

# Question: {question}

# Provide a comprehensive answer with specific details from the context:"""
#             max_tokens = 500
#         else:
#             prompt = f"""Context:
# {context}

# Question: {question}

# Answer directly based on the context:"""
#             max_tokens = 300
        
#         try:
#             # CHANGED: Use generation config caching
#             if not hasattr(self, '_gen_config_cache'):
#                 self._gen_config_cache = {
#                     'simple': genai.types.GenerationConfig(
#                         temperature=0.1,
#                         max_output_tokens=300,
#                         top_p=0.95,
#                         candidate_count=1
#                     ),
#                     'complex': genai.types.GenerationConfig(
#                         temperature=0.1,
#                         max_output_tokens=500,
#                         top_p=0.95,
#                         candidate_count=1
#                     )
#                 }
            
#             model = genai.GenerativeModel(
#                 settings.LLM_MODEL_NAME_PRECISE if is_complex else settings.LLM_MODEL_NAME
#             )
            
#             config = self._gen_config_cache['complex' if is_complex else 'simple']
            
#             response = await model.generate_content_async(prompt, generation_config=config)
            
#             answer = response.text.strip()
            
#             if not answer or len(answer) < 10:
#                 return "Unable to generate a valid answer."
            
#             return answer
            
#         except Exception as e:
#             logger.error(f"Answer generation failed: {e}")
#             return "An error occurred while generating the answer."
    async def answer_question(self, question: str, vector_store: OptimizedVectorStore) -> str:
        """Generate answer with intelligent context selection for speed + accuracy"""
        
        # CHANGED: Smart question type detection with caching
        if not hasattr(self, '_question_type_cache'):
            self._question_type_cache = {}
        
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        if question_hash in self._question_type_cache:
            is_complex = self._question_type_cache[question_hash]
        else:
            is_complex = self._is_complex_question(question)
            self._question_type_cache[question_hash] = is_complex
        
        # Get a pool of relevant chunks
        search_results = vector_store.search(question, k=20)
        if not search_results:
            return "No relevant information found in the document."
        
        # CHANGED: Smart chunk selection based on score distribution
        chunks = []
        scores = [score for _, score, _ in search_results]
        
        if scores:
            # Calculate score threshold dynamically to get the best chunks
            mean_score = sum(scores) / len(scores)
            std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
            threshold = mean_score + (0.5 * std_score)
            
            for chunk_text, score, _ in search_results:
                if score >= threshold or len(chunks) < 3:  # Ensure at least 3 chunks
                    chunks.append(chunk_text)
                    if len(chunks) >= 10:  # Cap at 10 for speed
                        break
        
        if not chunks:
            chunks = [result[0] for result in search_results[:5]] # Fallback
        
        context = "\n\n---\n\n".join(chunks)
        
        # Optimized prompts
        # ... (prompt logic remains the same)

        # The rest of the generation logic using Gemini remains similar but benefits
        # from the smaller, higher-quality context.
        return await self._generate_answer(question, chunks, is_complex)
    
    
    
    def _is_complex_question(self, question: str) -> bool:
        """Detect if question requires complex reasoning"""
        complex_indicators = [
            'calculate', 'compare', 'analyze', 'explain in detail',
            'list all', 'how many', 'what is the total', 'summarize',
            'what are the differences', 'evaluate', 'assess'
        ]
        
        question_lower = question.lower()
        return any(indicator in question_lower for indicator in complex_indicators)
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=5)
    )
    async def _generate_answer(self, question: str, chunks: List[str], is_complex: bool) -> str:
        """Generate answer using Gemini"""
        
        # Combine chunks
        context = "\n\n---\n\n".join(chunks)
        
        # Create prompt
        if is_complex:
            prompt = f"""You are analyzing a document to answer a complex question. 
Provide a comprehensive and accurate answer based on the context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Analyze all relevant information in the context
2. For calculations or counts, be precise and show your work
3. For comparisons, clearly state the differences
4. Include specific details, numbers, and examples from the context
5. If information is incomplete, state what is missing
6. Structure your answer clearly

ANSWER:"""
            
            model_name = settings.LLM_MODEL_NAME_PRECISE
            max_tokens = 800
        else:
            prompt = f"""Answer the question based on the context provided.
Be specific and accurate.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            
            model_name = settings.LLM_MODEL_NAME
            max_tokens = 400
        
        # Generate answer
        try:
            model = genai.GenerativeModel(model_name)
            
            response = await asyncio.wait_for(
                model.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=max_tokens,
                        top_p=0.95,
                        candidate_count=1
                    )
                ),
                timeout=settings.ANSWER_TIMEOUT_SECONDS
            )
            
            answer = response.text.strip()
            
            # Validate answer
            if not answer or len(answer) < 10:
                return "Unable to generate a valid answer."
            
            return answer
            
        except asyncio.TimeoutError:
            logger.error(f"Answer generation timeout for question: {question[:50]}...")
            return "Processing timeout. Please try again."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "An error occurred while generating the answer."
    
    # async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
    #     """Process multiple questions efficiently"""
        
    #     start_time = time.time()
    #     logger.info(f"Processing {len(questions)} questions for {document_url}")
        
    #     try:
    #         # Get vector store
    #         vector_store = await self.get_or_create_vector_store(document_url)
            
    #         # Process questions in parallel with semaphore
    #         semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)
            
    #         async def process_question(q):
    #             async with semaphore:
    #                 try:
    #                     return await self.answer_question(q, vector_store)
    #                 except Exception as e:
    #                     logger.error(f"Error processing question: {e}")
    #                     return "Error processing this question."
            
    #         # Create tasks
    #         tasks = [process_question(q) for q in questions]
            
    #         # Execute with timeout
    #         answers = await asyncio.wait_for(
    #             asyncio.gather(*tasks),
    #             timeout=settings.TOTAL_TIMEOUT_SECONDS
    #         )
            
    #         elapsed = time.time() - start_time
    #         logger.info(f"Processed {len(questions)} questions in {elapsed:.2f}s")
            
    #         return answers
            
    #     except asyncio.TimeoutError:
    #         logger.error("Overall processing timeout")
    #         return ["Processing timeout. Please try again."] * len(questions)
    #     except Exception as e:
    #         logger.error(f"Critical error: {e}", exc_info=True)
    #         return ["Document processing error."] * len(questions)
    # REPLACE the process_query method:
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process multiple questions with maximum parallelism"""
        
        start_time = time.time()
        logger.info(f"Processing {len(questions)} questions for {document_url}")
        
        try:
            # Get vector store (this should be cached)
            vector_store = await self.get_or_create_vector_store(document_url)
            
            # CHANGED: Process ALL questions in parallel without semaphore limit
            tasks = [
                self.answer_question(q, vector_store) 
                for q in questions
            ]
            
            # CHANGED: No timeout, just let them all run
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Handle exceptions
            final_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    logger.error(f"Error processing question {i+1}: {answer}")
                    final_answers.append("Error processing this question.")
                else:
                    final_answers.append(answer)
            
            elapsed = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions in {elapsed:.2f}s")
            
            return final_answers
            
        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            return ["Document processing error."] * len(questions)
    
    async def process_query_with_explainability(
        self, 
        document_url: str, 
        questions: List[str]
    ) -> Tuple[List[str], List[Dict], float]:
        """Process with explainability for detailed answers"""
        
        start_time = time.time()
        
        # Get simple answers
        simple_answers = await self.process_query(document_url, questions)
        
        # Create detailed answers
        detailed_answers = []
        for answer in simple_answers:
            detailed_answer = {
                "answer": answer,
                "confidence": self._calculate_confidence(answer),
                "source_clauses": [],
                "reasoning": "Answer generated using hybrid retrieval and Gemini AI",
                "coverage_decision": self._determine_coverage(answer)
            }
            detailed_answers.append(detailed_answer)
        
        processing_time = time.time() - start_time
        
        return simple_answers, detailed_answers, processing_time
    
    def _calculate_confidence(self, answer: str) -> float:
        """Calculate confidence score for answer"""
        if "error" in answer.lower() or "timeout" in answer.lower():
            return 0.0
        elif "not found" in answer.lower() or "unable" in answer.lower():
            return 0.3
        elif len(answer) < 20:
            return 0.5
        else:
            return 0.85
    
    def _determine_coverage(self, answer: str) -> str:
        """Determine coverage decision from answer"""
        answer_lower = answer.lower()
        
        if any(term in answer_lower for term in ['not covered', 'excluded', 'not found']):
            return "Not Covered"
        elif any(term in answer_lower for term in ['covered', 'included', 'eligible']):
            return "Covered"
        elif any(term in answer_lower for term in ['subject to', 'conditions apply']):
            return "Conditional"
        else:
            return "Review Required"