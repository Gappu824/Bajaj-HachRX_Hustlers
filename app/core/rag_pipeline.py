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

    def search(self, query: str, k: int = 15) -> List[Tuple[str, float, Dict]]:
        """Hybrid search combining semantic and keyword search"""
        
        # Semantic search
        query_embedding = self.model.encode([query], show_progress_bar=False).astype('float32')
        distances, indices = self.index.search(query_embedding, min(k * 2, len(self.chunks)))
        
        # Keyword search
        keyword_results = self.enhanced_retriever.retrieve(query, k=k * 2)
        
        # Combine results
        combined_scores = defaultdict(float)
        
        # Add semantic scores
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                # Convert distance to similarity score
                similarity = 1.0 / (1.0 + dist)
                combined_scores[idx] += similarity * 0.6  # 60% weight for semantic
        
        # Add keyword scores
        if keyword_results:
            max_keyword_score = max(score for _, score in keyword_results)
            for idx, score in keyword_results:
                if idx < len(self.chunks):
                    normalized_score = score / max_keyword_score if max_keyword_score > 0 else 0
                    combined_scores[idx] += normalized_score * 0.4  # 40% weight for keywords
        
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
        
        # Configure Gemini
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise
    
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
    async def get_or_create_vector_store(self, url: str) -> OptimizedVectorStore:
        """Get or create vector store, with special handling for large ZIP files.""" # --- MODIFIED ---
        
        # --- This cache check logic remains the same ---
        cache_key = f"vecstore_{hashlib.md5(url.encode()).hexdigest()}"
        cached_store = await cache.get(cache_key)
        if cached_store:
            logger.info(f"Using cached vector store for {url}")
            return cached_store
        
        logger.info(f"Creating new vector store for {url}")
        
        # --- ADDED: Detect file type to decide on processing strategy ---
        file_extension = os.path.splitext(url.split('?')[0])[1].lower()
        
        # --- ADDED: Conditional logic to handle ZIPs differently ---
        if file_extension == '.zip':
            # Use the new memory-safe incremental process for ZIPs
            vector_store = await self._process_zip_incrementally(url)
        else:
            # --- MODIFIED: This is your original logic, now used for non-ZIP files ---
            # Note: Assumes you have a download function that returns bytes for these files.
            content = await self.download_document(url) 
            
            text, metadata = DocumentParser.parse_document(content, file_extension)
            
            if not text or len(text) < 10:
                logger.error(f"Document parsing failed or empty: {url}")
                text = "Document could not be parsed or is empty."
                metadata = [{'type': 'error'}]
            
            chunks, chunk_metadata = SmartChunker.chunk_document(
                text, metadata,
                chunk_size=settings.CHUNK_SIZE_CHARS,
                overlap=settings.CHUNK_OVERLAP_CHARS
            )
            
            logger.info(f"Created {len(chunks)} chunks")
            embeddings = await self._generate_embeddings(chunks)
            
            # --- MODIFIED: Create the vector store using the new incremental approach ---
            # Note: Assumes self.embedding_model.get_sentence_embedding_dimension() exists
            dimension = self.embedding_model.get_sentence_embedding_dimension()
            vector_store = OptimizedVectorStore(self.embedding_model, dimension)
            # Add all data in one batch
            vector_store.add(chunks, embeddings, chunk_metadata)

        # --- MODIFIED: Caching is now done at the end, regardless of file type ---
        await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS)
        logger.info(f"Cached vector store for document: {url}")
        
        return vector_store
    

    # Add these methods inside the HybridRAGPipeline class

    async def download_document_path(self, url: str) -> str:
        """
        Downloads a document by streaming it to a temporary file on disk
        and returns the file path. This is essential for large files.
        """
        headers = {'User-Agent': 'Mozilla/5.0'}
        try:
            timeout = aiohttp.ClientTimeout(total=600)  # 10 minute timeout for large downloads
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
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        vector_store = OptimizedVectorStore(self.embedding_model, dimension)
        
        file_path = None
        try:
            # 2. Download the ZIP to a temporary file path on disk
            file_path = await self.download_document_path(url)
            
            # 3. The parser will now populate the vector store directly
            DocumentParser.parse_zip_incrementally(file_path, vector_store, self)
            
            return vector_store
        finally:
            # 4. CRITICAL: Clean up the temporary file from disk
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
    
    async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings in batches"""
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            # Generate embeddings
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.embedding_model.encode(
                    batch, 
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype('float32')
    
    async def answer_question(self, question: str, vector_store: OptimizedVectorStore) -> str:
        """Generate answer with adaptive strategy"""
        
        # Detect question complexity
        is_complex = self._is_complex_question(question)
        
        # Retrieve relevant chunks
        k = 20 if is_complex else 12
        search_results = vector_store.search(question, k=k)
        
        if not search_results:
            return "No relevant information found in the document."
        
        # Extract chunks
        chunks = [result[0] for result in search_results[:12 if is_complex else 8]]
        
        # Generate answer
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
    
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process multiple questions efficiently"""
        
        start_time = time.time()
        logger.info(f"Processing {len(questions)} questions for {document_url}")
        
        try:
            # Get vector store
            vector_store = await self.get_or_create_vector_store(document_url)
            
            # Process questions in parallel with semaphore
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)
            
            async def process_question(q):
                async with semaphore:
                    try:
                        return await self.answer_question(q, vector_store)
                    except Exception as e:
                        logger.error(f"Error processing question: {e}")
                        return "Error processing this question."
            
            # Create tasks
            tasks = [process_question(q) for q in questions]
            
            # Execute with timeout
            answers = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=settings.TOTAL_TIMEOUT_SECONDS
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions in {elapsed:.2f}s")
            
            return answers
            
        except asyncio.TimeoutError:
            logger.error("Overall processing timeout")
            return ["Processing timeout. Please try again."] * len(questions)
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