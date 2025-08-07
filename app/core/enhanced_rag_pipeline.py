# app/core/enhanced_rag_pipeline.py - Complete rewrite with all improvements
import io
import os
import re
import logging
import asyncio
import time
import hashlib
import json
from typing import List, Tuple, Dict, Optional, Any
from collections import defaultdict
import numpy as np
import tempfile
import aiofiles  # Missing import
import shutil
import pickle
import pdfplumber

# External imports
import aiohttp
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential
import nltk
from rapidfuzz import fuzz

# Local imports
from app.core.config import settings
from app.core.cache import cache
from app.core.universal_parser import UniversalDocumentParser
from app.core.smart_chunker import SmartChunker
from app.core.enhanced_retrieval import EnhancedRetriever
from app.core.question_analyzer import QuestionAnalyzer
from app.core.answer_validator import AnswerValidator
# Add these imports at the top of enhanced_rag_pipeline.py
import aiofiles  # Missing import
# from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

class AdvancedVectorStore:
    """Advanced vector store with hierarchical chunking and re-ranking"""
    
    # def __init__(self, model: SentenceTransformer, reranker: CrossEncoder, dimension: int):
    def __init__(self, model: SentenceTransformer, reranker: CrossEncoder, dimension: int, document_url: str = None):
        self.model = model
        self.reranker = reranker
        self.dimension = dimension
        self.document_url = document_url
        self.document_hash = hashlib.md5(document_url.encode()).hexdigest() if document_url else None

        self.max_chunks_in_memory = 500  # Keep only top 500 chunks in memory
        self.disk_cache_dir = tempfile.mkdtemp(prefix="vecstore_")
        
        # Hierarchical storage
        self.small_chunks: List[str] = []
        self.large_chunks: List[str] = []
        self.chunk_metadata: List[Dict] = []
        self.chunk_to_large_mapping: Dict[int, int] = {}  # Maps small chunk index to large chunk index
        
        # FAISS indices
        self.small_index = faiss.IndexFlatL2(dimension)
        self.large_index = faiss.IndexFlatL2(dimension)
        
        # Enhanced retriever
        self.enhanced_retriever = None
        
        logger.info("Initialized advanced vector store with hierarchical chunking")

    def clear(self):
        """Clear memory and disk cache"""
        # Clear chunks
        self.small_chunks.clear()
        self.large_chunks.clear()
        self.chunk_metadata.clear()
        self.chunk_to_large_mapping.clear()
        
        # Clear FAISS indices
        self.small_index = faiss.IndexFlatL2(self.dimension)
        self.large_index = faiss.IndexFlatL2(self.dimension)
        
        # Clean disk cache directory
        if hasattr(self, 'disk_cache_dir') and os.path.exists(self.disk_cache_dir):
            try:
                shutil.rmtree(self.disk_cache_dir)
                self.disk_cache_dir = tempfile.mkdtemp(prefix="vecstore_")
            except Exception as e:
                logger.warning(f"Failed to clear disk cache: {e}")
        
        # Clear retriever
        self.enhanced_retriever = None
        
        logger.info("Vector store cleared")    
    
    def add_hierarchical(self, small_chunks: List[str], small_embeddings: np.ndarray,
                         large_chunks: List[str], large_embeddings: np.ndarray,
                         metadata: List[Dict], chunk_mapping: Dict[int, int]):
        """Add hierarchical chunks to the store"""
        if len(self.small_chunks) + len(small_chunks) > self.max_chunks_in_memory:
            self._evict_to_disk()
        # Store chunks
        # base_idx = len(self.small_chunks)
        # self.small_chunks.extend(small_chunks)
        # self.large_chunks.extend(large_chunks)
        # self.chunk_metadata.extend(metadata)
        base_idx = len(self.small_chunks)
        self.small_chunks.extend(small_chunks[:self.max_chunks_in_memory - len(self.small_chunks)])
        self.large_chunks.extend(large_chunks[:self.max_chunks_in_memory // 2 - len(self.large_chunks)])
        self.chunk_metadata.extend(metadata[:len(self.small_chunks) - base_idx])
        
        # Update mapping
        # for small_idx, large_idx in chunk_mapping.items():
        #     self.chunk_to_large_mapping[base_idx + small_idx] = large_idx
        for small_idx, large_idx in chunk_mapping.items():
            if base_idx + small_idx < self.max_chunks_in_memory:
                self.chunk_to_large_mapping[base_idx + small_idx] = large_idx
        
        # Add to indices
        # self.small_index.add(small_embeddings)
        # self.large_index.add(large_embeddings)
        self.small_index.add(small_embeddings[:len(self.small_chunks) - base_idx])
        self.large_index.add(large_embeddings[:len(self.large_chunks)])
        
        # Reinitialize retriever
        self.enhanced_retriever = EnhancedRetriever(self.small_chunks, self.chunk_metadata)
        
        # logger.info(f"Added {len(small_chunks)} small and {len(large_chunks)} large chunks")
        logger.info(f"Added chunks (memory: {len(self.small_chunks)}, total: {base_idx + len(small_chunks)})")
    def _evict_to_disk(self):
        """Evict oldest chunks to disk"""
        # OLD: No eviction
        # NEW: Save to disk and clear from memory
        
        evict_count = len(self.small_chunks) // 2
        
        # Save to disk
        disk_file = os.path.join(self.disk_cache_dir, f"chunks_{time.time()}.pkl")
        # data_to_evict = {
        #     'chunks': self.small_chunks[:evict_count],
        #     'metadata': self.chunk_metadata[:evict_count]
        # }
        with open(disk_file, 'wb') as f:
            pickle.dump({
                'chunks': self.small_chunks[:evict_count],
                'metadata': self.chunk_metadata[:evict_count]
            }, f)
        
        # Remove from memory
        self.small_chunks = self.small_chunks[evict_count:]
        self.chunk_metadata = self.chunk_metadata[evict_count:]
        
        logger.info(f"Evicted {evict_count} chunks to disk")    
    
    def search_with_reranking(self, query: str, k: int = 15) -> List[Tuple[str, float, Dict]]:
        """Advanced search with re-ranking"""
        
        # 1. Initial retrieval - get more candidates
        candidates_k = min(k * 3, len(self.small_chunks))
        
        # Encode query
        query_embedding = self.model.encode([query], show_progress_bar=False).astype('float32')
        
        # Semantic search on small chunks
        distances, indices = self.small_index.search(query_embedding, candidates_k)
        
        # Keyword search
        keyword_results = []
        if self.enhanced_retriever:
            keyword_results = self.enhanced_retriever.retrieve(query, k=candidates_k)
        
        # 2. Combine candidates
        candidate_indices = set()
        for idx in indices[0]:
            if 0 <= idx < len(self.small_chunks):
                candidate_indices.add(idx)
        
        for idx, _ in keyword_results:
            if 0 <= idx < len(self.small_chunks):
                candidate_indices.add(idx)
        
        # 3. Include context from large chunks
        enriched_candidates = []
        for idx in candidate_indices:
            small_chunk = self.small_chunks[idx]
            
            # Get corresponding large chunk for context
            large_idx = self.chunk_to_large_mapping.get(idx)
            if large_idx is not None and large_idx < len(self.large_chunks):
                context = self.large_chunks[large_idx]
                # Combine small chunk with partial context
                enriched_text = f"{small_chunk}\n\n[CONTEXT]: {context[:500]}"
            else:
                enriched_text = small_chunk
            
            enriched_candidates.append((idx, enriched_text))
        
        # 4. Re-rank with cross-encoder
        if enriched_candidates and self.reranker:
            # Prepare pairs for re-ranking
            pairs = [[query, text] for _, text in enriched_candidates]
            
            try:
                # Get re-ranking scores
                scores = self.reranker.predict(pairs)
                
                # Combine with indices
                scored_candidates = [
                    (enriched_candidates[i][0], scores[i], enriched_candidates[i][1])
                    for i in range(len(enriched_candidates))
                ]
                
                # Sort by score
                scored_candidates.sort(key=lambda x: x[1], reverse=True)
                
                # Return top-k with metadata
                results = []
                for idx, score, text in scored_candidates[:k]:
                    results.append((
                        self.small_chunks[idx],  # Return original small chunk
                        float(score),
                        self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
                    ))
                
                return results
                
            except Exception as e:
                logger.warning(f"Re-ranking failed: {e}, falling back to basic search")
        
        # Fallback to basic search if re-ranking fails
        results = []
        for idx in list(candidate_indices)[:k]:
            results.append((
                self.small_chunks[idx],
               1.0,
               self.chunk_metadata[idx] if idx < len(self.chunk_metadata) else {}
           ))
       
        return results

class EnhancedRAGPipeline:
    """Complete RAG pipeline with all improvements"""
    
    def __init__(self, embedding_model: SentenceTransformer, reranker_model: CrossEncoder):
        # self.embedding_model = embedding_model
        # self.reranker_model = reranker_model
        # self.settings = settings
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.settings = settings
        self.active_stores = []  # Track active vector stores
        self.temp_dirs = []  # Track temp directories
        # Initialize components
        self.universal_parser = UniversalDocumentParser()
        self.question_analyzer = QuestionAnalyzer()
        self.answer_validator = AnswerValidator()

        self._configure_gemini()


    async def process_query_with_explainability(
    self, document_url: str, questions: List[str]
) -> Tuple[List[str], List[Dict], float]:
        """
        Process questions with detailed explainability.
        Returns simple answers, detailed answers with explanations, and processing time.
        """
        start_time = time.time()
        
        try:
            # Get or create vector store
            vector_store = await self.get_or_create_vector_store(document_url)
            
            simple_answers = []
            detailed_answers = []
            
            for question in questions:
                # Get detailed answer with all metadata
                result = await self.answer_question(question, vector_store)
                
                # Simple answer for backward compatibility
                simple_answers.append(result['answer'])
                
                # Detailed answer with explainability
                detailed = {
                    'answer': result['answer'],
                    'confidence': result.get('confidence', 0.5),
                    'source_clauses': [],
                    'reasoning': f"Question type: {result.get('question_type', 'general')}. ",
                    'coverage_decision': None
                }
                
                # Add source information
                for source in result.get('sources', [])[:3]:  # Top 3 sources
                    detailed['source_clauses'].append({
                        'text': source.get('preview', '')[:200],
                        'confidence_score': result.get('confidence', 0.5),
                        'page_number': source.get('chunk_index'),
                        'section': source.get('type', 'text')
                    })
                
                # Add reasoning
                if result.get('validation_notes'):
                    detailed['reasoning'] += result['validation_notes']
                else:
                    detailed['reasoning'] += f"Answer extracted from {len(result.get('sources', []))} relevant document sections."
                
                # Coverage decision for yes/no questions
                if result.get('question_type') == 'yes_no':
                    if 'yes' in result['answer'].lower():
                        detailed['coverage_decision'] = 'Covered'
                    elif 'no' in result['answer'].lower():
                        detailed['coverage_decision'] = 'Not Covered'
                    else:
                        detailed['coverage_decision'] = 'Conditional'
                
                detailed_answers.append(detailed)
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions with explainability in {processing_time:.2f}s")
            
            return simple_answers, detailed_answers, processing_time
            
        except Exception as e:
            logger.error(f"Error in explainability processing: {e}", exc_info=True)
            processing_time = time.time() - start_time
            error_answer = "Error processing question with explainability."
            return (
                [error_answer] * len(questions),
                [{'answer': error_answer, 'confidence': 0, 'reasoning': str(e)}] * len(questions),
                processing_time
            )



    async def _generate_embeddings_sequential(self, chunks: List[str]) -> np.ndarray:
        """Sequential embedding generation fallback"""
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True
                )
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype('float32') if all_embeddings else np.array([])

# Add this missing import in enhanced_rag_pipeline.py
# import pdfplumber  # Add this import at the top with other imports    

    async def _create_hierarchical_store(self, text: str, metadata: List[Dict]) -> AdvancedVectorStore:
        """Helper to create a vector store with hierarchical chunks."""
        small_chunks, small_meta = SmartChunker.chunk_document(
            text, metadata, settings.CHUNK_SIZE_CHARS, settings.CHUNK_OVERLAP_CHARS
        )
        large_chunks, large_meta = SmartChunker.chunk_document(
            text, metadata, settings.LARGE_CHUNK_SIZE_CHARS, settings.LARGE_CHUNK_OVERLAP_CHARS
        )
        chunk_mapping = self._create_chunk_mapping(small_chunks, large_chunks, text)
        small_embeds = await self._generate_embeddings(small_chunks)
        large_embeds = await self._generate_embeddings(large_chunks)
        
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        store = AdvancedVectorStore(self.embedding_model, self.reranker_model, dimension)
        store.add_hierarchical(
            small_chunks, small_embeds, large_chunks, large_embeds, small_meta, chunk_mapping
        )
        return store    
    async def _standard_processing(self, text: str, metadata: List[Dict]) -> AdvancedVectorStore:
        """Standard document processing for non-large files."""
        return await self._create_hierarchical_store(text, metadata)
    def _get_extension(self, url: str) -> str:
        """Helper to extract a file extension from a URL."""
        from urllib.parse import urlparse
        path = urlparse(url).path
        return os.path.splitext(path)[1].lower()
    async def _create_prioritized_store(self, prioritized_chunks) -> AdvancedVectorStore:
        """Creates a vector store from chunks that have already been scored and prioritized."""
        small_chunks = [chunk for chunk, meta, score in prioritized_chunks]
        metadata = [meta for chunk, meta, score in prioritized_chunks]

        # For a prioritized store, we can simplify by using the same chunks for both hierarchies
        embeddings = await self._generate_embeddings(small_chunks)
        
        dimension = self.embedding_model.get_sentence_embedding_dimension()
        store = AdvancedVectorStore(self.embedding_model, self.reranker_model, dimension)
        
        # A simple 1-to-1 mapping is sufficient here
        mapping = {i: i for i in range(len(small_chunks))}
        
        store.add_hierarchical(small_chunks, embeddings, small_chunks, embeddings, metadata, mapping)
        return store
    
    async def _process_questions_optimized(
    self, questions: List[str], vector_store: AdvancedVectorStore, question_analysis: Dict
) -> List[str]:
        """Processes questions using the optimized, question-aware vector store."""
        tasks = [self.answer_question(q, vector_store) for q in questions]
        results = await asyncio.gather(*tasks)
        return [res.get('answer', "Processing failed.") for res in results]
    # def _extract_section(self, content: bytes, section: Dict, url: str) -> Tuple[str, List[Dict]]:
    #     """A placeholder for a function that would extract specific sections.
    #     This logic would be highly dependent on the document format."""
    #     logger.warning("Section extraction is a placeholder and not fully implemented.")
    #     # For now, we just re-parse the whole document as a fallback.
    #     return self.universal_parser.parse_any_document(content, url)
    def _extract_section(self, content: bytes, section: Dict, url: str) -> Tuple[str, List[Dict]]:
        """
        Efficiently extracts text and tables only from specific pages of a PDF,
        as identified by the two-pass strategy's structural analysis.
        """
        text_parts = []
        metadata_parts = []
        
        # This logic is optimized for PDFs, which is the primary use case for the two-pass strategy.
        file_ext = self._get_extension(url)
        if file_ext != '.pdf':
            logger.warning(f"Two-pass section extraction is optimized for PDF, but got {file_ext}. Falling back to full parse.")
            return self.universal_parser.parse_any_document(content, url)

        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                pages_to_process = section.get('pages', [])
                
                for page_num in pages_to_process:
                    # Page numbers are 1-based, but pdf.pages is 0-indexed.
                    if 1 <= page_num <= len(pdf.pages):
                        page = pdf.pages[page_num - 1]
                        
                        # Extract text from the specific page
                        page_text = page.extract_text() or ""
                        if page_text.strip():
                            text_parts.append(f"--- START PAGE {page_num} ---\n{page_text}\n--- END PAGE {page_num} ---")
                            metadata_parts.append({'page': page_num, 'type': 'text_section'})
                            
                        # Extract tables from the specific page
                        tables = page.extract_tables()
                        for j, table in enumerate(tables):
                            if table:
                                table_text = self.universal_parser.fallback_parser._format_table(table)
                                text_parts.append(f"--- TABLE {j+1} ON PAGE {page_num} ---\n{table_text}")
                                metadata_parts.append({'page': page_num, 'type': 'table_section'})
                                
        except Exception as e:
            logger.error(f"Failed to extract section from PDF on pages {section.get('pages')}: {e}")
            # As a last resort, fall back to parsing the whole document.
            return self.universal_parser.parse_any_document(content, url)

        return "\n\n".join(text_parts), metadata_parts


        
    def _configure_gemini(self):
        """Configure Gemini with retry logic"""
        # NEW: Robust Gemini configuration
        max_retries = 3
        for attempt in range(max_retries):
            try:
                genai.configure(api_key=settings.GOOGLE_API_KEY)
                # Test the configuration
                model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
                logger.info("Gemini AI configured successfully")
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to configure Gemini after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Gemini configuration attempt {attempt + 1} failed, retrying...")
                time.sleep(2 ** attempt)  # Exponential backoff   
    async def cleanup(self):
        """Clean up resources"""
        # NEW: Proper resource cleanup
        logger.info("Cleaning up pipeline resources...")
        
        # Clear vector stores
        for store in self.active_stores:
            try:
                store.clear()
            except:
                pass
        self.active_stores.clear()
        
        # Clean temp directories
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            except:
                pass
        self.temp_dirs.clear()
        
        # Clear cache
        cache.clear()
        
        logger.info("Pipeline cleanup complete")             
        
        # Configure Gemini
        # try:
        #     genai.configure(api_key=settings.GOOGLE_API_KEY)
        #     logger.info("Gemini AI configured successfully")
        # except Exception as e:
        #     logger.error(f"Failed to configure Gemini: {e}")
        #     raise
    
    # async def download_document(self, url: str) -> bytes:
    #     """Enhanced download with better error handling"""
    #     headers = {
    #         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    #         'Accept': '*/*',
    #         'Accept-Encoding': 'gzip, deflate',
    #         'Connection': 'keep-alive'
    #     }
        
    #     try:
    #         # Async download with retries
    #         timeout = aiohttp.ClientTimeout(total=60)
    #         async with aiohttp.ClientSession(timeout=timeout) as session:
    #             async with session.get(url, headers=headers) as response:
    #                 response.raise_for_status()
                    
    #                 # Stream large files
    #                 chunks = []
    #                 total_size = 0
    #                 max_size = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
                    
    #                 async for chunk in response.content.iter_chunked(1024 * 1024):
    #                     chunks.append(chunk)
    #                     total_size += len(chunk)
                        
    #                     if total_size > max_size:
    #                         logger.warning(f"Document exceeds {settings.MAX_DOCUMENT_SIZE_MB}MB limit")
    #                         break
                    
    #                 content = b''.join(chunks)
    #                 logger.info(f"Downloaded {total_size / 1024 / 1024:.1f}MB from {url}")
    #                 return content
                    
    #     except Exception as e:
    #         logger.error(f"Download failed: {e}")
    #         raise
    async def download_document(self, url: str) -> bytes:
        """Enhanced download with streaming for large files"""
        # OLD: Load entire file in memory
        # NEW: Stream download with chunked processing
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        try:
            timeout = aiohttp.ClientTimeout(total=120)  # Increased timeout
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Stream large files
                    chunks = []
                    total_size = 0
                    # max_size = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
                    max_size = (settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024) if settings.MAX_DOCUMENT_SIZE_MB else float('inf')
                    
                    # Process in 1MB chunks
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        chunks.append(chunk)
                        total_size += len(chunk)
                        
                        if total_size > max_size:
                            logger.warning(f"Document exceeds {settings.MAX_DOCUMENT_SIZE_MB}MB limit")
                            break
                    
                    content = b''.join(chunks)
                    logger.info(f"Downloaded {total_size / 1024 / 1024:.1f}MB from {url}")
                    return content
                    
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise

    
    async def get_or_create_vector_store(self, url: str) -> AdvancedVectorStore:
        """Get or create advanced vector store with hierarchical chunking"""
        
        # Generate cache key
        cache_key = f"adv_vecstore_{hashlib.md5(url.encode()).hexdigest()}"
        
        # Check cache
        cached_store = await cache.get(cache_key)
        if cached_store:
            logger.info(f"Using cached vector store for {url}")
            return cached_store
        
        logger.info(f"Creating new advanced vector store for {url}")
        content = await self.download_document(url)
    
    # Check document size
        is_large_doc = len(content) > 5 * 1024 * 1024  # > 5MB
        
        if is_large_doc:
            logger.info("Using two-pass strategy for large document")
            vector_store = await self._two_pass_processing(content, url)
        else:
            # Standard processing for smaller documents
            text, metadata = self.universal_parser.parse_any_document(content, url)
            vector_store = await self._standard_processing(text, metadata)
        
        await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS)
        return vector_store
    
    async def _two_pass_processing(self, content: bytes, url: str) -> AdvancedVectorStore:
        """Two-pass processing for large documents"""
        # NEW: Implement two-pass strategy
        
        # First pass: Quick scan for structure
        logger.info("First pass: Extracting document structure")
        
        # Extract headers, TOC, summary
        structure_info = self._extract_structure(content, url)
        
        # Identify important sections based on structure
        important_sections = self._identify_important_sections(structure_info)
        
        logger.info(f"Identified {len(important_sections)} important sections")
        
        # Second pass: Detailed extraction of important parts
        logger.info("Second pass: Detailed extraction")
        
        text_parts = []
        metadata_parts = []
        
        for section in important_sections:
            section_text, section_meta = self._extract_section(content, section, url)
            text_parts.append(section_text)
            metadata_parts.extend(section_meta)
        
        full_text = "\n\n".join(text_parts)
        
        # Create vector store with extracted content
        return await self._create_hierarchical_store(full_text, metadata_parts)

    def _extract_structure(self, content: bytes, url: str) -> Dict:
        """Extract document structure quickly"""
        # NEW: Fast structure extraction
        
        file_ext = self._get_extension(url)
        structure = {
            'headers': [],
            'sections': [],
            'tables': [],
            'total_pages': 0
        }
        
        if file_ext == '.pdf':
            try:
                with pdfplumber.open(io.BytesIO(content)) as pdf:
                    structure['total_pages'] = len(pdf.pages)
                    
                    # Sample first 10 pages for structure
                    for i, page in enumerate(pdf.pages[:10]):
                        text = page.extract_text() or ""
                        
                        # Extract headers (lines with specific patterns)
                        headers = re.findall(r'^[A-Z][A-Z\s]{2,}$', text, re.MULTILINE)
                        structure['headers'].extend([(h, i+1) for h in headers])
                        
                        # Check for tables
                        tables = page.extract_tables()
                        if tables:
                            structure['tables'].append((i+1, len(tables)))
            except:
                pass
        
        return structure

    def _identify_important_sections(self, structure: Dict) -> List[Dict]:
        """Identify important sections to process"""
        # NEW: Smart section identification
        
        important = []
        
        # Always include first few pages (usually summary/intro)
        # important.append({'type': 'intro', 'pages': range(1, min(6, structure['total_pages'] + 1))})
        important.append({'type': 'intro', 'pages': range(1, min(25, structure['total_pages'] + 1))}) # Increased from 6 to 25

        
        # Include pages with tables
        for page, count in structure['tables']:
            important.append({'type': 'table', 'pages': [page]})
        
        # Include sections with important headers
        important_keywords = ['summary', 'conclusion', 'total', 'result', 'key', 'important']
        for header, page in structure['headers']:
            if any(kw in header.lower() for kw in important_keywords):
                important.append({'type': 'header', 'pages': [page], 'header': header})
        
        return important
        
        # Download and parse document
        # content = await self.download_document(url)
        # text, metadata = self.universal_parser.parse_any_document(content, url)
        
        # if not text or len(text) < 10:
        #     logger.error(f"Document parsing failed or empty: {url}")
        #     text = "Document could not be parsed or is empty."
        #     metadata = [{'type': 'error'}]
        
        # # Hierarchical chunking
        # small_chunks, small_metadata = SmartChunker.chunk_document(
        #     text, metadata,
        #     chunk_size=settings.CHUNK_SIZE_CHARS,
        #     overlap=settings.CHUNK_OVERLAP_CHARS
        # )
        
        # large_chunks, large_metadata = SmartChunker.chunk_document(
        #     text, metadata,
        #     chunk_size=settings.LARGE_CHUNK_SIZE_CHARS,
        #     overlap=settings.LARGE_CHUNK_OVERLAP_CHARS
        # )
        
        # # Create chunk mapping
        # chunk_mapping = self._create_chunk_mapping(small_chunks, large_chunks, text)
        
        # logger.info(f"Created {len(small_chunks)} small and {len(large_chunks)} large chunks")
        
        # # Generate embeddings
        # small_embeddings = await self._generate_embeddings(small_chunks)
        # large_embeddings = await self._generate_embeddings(large_chunks)
        
        # # Create vector store
        # dimension = small_embeddings.shape[1]
        # vector_store = AdvancedVectorStore(
        #     self.embedding_model,
        #     self.reranker_model,
        #     dimension
        # )
        
        # # Add hierarchical data
        # vector_store.add_hierarchical(
        #     small_chunks, small_embeddings,
        #     large_chunks, large_embeddings,
        #     small_metadata, chunk_mapping
        # )
        
        # # Cache the store
        # await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS)
        
        # return vector_store
    
    def _create_chunk_mapping(self, small_chunks: List[str], large_chunks: List[str], 
                                full_text: str) -> Dict[int, int]:
        """Map small chunks to their corresponding large chunks"""
        mapping = {}
        
        for i, small_chunk in enumerate(small_chunks):
            # Find which large chunk contains this small chunk
            small_start = full_text.find(small_chunk[:50])  # Use first 50 chars to locate
            
            for j, large_chunk in enumerate(large_chunks):
                large_start = full_text.find(large_chunk[:50])
                large_end = large_start + len(large_chunk)
                
                if large_start <= small_start < large_end:
                    mapping[i] = j
                    break
        
        return mapping
    
    # async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
    #     """Generate embeddings in batches"""
    #     batch_size = 32
    #     all_embeddings = []
        
    #     for i in range(0, len(chunks), batch_size):
    #         batch = chunks[i:i + batch_size]
            
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
    # enhanced_rag_pipeline.py - Update _generate_embeddings method
    # async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
    #     """Generate embeddings in optimized batches"""
    #     # OLD: Fixed batch size of 32
    #     # NEW: Dynamic batch size with caching
        
    #     batch_size = 16  # Reduced from 32 for stability
    #     all_embeddings = []
        
    #     # Check cache first
    #     cache_key_prefix = "emb_"
    #     cached_embeddings = []
    #     chunks_to_embed = []
    #     chunks_indices = []
        
    #     for i, chunk in enumerate(chunks):
    #         chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
    #         cache_key = f"{cache_key_prefix}{chunk_hash}"
            
    #         cached = await cache.get(cache_key)
    #         if cached is not None:
    #             cached_embeddings.append((i, cached))
    #         else:
    #             chunks_to_embed.append(chunk)
    #             chunks_indices.append(i)
        
    #     # Generate embeddings for non-cached chunks
    #     if chunks_to_embed:
    #         for i in range(0, len(chunks_to_embed), batch_size):
    #             batch = chunks_to_embed[i:i + batch_size]
                
    #             embeddings = await asyncio.get_event_loop().run_in_executor(
    #                 None,
    #                 lambda: self.embedding_model.encode(
    #                     batch,
    #                     convert_to_numpy=True,
    #                     show_progress_bar=False,
    #                     normalize_embeddings=True  # Add normalization
    #                 )
    #             )
                
    #             # Cache the embeddings
    #             for j, emb in enumerate(embeddings):
    #                 chunk_idx = chunks_indices[i + j]
    #                 chunk_hash = hashlib.md5(chunks[chunk_idx].encode()).hexdigest()[:8]
    #                 cache_key = f"{cache_key_prefix}{chunk_hash}"
    #                 await cache.set(cache_key, emb, ttl=86400)  # Cache for 24 hours
                
    #             all_embeddings.append(embeddings)
        
    #     # Combine cached and new embeddings in correct order
    #     if cached_embeddings:
    #         # Merge cached and new embeddings
    #         result = np.zeros((len(chunks), self.embedding_model.get_sentence_embedding_dimension()))
            
    #         for idx, emb in cached_embeddings:
    #             result[idx] = emb
            
    #         if all_embeddings:
    #             new_embeddings = np.vstack(all_embeddings)
    #             for i, idx in enumerate(chunks_indices):
    #                 result[idx] = new_embeddings[i]
            
    #         return result.astype('float32')
    #     else:
    #         return np.vstack(all_embeddings).astype('float32') if all_embeddings else np.array([])
    # async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
    #     """Generate embeddings with parallel processing and caching"""
    #     # OLD: Sequential embedding generation
    #     # NEW: Parallel generation with comprehensive caching
        
    #     if not settings.PARALLEL_EMBEDDING_GENERATION:
    #         return await self._generate_embeddings_sequential(chunks)
        
    #     batch_size = 16
    #     cache_key_prefix = "emb_"
        
    #     # Prepare for parallel processing
    #     chunks_to_process = []
    #     cached_embeddings = {}
    #     chunk_indices = {}
        
    #     # Check cache in parallel
    #     cache_tasks = []
    #     for i, chunk in enumerate(chunks):
    #         chunk_hash = hashlib.md5(chunk.encode()).hexdigest()[:8]
    #         cache_key = f"{cache_key_prefix}{chunk_hash}"
    #         cache_tasks.append((i, cache_key, cache.get(cache_key)))
        
    #     # Gather cache results
    #     cache_results = await asyncio.gather(*[task[2] for task in cache_tasks])
        
    #     for (i, cache_key, _), result in zip(cache_tasks, cache_results):
    #         if result is not None:
    #             cached_embeddings[i] = result
    #         else:
    #             chunks_to_process.append((i, chunks[i], cache_key))
        
    #     # Process uncached chunks in parallel batches
    #     if chunks_to_process:
    #         # Split into batches
    #         batches = []
    #         for i in range(0, len(chunks_to_process), batch_size):
    #             batch = chunks_to_process[i:i + batch_size]
    #             batches.append(batch)
            
    #         # Process batches in parallel
    #         async def process_batch(batch):
    #             chunk_texts = [item[1] for item in batch]
                
    #             # Generate embeddings for batch
    #             embeddings = await asyncio.get_event_loop().run_in_executor(
    #                 None,
    #                 lambda: self.embedding_model.encode(
    #                     chunk_texts,
    #                     convert_to_numpy=True,
    #                     show_progress_bar=False,
    #                     normalize_embeddings=True
    #                 )
    #             )
                
    #             # Cache embeddings
    #             cache_tasks = []
    #             for (idx, _, cache_key), emb in zip(batch, embeddings):
    #                 cache_tasks.append(cache.set(cache_key, emb, ttl=settings.EMBEDDING_CACHE_TTL))
                
    #             await asyncio.gather(*cache_tasks, return_exceptions=True)
                
    #             return [(item[0], emb) for item, emb in zip(batch, embeddings)]
            
    #         # Process all batches in parallel
    #         batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
            
    #         # Flatten results
    #         for batch_result in batch_results:
    #             for idx, emb in batch_result:
    #                 cached_embeddings[idx] = emb
        
    #     # Combine in correct order
    #     dimension = self.embedding_model.get_sentence_embedding_dimension()
    #     result = np.zeros((len(chunks), dimension), dtype='float32')
        
    #     for idx, emb in cached_embeddings.items():
    #         result[idx] = emb
        
    #     return result
    # In EnhancedRAGPipeline class in enhanced_rag_pipeline.py

# --- REPLACE THIS ENTIRE METHOD ---
    # async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
    #     """
    #     Generate embeddings with a robust, simplified caching strategy.
    #     This version correctly handles batching and async operations.
    #     """
    #     all_embeddings = {}
    #     chunks_to_process = {}  # Maps original index to chunk text

    #     # Step 1: Check cache for existing embeddings
    #     cache_keys = {i: f"emb_{hashlib.md5(chunk.encode()).hexdigest()}" for i, chunk in enumerate(chunks)}
    #     cached_results = await cache.get_many(list(cache_keys.values()))

    #     for i, chunk in enumerate(chunks):
    #         key = cache_keys[i]
    #         if key in cached_results:
    #             all_embeddings[i] = cached_results[key]
    #         else:
    #             chunks_to_process[i] = chunk

    #     # Step 2: Process only the chunks that were not in the cache
    #     if chunks_to_process:
    #         logger.info(f"Generating embeddings for {len(chunks_to_process)} new chunks.")
            
    #         indices = list(chunks_to_process.keys())
    #         texts = list(chunks_to_process.values())

    #         # Run the CPU-bound encoding in a separate thread to avoid blocking the event loop
    #         new_embeddings = await asyncio.to_thread(
    #             self.embedding_model.encode,
    #             texts,
    #             batch_size=settings.EMBEDDING_BATCH_SIZE,
    #             show_progress_bar=False,
    #             convert_to_numpy=True
    #         )

    #         # Step 3: Update the cache with the newly generated embeddings
    #         items_to_cache = {}
    #         for i, embedding in enumerate(new_embeddings):
    #             original_index = indices[i]
    #             all_embeddings[original_index] = embedding
    #             items_to_cache[cache_keys[original_index]] = embedding
            
    #         if items_to_cache:
    #             await cache.set_many(items_to_cache, ttl=settings.EMBEDDING_CACHE_TTL)
                
    #     # Step 4: Assemble the final numpy array in the correct order
    #     final_embeddings = np.zeros((len(chunks), self.embedding_model.get_sentence_embedding_dimension()))
    #     for i, embedding in all_embeddings.items():
    #         final_embeddings[i] = embedding
            
    #     return final_embeddings.astype('float32')
    async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings with a robust caching strategy.
        """
        all_embeddings = np.zeros((len(chunks), self.embedding_model.get_sentence_embedding_dimension()), dtype=np.float32)
        chunks_to_process = {}  # Maps index -> chunk

        # Step 1: Check cache for existing embeddings
        cache_keys = {i: f"emb_{hashlib.md5(chunk.encode()).hexdigest()}" for i, chunk in enumerate(chunks)}
        # Use the corrected get_many from your cache.py
        cached_results = await cache.get_many(list(cache_keys.values()))

        for i, chunk in enumerate(chunks):
            key = cache_keys[i]
            if key in cached_results and cached_results[key] is not None:
                all_embeddings[i] = cached_results[key]
            else:
                chunks_to_process[i] = chunk

        # Step 2: Process only the chunks that were not found in the cache
        if chunks_to_process:
            logger.info(f"Generating embeddings for {len(chunks_to_process)} new chunks.")
            indices = list(chunks_to_process.keys())
            texts_to_embed = list(chunks_to_process.values())

            # Run the CPU-bound encoding in a separate thread
            new_embeddings = await asyncio.to_thread(
                self.embedding_model.encode,
                texts_to_embed,
                batch_size=settings.EMBEDDING_BATCH_SIZE,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Step 3: Update the main embeddings array and cache the new results
            items_to_cache = {}
            for i, embedding in enumerate(new_embeddings):
                original_index = indices[i]
                all_embeddings[original_index] = embedding
                items_to_cache[cache_keys[original_index]] = embedding
            
            if items_to_cache:
                await cache.set_many(items_to_cache, ttl=settings.EMBEDDING_CACHE_TTL)
                
        return all_embeddings
    
    async def answer_question(self, question: str, vector_store: AdvancedVectorStore) -> Dict[str, Any]:
        """Generate answer with multi-step reasoning and validation"""
        
        # 1. Analyze question type
        question_info = self.question_analyzer.analyze(question)
        min_chunks = 3
        max_chunks = settings.MAX_CHUNKS_PER_QUERY
        
        if question_info['type'] == 'yes_no':
            max_chunks = 5  # Yes/no needs less context
        elif question_info['type'] == 'numerical':
            max_chunks = 8  # Numbers need moderate context
        elif question_info['type'] == 'list':
            max_chunks = 15  # Lists need more context
        
        # Progressive retrieval
        all_chunks = []
        all_scores = []
        confidence_threshold = 0.85
        
        for k in [min_chunks, max_chunks // 2, max_chunks]:
            search_results = vector_store.search_with_reranking(question, k=k)
            
            new_chunks = []
            for chunk, score, meta in search_results:
                if chunk not in all_chunks:  # Avoid duplicates
                    all_chunks.append(chunk)
                    all_scores.append(score)
                    new_chunks.append(chunk)
            
            # Check if we have enough context
            if len(all_chunks) >= min_chunks:
                context_quality = self._assess_context_quality(
                    question, all_chunks, all_scores, question_info
                )
                
                if context_quality > confidence_threshold:
                    logger.info(f"Early termination with {len(all_chunks)} chunks (quality: {context_quality:.2f})")
                    break
        
        # 2. Query expansion
        # expanded_queries = self._expand_query(question, question_info)
        
        # 3. Retrieve relevant chunks for all queries
        # all_chunks = []
        # all_scores = []
        # confidence_threshold = 0.8
        
        # for query in expanded_queries:
        #     search_results = vector_store.search_with_reranking(
        #         query,
        #         k=settings.MAX_CHUNKS_PER_QUERY
        #     )
            
        #     for chunk, score, meta in search_results:
        #         all_chunks.append(chunk)
        #         all_scores.append(score)
        #         if len(all_chunks) >= 5 and np.mean(all_scores[-5:]) > confidence_threshold:
        #             logger.info(f"Early termination with {len(all_chunks)} chunks")
        #             break
        
        # 4. Deduplicate and rank chunks
        unique_chunks = self._deduplicate_chunks(all_chunks, all_scores)
        return await self._generate_answer_with_validation(
        question, unique_chunks, question_info, all_scores
    )
    # In EnhancedRAGPipeline class in app/core/enhanced_rag_pipeline.py

# --- NEW ---
    async def _generate_answer_with_validation(
        self, question: str, chunks: List[str], question_info: Dict, scores: List[float]
    ) -> Dict[str, Any]:
        """
        Generates an answer, validates it, and calculates confidence in a single step.
        This replaces the separate calls to generate, validate, and calculate confidence.
        """
        # Step 1: Generate the initial answer
        if question_info['requires_multi_step']:
            answer_text = await self._multi_step_reasoning(question, chunks, question_info)
        else:
            answer_text = await self._generate_answer(question, chunks, question_info)

        # Step 2: Validate the generated answer
        validated_answer = self.answer_validator.validate(
            question, answer_text, question_info, chunks
        )

        # Step 3: Calculate a final confidence score
        confidence = self._calculate_confidence(validated_answer, scores)

        # Step 4: Assemble and return the final, detailed response object
        return {
            'answer': validated_answer.get('answer', "No answer could be generated."),
            'confidence': confidence,
            'question_type': question_info.get('type'),
            'sources': validated_answer.get('sources', []),
            'validation_notes': validated_answer.get('notes', '')
        }
        
        # # 5. Multi-step reasoning if needed
        # if question_info['requires_multi_step']:
        #     answer = await self._multi_step_reasoning(question, unique_chunks, question_info)
        # else:
        #     answer = await self._generate_answer(question, unique_chunks, question_info)
        
        # # 6. Validate answer
        # validated_answer = self.answer_validator.validate(
        #     question, answer, question_info, unique_chunks
        # )
        
        # # 7. Calculate confidence
        # confidence = self._calculate_confidence(validated_answer, all_scores)
        
        # return {
        #     'answer': validated_answer['answer'],
        #     'confidence': confidence,
        #     'question_type': question_info['type'],
        #     'sources': validated_answer.get('sources', []),
        #     'validation_notes': validated_answer.get('notes', '')
        # }
    def _assess_context_quality(self, question: str, chunks: List[str], 
                           scores: List[float], question_info: Dict) -> float:
        """Assess if we have enough context to answer"""
        # NEW: Context quality assessment
        
        quality = 0.0
        
        # Factor 1: Retrieval scores
        if scores:
            avg_score = np.mean(scores)
            quality += min(avg_score, 0.3)
        
        # Factor 2: Keyword coverage
        question_keywords = set(question.lower().split())
        context_text = ' '.join(chunks).lower()
        keyword_coverage = sum(1 for kw in question_keywords if kw in context_text) / len(question_keywords)
        quality += keyword_coverage * 0.3
        
        # Factor 3: Type-specific checks
        if question_info['type'] == 'numerical':
            # Check if numbers are present
            numbers = re.findall(r'\d+', context_text)
            if numbers:
                quality += 0.2
        
        elif question_info['type'] == 'yes_no':
            # Check for definitive statements
            definitive_words = ['yes', 'no', 'true', 'false', 'correct', 'incorrect']
            if any(word in context_text for word in definitive_words):
                quality += 0.2
        
        # Factor 4: Entity coverage
        entities = question_info.get('entities', [])
        if entities:
            entity_coverage = sum(1 for e in entities if e.lower() in context_text) / len(entities)
            quality += entity_coverage * 0.2
        
        return min(quality, 1.0)
    
    # def retrieve(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
    #     """Multi-strategy retrieval with better combination"""
    #     # OLD: Simple score addition
    #     # NEW: Weighted combination with normalization
        
    #     query_lower = query.lower()
    #     scores = np.zeros(len(self.chunks))
        
    #     # 1. Exact phrase matching (highest weight)
    #     for idx, chunk in enumerate(self.chunks):
    #         if query_lower in chunk.lower():
    #             scores[idx] += 10.0
                
    #             # Bonus for exact case match
    #             if query in chunk:
    #                 scores[idx] += 2.0
        
    #     # 2. BM25 scoring (moderate weight)
    #     if hasattr(self, 'bm25'):
    #         query_tokens = self._tokenize(query_lower)
    #         if query_tokens:
    #             bm25_scores = self.bm25.get_scores(query_tokens)
    #             # Normalize BM25 scores
    #             if max(bm25_scores) > 0:
    #                 bm25_scores = bm25_scores / max(bm25_scores)
    #             scores += bm25_scores * 5.0
        
    #     # 3. TF-IDF scoring (lower weight)
    #     if self.tfidf_vectorizer and self.tfidf_matrix is not None:
    #         try:
    #             query_vec = self.tfidf_vectorizer.transform([query])
    #             tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
    #             # Normalize TF-IDF scores
    #             if max(tfidf_scores) > 0:
    #                 tfidf_scores = tfidf_scores / max(tfidf_scores)
    #             scores += tfidf_scores * 3.0
    #         except:
    #             pass
        
    #     # 4. Keyword and entity matching
    #     entities = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
    #     for entity in entities:
    #         entity_lower = entity.lower()
    #         for idx, chunk in enumerate(self.chunks):
    #             if entity_lower in chunk.lower():
    #                 scores[idx] += 4.0
        
    #     # Get top-k with score threshold
    #     score_threshold = 0.1
    #     top_indices = np.argsort(scores)[-k:][::-1]
    #     results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > score_threshold]
        
    #     # Ensure minimum results
    #     if len(results) < 3 and len(self.chunks) > 0:
    #         results = [(i, 1.0) for i in range(min(3, len(self.chunks)))]
        
    #     return results
    
    def _expand_query(self, question: str, question_info: Dict) -> List[str]:
        """Expand query for better retrieval"""
        queries = [question]
        
        if not settings.ENABLE_QUERY_EXPANSION:
            return queries
        
        # Add variations based on question type
        if question_info['type'] == 'numerical':
            # Add keywords for numbers
            queries.append(re.sub(r'[^\w\s]', ' ', question) + ' number amount value')
            
        elif question_info['type'] == 'list':
            # Add enumeration keywords
            queries.append(question + ' list items all types')
            
        elif question_info['type'] == 'comparison':
            # Extract entities and search separately
            entities = question_info.get('entities', [])
            for entity in entities:
                queries.append(f"{entity} characteristics features")
        
        elif question_info['type'] == 'definition':
            # Add definition keywords
            main_term = question_info.get('main_term', '')
            if main_term:
                queries.append(f"what is {main_term} definition meaning")
        
        return queries[:3]  # Limit to avoid too many queries
    
    # Complete the _deduplicate_chunks method that was cut off
    def _deduplicate_chunks(self, chunks: List[str], scores: List[float]) -> List[str]:
        """Deduplicate similar chunks while preserving best scores"""
        from rapidfuzz import fuzz  # Add this import
        
        unique_chunks = []
        seen_hashes = set()
        
        # Sort by score
        sorted_pairs = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        
        for chunk, score in sorted_pairs:
            # Use fuzzy matching to detect near-duplicates
            is_duplicate = False
            for seen_chunk in unique_chunks:
                similarity = fuzz.ratio(chunk[:200], seen_chunk[:200])
                if similarity > 85:  # 85% similarity threshold
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_chunks.append(chunk)
                chunk_hash = hashlib.md5(chunk[:100].encode()).hexdigest()
                seen_hashes.add(chunk_hash)
            
            if len(unique_chunks) >= settings.MAX_CHUNKS_PER_QUERY:
                break
        
        return unique_chunks
        
    async def _multi_step_reasoning(self, question: str, chunks: List[str], 
                                    question_info: Dict) -> str:
        """Multi-step reasoning for complex questions"""
        
        # Break down into sub-questions
        sub_questions = self._generate_sub_questions(question, question_info)
        
        # Answer each sub-question
        sub_answers = []
        for sub_q in sub_questions:
            sub_answer = await self._generate_answer(sub_q, chunks, {'type': 'simple'})
            sub_answers.append(f"Q: {sub_q}\nA: {sub_answer}")
        
        # Combine sub-answers for final answer
        context = "\n\n".join(chunks[:10])
        sub_answers_text = "\n\n".join(sub_answers)
        
        prompt = f"""Based on the following context and intermediate answers, provide a comprehensive answer to the main question.

    CONTEXT:
    {context}

    INTERMEDIATE ANSWERS:
    {sub_answers_text}

    MAIN QUESTION: {question}

    Synthesize the information to provide a complete, accurate answer:"""
        
        try:
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME_PRECISE)
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=1000,
                    top_p=0.95
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Multi-step reasoning failed: {e}")
            return await self._generate_answer(question, chunks, question_info)
    
    def _generate_sub_questions(self, question: str, question_info: Dict) -> List[str]:
        """Generate sub-questions for complex queries"""
        sub_questions = []
        
        if question_info['type'] == 'comparison':
            entities = question_info.get('entities', [])
            for entity in entities:
                sub_questions.append(f"What are the key features of {entity}?")
            sub_questions.append("What are the main differences between them?")
            
        elif question_info['type'] == 'multi_part':
            # Extract individual parts
            parts = re.split(r'[,;]|and', question)
            for part in parts:
                if len(part.strip()) > 10:
                    sub_questions.append(part.strip() + "?")
        
        elif 'calculate' in question.lower() or 'total' in question.lower():
            sub_questions.append("What are the individual values mentioned?")
            sub_questions.append("What operation should be performed?")
        
        # Default sub-questions if none generated
        if not sub_questions:
            sub_questions = [
                f"What is the main topic of '{question}'?",
                f"What specific information is requested?",
                f"What details support the answer?"
            ]
        
        return sub_questions[:4]  # Limit sub-questions
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def _generate_answer(self, question: str, chunks: List[str], 
                                question_info: Dict) -> str:
        """Generate answer using appropriate prompt based on question type"""
        
        # Select prompt template based on question type
        prompt = self._create_prompt(question, chunks, question_info)
        
        # Select model based on complexity
        if question_info.get('requires_precision', False):
            model_name = settings.LLM_MODEL_NAME_PRECISE
            max_tokens = 1000
        else:
            model_name = settings.LLM_MODEL_NAME
            max_tokens = 600
        
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
            
            if not answer or len(answer) < 10:
                return "Unable to generate a valid answer."
            
            return answer
            
        except asyncio.TimeoutError:
            logger.error(f"Answer generation timeout for question: {question[:50]}...")
            return "Processing timeout. Please try again."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "An error occurred while generating the answer."
    
    def _create_prompt(self, question: str, chunks: List[str], 
                        question_info: Dict) -> str:
        """Create specialized prompt based on question type"""
        
        context = "\n\n---\n\n".join(chunks)
        
        # Type-specific prompts
        if question_info['type'] == 'numerical':
            return f"""Extract and calculate the numerical answer from the context.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Identify all relevant numbers
    2. Show the calculation step by step
    3. Provide the final numerical answer
    4. Include units if applicable

    ANSWER:"""
        
        elif question_info['type'] == 'list':
            return f"""List all items requested in the question based on the context.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Extract ALL relevant items
    2. Present as a numbered or bulleted list
    3. Ensure completeness
    4. Include brief descriptions if available

    ANSWER:"""
        
        elif question_info['type'] == 'yes_no':
            return f"""Answer with Yes or No based on the context, then provide explanation.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Start with clear Yes or No
    2. Provide supporting evidence from context
    3. Be definitive in your answer

    ANSWER:"""
        
        elif question_info['type'] == 'comparison':
            return f"""Compare the items mentioned in the question based on the context.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Identify items being compared
    2. List key characteristics of each
    3. Highlight differences and similarities
    4. Provide a clear comparison summary

    ANSWER:"""
        
        else:
            # Default comprehensive prompt
            return f"""Answer the question accurately based on the context provided.

    CONTEXT:
    {context}

    QUESTION: {question}

    INSTRUCTIONS:
    1. Be specific and accurate
    2. Include relevant details from the context
    3. If information is incomplete, state what is missing
    4. Structure your answer clearly

    ANSWER:"""
    
    def _calculate_confidence(self, validated_answer: Dict, scores: List[float]) -> float:
        """Calculate confidence score for the answer"""
        
        base_confidence = 0.5
        
        # Factor 1: Retrieval scores
        if scores:
            avg_score = np.mean(scores[:10])  # Top 10 scores
            base_confidence += avg_score * 0.2
        
        # Factor 2: Validation results
        if validated_answer.get('validation_passed', False):
            base_confidence += 0.2
        
        # Factor 3: Answer completeness
        answer = validated_answer.get('answer', '')
        if len(answer) > 100:
            base_confidence += 0.1
        
        # Factor 4: Question type confidence
        question_type = validated_answer.get('question_type', 'unknown')
        type_confidence = {
            'simple': 0.1,
            'numerical': 0.15,
            'yes_no': 0.15,
            'list': 0.05,
            'comparison': 0.0,
            'complex': -0.05
        }
        base_confidence += type_confidence.get(question_type, 0)
        
        return min(max(base_confidence, 0.0), 1.0)
    
    # async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
    #     """Process multiple questions with parallel execution"""
        
    #     start_time = time.time()
    #     logger.info(f"Processing {len(questions)} questions for {document_url}")
        
    #     try:
    #         # Get vector store
    #         vector_store = await self.get_or_create_vector_store(document_url)
            
    #         # Process questions in parallel
    #         semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)
            
    #         async def process_question(q):
    #             async with semaphore:
    #                 try:
    #                     result = await self.answer_question(q, vector_store)
    #                     return result['answer']
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
    # enhanced_rag_pipeline.py - Add question-aware processing
    # async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
    #     """Process with question awareness"""
    #     # OLD: Process document fully before answering
    #     # NEW: Analyze questions first, then process relevant parts
        
    #     start_time = time.time()
    #     logger.info(f"Processing {len(questions)} questions for {document_url}")
        
    #     try:
    #         # Analyze all questions first
    #         question_analysis = self._analyze_all_questions(questions)
            
    #         # Get or create vector store with question context
    #         vector_store = await self._get_or_create_question_aware_store(
    #             document_url, question_analysis
    #         )
            
    #         # Process questions with optimized retrieval
    #         answers = await self._process_questions_optimized(
    #             questions, vector_store, question_analysis
    #         )
            
    #         elapsed = time.time() - start_time
    #         logger.info(f"Processed {len(questions)} questions in {elapsed:.2f}s")
            
    #         return answers
            
    #     except Exception as e:
    #         logger.error(f"Critical error: {e}", exc_info=True)
    #         return ["Document processing error."] * len(questions)
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process multiple questions with optimized parallel execution and caching"""
        # OLD: Simple parallel processing with fixed concurrency
        # NEW: Smart batching with comprehensive caching
        
        start_time = time.time()
        logger.info(f"Processing {len(questions)} questions for {document_url}")
        
        try:
            # Check cache for previously answered questions
            cached_answers = await self._get_cached_answers(document_url, questions)
            questions_to_process = []
            question_indices = []
            
            for i, q in enumerate(questions):
                if cached_answers[i] is None:
                    questions_to_process.append(q)
                    question_indices.append(i)
            
            logger.info(f"Found {len(questions) - len(questions_to_process)} cached answers")
            
            # Process uncached questions
            if questions_to_process:
                vector_store = await self.get_or_create_vector_store(document_url)
                
                # Process in optimized batches
                batch_size = settings.QUESTION_BATCH_SIZE or 3
                new_answers = []
                
                for i in range(0, len(questions_to_process), batch_size):
                    batch = questions_to_process[i:i + batch_size]
                    batch_answers = await self._process_question_batch(batch, vector_store)
                    new_answers.extend(batch_answers)
                    
                    # Cache the answers immediately
                    for q, a in zip(batch, batch_answers):
                        await self._cache_answer(document_url, q, a)
                
                # Merge cached and new answers
                final_answers = list(cached_answers)
                for idx, answer in zip(question_indices, new_answers):
                    final_answers[idx] = answer
            else:
                final_answers = cached_answers
            
            elapsed = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions in {elapsed:.2f}s")
            
            return final_answers
            
        except asyncio.TimeoutError:
            logger.error("Overall processing timeout")
            return ["Processing timeout. Please try again."] * len(questions)
        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            return ["Document processing error."] * len(questions)
    async def _answer_with_timeout(self, question: str, vector_store: AdvancedVectorStore) -> str:
        """Answer question with individual timeout and caching"""
        # NEW: Individual timeout per question with cache check
        
        # Check if this specific question-document pair is cached
        cache_key = self._get_question_cache_key(vector_store.document_hash, question)
        cached_result = await cache.get(cache_key)
        
        if cached_result:
            logger.info(f"Using cached result for question: {question[:50]}...")
            return cached_result
        
        try:
            result = await asyncio.wait_for(
                self.answer_question(question, vector_store),
                timeout=settings.ANSWER_TIMEOUT_SECONDS
            )
            
            answer = result['answer']
            
            # Cache the result
            await cache.set(cache_key, answer, ttl=settings.ANSWER_CACHE_TTL or 7200)
            
            return answer
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout for question: {question[:50]}...")
            return "Processing timeout. Question too complex."
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Error generating answer."    
        
    async def _process_question_batch(self, questions: List[str], vector_store: AdvancedVectorStore) -> List[str]:
        """Process a batch of questions in parallel with optimal concurrency"""
        # NEW: Optimized batch processing with semaphore control
        
        # Use semaphore to control concurrency
        semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS or 1)
        
        async def process_single(q):
            async with semaphore:
                return await self._answer_with_timeout(q, vector_store)
        
        # Create tasks for all questions in batch
        tasks = [asyncio.create_task(process_single(q)) for q in questions]
        
        # Wait for all with individual timeout handling
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {i}: {result}")
                answers.append("Error processing this question.")
            else:
                answers.append(result)
        
        return answers    
    async def _get_cached_answers(self, doc_url: str, questions: List[str]) -> List[Optional[str]]:
        """Get cached answers for questions with batch retrieval"""
        # NEW: Efficient batch cache retrieval
        
        doc_hash = hashlib.md5(doc_url.encode()).hexdigest()
        cached = []
        
        # Batch get from cache if supported
        cache_keys = [self._get_question_cache_key(doc_hash, q) for q in questions]
        
        # Try batch get
        try:
            # If cache supports batch get
            if hasattr(cache, 'get_many'):
                cached_dict = await cache.get_many(cache_keys)
                cached = [cached_dict.get(key) for key in cache_keys]
            else:
                # Fall back to individual gets
                cached = await asyncio.gather(*[cache.get(key) for key in cache_keys])
        except Exception as e:
            logger.warning(f"Cache retrieval error: {e}")
            cached = [None] * len(questions)
        
        return cached
    async def _cache_answer(self, doc_url: str, question: str, answer: str):
        """Cache an answer with error handling"""
        # NEW: Robust answer caching
        
        if not settings.USE_ANSWER_CACHE:
            return
        
        try:
            doc_hash = hashlib.md5(doc_url.encode()).hexdigest()
            cache_key = self._get_question_cache_key(doc_hash, question)
            
            # Don't cache error responses
            if not self._is_error_answer(answer):
                await cache.set(cache_key, answer, ttl=settings.ANSWER_CACHE_TTL or 7200)
                logger.debug(f"Cached answer for: {question[:30]}...")
        except Exception as e:
            logger.warning(f"Failed to cache answer: {e}") 

    def _get_question_cache_key(self, doc_hash: str, question: str) -> str:
        """Generate cache key for question-document pair"""
        # NEW: Consistent cache key generation
        
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        return f"qa_{doc_hash[:8]}_{question_hash}" 

    def _is_error_answer(self, answer: str) -> bool:
        """Check if answer is an error response"""
        # NEW: Don't cache error responses
        
        error_indicators = [
            "error", "timeout", "failed", "unable to",
            "processing error", "document processing error"
        ]
        
        answer_lower = answer.lower()
        return any(indicator in answer_lower for indicator in error_indicators) or len(answer) < 10  
         
    def _analyze_all_questions(self, questions: List[str]) -> Dict:
        """Analyze all questions to understand what to look for"""
        # NEW: Comprehensive question analysis
        
        analysis = {
            'types': [],
            'keywords': set(),
            'entities': set(),
            'needs_tables': False,
            'needs_numbers': False,
            'sections_needed': set()
        }
        
        for question in questions:
            q_info = self.question_analyzer.analyze(question)
            analysis['types'].append(q_info['type'])
            
            # Extract keywords
            keywords = re.findall(r'\b[A-Za-z]{4,}\b', question.lower())
            analysis['keywords'].update(keywords)
            
            # Extract entities
            analysis['entities'].update(q_info.get('entities', []))
            
            # Check specific needs
            if q_info['type'] == 'numerical':
                analysis['needs_numbers'] = True
            if 'table' in question.lower() or 'list' in question.lower():
                analysis['needs_tables'] = True
            
            # Identify sections
            section_keywords = ['introduction', 'summary', 'conclusion', 'method', 'result']
            for kw in section_keywords:
                if kw in question.lower():
                    analysis['sections_needed'].add(kw)
        
        return analysis

    async def _get_or_create_question_aware_store(self, url: str, question_analysis: Dict) -> AdvancedVectorStore:
        """Create vector store focused on question needs"""
        # NEW: Question-aware document processing
        
        # Create unique cache key including question context
        context_hash = hashlib.md5(
            str(sorted(question_analysis['keywords'])).encode()
        ).hexdigest()[:8]
        cache_key = f"qa_vecstore_{hashlib.md5(url.encode()).hexdigest()}_{context_hash}"
        
        cached_store = await cache.get(cache_key)
        if cached_store:
            logger.info(f"Using cached question-aware vector store")
            return cached_store
        
        logger.info(f"Creating question-aware vector store")
        
        content = await self.download_document(url)
        text, metadata = self.universal_parser.parse_any_document(content, url)
        
        # Filter and prioritize chunks based on question analysis
        prioritized_chunks = self._prioritize_chunks_for_questions(
            text, metadata, question_analysis
        )
        
        # Create vector store with prioritized chunks
        vector_store = await self._create_prioritized_store(prioritized_chunks)
        
        await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS // 2)
        return vector_store

    def _prioritize_chunks_for_questions(self, text: str, metadata: List[Dict], 
                                        question_analysis: Dict) -> List[Tuple[str, Dict, float]]:
        """Prioritize chunks based on question needs"""
        # NEW: Smart chunk prioritization
        
        chunks, chunk_meta = SmartChunker.chunk_document(
            text, metadata,
            chunk_size=settings.CHUNK_SIZE_CHARS,
            overlap=settings.CHUNK_OVERLAP_CHARS
        )
        
        prioritized = []
        
        for chunk, meta in zip(chunks, chunk_meta):
            score = 0.0
            
            # Score based on keyword matches
            chunk_lower = chunk.lower()
            for keyword in question_analysis['keywords']:
                if keyword in chunk_lower:
                    score += 1.0
            
            # Boost tables if needed
            if question_analysis['needs_tables'] and meta.get('type') == 'table':
                score += 5.0
            
            # Boost numerical content if needed
            if question_analysis['needs_numbers']:
                numbers = re.findall(r'\d+', chunk)
                score += len(numbers) * 0.5
            
            # Boost specific sections
            for section in question_analysis['sections_needed']:
                if section in chunk_lower:
                    score += 3.0
            
            prioritized.append((chunk, meta, score))
        
        # Sort by score and keep top chunks
        prioritized.sort(key=lambda x: x[2], reverse=True)
        max_chunks = min(len(prioritized), settings.MAX_TOTAL_CHUNKS)
        
        logger.info(f"Keeping top {max_chunks} chunks out of {len(prioritized)}")
        return prioritized[:max_chunks]