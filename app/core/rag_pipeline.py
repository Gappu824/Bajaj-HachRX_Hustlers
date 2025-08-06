# app/core/rag_pipeline.py - Optimized version
import io
import re
import os
import logging
import requests
import asyncio
import aiohttp
import time
import json
import hashlib
import msgpack
import lz4.frame
from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict

# Document parsing imports
import pdfplumber
from docx import Document
from odf.text import P
from odf.opendocument import load
import PyPDF2
import pypdf
import pandas as pd
try:
    from pptx import Presentation
    PPTX_AVAILABLE = True
except ImportError:
    PPTX_AVAILABLE = False
    logging.warning("python-pptx not available, PowerPoint parsing disabled")

# ML and AI imports
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import faiss
import numpy as np
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions

# Local imports
from app.core.config import settings
from app.core.cache import cache
from app.core.enhanced_retrieval import EnhancedRetriever
from app.core.contextual_embeddings import ContextualRetrieval
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


class SmartChunker:
    """Intelligent chunking that preserves semantic boundaries"""
    
    @staticmethod
    def chunk_with_boundaries(text: str, metadata: List[Dict], 
                            chunk_size: int = 800,
                            overlap: int = 100) -> Tuple[List[str], List[Dict]]:
        """Create chunks with adaptive sizing based on document length"""
        
        # Adaptive chunk size based on document length
        doc_length = len(text)
        if doc_length < 5000:
            chunk_size = 500
            overlap = 50
        elif doc_length < 20000:
            chunk_size = 800
            overlap = 100
        else:
            chunk_size = 1200
            overlap = 150
        
        chunks = []
        chunk_metadata = []
        
        # Fast path for very small documents
        if doc_length < 1000:
            chunks.append(text)
            chunk_metadata.append({'type': 'full_document'})
            return chunks, chunk_metadata
        
        # Split by major sections
        sections = re.split(r'\n(?=(?:CHAPTER|SECTION|ARTICLE|PART)\s+[IVXLCDM\d]+)', text)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Split by paragraphs
            paragraphs = re.split(r'\n\n+', section)
            current_chunk = ""
            current_meta = {}
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if adding this paragraph exceeds chunk size
                if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    chunk_metadata.append(current_meta)
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap:
                        overlap_text = current_chunk[-overlap:]
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                    current_meta = SmartChunker._extract_metadata(para)
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
                        current_meta = SmartChunker._extract_metadata(para)
            
            # Add final chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                chunk_metadata.append(current_meta)
        
        return chunks, chunk_metadata
    
    @staticmethod
    def _extract_metadata(text: str) -> Dict:
        """Extract metadata from text"""
        metadata = {'type': 'text'}
        
        # Check for headers
        if re.match(r'^(?:CHAPTER|SECTION|ARTICLE|PART)\s+[IVXLCDM\d]+', text):
            metadata['type'] = 'header'
            metadata['level'] = 1
        elif re.match(r'^\d+\.\s+[A-Z]', text):
            metadata['type'] = 'section'
            metadata['level'] = 2
        
        # Check for lists
        if re.match(r'^(?:\d+\.|[a-z]\)|•|-|\*)\s+', text):
            metadata['type'] = 'list_item'
        
        # Check for definitions
        if 'means' in text or 'defined as' in text or 'refers to' in text:
            metadata['type'] = 'definition'
        
        return metadata


class OptimizedVectorStore:
    """Enhanced vector store with hybrid retrieval"""
    
    def __init__(self, chunks: List[str], embeddings: np.ndarray, 
                 model: SentenceTransformer, chunk_metadata: Optional[List[Dict]] = None):
        self.chunks = chunks
        self.model = model
        self.chunk_metadata = chunk_metadata or [{}] * len(chunks)
        self.embeddings = embeddings
        dimension = embeddings.shape[1]
        
        # Use simple flat index for speed (good for < 100k chunks)
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Initialize enhanced retriever only if we have multiple chunks
        if len(chunks) > 1:
            self.enhanced_retriever = EnhancedRetriever(chunks, chunk_metadata)
        else:
            self.enhanced_retriever = None
        
        # Build question patterns
        self.question_patterns = self._build_question_patterns()
        
        logger.info(f"Built optimized vector store with {len(chunks)} chunks")
    
    def _build_question_patterns(self) -> Dict[str, Dict]:
        """Build patterns for question classification"""
        patterns = {
            'definitional': {
                'patterns': [
                    r'what is', r'define', r'meaning of', r'explain the term',
                    r'definition of', r'what does .* mean'
                ],
                'boost_terms': ['is defined as', 'refers to', 'means', 'definition', 'is called']
            },
            'computational': {
                'patterns': [
                    r'calculate', r'how much', r'what is the total', r'compute',
                    r'what.*cost', r'what.*price', r'what.*amount'
                ],
                'boost_terms': ['formula', 'calculation', 'total', 'sum', '%', 'percentage', 'equals']
            },
            'procedural': {
                'patterns': [
                    r'how to', r'what is the process', r'steps to', r'explain the procedure',
                    r'what.*procedure', r'how do'
                ],
                'boost_terms': ['process', 'procedure', 'steps', 'first', 'then', 'finally', 'instructions']
            },
            'list_based': {
                'patterns': [
                    r'list all', r'what are all', r'mention the', r'enumerate',
                    r'what are the.*types', r'give.*examples'
                ],
                'boost_terms': ['include:', 'are as follows', 'following', 'such as', 'for example', 'includes']
            },
            'conditional': {
                'patterns': [
                    r'what if', r'under what circumstances', r'conditions',
                    r'when.*covered', r'is.*covered'
                ],
                'boost_terms': ['if', 'subject to', 'provided that', 'conditions', 'unless', 'except']
            }
        }
        return patterns
    
    def hybrid_search(self, query: str, k: int = 15) -> List[Tuple[str, float, Dict]]:
        """Fast hybrid search combining multiple retrieval methods"""
        # Detect question type
        question_type = self._detect_question_type(query.lower())
        
        # For single chunk documents, return immediately
        if len(self.chunks) == 1:
            return [(self.chunks[0], 1.0, {
                'chunk_idx': 0,
                'metadata': self.chunk_metadata[0],
                'question_type': question_type,
                'retrieval_method': 'single_chunk'
            })]
        
        # Get candidates from enhanced retriever if available
        if self.enhanced_retriever:
            bm25_candidates = self.enhanced_retriever.retrieve(
                query, k=min(k*2, len(self.chunks)), question_type=question_type
            )
        else:
            bm25_candidates = []
        
        # Get candidates from vector search
        query_embedding = self.model.encode([query]).astype('float32')
        search_k = min(k*2, len(self.chunks))
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Combine and deduplicate results
        candidate_scores = defaultdict(float)
        
        # Add BM25 scores
        if bm25_candidates:
            max_bm25_score = max(score for _, score in bm25_candidates) if bm25_candidates else 1.0
            for idx, score in bm25_candidates:
                if idx < len(self.chunks):
                    candidate_scores[idx] += (score / max_bm25_score) * 0.5
        
        # Add vector similarity scores
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                similarity = 1 / (1 + dist)
                candidate_scores[idx] += similarity * 0.5
        
        # Apply question-type specific boosting
        for idx in candidate_scores:
            chunk = self.chunks[idx]
            chunk_lower = chunk.lower()
            
            if question_type in self.question_patterns:
                boost_terms = self.question_patterns[question_type]['boost_terms']
                boost_score = sum(0.1 for term in boost_terms if term in chunk_lower)
                candidate_scores[idx] += boost_score
        
        # Sort by combined score
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top-k results
        results = []
        for idx, score in sorted_candidates[:k]:
            results.append((
                self.chunks[idx],
                score,
                {
                    'chunk_idx': idx,
                    'metadata': self.chunk_metadata[idx],
                    'question_type': question_type,
                    'retrieval_method': 'hybrid'
                }
            ))
        
        return results
    
    def _detect_question_type(self, query: str) -> str:
        """Detect question type"""
        for q_type, config in self.question_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, query, re.IGNORECASE):
                    return q_type
        return 'general'


class HybridFastTrackRAGPipeline:
    """Optimized pipeline for speed and accuracy"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.cross_encoder = None
        
        # Configure Gemini
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Google Generative AI configured successfully")
        except Exception as e:
            logger.critical(f"Failed to configure Google AI: {e}")
            raise
        
        # Thread pools
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
    async def download_document_async(self, url: str) -> bytes:
        """Async document download with better error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; RAGPipeline/3.0)',
            'Accept': '*/*'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, 
                                     timeout=aiohttp.ClientTimeout(total=30)) as response:
                    response.raise_for_status()
                    content = await response.read()
                    logger.info(f"Downloaded {len(content)/1024/1024:.1f}MB from {url}")
                    return content
        except Exception as e:
            logger.error(f"Async download failed: {e}, trying sync download")
            # Fallback to sync download
            return await asyncio.get_event_loop().run_in_executor(
                None, self._download_sync, url
            )
    
    def _download_sync(self, url: str) -> bytes:
        """Sync download fallback"""
        try:
            response = requests.get(url, timeout=30, stream=True)
            response.raise_for_status()
            
            content = b""
            for chunk in response.iter_content(chunk_size=1024*1024):
                content += chunk
                if len(content) > 200*1024*1024:  # 200MB limit
                    break
            
            return content
        except Exception as e:
            logger.error(f"Sync download also failed: {e}")
            raise
    
    def _parse_pdf_enhanced(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Enhanced PDF parsing with multiple fallbacks"""
        logger.info("Starting enhanced PDF parsing")
        full_text = ""
        chunk_metadata = []
        
        # Try pdfplumber first
        try:
            temp_file = io.BytesIO(content)
            with pdfplumber.open(temp_file) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing {total_pages} pages with pdfplumber")
                
                for i, page in enumerate(pdf.pages):
                    if i % 50 == 0 and i > 0:
                        logger.info(f"Processing page {i}/{total_pages}")
                    
                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        full_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                        chunk_metadata.append({
                            'page': i+1,
                            'type': 'text'
                        })
                    
                    # Extract tables (limit to avoid memory issues)
                    if i < 100:  # Only extract tables from first 100 pages
                        tables = page.extract_tables()
                        for j, table in enumerate(tables[:3]):  # Max 3 tables per page
                            if table:
                                table_text = self._format_table(table)
                                if table_text:
                                    full_text += f"\n=== TABLE {j+1} (Page {i+1}) ===\n{table_text}\n"
                                    chunk_metadata.append({
                                        'page': i+1,
                                        'type': 'table',
                                        'table_idx': j+1
                                    })
                
                logger.info(f"Successfully parsed {total_pages} pages")
                return self._clean_text(full_text), chunk_metadata
               
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying pypdf")
        
        # Fallback to pypdf
        try:
            temp_file = io.BytesIO(content)
            reader = pypdf.PdfReader(temp_file)
            total_pages = len(reader.pages)
            logger.info(f"Processing {total_pages} pages with pypdf")
            
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        full_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                        chunk_metadata.append({
                            'page': i+1,
                            'type': 'text_fallback'
                        })
                except Exception as e:
                    logger.warning(f"Page {i+1} extraction failed: {e}")
            
            return self._clean_text(full_text), chunk_metadata
            
        except Exception as e:
            logger.error(f"All PDF parsing methods failed: {e}")
            raise ValueError("Could not parse PDF document")
    
    def _format_table(self, table: List[List]) -> str:
        """Format table with better structure"""
        if not table:
            return ""
        
        formatted_rows = []
        
        # Add header separator if first row looks like headers
        if table and len(table) > 1:
            first_row = table[0]
            if all(cell and isinstance(cell, str) for cell in first_row):
                formatted_rows.append(" | ".join(str(cell) for cell in first_row))
                formatted_rows.append("-" * min(len(formatted_rows[0]), 100))
                
                for row in table[1:]:
                    if any(cell for cell in row):
                        formatted_rows.append(" | ".join(str(cell) if cell else "" for cell in row))
            else:
                for row in table:
                    if any(cell for cell in row):
                        formatted_rows.append(" | ".join(str(cell) if cell else "" for cell in row))
        
        return "\n".join(formatted_rows)
    
    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common OCR issues
        replacements = {
            'ﬁ': 'fi', 'ﬂ': 'fl', '–': '-', '"': '"', '"': '"',
            ''': "'", ''': "'", '…': '...', '—': '--'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix broken words
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        
        return text.strip()
    
    def _parse_docx(self, temp_file: io.BytesIO) -> str:
        """Parse DOCX files"""
        try:
            doc = Document(temp_file)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            # Extract tables
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    if any(row_text):
                        table_text.append(" | ".join(row_text))
                if table_text:
                    full_text.append("\n=== TABLE ===\n" + "\n".join(table_text) + "\n=== END TABLE ===\n")
            
            return "\n\n".join(full_text)
        except Exception as e:
            logger.error(f"DOCX parsing error: {e}")
            return ""
    
    def _parse_odt(self, temp_file: io.BytesIO) -> str:
        """Parse ODT files"""
        try:
            doc = load(temp_file)
            full_text = []
            
            for element in doc.getElementsByType(P):
                text = str(element)
                if text.strip():
                    full_text.append(text)
            
            return "\n\n".join(full_text)
        except Exception as e:
            logger.error(f"ODT parsing error: {e}")
            return ""
    
    def _parse_other_formats(self, content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
        """Parse non-PDF formats with better handling"""
        logger.info(f"Parsing document with extension: {file_extension}")
        
        # Skip binary formats that can't contain text
        binary_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.zip', '.rar', 
                            '.7z', '.mp3', '.mp4', '.avi', '.mov', '.bin']
        if file_extension.lower() in binary_extensions:
            logger.warning(f"Skipping binary file format: {file_extension}")
            return "This appears to be a binary file that cannot be processed for text content.", [{'type': 'binary_skip'}]
        
        # Handle Excel files
        if file_extension in ['.xlsx', '.xls']:
            try:
                temp_file = io.BytesIO(content)
                df = pd.read_excel(temp_file, engine='openpyxl' if file_extension == '.xlsx' else None)
                # Convert to string representation
                text = df.to_string(max_rows=1000)  # Limit rows for large files
                return self._clean_text(text), [{'type': 'spreadsheet'}]
            except Exception as e:
                logger.warning(f"Failed to parse Excel file: {e}")
                return "Unable to parse Excel file content.", [{'type': 'excel_error'}]
        
        # Handle PowerPoint
        if file_extension in ['.pptx', '.ppt'] and PPTX_AVAILABLE:
            try:
                temp_file = io.BytesIO(content)
                prs = Presentation(temp_file)
                text_runs = []
                for slide_num, slide in enumerate(prs.slides):
                    slide_text = f"\n--- SLIDE {slide_num + 1} ---\n"
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text:
                            slide_text += shape.text + "\n"
                    text_runs.append(slide_text)
                text = "\n".join(text_runs)
                return self._clean_text(text), [{'type': 'presentation'}]
            except Exception as e:
                logger.warning(f"Failed to parse PowerPoint file: {e}")
                return "Unable to parse PowerPoint file content.", [{'type': 'pptx_error'}]
        
        # Handle DOCX
        if file_extension in ['.docx', '.doc']:
            try:
                temp_file = io.BytesIO(content)
                text = self._parse_docx(temp_file)
                return self._clean_text(text), [{'type': 'docx'}]
            except Exception as e:
                logger.warning(f"Failed to parse DOCX: {e}")
        
        # Handle ODT
        if file_extension == '.odt':
            try:
                temp_file = io.BytesIO(content)
                text = self._parse_odt(temp_file)
                return self._clean_text(text), [{'type': 'odt'}]
            except Exception as e:
                logger.warning(f"Failed to parse ODT: {e}")
        
        # Default to text extraction
        try:
            text = content.decode('utf-8', errors='ignore')
            return self._clean_text(text), [{'type': 'text'}]
        except Exception as e:
            logger.error(f"Failed to decode as text: {e}")
            return "Unable to extract text from this file format.", [{'type': 'unknown'}]
    
    async def get_or_create_optimized_vector_store(self, url: str) -> OptimizedVectorStore:
        """Create vector store with enhanced caching"""
        # Generate cache key
        cache_key = f"vs_{hashlib.md5(url.encode()).hexdigest()}"
        
        # Check cache
        cached_store = await cache.get(cache_key)
        if cached_store:
            logger.info(f"Using cached vector store for {url}")
            return cached_store
        
        # Check for cached intermediate embeddings
        embed_cache_key = f"emb_{hashlib.md5(url.encode()).hexdigest()}"
        cached_embeddings = await cache.get(embed_cache_key)
        
        if cached_embeddings:
            logger.info("Using cached embeddings")
            chunks, embeddings, chunk_metadata = cached_embeddings
        else:
            # Download and parse document
            logger.info(f"Downloading document from {url}")
            try:
                content = await self.download_document_async(url)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise ValueError(f"Could not download document: {str(e)}")
            
            # Determine file type and parse
            file_extension = os.path.splitext(url.split('?')[0])[1].lower()
            
            if not file_extension or file_extension == '.pdf':
                text, doc_metadata = self._parse_pdf_enhanced(content)
            else:
                text, doc_metadata = self._parse_other_formats(content, file_extension)
            
            # Check if we got valid text
            if not text or len(text) < 10:
                logger.warning(f"No valid text extracted from {url}")
                text = "Document appears to be empty or could not be parsed."
                doc_metadata = [{'type': 'empty'}]
            
            # Smart chunking
            chunks, chunk_metadata = SmartChunker.chunk_with_boundaries(text, doc_metadata)
            
            if not chunks:
                chunks = [text]  # Fallback to full text as single chunk
                chunk_metadata = doc_metadata
            
            # Add context to chunks for better retrieval
            contextualized_chunks = ContextualRetrieval.add_context_to_chunks(chunks, chunk_metadata)
            
            logger.info(f"Created {len(chunks)} chunks, generating embeddings...")
            
            # Generate embeddings in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(contextualized_chunks), batch_size):
                batch = contextualized_chunks[i:i + batch_size]
                try:
                    batch_embeddings = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda b: self.embedding_model.encode(b, convert_to_numpy=True, show_progress_bar=False),
                            batch
                        ),
                        timeout=30.0
                    )
                    all_embeddings.append(batch_embeddings)
                except asyncio.TimeoutError:
                    logger.error(f"Embedding generation timeout for batch {i//batch_size}")
                    # Use zero embeddings as fallback
                    zero_embeddings = np.zeros((len(batch), self.embedding_model.get_sentence_embedding_dimension()))
                    all_embeddings.append(zero_embeddings)
            
            embeddings = np.vstack(all_embeddings).astype('float32')
            
            # Cache embeddings
            await cache.set(embed_cache_key, (chunks, embeddings, chunk_metadata))
            logger.info(f"Cached embeddings for {url}")
        
        # Create vector store
        vector_store = OptimizedVectorStore(chunks, embeddings, self.embedding_model, chunk_metadata)
        
        # Cache the store
        await cache.set(cache_key, vector_store)
        
        return vector_store
    
    async def _answer_question_fast(self, question: str, vector_store: OptimizedVectorStore) -> str:
        """Fast answer generation with caching"""
        # Check answer cache first
        answer_cache_key = f"ans_{hashlib.md5(question.encode()).hexdigest()[:16]}"
        cached_answer = await cache.get(answer_cache_key)
        if cached_answer:
            return cached_answer
        
        # Get relevant chunks using hybrid search
        retrieved_results = vector_store.hybrid_search(question, k=settings.MAX_CHUNKS_PER_QUERY)
        
        if not retrieved_results:
            return "Based on the available document content, I could not find specific information to answer this question."
        
        # Extract just the text chunks
        top_chunks = [result[0] for result in retrieved_results[:8]]
        
        # Use a single, focused prompt for speed
        prompt = f"""Answer the question based ONLY on the provided context. Be specific and direct.

    CONTEXT:
    {chr(10).join(top_chunks)}

    QUESTION: {question}

    INSTRUCTIONS:
    - Answer directly with specific information from the context
    - Include exact values, numbers, or terms found
    - If information is not in the context, say "Information not found in the provided context"
    - Keep answer concise but complete

    ANSWER:"""

        try:
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)  # Use flash model by default
            
            response = await asyncio.wait_for(
                model.generate_content_async(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        max_output_tokens=400,
                        top_p=0.95,
                        candidate_count=1
                    )
                ),
                timeout=settings.ANSWER_TIMEOUT_SECONDS
            )
            
            answer = response.text.strip()
            
            # Validate answer
            if not answer or len(answer) < 10:
                answer = "Unable to generate a valid answer from the document."
            
            # Cache the answer
            if len(answer) > 20:
                await cache.set(answer_cache_key, answer)
            
            return answer
            
        except asyncio.TimeoutError:
            logger.warning(f"Answer generation timeout for question: {question[:50]}...")
            return "Processing timeout. Unable to generate answer."
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer due to an error."
    
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process queries with optimized parallel execution"""
        start_time = time.time()
        logger.info(f"Processing {len(questions)} questions for document: {document_url}")
        
        try:
            # Create vector store once
            vector_store = await self.get_or_create_optimized_vector_store(document_url)
            
            # Process all questions in parallel with controlled concurrency
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)
            
            async def process_with_semaphore(q):
                async with semaphore:
                    return await self._answer_question_safe(q, vector_store)
            
            # Create all tasks
            tasks = [process_with_semaphore(q) for q in questions]
            
            # Execute with overall timeout
            try:
                answers = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=settings.TOTAL_TIMEOUT_SECONDS
                )
                
                # Process results
                final_answers = []
                for i, answer in enumerate(answers):
                    if isinstance(answer, Exception):
                        logger.error(f"Question {i+1} error: {answer}")
                        final_answers.append("Error processing this question.")
                    else:
                        final_answers.append(answer)
                
                processing_time = time.time() - start_time
                logger.info(f"Completed {len(questions)} questions in {processing_time:.2f}s "
                            f"(avg: {processing_time/len(questions):.2f}s per question)")
                
                return final_answers
                
            except asyncio.TimeoutError:
                logger.error("Overall processing timeout")
                return ["Processing timeout. Please try again."] * len(questions)
                
        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
            return ["Document processing error."] * len(questions)
    
    async def _answer_question_safe(self, question: str, vector_store: OptimizedVectorStore) -> str:
        """Safe wrapper for question answering"""
        try:
            return await self._answer_question_fast(question, vector_store)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Unable to process this question due to an error."
    
    async def process_query_with_explainability(self, document_url: str, questions: List[str]) -> Tuple[List[str], List[Dict], float]:
        """Process with detailed explanations (backward compatibility)"""
        start_time = time.time()
        
        # Use the fast processing method
        simple_answers = await self.process_query(document_url, questions)
        
        # Create basic detailed answers
        detailed_answers = []
        for answer in simple_answers:
            detailed_answer = {
                "answer": answer,
                "confidence": 0.8 if answer and "error" not in answer.lower() else 0.0,
                "source_clauses": [],
                "reasoning": "Answer extracted using optimized hybrid retrieval.",
                "coverage_decision": self._determine_coverage(answer)
            }
            detailed_answers.append(detailed_answer)
        
        processing_time = time.time() - start_time
        return simple_answers, detailed_answers, processing_time
    
    def _determine_coverage(self, answer: str) -> str:
        """Determine coverage decision"""
        answer_lower = answer.lower()
        
        if any(term in answer_lower for term in ['not covered', 'excluded', 'not available', 'cannot find', 'not found']):
            return "Not Covered"
        elif any(term in answer_lower for term in ['covered', 'included', 'eligible', 'yes']):
            return "Covered"
        elif any(term in answer_lower for term in ['conditions apply', 'subject to', 'depending on']):
            return "Conditional"
        elif any(term in answer_lower for term in ['error', 'timeout', 'unable']):
            return "Error"
        else:
            return "Review Required"