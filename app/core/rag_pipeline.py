# app/core/rag_pipeline.py - Enhanced version
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing

logger = logging.getLogger(__name__)


class SmartChunker:
    """Intelligent chunking that preserves semantic boundaries"""
    
    @staticmethod
    def chunk_with_boundaries(text: str, metadata: List[Dict], 
                            chunk_size: int = 1200, 
                            overlap: int = 200) -> Tuple[List[str], List[Dict]]:
        """Create chunks that respect semantic boundaries"""
        chunks = []
        chunk_metadata = []
        
        # Split by major sections first
        sections = re.split(r'\n(?=(?:CHAPTER|SECTION|ARTICLE|PART)\s+[IVXLCDM\d]+)', text)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Further split by paragraphs
            paragraphs = re.split(r'\n\n+', section)
            current_chunk = ""
            current_meta = {}
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check if adding this paragraph exceeds chunk size
                if len(current_chunk) + len(para) + 2 > chunk_size and current_chunk:
                    # Save current chunk
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
        
        # Initialize FAISS index
        if len(chunks) > 5000:
            nlist = min(256, len(chunks) // 20)
            m = 8
            self.index = faiss.IndexIVFPQ(
                faiss.IndexFlatL2(dimension), dimension, nlist, m, 8
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(64, nlist // 2)
        elif len(chunks) > 1000:
            nlist = min(100, len(chunks) // 10)
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(dimension), dimension, nlist
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(32, nlist // 2)
        else:
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        
        # Initialize enhanced retriever
        self.enhanced_retriever = EnhancedRetriever(chunks, chunk_metadata)
        
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
            },
            'temporal': {
                'patterns': [
                    r'when', r'how long', r'what.*period', r'duration',
                    r'waiting period', r'from when'
                ],
                'boost_terms': ['days', 'months', 'years', 'period', 'duration', 'from', 'after']
            }
        }
        return patterns
    
    def hybrid_search(self, query: str, k: int = 30) -> List[Tuple[str, float, Dict]]:
        """Hybrid search combining multiple retrieval methods"""
        # Detect question type
        question_type = self._detect_question_type(query.lower())
        
        # Get candidates from enhanced retriever
        bm25_candidates = self.enhanced_retriever.retrieve(query, k=k*2, question_type=question_type)
        
        # Get candidates from vector search
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, k*2)
        
        # Combine and deduplicate results
        candidate_scores = defaultdict(float)
        
        # Add BM25 scores (normalized)
        if bm25_candidates:
            max_bm25_score = max(score for _, score in bm25_candidates)
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
    """Enhanced pipeline with better accuracy"""
    
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
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Enhanced cache for large documents
        self.embedding_cache = {}
        
    async def download_document_async(self, url: str) -> bytes:
        """Async document download with better error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; RAGPipeline/3.0)',
            'Accept': '*/*'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)) as response:
                    response.raise_for_status()
                    content = await response.read()
                    logger.info(f"Downloaded {len(content)/1024/1024:.1f}MB from {url}")
                    return content
            except Exception as e:
                logger.error(f"Download failed: {e}")
                # Fallback to sync download
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._download_sync, url
                )
    
    def _download_sync(self, url: str) -> bytes:
        """Sync download fallback"""
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()
        
        content = b""
        for chunk in response.iter_content(chunk_size=1024*1024):
            content += chunk
            if len(content) > 200*1024*1024:  # 200MB limit
                break
        
        return content
    
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
                    if i % 50 == 0:
                        logger.info(f"Processing page {i}/{total_pages}")
                    
                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        full_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                        chunk_metadata.append({
                            'page': i+1,
                            'type': 'text'
                        })
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        if table:
                            table_text = self._format_table(table)
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
                formatted_rows.append("-" * len(formatted_rows[0]))
                
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
    
    async def get_or_create_optimized_vector_store(self, url: str) -> OptimizedVectorStore:
        """Create vector store with enhanced caching"""
        # Generate cache key
        cache_key = f"vs_{hashlib.md5(url.encode()).hexdigest()}"
        
        # Check cache
        cached_store = await cache.get(cache_key)
        if cached_store:
            logger.info(f"Using cached vector store for {url}")
            return cached_store
        
        # # Check if we have cached embeddings
        # embed_cache_key = f"emb_{hashlib.md5(url.encode()).hexdigest()}"
        # cached_embeddings = self.embedding_cache.get(embed_cache_key)
        # Check for cached intermediate embeddings in the main shared cache
        embed_cache_key = f"emb_{hashlib.md5(url.encode()).hexdigest()}"
        cached_embeddings_data = await cache.get(embed_cache_key)
        
        if cached_embeddings:
            logger.info("Using cached embeddings")
            chunks, embeddings, chunk_metadata = cached_embeddings
        else:
            # Download and parse document
            logger.info(f"Downloading document from {url}")
            content = await self.download_document_async(url)
            
            # Determine file type and parse
            file_extension = os.path.splitext(url.split('?')[0])[1].lower()
            if not file_extension or file_extension == '.pdf':
                text, doc_metadata = self._parse_pdf_enhanced(content)
            else:
                # Handle other formats
                text, doc_metadata = self._parse_other_formats(content, file_extension)
            
            # Smart chunking
            chunks, chunk_metadata = SmartChunker.chunk_with_boundaries(text, doc_metadata)
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            logger.info(f"Created {len(chunks)} chunks, generating embeddings...")
            
            # Generate embeddings in batches
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda b: self.embedding_model.encode(b, convert_to_numpy=True, show_progress_bar=False),
                    batch
                )
                all_embeddings.append(batch_embeddings)
                
                if i % (batch_size * 10) == 0:
                    logger.info(f"Generated embeddings for {i}/{len(chunks)} chunks")
            
            embeddings = np.vstack(all_embeddings).astype('float32')
            
            # Cache embeddings for large documents
            # if len(chunks) > 1000:
            #     self.embedding_cache[embed_cache_key] = (chunks, embeddings, chunk_metadata)
            #     logger.info("Cached embeddings for future use")
            # Cache the expensive-to-create embeddings data in the main shared cache
            embedding_data_to_cache = (chunks, embeddings, chunk_metadata)
            await cache.set(embed_cache_key, embedding_data_to_cache)
            logger.info(f"Cached intermediate embeddings for {url} under key {embed_cache_key}")

        
        # Create vector store
        vector_store = OptimizedVectorStore(chunks, embeddings, self.embedding_model, chunk_metadata)
        
        # Cache the store
        await cache.set(cache_key, vector_store)
        
        return vector_store
    
    def _parse_other_formats(self, content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
        """Parse non-PDF formats"""
        if file_extension in ['.docx', '.doc']:
            temp_file = io.BytesIO(content)
            return self._parse_docx(temp_file), [{'type': 'docx'}]
        elif file_extension == '.odt':
            temp_file = io.BytesIO(content)
            return self._parse_odt(temp_file), [{'type': 'odt'}]
        else:
            # Default to text
            try:
                text = content.decode('utf-8', errors='ignore')
                return text, [{'type': 'text'}]
            except:
                raise ValueError(f"Unsupported file format: {file_extension}")
    
    def _parse_docx(self, temp_file: io.BytesIO) -> str:
        """Parse DOCX files"""
        doc = Document(temp_file)
        full_text = []
        
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                full_text.append(paragraph.text)
        
        for table in doc.tables:
            table_text = []
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells]
                if any(row_text):
                    table_text.append(" | ".join(row_text))
            if table_text:
                full_text.append("\n=== TABLE ===\n" + "\n".join(table_text) + "\n=== END TABLE ===\n")
        
        return "\n\n".join(full_text)
    
    def _parse_odt(self, temp_file: io.BytesIO) -> str:
        """Parse ODT files"""
        doc = load(temp_file)
        full_text = []
        
        for element in doc.getElementsByType(P):
            text = str(element)
            if text.strip():
                full_text.append(text)
        
        return "\n\n".join(full_text)

    def _parse_other_formats(self, content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
        """
        Parses non-PDF document formats like DOCX, ODT, and plain text.
        This function was previously missing.
        """
        logger.info(f"Parsing document with extension: {file_extension}")
        metadata = [{'type': file_extension.strip('.')}]
        
        try:
            if file_extension in ['.docx', '.doc']:
                temp_file = io.BytesIO(content)
                text = self._parse_docx(temp_file)
                return self._clean_text(text), metadata
            
            elif file_extension == '.odt':
                temp_file = io.BytesIO(content)
                text = self._parse_odt(temp_file)
                return self._clean_text(text), metadata

            else:
                # Default to decoding as text, ignoring errors
                logger.warning(f"Unsupported extension '{file_extension}', attempting to read as plain text.")
                text = content.decode('utf-8', errors='ignore')
                return self._clean_text(text), metadata

        except Exception as e:
            logger.error(f"Failed to parse file with extension {file_extension}: {e}")
            raise ValueError(f"Could not parse file with extension: {file_extension}") 
    
    async def _answer_question_enhanced(self, question: str, vector_store: OptimizedVectorStore) -> str:
        """Enhanced answer generation with multi-stage approach"""
        logger.info(f"Answering question: {question[:100]}...")
        
        # Get relevant chunks using hybrid search
        retrieved_results = vector_store.hybrid_search(question, k=40)
        
        if not retrieved_results:
            return "Based on the available document content, I could not find specific information to answer this question."
        
        # Use cross-encoder for reranking if available
        if len(retrieved_results) > 20:
            if self.cross_encoder is None:
                try:
                    self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                except:
                    logger.warning("Could not load cross-encoder")
            
            if self.cross_encoder:
                chunks = [result[0] for result in retrieved_results[:25]]
                pairs = [[question, chunk] for chunk in chunks]
                
                try:
                    scores = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, self.cross_encoder.predict, pairs
                        ),
                        timeout=10.0
                    )
                    
                    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
                    top_chunks = [chunk for chunk, _ in reranked[:15]]
                except:
                    top_chunks = [result[0] for result in retrieved_results[:15]]
            else:
                top_chunks = [result[0] for result in retrieved_results[:15]]
        else:
            top_chunks = [result[0] for result in retrieved_results[:15]]
        
        # Multi-stage answer generation
        answer = await self._generate_answer_multistage(question, top_chunks)
        
        logger.info(f"Generated answer: {answer[:200]}...")
        return answer
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted)
    )
    async def _generate_answer_multistage(self, question: str, context: List[str]) -> str:
        """Multi-stage answer generation for better accuracy"""
        if not context:
            return "No relevant information found to answer this question."
        
        # Stage 1: Extract relevant information
        extraction_prompt = f"""You are an expert at extracting specific information from documents.

CONTEXT (from document):
{chr(10).join(context[:10])}

QUESTION: {question}

TASK: Extract ALL relevant information that helps answer the question. Include:
- Direct answers if present
- Related facts and details
- Specific numbers, dates, or values
- Any conditions or exceptions mentioned
- Lists or enumerations if the question asks for multiple items

If the information is not in the context, say "Information not found in the provided context."

EXTRACTED INFORMATION:"""

        try:
            model = genai.GenerativeModel('gemini-1.5-pro-latest')
            
            # First stage - extraction
            extraction_response = await model.generate_content_async(
                extraction_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=800,
                    top_p=0.95
                )
            )
            
            extracted_info = extraction_response.text.strip()
            
            # Check if information was found
            if "Information not found" in extracted_info or len(extracted_info) < 20:
                # Try with more context
                extended_context = chr(10).join(context[:15])
                extended_prompt = extraction_prompt.replace(
                    chr(10).join(context[:10]), 
                    extended_context
                )
                
                extraction_response = await model.generate_content_async(
                    extended_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=1000,
                        top_p=0.95
                    )
                )
                
                extracted_info = extraction_response.text.strip()
            
            # Stage 2: Formulate final answer
            answer_prompt = f"""Based on the extracted information, provide a direct and complete answer to the question.

QUESTION: {question}

EXTRACTED INFORMATION:
{extracted_info}

INSTRUCTIONS:
1. Answer directly and specifically
2. Include ALL relevant details found
3. Use exact values, numbers, or terms from the information
4. For list questions, enumerate all items found
5. For calculation questions, show the calculation
6. If information is incomplete, mention what's missing

ANSWER:"""

            answer_response = await model.generate_content_async(
                answer_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=600,
                    top_p=0.95
                )
            )
            
            final_answer = answer_response.text.strip()
            
            # Validate answer quality
            if len(final_answer) < 10 or "information not found" in final_answer.lower():
                return f"Based on the available document content, I could not find specific information to answer: '{question}'"
            
            return final_answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Unable to generate answer due to an error: {str(e)}"
    
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process queries with improved accuracy"""
        start_time = time.time()
        logger.info(f"Processing {len(questions)} questions for document: {document_url}")
        
        try:
            # Create vector store
            vector_store = await self.get_or_create_optimized_vector_store(document_url)
            
            # Process questions in parallel batches
            batch_size = 10
            all_answers = []
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                
                # Create tasks for batch
                tasks = [
                    self._answer_question_safe(q, vector_store) 
                    for q in batch
                ]
                
                # Execute batch with timeout
                try:
                    batch_answers = await asyncio.wait_for(
                        asyncio.gather(*tasks, return_exceptions=True),
                        timeout=120.0
                    )
                    
                    # Process results
                    for j, answer in enumerate(batch_answers):
                        if isinstance(answer, Exception):
                            logger.error(f"Question {i+j+1} error: {answer}")
                            all_answers.append("An error occurred while processing this question.")
                        else:
                            all_answers.append(answer)
                    
                except asyncio.TimeoutError:
                    logger.error(f"Batch timeout for questions {i+1}-{i+len(batch)}")
                    for _ in batch:
                        all_answers.append("Processing timeout. Please try again.")
                
                # Log progress
                logger.info(f"Processed {min(i + batch_size, len(questions))}/{len(questions)} questions")
            
            processing_time = time.time() - start_time
            logger.info(f"Completed processing in {processing_time:.2f} seconds")
            
            return all_answers
            
        except Exception as e:
            logger.critical(f"Critical error: {e}", exc_info=True)
            return ["Document processing error. Please try again."] * len(questions)
    
    async def _answer_question_safe(self, question: str, vector_store: OptimizedVectorStore) -> str:
        """Safe wrapper for question answering"""
        try:
            return await self._answer_question_enhanced(question, vector_store)
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return "Unable to process this question due to an error."
    
    async def process_query_with_explainability(self, document_url: str, questions: List[str]) -> Tuple[List[str], List[Dict], float]:
        """Process with detailed explanations"""
        start_time = time.time()
        
        try:
            vector_store = await self.get_or_create_optimized_vector_store(document_url)
            
            simple_answers = []
            detailed_answers = []
            
            for question in questions:
                # Get retrieval results
                retrieved_results = vector_store.hybrid_search(question, k=30)
                
                if retrieved_results:
                    # Generate answer
                    top_chunks = [result[0] for result in retrieved_results[:15]]
                    answer = await self._generate_answer_multistage(question, top_chunks)
                    
                    # Create detailed answer
                    source_clauses = []
                    for i, (chunk, score, metadata) in enumerate(retrieved_results[:5]):
                        source_clauses.append({
                            "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            "confidence_score": float(score),
                            "page_number": metadata.get('metadata', {}).get('page'),
                            "section": metadata.get('metadata', {}).get('type', 'general'),
                            "retrieval_type": metadata.get('retrieval_method', 'unknown'),
                            "question_type": metadata.get('question_type', 'general')
                        })
                    
                    detailed_answer = {
                        "answer": answer,
                        "confidence": min(0.95, float(retrieved_results[0][1])) if retrieved_results else 0.0,
                        "source_clauses": source_clauses,
                        "reasoning": f"Answer extracted using hybrid retrieval with {len(retrieved_results)} relevant chunks.",
                        "coverage_decision": self._determine_coverage(answer)
                    }
                else:
                    answer = "Information not available in the document."
                    detailed_answer = {
                        "answer": answer,
                        "confidence": 0.0,
                        "source_clauses": [],
                        "reasoning": "No relevant information found.",
                        "coverage_decision": "Not Found"
                    }
                
                simple_answers.append(answer)
                detailed_answers.append(detailed_answer)
            
            processing_time = time.time() - start_time
            return simple_answers, detailed_answers, processing_time
            
        except Exception as e:
            logger.error(f"Error in explainable processing: {e}")
            processing_time = time.time() - start_time
            error_answers = ["Processing error"] * len(questions)
            detailed_errors = [
                {
                    "answer": "Processing error",
                    "confidence": 0.0,
                    "source_clauses": [],
                    "reasoning": str(e),
                    "coverage_decision": "Error"
                }
                for _ in questions
            ]
            return error_answers, detailed_errors, processing_time
    
    def _determine_coverage(self, answer: str) -> str:
        """Determine coverage decision"""
        answer_lower = answer.lower()
        
        if any(term in answer_lower for term in ['not covered', 'excluded', 'not available', 'cannot find']):
            return "Not Covered"
        elif any(term in answer_lower for term in ['covered', 'included', 'eligible', 'yes']):
            return "Covered"
        elif any(term in answer_lower for term in ['conditions apply', 'subject to', 'depending on']):
            return "Conditional"
        else:
            return "Review Required"