# app/core/rag_pipeline.py - FAST & ACCURATE VERSION
import io
import re
import os
import logging
import requests
import asyncio
import email
import time
from typing import List, Tuple, Dict, Set

# Imports for multi-format support
import pdfplumber
from docx import Document
from odf.text import P
from odf.opendocument import load

# Imports for resilience and cloud integration
from google.cloud import storage
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core import exceptions as google_exceptions

# Imports for core RAG logic
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
import faiss
import numpy as np
import google.generativeai as genai

from app.core.config import settings

logger = logging.getLogger(__name__)


class FastVectorStore:
    """FAST vector store optimized for speed while maintaining accuracy"""
    def __init__(self, chunks: List[str], embeddings: np.ndarray, model: SentenceTransformer):
        self.chunks = chunks
        self.model = model
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Lightweight keyword index for critical terms
        self.critical_terms = self._extract_critical_terms(chunks)
        logger.info(f"Built FAST FAISS index for {len(chunks)} chunks with {len(self.critical_terms)} critical terms.")

    def _extract_critical_terms(self, chunks: List[str]) -> Dict[str, List[int]]:
        """Extract only the most critical terms for fast lookup"""
        critical_index = {}
        
        # Focus on high-value terms only
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            
            # Numbers and percentages - critical for accuracy
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|percent|million|billion|thousand|crore|lakh)?\b', chunk_lower)
            
            # Years - critical for temporal questions
            years = re.findall(r'\b(19|20)\d{2}\b', chunk)
            
            # Monetary amounts
            amounts = re.findall(r'(?:rs\.|₹|inr|dollar|\$)\s*\d+(?:,\d{3})*(?:\.\d+)?(?:million|billion|thousand|crore|lakh)?\b', chunk_lower)
            
            # Key insurance/policy terms
            key_terms = []
            important_keywords = [
                'coverage', 'premium', 'deductible', 'waiting period', 'grace period',
                'claim', 'benefit', 'exclusion', 'maternity', 'pre-existing',
                'renewal', 'discount', 'hospital', 'treatment', 'surgery'
            ]
            
            for keyword in important_keywords:
                if keyword in chunk_lower:
                    key_terms.append(keyword)
            
            # Index all critical terms
            all_terms = numbers + years + amounts + key_terms
            for term in set(all_terms):  # Remove duplicates
                if term not in critical_index:
                    critical_index[term] = []
                critical_index[term].append(i)
        
        return critical_index

    def fast_search(self, query: str, k: int = 20) -> List[Tuple[str, float]]:
        """Fast multi-strategy search optimized for speed"""
        results = []
        
        # Strategy 1: Vector similarity (primary)
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        for distance, idx in zip(distances[0], indices[0]):
            if idx != -1:
                similarity = 1 / (1 + distance)
                results.append((self.chunks[idx], similarity))
        
        # Strategy 2: Critical term boosting (fast)
        query_lower = query.lower()
        bonus_chunks = set()
        
        # Boost chunks with matching critical terms
        for term, chunk_indices in self.critical_terms.items():
            if term in query_lower:
                for chunk_idx in chunk_indices[:3]:  # Limit to avoid slowdown
                    if chunk_idx < len(self.chunks):
                        bonus_chunks.add(chunk_idx)
        
        # Add bonus chunks if not already included
        for chunk_idx in bonus_chunks:
            chunk = self.chunks[chunk_idx]
            if not any(chunk == existing_chunk for existing_chunk, _ in results):
                results.append((chunk, 0.85))  # High relevance score
        
        # Sort by relevance and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]


class FastAccurateRAGPipeline:
    """FAST & ACCURATE RAG pipeline optimized for production speed"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Google Generative AI client configured successfully.")
        except Exception as e:
            logger.critical(f"Failed to configure Google Generative AI: {e}")
            raise RuntimeError("Google API Key is not configured correctly.")
        
        # Use lighter cross-encoder for speed
        try:
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            logger.info("Fast CrossEncoder model loaded.")
        except Exception as e:
            logger.warning(f"CrossEncoder loading failed, will use vector similarity only: {e}")
            self.cross_encoder = None

    def _download_and_parse_document(self, url: str) -> str:
        """Fast document parsing - reuse your existing method but with timeout optimization"""
        logger.info(f"Processing document from {url}")
        
        try:
            temp_file = None
            file_extension = ""
            
            if url.startswith(("http://", "https://")):
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (compatible; DocumentProcessor/1.0)',
                        'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*'
                    }
                    
                    logger.info(f"Downloading from URL: {url}")
                    response = requests.get(url, headers=headers, timeout=30, stream=True, allow_redirects=True)
                    response.raise_for_status()
                    
                    content = b""
                    max_size = 50 * 1024 * 1024  # 50MB limit for speed
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content += chunk
                            if len(content) > max_size:
                                break
                    
                    temp_file = io.BytesIO(content)
                    file_extension = os.path.splitext(url.split('?')[0])[1].lower()
                    
                    if not file_extension and 'pdf' in response.headers.get('content-type', ''):
                        file_extension = '.pdf'
                    
                    logger.info(f"Downloaded {len(content)} bytes, extension: {file_extension}")
                    
                except Exception as e:
                    raise ValueError(f"Error downloading: {str(e)}")
            else:
                raise ValueError("Only HTTP/HTTPS URLs supported in fast mode")

            # Fast PDF parsing
            full_text = ""
            temp_file.seek(0)
            
            if file_extension == '.pdf':
                try:
                    with pdfplumber.open(temp_file) as pdf:
                        logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                        
                        # Process pages in batches for speed
                        for page_num, page in enumerate(pdf.pages):
                            if page_num > 50:  # Limit pages for speed
                                break
                                
                            try:
                                page_text = page.extract_text()
                                if page_text and page_text.strip():
                                    full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                                
                                # Quick table extraction
                                tables = page.extract_tables()
                                for table in tables[:2]:  # Limit tables per page
                                    if table:
                                        table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table if row])
                                        if table_text.strip():
                                            full_text += f"\n=== TABLE ===\n{table_text}\n=== END TABLE ===\n"
                                            
                            except Exception as e:
                                logger.warning(f"Error on page {page_num}: {e}")
                                continue
                                
                except Exception as e:
                    raise ValueError(f"PDF parsing error: {str(e)}")
            
            # Validate and clean
            if not full_text or len(full_text) < 50:
                raise ValueError("Document too short or empty")
            
            # Fast cleaning
            full_text = re.sub(r'\n\s*\n\s*\n', '\n\n', full_text)
            full_text = re.sub(r'[ \t]+', ' ', full_text).strip()
            
            logger.info(f"Extracted {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise ValueError(f"Could not process document: {str(e)}")

    def _fast_chunk_text(self, text: str) -> List[str]:
        """Fast chunking optimized for speed and accuracy balance"""
        
        # Quick paragraph-based chunking
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 20]
        
        chunks = []
        current_chunk = ""
        target_size = 800  # Smaller for speed
        overlap_size = 150  # Reasonable overlap
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > target_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Smart overlap
                if len(current_chunk) > overlap_size:
                    sentences = current_chunk.split('. ')
                    if len(sentences) > 2:
                        overlap = '. '.join(sentences[-2:])
                        current_chunk = overlap + " " + paragraph
                    else:
                        current_chunk = paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter and validate
        final_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]
        
        logger.info(f"Created {len(final_chunks)} fast chunks")
        return final_chunks

    async def get_or_create_fast_vector_store(self, url: str) -> FastVectorStore:
        """Create vector store with speed optimization"""
        from app.core.cache import cache
        cached_store = await cache.get(url)
        if cached_store:
            logger.info(f"Using cached vector store for {url}")
            return cached_store
            
        logger.info(f"Creating fast vector store for {url}...")
        text = self._download_and_parse_document(url)
        chunks = self._fast_chunk_text(text)
        
        if not chunks:
            raise ValueError("No chunks created")
            
        # Fast embedding generation
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False, batch_size=32).astype('float32')
        vector_store = FastVectorStore(chunks, embeddings, self.embedding_model)
        
        await cache.set(url, vector_store)
        return vector_store

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted)
    )
    async def _generate_fast_answer(self, question: str, context: List[str]) -> str:
        """Fast answer generation with accuracy focus"""
        
        # Limit context for speed
        context_text = "\n\n---\n\n".join(context[:6])
        
        # Smart prompt based on question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['how many', 'percentage', 'amount', 'number']):
            instruction = "Focus on finding specific numbers, amounts, and quantitative data. Provide exact figures with context."
        elif any(word in question_lower for word in ['does', 'is', 'are', 'can', 'will']):
            instruction = "Provide a clear yes/no answer followed by supporting details and conditions."
        else:
            instruction = "Provide a comprehensive answer with specific details from the document."
        
        prompt = f"""You are an expert document analyst. {instruction}

IMPORTANT: Extract information directly from the context below. If you find relevant information, provide it clearly. Only say "information not available" if you truly cannot find any related content.

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

        try:
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=400,
                    top_p=0.8
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Error generating answer."

    async def _answer_question_fast(self, question: str, vector_store: FastVectorStore) -> str:
        """Fast question answering with accuracy optimization"""
        
        # Fast retrieval
        retrieved_results = vector_store.fast_search(question, k=15)
        
        if not retrieved_results:
            return "Based on the provided documents, the information to answer this question is not available."

        # Optional re-ranking (skip if too slow)
        if self.cross_encoder and len(retrieved_results) <= 10:
            try:
                chunks = [chunk for chunk, _ in retrieved_results]
                pairs = [[question, chunk] for chunk in chunks]
                
                # Use executor with timeout
                loop = asyncio.get_running_loop()
                scores = await asyncio.wait_for(
                    loop.run_in_executor(None, self.cross_encoder.predict, pairs),
                    timeout=5.0  # 5 second timeout
                )
                
                # Combine scores
                final_results = []
                for i, ((chunk, initial_score), cross_score) in enumerate(zip(retrieved_results, scores)):
                    combined_score = 0.4 * initial_score + 0.6 * float(cross_score)
                    final_results.append((combined_score, chunk))
                
                final_results.sort(reverse=True)
                top_chunks = [chunk for score, chunk in final_results[:8]]
                
            except asyncio.TimeoutError:
                logger.warning("Cross-encoder timeout, using vector similarity only")
                top_chunks = [chunk for chunk, _ in retrieved_results[:8]]
            except Exception as e:
                logger.warning(f"Cross-encoder failed: {e}")
                top_chunks = [chunk for chunk, _ in retrieved_results[:8]]
        else:
            top_chunks = [chunk for chunk, _ in retrieved_results[:8]]

        # Generate answer
        return await self._generate_fast_answer(question, top_chunks)

    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Fast query processing optimized for production"""
        start_time = time.time()
        
        try:
            vector_store = await self.get_or_create_fast_vector_store(document_url)
            
            # Process questions with reasonable concurrency
            answers = []
            batch_size = 3  # Process 3 at a time to avoid overwhelming the system
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                batch_tasks = [self._answer_question_fast(q, vector_store) for q in batch]
                
                try:
                    batch_answers = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=30  # 30 seconds per batch
                    )
                    
                    for ans in batch_answers:
                        if isinstance(ans, Exception):
                            logger.error(f"Question processing error: {ans}")
                            answers.append("Error processing this question.")
                        else:
                            answers.append(str(ans))
                            
                except asyncio.TimeoutError:
                    logger.error(f"Batch timeout for questions {i}-{i+batch_size}")
                    answers.extend(["Processing timeout."] * len(batch))
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions in {processing_time:.2f} seconds")
            
            return answers
            
        except Exception as e:
            logger.error(f"Critical processing error: {e}")
            return ["Error processing query."] * len(questions)

# For backward compatibility, create alias
AccuracyFirstRAGPipeline = FastAccurateRAGPipeline