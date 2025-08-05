# app/core/rag_pipeline.py - Enhanced RAG Pipeline with Improved Parsing and Error Handling
import io
import re
import os
import logging
import requests
import asyncio
import email
import time
from typing import List, Tuple, Dict, Set, Optional
import json

# Imports for multi-format support
import pdfplumber
from docx import Document
from odf.text import P
from odf.opendocument import load

# Additional PDF parsing libraries
try:
    import PyPDF2
except ImportError:
    PyPDF2 = None
    logging.warning("PyPDF2 not available - using pdfplumber only")

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
    """Enhanced RAG pipeline with improved error handling and logging"""
    
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
        """Enhanced document parsing with better error handling and multiple fallbacks"""
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
                    response = requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True)
                    response.raise_for_status()
                    
                    content = b""
                    max_size = 100 * 1024 * 1024  # 100MB limit
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            content += chunk
                            if len(content) > max_size:
                                logger.warning(f"Document exceeds {max_size} bytes, truncating")
                                break
                    
                    temp_file = io.BytesIO(content)
                    file_extension = os.path.splitext(url.split('?')[0])[1].lower()
                    
                    if not file_extension and 'pdf' in response.headers.get('content-type', ''):
                        file_extension = '.pdf'
                    
                    logger.info(f"Downloaded {len(content)} bytes, extension: {file_extension}")
                    
                except Exception as e:
                    raise ValueError(f"Error downloading: {str(e)}")
            else:
                raise ValueError("Only HTTP/HTTPS URLs supported")

            # Enhanced PDF parsing with multiple strategies
            full_text = ""
            temp_file.seek(0)
            
            if file_extension == '.pdf':
                # Try multiple PDF parsing strategies
                full_text = self._parse_pdf_with_fallbacks(temp_file, url)
            elif file_extension in ['.docx', '.doc']:
                full_text = self._parse_docx(temp_file)
            elif file_extension == '.odt':
                full_text = self._parse_odt(temp_file)
            else:
                # Try PDF parsing as default
                logger.warning(f"Unknown extension {file_extension}, attempting PDF parsing")
                full_text = self._parse_pdf_with_fallbacks(temp_file, url)
            
            # Validate and clean
            if not full_text or len(full_text) < 50:
                raise ValueError("Document too short or empty")
            
            # Enhanced cleaning
            full_text = self._clean_text(full_text)
            
            logger.info(f"Successfully extracted {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise ValueError(f"Could not process document: {str(e)}")

    def _parse_pdf_with_fallbacks(self, temp_file: io.BytesIO, url: str) -> str:
        """Try multiple PDF parsing strategies with fallbacks"""
        full_text = ""
        
        # Strategy 1: pdfplumber with enhanced settings
        try:
            temp_file.seek(0)
            with pdfplumber.open(temp_file) as pdf:
                logger.info(f"Processing PDF with {len(pdf.pages)} pages using pdfplumber")
                
                for page_num, page in enumerate(pdf.pages):
                    # if page_num > 200:  # Reasonable limit
                    #     logger.warning(f"Stopping at page {page_num} to prevent timeout")
                    #     break
                    
                    try:
                        # Try with layout preservation for complex documents
                        page_text = page.extract_text(
                            x_tolerance=3,
                            y_tolerance=3,
                            layout=False,  # Disable layout for speed
                            x_density=7.25,
                            y_density=13
                        )
                        
                        if page_text and page_text.strip():
                            full_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                        
                        # Extract tables
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables[:5]):  # Limit tables per page
                            if table:
                                table_text = self._format_table(table)
                                if table_text.strip():
                                    full_text += f"\n=== TABLE {table_idx + 1} (Page {page_num + 1}) ===\n{table_text}\n"
                                    
                    except Exception as e:
                        logger.warning(f"Error on page {page_num} with pdfplumber: {e}")
                        continue
                
                if len(full_text) > 1000:
                    logger.info("Successfully parsed with pdfplumber")
                    return full_text
                    
        except Exception as e:
            logger.warning(f"pdfplumber parsing failed: {e}")
        
        # Strategy 2: PyPDF2 fallback
        if PyPDF2:
            try:
                temp_file.seek(0)
                logger.info("Attempting PyPDF2 fallback parsing")
                pdf_reader = PyPDF2.PdfReader(temp_file)
                
                for page_num in range(min(len(pdf_reader.pages), 200)):
                    try:
                        page = pdf_reader.pages[page_num]
                        text = page.extract_text()
                        if text and text.strip():
                            full_text += f"\n--- PAGE {page_num + 1} ---\n{text}\n"
                    except Exception as e:
                        logger.warning(f"PyPDF2 error on page {page_num}: {e}")
                        continue
                
                if len(full_text) > 1000:
                    logger.info("Successfully parsed with PyPDF2")
                    return full_text
                    
            except Exception as e:
                logger.warning(f"PyPDF2 parsing failed: {e}")
        
        # Strategy 3: Re-attempt pdfplumber with minimal settings
        try:
            temp_file.seek(0)
            logger.info("Final attempt with minimal pdfplumber settings")
            with pdfplumber.open(temp_file) as pdf:
                for page_num, page in enumerate(pdf.pages[:100]):
                    try:
                        # Minimal extraction
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n"
                    except:
                        continue
                        
        except Exception as e:
            logger.error(f"All PDF parsing strategies failed: {e}")
        
        if not full_text:
            raise ValueError("All PDF parsing methods failed")
            
        return full_text

    def _format_table(self, table: List[List]) -> str:
        """Format table data into readable text"""
        if not table:
            return ""
        
        formatted_rows = []
        for row in table:
            if row and any(cell for cell in row if cell):
                formatted_row = " | ".join([str(cell) if cell else "" for cell in row])
                formatted_rows.append(formatted_row)
        
        return "\n".join(formatted_rows)

    def _parse_docx(self, temp_file: io.BytesIO) -> str:
        """Parse DOCX files"""
        try:
            doc = Document(temp_file)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            # Also extract tables
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
            raise

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
            raise

    def _clean_text(self, text: str) -> str:
        """Enhanced text cleaning"""
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix common OCR issues
        text = text.replace('ﬁ', 'fi')
        text = text.replace('ﬂ', 'fl')
        text = text.replace('–', '-')
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Handle encoding issues
        try:
            text = text.encode('utf-8', 'ignore').decode('utf-8')
        except:
            pass
        
        return text.strip()

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
        """Create vector store with speed optimization and error handling"""
        from app.core.cache import cache
        
        # Check cache first
        cached_store = await cache.get(url)
        if cached_store:
            logger.info(f"Using cached vector store for {url}")
            return cached_store
        
        logger.info(f"Creating fast vector store for {url}...")
        
        try:
            text = self._download_and_parse_document(url)
            chunks = self._fast_chunk_text(text)
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Fast embedding generation
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            embeddings = self.embedding_model.encode(chunks, show_progress_bar=False, batch_size=32).astype('float32')
            vector_store = FastVectorStore(chunks, embeddings, self.embedding_model)
            
            # Cache the store
            await cache.set(url, vector_store)
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=5),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted)
    )
    async def _generate_fast_answer(self, question: str, context: List[str]) -> str:
        """Enhanced answer generation with better prompts"""
        
        # Check if we have sufficient context
        if not context or all(len(c.strip()) < 50 for c in context):
            logger.warning(f"Insufficient context for question: {question[:50]}...")
            # Try to provide a more helpful response
            return await self._generate_answer_with_limited_context(question)
        
        # Use more context for better answers
        context_text = "\n\n---\n\n".join(context[:8])
        
        # Enhanced prompt based on question type
        question_lower = question.lower()
        
        # Determine instruction based on question type
        if any(word in question_lower for word in ['how many', 'percentage', 'amount', 'number', 'calculate']):
            instruction = "Focus on finding specific numbers, amounts, and quantitative data. Provide exact figures with context. If you need to calculate, show your work."
        elif any(word in question_lower for word in ['list', 'all', 'every', 'each']):
            instruction = "Provide a comprehensive list of all items requested. Even if you can't find everything, list what you can find."
        elif any(word in question_lower for word in ['does', 'is', 'are', 'can', 'will']):
            instruction = "Provide a clear yes/no answer followed by supporting details and any relevant conditions."
        else:
            instruction = "Provide a comprehensive answer with specific details from the document."
        
        prompt = f"""You are an expert document analyst. {instruction}

IMPORTANT INSTRUCTIONS:
1. Extract information directly from the context below
2. If you find relevant information, provide it clearly
3. If information is partial, share what you found and note what's missing
4. Only say "information not available" if you've thoroughly searched and found NOTHING related
5. For calculations, use any numbers you find and show your work
6. Make reasonable inferences from available data

CONTEXT:
{context_text}

QUESTION: {question}

ANSWER:"""

        try:
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=500,
                    top_p=0.9
                )
            )
            
            answer = response.text.strip()
            
            # Validate answer quality
            if len(answer) < 20 or "error" in answer.lower():
                logger.warning(f"Poor quality answer for question: {question[:50]}...")
                return await self._generate_answer_with_limited_context(question)
                
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return await self._generate_answer_with_limited_context(question)

    async def _generate_answer_with_limited_context(self, question: str) -> str:
        """Generate a helpful response when context is limited"""
        return f"Based on the available document content, I could not find specific information to answer: '{question}'. This may be due to the information not being present in the document or not being extracted properly from the source material."

    async def _answer_question_fast(self, question: str, vector_store: FastVectorStore) -> str:
        """Fast question answering with accuracy optimization"""
        
        logger.debug(f"Answering question: {question}")
        
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
        """Enhanced query processing with better error handling and logging"""
        start_time = time.time()
        
        # Log the questions being processed
        logger.info(f"Processing {len(questions)} questions for document: {document_url}")
        for i, q in enumerate(questions):
            logger.info(f"Question {i+1}: {q}")
        
        try:
            # Create vector store with retries
            vector_store = None
            for attempt in range(3):
                try:
                    vector_store = await self.get_or_create_fast_vector_store(document_url)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed for document processing: {e}")
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        # Final attempt failed
                        logger.critical(f"All attempts failed to process document: {document_url}")
                        return self._generate_error_responses(questions, "Document processing failed")
            
            if not vector_store:
                return self._generate_error_responses(questions, "Could not create vector store")
            
            # Process questions with reasonable concurrency
            answers = []
            batch_size = 3  # Process 3 at a time
            
            for i in range(0, len(questions), batch_size):
                batch = questions[i:i + batch_size]
                batch_tasks = []
                
                for q in batch:
                    task = self._answer_question_safe(q, vector_store)
                    batch_tasks.append(task)
                
                try:
                    batch_answers = await asyncio.wait_for(
                        asyncio.gather(*batch_tasks, return_exceptions=True),
                        timeout=30  # 30 seconds per batch
                    )
                    
                    for j, ans in enumerate(batch_answers):
                        if isinstance(ans, Exception):
                            logger.error(f"Question {i+j+1} processing error: {ans}")
                            answers.append(f"An error occurred while processing this question: {questions[i+j][:50]}...")
                        else:
                            answers.append(str(ans))
                            logger.debug(f"Answer {i+j+1}: {str(ans)[:100]}...")
                            
                except asyncio.TimeoutError:
                    logger.error(f"Batch timeout for questions {i+1}-{i+batch_size}")
                    for q in batch:
                        answers.append("Processing timeout. Please try again with a shorter question.")
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions in {processing_time:.2f} seconds")
            
            return answers
            
        except Exception as e:
            logger.critical(f"Critical processing error: {e}", exc_info=True)
            return self._generate_error_responses(questions, str(e))

    async def _answer_question_safe(self, question: str, vector_store: FastVectorStore) -> str:
        """Safe wrapper for question answering"""
        try:
            return await self._answer_question_fast(question, vector_store)
        except Exception as e:
            logger.error(f"Error answering question '{question[:50]}...': {e}")
            return f"Unable to process this question due to an error. Question: {question[:100]}..."

    def _generate_error_responses(self, questions: List[str], error_msg: str) -> List[str]:
        """Generate appropriate error responses"""
        logger.error(f"Generating error responses: {error_msg}")
        return [f"Unable to answer due to document processing error: {error_msg}"] * len(questions)

    async def process_query_with_explainability(self, document_url: str, questions: List[str]) -> Tuple[List[str], List[Dict], float]:
        """Process queries with explainability - for future enhancement"""
        start_time = time.time()
        
        # For now, just use the regular process_query
        simple_answers = await self.process_query(document_url, questions)
        
        # Placeholder for detailed answers
        detailed_answers = []
        for i, answer in enumerate(simple_answers):
            detailed_answers.append({
                "answer": answer,
                "confidence": 0.8,  # Placeholder
                "source_clauses": [],
                "reasoning": "Answer extracted from document using semantic search",
                "coverage_decision": None
            })
        
        processing_time = time.time() - start_time
        return simple_answers, detailed_answers, processing_time


# For backward compatibility
AccuracyFirstRAGPipeline = FastAccurateRAGPipeline