# app/core/rag_pipeline.py - ACCURACY-FOCUSED VERSION
import io
import re
import os
import logging
import requests
import asyncio
import email
from typing import List, Tuple, Dict
import time

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


class EnhancedVectorStore:
    """ACCURACY-FOCUSED vector store with multiple search strategies"""
    def __init__(self, chunks: List[str], embeddings: np.ndarray, model: SentenceTransformer):
        self.chunks = chunks
        self.model = model
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        # Create keyword index for numerical/factual queries
        self.keyword_index = self._build_keyword_index(chunks)
        logger.info(f"Built enhanced FAISS index for {len(chunks)} chunks with keyword support.")

    def _build_keyword_index(self, chunks: List[str]) -> Dict[str, List[int]]:
        """Build keyword index for better factual retrieval"""
        keyword_index = {}
        for i, chunk in enumerate(chunks):
            # Extract numbers, percentages, years, monetary amounts
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|million|billion|thousand)?\b', chunk.lower())
            years = re.findall(r'\b(19|20)\d{2}\b', chunk)
            amounts = re.findall(r'\$\d+(?:,\d{3})*(?:\.\d+)?(?:million|billion|k)?\b', chunk.lower())
            
            for term in numbers + years + amounts:
                if term not in keyword_index:
                    keyword_index[term] = []
                keyword_index[term].append(i)
        
        return keyword_index

    def enhanced_search(self, query: str, k: int = 25) -> List[Tuple[str, float]]:
        """Multi-strategy search combining vector, keyword, and semantic approaches"""
        # Strategy 1: Vector similarity search
        query_embedding = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_embedding, min(k, len(self.chunks)))
        
        vector_results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                vector_results.append((self.chunks[idx], similarity, 'vector'))
        
        # Strategy 2: Keyword matching for numerical queries
        keyword_results = []
        query_lower = query.lower()
        query_numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:%|million|billion|thousand)?\b', query_lower)
        query_years = re.findall(r'\b(19|20)\d{2}\b', query)
        
        for term in query_numbers + query_years:
            if term in self.keyword_index:
                for chunk_idx in self.keyword_index[term]:
                    chunk = self.chunks[chunk_idx]
                    keyword_results.append((chunk, 0.9, 'keyword'))  # High confidence for exact matches
        
        # Strategy 3: Fuzzy keyword matching
        fuzzy_results = []
        important_terms = ['million', 'billion', 'percent', '%', 'by 2025', 'by 2030', 'target', 'goal', 'aims']
        for term in important_terms:
            if term in query_lower:
                for i, chunk in enumerate(self.chunks):
                    if term in chunk.lower():
                        fuzzy_results.append((chunk, 0.7, 'fuzzy'))
        
        # Combine and deduplicate results
        all_results = vector_results + keyword_results + fuzzy_results
        seen_chunks = set()
        unique_results = []
        
        for chunk, score, strategy in all_results:
            if chunk not in seen_chunks:
                seen_chunks.add(chunk)
                unique_results.append((chunk, score))
        
        # Sort by score and return top k
        unique_results.sort(key=lambda x: x[1], reverse=True)
        return unique_results[:k]


class AccuracyFirstRAGPipeline:
    """ACCURACY-FOCUSED RAG pipeline optimized for high precision retrieval"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Google Generative AI client configured successfully.")
        except Exception as e:
            logger.critical(f"Failed to configure Google Generative AI: {e}")
            raise RuntimeError("Google API Key is not configured correctly.")
        
        # Use more accurate cross-encoder
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
        logger.info("Enhanced CrossEncoder model for re-ranking loaded.")

    def _download_and_parse_document_enhanced(self, url: str) -> str:
        """Enhanced document parsing with better structure preservation"""
        logger.info(f"Processing document with enhanced parsing from {url}")
        
        try:
            temp_file = None
            file_extension = ""
            
            if url.startswith("gs://"):
                try:
                    storage_client = storage.Client()
                    bucket_name, blob_name = url.replace("gs://", "").split("/", 1)
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    
                    if not blob.exists():
                        raise ValueError(f"File not found in Google Cloud Storage: {url}")
                    
                    temp_file = io.BytesIO(blob.download_as_bytes())
                    file_extension = os.path.splitext(blob_name)[1].lower()
                    logger.info(f"Successfully downloaded from GCS: {blob_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to download from Google Cloud Storage: {e}")
                    raise ValueError(f"Could not access Google Cloud Storage file: {str(e)}")
                    
            elif url.startswith(("http://", "https://")):
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                        'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Connection': 'keep-alive'
                    }
                    
                    logger.info(f"Downloading from URL: {url}")
                    response = requests.get(url, headers=headers, timeout=45, stream=True, allow_redirects=True)
                    response.raise_for_status()
                    
                    content = b""
                    max_size = 100 * 1024 * 1024  # 100MB
                    size = 0
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            size += len(chunk)
                            if size > max_size:
                                raise ValueError(f"File too large (>{max_size/1024/1024:.1f}MB)")
                            content += chunk
                    
                    if not content:
                        raise ValueError("Empty response received from server")
                    
                    temp_file = io.BytesIO(content)
                    url_path = url.split('?')[0].split('#')[0]
                    file_extension = os.path.splitext(url_path)[1].lower()
                    
                    if not file_extension:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'pdf' in content_type:
                            file_extension = '.pdf'
                        elif 'wordprocessingml' in content_type:
                            file_extension = '.docx'
                        else:
                            file_extension = '.pdf'  # Default assumption
                    
                    logger.info(f"Downloaded {len(content)} bytes, extension: {file_extension}")
                    
                except Exception as e:
                    raise ValueError(f"Error downloading from URL: {str(e)}")
            else:
                raise ValueError("Invalid document URL scheme")

            # Enhanced parsing with structure preservation
            full_text = ""
            temp_file.seek(0)
            
            if file_extension == '.pdf':
                try:
                    with pdfplumber.open(temp_file) as pdf:
                        if len(pdf.pages) == 0:
                            raise ValueError("PDF file contains no pages")
                        
                        logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                # Add page marker for better context
                                full_text += f"\n\n--- PAGE {page_num + 1} ---\n"
                                
                                # Extract text with better formatting
                                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                                if page_text and page_text.strip(): 
                                    full_text += page_text + "\n"
                                
                                # Extract tables with better formatting
                                tables = page.extract_tables()
                                for table in tables:
                                    if table and len(table) > 0:
                                        full_text += "\n=== TABLE START ===\n"
                                        for row in table:
                                            if row:
                                                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                                                full_text += row_text + "\n"
                                        full_text += "=== TABLE END ===\n"
                                        
                            except Exception as e:
                                logger.warning(f"Error processing PDF page {page_num}: {e}")
                                continue
                                
                except Exception as e:
                    raise ValueError(f"Error parsing PDF file: {str(e)}")
                    
            elif file_extension == '.docx':
                try:
                    doc = Document(temp_file)
                    paragraphs = []
                    for para in doc.paragraphs:
                        if para.text and para.text.strip():
                            paragraphs.append(para.text)
                    
                    if not paragraphs:
                        raise ValueError("DOCX file contains no readable text")
                    
                    full_text = "\n".join(paragraphs)
                    logger.info(f"Extracted {len(paragraphs)} paragraphs from DOCX")
                except Exception as e:
                    raise ValueError(f"Error parsing DOCX file: {str(e)}")
            
            # [ODT and EML parsing remains the same as your original]
            
            # Validate extracted text
            if not full_text or not full_text.strip():
                raise ValueError("Document appears to be empty")
            
            # Enhanced text cleaning - preserve structure better
            full_text = re.sub(r'\n\s*\n\s*\n', '\n\n', full_text)  # Reduce multiple newlines
            full_text = re.sub(r'[ \t]+', ' ', full_text)  # Normalize spaces
            full_text = full_text.strip()
            
            if len(full_text) < 50:
                raise ValueError("Extracted text is too short")
            
            logger.info(f"Successfully extracted {len(full_text)} characters with enhanced structure")
            return full_text
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            raise ValueError(f"Could not process document: {str(e)}")

    def _chunk_text_for_accuracy(self, text: str) -> List[str]:
        """ACCURACY-FOCUSED chunking with larger chunks and better overlap"""
        
        # Split into paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        max_chunk_size = 1200  # Larger chunks for better context
        overlap_size = 300     # Substantial overlap
        
        for paragraph in paragraphs:
            # If adding this paragraph would exceed max size
            if len(current_chunk) + len(paragraph) > max_chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Create overlap from end of current chunk
                if len(current_chunk) > overlap_size:
                    overlap_text = current_chunk[-overlap_size:]
                    # Find a good break point
                    break_point = overlap_text.find('. ')
                    if break_point > 50:  # If we find a sentence break
                        overlap_text = overlap_text[break_point + 2:]
                    current_chunk = overlap_text + " " + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        # Filter very short chunks and split very long ones
        final_chunks = []
        for chunk in chunks:
            if len(chunk) < 100:  # Skip very short chunks
                continue
            elif len(chunk) > 2000:  # Split very long chunks
                # Split by sentences
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                temp_chunk = ""
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > 1200 and temp_chunk:
                        final_chunks.append(temp_chunk.strip())
                        temp_chunk = sentence
                    else:
                        temp_chunk += " " + sentence if temp_chunk else sentence
                if temp_chunk.strip():
                    final_chunks.append(temp_chunk.strip())
            else:
                final_chunks.append(chunk)
        
        logger.info(f"Created {len(final_chunks)} accuracy-focused chunks")
        return final_chunks

    async def get_or_create_enhanced_vector_store(self, url: str) -> EnhancedVectorStore:
        """Create enhanced vector store with multiple search strategies"""
        from app.core.cache import cache
        cached_store = await cache.get(url)
        if cached_store:
            logger.info(f"Found cached enhanced vector store for {url}")
            return cached_store
            
        logger.info(f"Creating enhanced vector store for {url}...")
        text = self._download_and_parse_document_enhanced(url)
        
        if not text:
            raise ValueError("Document is empty or could not be processed.")
            
        chunks = self._chunk_text_for_accuracy(text)
        if not chunks:
            raise ValueError("Could not extract meaningful text chunks.")
            
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).astype('float32')
        vector_store = EnhancedVectorStore(chunks, embeddings, self.embedding_model)
        
        await cache.set(url, vector_store)
        return vector_store

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=(retry_if_exception_type(google_exceptions.ResourceExhausted))
    )
    async def _generate_accurate_answer(self, question: str, context: List[str]) -> str:
        """ACCURACY-FOCUSED answer generation with detailed prompting"""
        context_str = "\n\n---CONTEXT SECTION---\n\n".join(context)
        
        # Enhanced prompt focused on finding information
        prompt = f"""You are an expert document analyst specializing in extracting precise information from complex documents.

**CRITICAL INSTRUCTION**: Your primary goal is to find and extract the requested information from the provided context. Be thorough and look for:
- Direct statements that answer the question
- Numerical data, percentages, targets, and goals
- Implied information that can be reasonably inferred
- Partial information that relates to the question

**SEARCH STRATEGY**:
1. Look for exact matches to the question topic
2. Search for related terms and synonyms
3. Check tables, lists, and structured data
4. Look for numerical patterns and targets

**RESPONSE GUIDELINES**:
- Provide specific, factual answers with numbers when available
- If you find partial information, provide what you can find
- Only say "information not available" if you truly cannot find ANY related content
- Include relevant context and details that support your answer
- For numerical questions, provide the specific numbers found

**DOCUMENT CONTEXT**:
{context_str}

**QUESTION**: {question}

**DETAILED ANSWER** (Be thorough and specific):"""

        try:
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Low temperature for factual accuracy
                    max_output_tokens=500,
                    top_p=0.8
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return "An error occurred while generating the answer."

    async def _answer_question_with_enhanced_retrieval(self, question: str, vector_store: EnhancedVectorStore) -> str:
        """Enhanced question answering with multiple retrieval strategies"""
        # Step 1: Enhanced retrieval with multiple strategies
        retrieved_results = vector_store.enhanced_search(question, k=25)
        
        if not retrieved_results:
            return "Based on the provided documents, the information to answer this question is not available."

        # Step 2: Advanced re-ranking
        chunks_with_scores = [(chunk, score) for chunk, score in retrieved_results]
        
        # Cross-encoder re-ranking
        loop = asyncio.get_running_loop()
        chunks = [chunk for chunk, _ in chunks_with_scores]
        pairs = [[question, chunk] for chunk in chunks]
        
        try:
            cross_scores = await loop.run_in_executor(None, self.cross_encoder.predict, pairs)
            
            # Combine scores (weighted combination)
            final_scores = []
            for i, ((chunk, initial_score), cross_score) in enumerate(zip(chunks_with_scores, cross_scores)):
                # Weighted combination: 40% initial similarity + 60% cross-encoder
                combined_score = 0.4 * initial_score + 0.6 * float(cross_score)
                final_scores.append((combined_score, chunk))
            
            # Sort by combined score and take top chunks
            final_scores.sort(reverse=True)
            top_chunks = [chunk for score, chunk in final_scores[:8]]  # More context
            
        except Exception as e:
            logger.warning(f"Cross-encoder failed, using initial ranking: {e}")
            top_chunks = [chunk for chunk, _ in chunks_with_scores[:8]]

        # Step 3: Generate answer with enhanced prompt
        return await self._generate_accurate_answer(question, top_chunks)

    # async def process_query_for_accuracy(self, document_url: str, questions: List[str]) -> List[str]:
    #     """Process queries with focus on maximum accuracy"""
    #     start_time = time.time()
        
    #     try:
    #         vector_store = await self.get_or_create_enhanced_vector_store(document_url)
            
    #         # Process questions with longer timeout for accuracy
    #         tasks = []
    #         for question in questions:
    #             task = self._answer_question_with_enhanced_retrieval(question, vector_store)
    #             tasks.append(task)
            
    #         # Wait for all answers with extended timeout
    #         answers = await asyncio.wait_for(
    #             asyncio.gather(*tasks, return_exceptions=True), 
    #             timeout=len(questions) * 15  # 15 seconds per question
    #         )
            
    #         # Process results
    #         final_answers = []
    #         for i, ans in enumerate(answers):
    #             if isinstance(ans, Exception):
    #                 logger.error(f"Error processing question {i}: {ans}")
    #                 final_answers.append("Error processing this question.")
    #             else:
    #                 final_answers.append(str(ans))
            
    #         processing_time = time.time() - start_time
    #         logger.info(f"Processed {len(questions)} questions in {processing_time:.2f} seconds")
            
    #         return final_answers
            
    #     except Exception as e:
    #         logger.error(f"Critical error in accuracy processing: {e}")
    #         return ["Error processing query."] * len(questions)
    # app/core/rag_pipeline.py

    async def process_query_for_accuracy(self, query: str) -> str:
        """
        Processes a query with a focus on accuracy, using a RAG pipeline.

        Args:
            query: The user's query.

        Returns:
            The processed response.
        """
        try:
            logger.info(f"Processing query for accuracy: {query}")
            
            retrieved_docs = await self.vector_store.asimilarity_search(query)
            
            if not retrieved_docs:
                logger.warning("No documents retrieved for the query.")
                return "No relevant information found."
            
            response = f"Retrieved {len(retrieved_docs)} documents. The most relevant is: {retrieved_docs[0].page_content}"
            
            return response
        except Exception as e:
            # Log the full, detailed error message.
            logger.critical(f"A critical, unhandled error occurred in the accuracy processing pipeline: {e}", exc_info=True)
            # Re-raise the exception to get a full traceback for debugging.
            raise

    # Keep your existing process_query method for backward compatibility
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Backward compatibility - redirect to accuracy-focused version"""
        return await self.process_query_for_accuracy(document_url, questions)