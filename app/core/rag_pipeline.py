# app/core/rag_pipeline.py
import io
import re
import os
import logging
import requests
import asyncio
import email
from typing import List

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


class VectorStore:
    def __init__(self, chunks: List[str], embeddings: np.ndarray, model: SentenceTransformer):
        self.chunks = chunks
        self.model = model
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        logger.info(f"Built FAISS index for {len(chunks)} chunks.")

    def search(self, query: str, k: int = 12) -> List[str]:
        query_embedding = self.model.encode([query]).astype('float32')
        _, indices = self.index.search(query_embedding, k)
        return [self.chunks[i] for i in indices[0]]


class RAGPipeline:
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Google Generative AI client configured successfully.")
        except Exception as e:
            logger.critical(f"Failed to configure Google Generative AI: {e}")
            raise RuntimeError("Google API Key is not configured correctly.")
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        logger.info("CrossEncoder model for re-ranking loaded.") 
        


    
        

    def _download_and_parse_document(self, url: str) -> str:
        """
        Downloads and parses a document from a URL, supporting PDF, DOCX, ODT, EML,
        and secure gs:// paths with robust error handling.
        """
        logger.info(f"Processing document from {url}")
        
        try:
            temp_file = None
            file_extension = ""
            
            if url.startswith("gs://"):
                # Google Cloud Storage handling
                try:
                    storage_client = storage.Client()
                    bucket_name, blob_name = url.replace("gs://", "").split("/", 1)
                    bucket = storage_client.bucket(bucket_name)
                    blob = bucket.blob(blob_name)
                    
                    # Check if blob exists
                    if not blob.exists():
                        raise ValueError(f"File not found in Google Cloud Storage: {url}")
                    
                    temp_file = io.BytesIO(blob.download_as_bytes())
                    file_extension = os.path.splitext(blob_name)[1].lower()
                    logger.info(f"Successfully downloaded from GCS: {blob_name}")
                    
                except Exception as e:
                    logger.error(f"Failed to download from Google Cloud Storage: {e}")
                    raise ValueError(f"Could not access Google Cloud Storage file: {str(e)}")
                    
            elif url.startswith(("http://", "https://")):
                # HTTP/HTTPS handling with improved robustness
                try:
                    headers = {
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                        'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,application/vnd.oasis.opendocument.text,message/rfc822,*/*',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Accept-Encoding': 'gzip, deflate, br',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1',
                        'Sec-Fetch-Dest': 'document',
                        'Sec-Fetch-Mode': 'navigate',
                        'Sec-Fetch-Site': 'none'
                    }
                    
                    # Make the request with retries
                    max_retries = 3
                    for attempt in range(max_retries):
                        try:
                            logger.info(f"Downloading from URL (attempt {attempt + 1}/{max_retries}): {url}")
                            response = requests.get(
                                url, 
                                headers=headers, 
                                timeout=60, 
                                stream=True, 
                                allow_redirects=True,
                                verify=True  # Verify SSL certificates
                            )
                            response.raise_for_status()
                            break
                        except requests.exceptions.Timeout:
                            if attempt == max_retries - 1:
                                raise ValueError(f"Request timed out after {max_retries} attempts: {url}")
                            logger.warning(f"Timeout on attempt {attempt + 1}, retrying...")
                            continue
                        except requests.exceptions.ConnectionError as e:
                            if attempt == max_retries - 1:
                                raise ValueError(f"Connection error after {max_retries} attempts: {url} - {str(e)}")
                            logger.warning(f"Connection error on attempt {attempt + 1}, retrying...")
                            continue
                        except requests.exceptions.HTTPError as e:
                            if e.response.status_code in [429, 503, 502, 504]:  # Retryable errors
                                if attempt == max_retries - 1:
                                    raise ValueError(f"HTTP error {e.response.status_code} after {max_retries} attempts: {url}")
                                logger.warning(f"HTTP error {e.response.status_code} on attempt {attempt + 1}, retrying...")
                                continue
                            else:
                                raise ValueError(f"HTTP error {e.response.status_code}: {url}")
                    
                    # Check response content type
                    content_type = response.headers.get('content-type', '').lower()
                    logger.info(f"Response Content-Type: {content_type}, Status: {response.status_code}")
                    
                    # Verify we got actual content, not HTML error page
                    if 'text/html' in content_type and 'pdf' not in content_type and len(response.content) < 10000:
                        # Likely an error page, not the actual document
                        logger.warning(f"Received HTML content instead of document. URL might be incorrect or require authentication: {url}")
                    
                    # Read content with size limit (100MB max)
                    max_size = 100 * 1024 * 1024  # 100MB
                    content = b""
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
                    
                    # Determine file extension from URL, removing query parameters
                    url_path = url.split('?')[0].split('#')[0]
                    file_extension = os.path.splitext(url_path)[1].lower()
                    
                    # If no extension in URL, try to determine from content-type
                    if not file_extension:
                        if 'pdf' in content_type:
                            file_extension = '.pdf'
                        elif 'wordprocessingml' in content_type or 'msword' in content_type:
                            file_extension = '.docx'
                        elif 'opendocument.text' in content_type:
                            file_extension = '.odt'
                        elif 'message' in content_type or 'rfc822' in content_type:
                            file_extension = '.eml'
                        else:
                            # Try to detect from content magic bytes
                            content_start = content[:10]
                            if content_start.startswith(b'%PDF'):
                                file_extension = '.pdf'
                            elif content_start.startswith(b'PK'):  # ZIP-based formats (DOCX, ODT)
                                # Default to .docx for ZIP-based files
                                file_extension = '.docx'
                            else:
                                logger.warning(f"Could not determine file type from URL or content-type: {url}")
                                file_extension = '.pdf'  # Default assumption
                    
                    logger.info(f"Successfully downloaded {len(content)} bytes, detected extension: {file_extension}")
                    
                except requests.exceptions.RequestException as e:
                    raise ValueError(f"Request error while downloading from: {url}: {str(e)}")
                except Exception as e:
                    raise ValueError(f"Unexpected error downloading from: {url}: {str(e)}")
            else:
                raise ValueError("Invalid document URL scheme. Must be http, https, or gs://")

            if not temp_file:
                raise ValueError("Failed to create temporary file from downloaded content")

            logger.info(f"Processing file with extension: {file_extension}")

            # Parse the document based on file type
            full_text = ""
            temp_file.seek(0)  # Reset file pointer
            
            try:
                if file_extension == '.pdf':
                    try:
                        with pdfplumber.open(temp_file) as pdf:
                            if len(pdf.pages) == 0:
                                raise ValueError("PDF file contains no pages")
                            
                            logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                            for page_num, page in enumerate(pdf.pages):
                                try:
                                    page_text = page.extract_text(x_tolerance=3)
                                    if page_text and page_text.strip(): 
                                        full_text += page_text + "\n"
                                    
                                    # Extract tables
                                    tables = page.extract_tables()
                                    for table in tables:
                                        if table and len(table) > 0:
                                            table_text = "\n".join([
                                                "\t".join(map(str, (c if c is not None else "" for c in r))) 
                                                for r in table if r
                                            ])
                                            if table_text.strip():
                                                full_text += "\n--- TABLE ---\n" + table_text + "\n--- END TABLE ---\n"
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
                        
                elif file_extension == '.odt':
                    try:
                        doc = load(temp_file)
                        paragraphs = doc.getElementsByType(P)
                        if not paragraphs:
                            raise ValueError("ODT file contains no readable paragraphs")
                        
                        text_parts = []
                        for p in paragraphs:
                            text_content = " ".join(
                                node.data for node in p.childNodes 
                                if node.nodeType == node.TEXT_NODE and node.data
                            )
                            if text_content.strip():
                                text_parts.append(text_content)
                        
                        if not text_parts:
                            raise ValueError("ODT file contains no extractable text")
                        
                        full_text = "\n".join(text_parts)
                        logger.info(f"Extracted text from {len(text_parts)} ODT paragraphs")
                    except Exception as e:
                        raise ValueError(f"Error parsing ODT file: {str(e)}")
                        
                elif file_extension == '.eml':
                    try:
                        temp_file.seek(0)
                        content = temp_file.read()
                        msg = email.message_from_bytes(content)
                        
                        # Extract headers safely
                        subject = msg.get('subject', 'N/A') or 'N/A'
                        from_addr = msg.get('from', 'N/A') or 'N/A'
                        to_addr = msg.get('to', 'N/A') or 'N/A'
                        date = msg.get('date', 'N/A') or 'N/A'
                        
                        full_text += f"Subject: {subject}\nFrom: {from_addr}\nTo: {to_addr}\nDate: {date}\n\n"
                        
                        # Extract body
                        body_parts = []
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    try:
                                        payload = part.get_payload(decode=True)
                                        if payload:
                                            decoded_text = payload.decode('utf-8', errors='ignore')
                                            if decoded_text.strip():
                                                body_parts.append(decoded_text)
                                    except Exception as e:
                                        logger.warning(f"Error decoding email part: {e}")
                        else:
                            try:
                                payload = msg.get_payload(decode=True)
                                if payload:
                                    decoded_text = payload.decode('utf-8', errors='ignore')
                                    if decoded_text.strip():
                                        body_parts.append(decoded_text)
                            except Exception as e:
                                logger.warning(f"Error decoding email payload: {e}")
                        
                        if body_parts:
                            full_text += "\n".join(body_parts)
                        
                        logger.info("Successfully extracted EML content")
                    except Exception as e:
                        raise ValueError(f"Error parsing EML file: {str(e)}")
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
            
            except ValueError:
                # Re-raise ValueError as-is
                raise
            except Exception as e:
                logger.error(f"Error during document parsing: {e}")
                raise ValueError(f"Could not parse document: {str(e)}")

            # Validate extracted text
            if not full_text or not full_text.strip():
                logger.error(f"No text extracted from document at {url}")
                raise ValueError("Document appears to be empty or contains no extractable text")
            
            # Basic text cleaning
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
            # Final validation
            if len(full_text) < 10:
                raise ValueError("Extracted text is too short to be meaningful")
            
            logger.info(f"Successfully extracted {len(full_text)} characters from document")
            return full_text
            
        except ValueError:
            # Re-raise ValueError as-is
            raise
        except Exception as e:
            logger.error(f"Unexpected error processing document from {url}: {e}", exc_info=True)
            raise ValueError(f"Could not retrieve or parse document from URL: {str(e)}")
    def _chunk_text(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text).strip()
        chunk_size = 1000
        overlap = 200
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]
        final_chunks = [chunk for chunk in chunks if len(chunk) > 100]
        logger.info(f"Chunked document into {len(final_chunks)} overlapping parts.")
        return final_chunks

    async def get_or_create_vector_store(self, url: str) -> VectorStore:
        from app.core.cache import cache
        cached_store = await cache.get(url)
        if cached_store:
            logger.info(f"Found cached vector store for {url}")
            return cached_store
        logger.info(f"No cache found. Processing document for {url}...")
        text = self._download_and_parse_document(url) # Updated function call
        if not text:
            raise ValueError("Document is empty or could not be processed.")
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Could not extract meaningful text chunks from the document.")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).astype('float32')
        vector_store = VectorStore(chunks, embeddings, self.embedding_model)
        await cache.set(url, vector_store)
        return vector_store
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=(retry_if_exception_type(google_exceptions.ResourceExhausted) | retry_if_exception_type(google_exceptions.InternalServerError))
    )
    async def _generate_answer(self, question: str, context: List[str]) -> str:
        context_str = "\n\n---\n\n".join(context)
        prompt = f"""
    You are a specialized AI Analyst with expertise in dissecting complex legal, insurance, HR, and compliance documents. Your responses must be precise, objective, and grounded exclusively in the provided text.

    **Core Directive:** Answer the user's question using ONLY the information available in the 'CONTEXT' section below.

    **Strict Constraints:**
    - Do NOT use any external or prior knowledge.
    - Do NOT make assumptions, inferences, or interpretations that are not explicitly supported by the text.
    - If the context does not contain the answer, you are required to state: "Based on the provided documents, the information to answer this question is not available." Do not attempt to answer.

    **Output Format:**
    - Your answer must be a direct and factual response.
    - Extract the key information and present it clearly and concisely.

    ---
    **CONTEXT:**
    {context_str}
    ---

    **QUESTION:**
    {question}

    **PRECISE ANSWER:**
    """
        try:
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            response = await model.generate_content_async(prompt)
            return response.text.strip()
        except google_exceptions.ResourceExhausted as e:
            logger.warning(f"Rate limit hit for question '{question}'. Retrying...")
            raise e
        except Exception as e:
            logger.error(f"Google Gemini generation failed for question '{question}': {e}")
            return "An error occurred while generating the answer using the Gemini API."

    # app/core/rag_pipeline.py
    # Add this new method inside the RAGPipeline class

    async def _generate_hypothetical_answer(self, question: str) -> str:
        """
        Generates a concise, hypothetical answer to a question to improve retrieval accuracy (HyDE).
        This answer is used for searching and is then discarded.
        """
        hyde_prompt = f"""You are a helpful assistant. Generate a plausible, one-sentence, hypothetical answer to the following user question. Do not say you don't know the answer.

QUESTION: {question}

HYPOTHETICAL ANSWER:"""
        
        try:
            # Use a fast model for this task as it doesn't need to be perfect.
            model = genai.GenerativeModel("gemini-1.5-flash-001")
            response = await model.generate_content_async(hyde_prompt)
            hypothetical_answer = response.text.strip()
            logger.info(f"Generated HyDE answer for '{question}': '{hypothetical_answer}'")
            return hypothetical_answer
        except Exception as e:
            logger.error(f"HyDE answer generation failed for question '{question}': {e}. Falling back to original question for search.")
            # If HyDE fails, we can safely fall back to using the original question.
            return question

    # --- REPLACEMENT FOR process_query ---

    # Replace your existing _answer_question_with_rerank method with this one

    async def _answer_question_with_rerank(self, question: str, vector_store: VectorStore) -> str:
        """
        Answers a question using a full retrieve-and-rerank strategy, enhanced with HyDE for better retrieval.
        """
        # --- HYDE IMPLEMENTATION ---
        # 1. Generate a hypothetical answer to use for semantic search.
        hypothetical_answer = await self._generate_hypothetical_answer(question)
        # --- END HYDE ---

        # 2. RETRIEVE: Use the hypothetical answer to find semantically similar chunks.
        retrieved_chunks = vector_store.search(hypothetical_answer, k=20)
        
        if not retrieved_chunks:
            logger.warning(f"No relevant chunks found for question: '{question}'")
            return "Based on the provided documents, the information to answer this question is not available."

        # 3. RE-RANK: Use the original, precise question with the CrossEncoder for high-accuracy re-ranking.
        loop = asyncio.get_running_loop()
        pairs = [[question, chunk] for chunk in retrieved_chunks]
        scores = await loop.run_in_executor(None, self.cross_encoder.predict, pairs)
        
        reranked_chunks = sorted(zip(scores, retrieved_chunks), reverse=True)
        
        # 4. SELECT: Take the top 4 most relevant chunks to create a clean, focused context.
        top_chunks = [chunk for score, chunk in reranked_chunks[:4]]

        # 5. GENERATE: Pass the high-quality context and original question to the powerful LLM for the final, factual answer.
        return await self._generate_answer(question, top_chunks)

    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Processes a list of questions concurrently using the retrieve-and-rerank strategy.
        """
        vector_store = await self.get_or_create_vector_store(document_url)
        
        # Create a list of tasks, where each task will answer one question using the re-ranking logic.
        tasks = [self._answer_question_with_rerank(question, vector_store) for question in questions]
        
        # Run all tasks in parallel and gather the results.
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert all results to string, handling any potential errors gracefully.
        return [str(ans) if not isinstance(ans, Exception) else "Error processing question." for ans in answers]

    # --- END REPLACEMENT ---