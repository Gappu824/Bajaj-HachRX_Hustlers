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

    def search(self, query: str, k: int = 8) -> List[str]:
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
        


    # Add this to your RAGPipeline class for debugging

    def debug_url_access(self, url: str) -> dict:
        """
        Debug function to test URL accessibility and gather information.
        Add this method to your RAGPipeline class for troubleshooting.
        """
        debug_info = {
            "url": url,
            "accessible": False,
            "status_code": None,
            "content_type": None,
            "content_length": None,
            "error": None,
            "redirects": []
        }
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Test with HEAD request first
            response = requests.head(url, headers=headers, timeout=30, allow_redirects=True)
            debug_info["status_code"] = response.status_code
            debug_info["content_type"] = response.headers.get('content-type')
            debug_info["content_length"] = response.headers.get('content-length')
            
            if response.history:
                debug_info["redirects"] = [r.url for r in response.history]
            
            if response.status_code == 200:
                debug_info["accessible"] = True
            
            logger.info(f"Debug info for {url}: {debug_info}")
            
        except Exception as e:
            debug_info["error"] = str(e)
            logger.error(f"Debug error for {url}: {e}")
        
        return debug_info

    # Add this test endpoint to your query.py for debugging

    @router.get("/debug/url", tags=["Debug"])
    async def debug_url(url: str, fastapi_req: Request, token: str = Depends(validate_token)):
        """Debug endpoint to test URL accessibility"""
        rag_pipeline: RAGPipeline = fastapi_req.app.state.rag_pipeline
        debug_info = rag_pipeline.debug_url_access(url)
        return debug_info    

    def _download_and_parse_document(self, url: str) -> str:
        """
        Downloads and parses a document from a URL, supporting PDF, DOCX, ODT, and EML formats.
        Improved version with better error handling and robustness.
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
                        'Accept-Encoding': 'gzip, deflate',
                        'Connection': 'keep-alive',
                        'Upgrade-Insecure-Requests': '1'
                    }
                    
                    # First, make a HEAD request to check if the resource exists and get content info
                    try:
                        head_response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
                        if head_response.status_code >= 400:
                            logger.warning(f"HEAD request failed with status {head_response.status_code}, trying GET request anyway")
                        
                        # Try to get content-type from headers
                        content_type = head_response.headers.get('content-type', '').lower()
                        logger.info(f"Content-Type from server: {content_type}")
                        
                    except Exception as e:
                        logger.warning(f"HEAD request failed: {e}, proceeding with GET request")
                    
                    # Now make the actual GET request
                    response = requests.get(url, headers=headers, timeout=60, stream=True, allow_redirects=True)
                    response.raise_for_status()
                    
                    # Check response content type
                    content_type = response.headers.get('content-type', '').lower()
                    logger.info(f"Response Content-Type: {content_type}, Status: {response.status_code}")
                    
                    # Verify we got actual content
                    if 'text/html' in content_type and 'pdf' not in content_type:
                        logger.warning(f"Received HTML content instead of document. URL might be incorrect: {url}")
                    
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
                                # This is a rough heuristic
                                file_extension = '.docx'  # Default to .docx
                            else:
                                logger.warning(f"Could not determine file type from URL or content-type: {url}")
                                file_extension = '.pdf'  # Default assumption
                    
                    logger.info(f"Successfully downloaded {len(content)} bytes, detected extension: {file_extension}")
                    
                except requests.exceptions.Timeout:
                    raise ValueError(f"Request timed out while downloading from: {url}")
                except requests.exceptions.ConnectionError:
                    raise ValueError(f"Connection error while downloading from: {url}")
                except requests.exceptions.HTTPError as e:
                    raise ValueError(f"HTTP error {e.response.status_code} while downloading from: {url}")
                except requests.exceptions.RequestException as e:
                    raise ValueError(f"Request error while downloading from: {url}: {str(e)}")
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
                    with pdfplumber.open(temp_file) as pdf:
                        if len(pdf.pages) == 0:
                            raise ValueError("PDF file contains no pages")
                        
                        logger.info(f"Processing PDF with {len(pdf.pages)} pages")
                        for page_num, page in enumerate(pdf.pages):
                            try:
                                page_text = page.extract_text()
                                if page_text: 
                                    full_text += page_text + "\n"
                                
                                # Extract tables
                                tables = page.extract_tables()
                                for table in tables:
                                    if table:
                                        table_text = "\n".join([
                                            "\t".join(map(str, (c if c is not None else "" for c in r))) 
                                            for r in table
                                        ])
                                        full_text += "\n--- TABLE ---\n" + table_text + "\n--- END TABLE ---\n"
                            except Exception as e:
                                logger.warning(f"Error processing PDF page {page_num}: {e}")
                                continue
                                
                elif file_extension == '.docx':
                    try:
                        doc = Document(temp_file)
                        paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
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
                        
                        for p in paragraphs:
                            text_content = " ".join(
                                node.data for node in p.childNodes 
                                if node.nodeType == node.TEXT_NODE
                            )
                            if text_content.strip():
                                full_text += text_content + "\n"
                        logger.info(f"Extracted text from {len(paragraphs)} ODT paragraphs")
                    except Exception as e:
                        raise ValueError(f"Error parsing ODT file: {str(e)}")
                        
                elif file_extension == '.eml':
                    try:
                        temp_file.seek(0)
                        content = temp_file.read()
                        msg = email.message_from_bytes(content)
                        
                        # Extract headers
                        full_text += f"Subject: {msg.get('subject', 'N/A')}\n"
                        full_text += f"From: {msg.get('from', 'N/A')}\n"
                        full_text += f"To: {msg.get('to', 'N/A')}\n"
                        full_text += f"Date: {msg.get('date', 'N/A')}\n\n"
                        
                        # Extract body
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    try:
                                        payload = part.get_payload(decode=True)
                                        if payload:
                                            full_text += payload.decode('utf-8', errors='ignore')
                                    except Exception as e:
                                        logger.warning(f"Error decoding email part: {e}")
                        else:
                            try:
                                payload = msg.get_payload(decode=True)
                                if payload:
                                    full_text += payload.decode('utf-8', errors='ignore')
                            except Exception as e:
                                logger.warning(f"Error decoding email payload: {e}")
                        
                        logger.info("Successfully extracted EML content")
                    except Exception as e:
                        raise ValueError(f"Error parsing EML file: {str(e)}")
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")
            
            except Exception as e:
                logger.error(f"Error during document parsing: {e}")
                raise

            # Validate extracted text
            if not full_text.strip():
                logger.error(f"No text extracted from document at {url}")
                raise ValueError("Document appears to be empty or contains no extractable text")
            
            # Basic text cleaning
            full_text = re.sub(r'\s+', ' ', full_text).strip()
            
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
        prompt = (
            "You are an expert AI assistant for analyzing complex policy documents. "
            "Your task is to answer the user's question based ONLY on the provided text context. "
            "Do not infer, imagine, or use any external knowledge. "
            "If the answer is not found in the context, you MUST state: "
            "'Based on the provided documents, there is no information on this topic.' "
            "Be precise and concise in your answer.\n\n"
            f"CONTEXT:\n{context_str}\n\nQUESTION:\n{question}"
        )
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

    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Processes a list of question strings concurrently for maximum speed.
        This is the final version for the hackathon submission.
        """
        vector_store = await self.get_or_create_vector_store(document_url)
        tasks = [self._generate_answer(question, vector_store.search(question, k=8)) for question in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        return [str(ans) for ans in answers]