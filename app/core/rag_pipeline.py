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

    def _download_and_parse_document(self, url: str) -> str:
        """
        Downloads and parses a document from a URL, supporting PDF, DOCX, ODT, and EML formats.
        """
        logger.info(f"Processing document from {url}")
        try:
            if url.startswith("gs://"):
                storage_client = storage.Client()
                bucket_name, blob_name = url.replace("gs://", "").split("/", 1)
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                temp_file = io.BytesIO(blob.download_as_bytes())
                file_extension = os.path.splitext(blob_name)[1].lower()
            elif url.startswith("http"):
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, timeout=30, headers=headers, stream=True)
                response.raise_for_status()
                temp_file = io.BytesIO(response.content)
                file_extension = os.path.splitext(url.split('?')[0])[-1].lower()
            else:
                raise ValueError("Invalid document URL scheme. Must be http or gs.")

            full_text = ""
            if file_extension == '.pdf':
                with pdfplumber.open(temp_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text: full_text += page_text + "\n"
                        for table in page.extract_tables():
                            if table:
                                table_text = "\n".join(["\t".join(map(str, (c if c is not None else "" for c in r))) for r in table])
                                full_text += "\n--- TABLE ---\n" + table_text + "\n--- END TABLE ---\n"
            elif file_extension == '.docx':
                doc = Document(temp_file)
                full_text = "\n".join([para.text for para in doc.paragraphs if para.text])
            elif file_extension == '.odt':
                doc = load(temp_file)
                for p in doc.getElementsByType(P):
                    full_text += " ".join(node.data for node in p.childNodes if node.nodeType == node.TEXT_NODE) + "\n"
            elif file_extension == '.eml':
                msg = email.message_from_bytes(temp_file.read())
                full_text += f"Subject: {msg['subject']}\nFrom: {msg['from']}\nTo: {msg['to']}\nDate: {msg['date']}\n\n"
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            full_text += part.get_payload(decode=True).decode()
                else:
                    full_text += msg.get_payload(decode=True).decode()
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")

            if not full_text.strip():
                logger.warning(f"No text extracted from document at {url}.")
                return ""
            return full_text
        except Exception as e:
            logger.error(f"Failed to process document from {url}: {e}")
            raise ValueError(f"Could not retrieve or parse document from URL.")

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