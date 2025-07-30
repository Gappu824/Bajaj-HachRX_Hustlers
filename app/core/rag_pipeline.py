# app/core/rag_pipeline.py
import io
import re
import os
import logging
import requests
import asyncio
import numpy as np
import faiss
import google.generativeai as genai
from typing import List

# --- CHANGE 1: Add new imports ---
from tenacity import retry, stop_after_attempt, wait_exponential
from google.api_core import exceptions as google_exceptions

from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorStore:
    # ... (no changes in this class)
    def __init__(self, chunks: List[str], embeddings: np.ndarray, model: SentenceTransformer):
        self.chunks = chunks
        self.model = model
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        logger.info(f"Built FAISS index for {len(chunks)} chunks.")

    def search(self, query: str, k: int = 5) -> List[str]:
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

    def _download_and_parse_pdf(self, url: str) -> str:
        # ... (no changes in this method)
        try:
            logger.info(f"Downloading document from {url}")
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=20, headers=headers)
            response.raise_for_status()
            reader = PdfReader(io.BytesIO(response.content))
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            if not text.strip():
                logger.warning(f"No text extracted from PDF at {url}. It might be image-based or empty.")
                return ""
            return text
        except requests.RequestException as e:
            logger.error(f"Failed to download PDF from {url}: {e}")
            raise ValueError(f"Could not retrieve document from URL.")
        except Exception as e:
            logger.error(f"Failed to parse PDF from {url}: {e}")
            raise ValueError(f"Could not parse PDF document.")

    def _chunk_text(self, text: str) -> List[str]:
        # ... (no changes in this method)
        text = re.sub(r'\s+', ' ', text).strip()
        paragraphs = text.split('\n\n')
        final_chunks = []
        for para in paragraphs:
            para = para.strip()
            if len(para) > 1500:
                final_chunks.extend(p.strip() for p in para.split('. ') if len(p.strip()) > 50)
            elif len(para) > 50:
                final_chunks.append(para)
        logger.info(f"Chunked document into {len(final_chunks)} parts.")
        return final_chunks

    async def get_or_create_vector_store(self, url: str) -> VectorStore:
        # ... (no changes in this method)
        from app.core.cache import cache
        cached_store = await cache.get(url)
        if cached_store:
            logger.info(f"Found cached vector store for {url}")
            return cached_store
        logger.info(f"No cache found. Processing document for {url}...")
        text = self._download_and_parse_pdf(url)
        if not text:
            raise ValueError("Document is empty or could not be processed.")
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("Could not extract meaningful text chunks from the document.")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=False).astype('float32')
        vector_store = VectorStore(chunks, embeddings, self.embedding_model)
        await cache.set(url, vector_store)
        return vector_store
    
    # --- CHANGE 2: Add the @retry decorator ---
    # This will automatically retry the function if it fails with a ResourceExhausted error (HTTP 429).
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), # Waits 2s, then 4s, etc.
        stop=stop_after_attempt(3), # Stops after 3 attempts
        retry=(retry_if_exception_type(google_exceptions.ResourceExhausted) | retry_if_exception_type(google_exceptions.InternalServerError)) # Only retry on these specific API errors
    )
    async def _generate_answer(self, question: str, context: List[str]) -> str:
        # ... (no changes to the logic inside this method)
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
            raise e # Reraise the exception to trigger tenacity's retry mechanism
        except Exception as e:
            logger.error(f"Google Gemini generation failed for question '{question}': {e}")
            return "An error occurred while generating the answer using the Gemini API."

    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        # ... (no changes in this method)
        vector_store = await self.get_or_create_vector_store(document_url)
        tasks = [self._generate_answer(question, vector_store.search(question, k=5)) for question in questions]
        answers = await asyncio.gather(*tasks, return_exceptions=True)
        return [str(ans) for ans in answers]