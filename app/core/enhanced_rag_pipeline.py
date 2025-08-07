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
from typing import List, Tuple, Dict, Optional, Any

logger = logging.getLogger(__name__)

class AdvancedVectorStore:
    """Advanced vector store with hierarchical chunking and re-ranking"""
    
    def __init__(self, model: SentenceTransformer, reranker: CrossEncoder, dimension: int):
        self.model = model
        self.reranker = reranker
        self.dimension = dimension
        
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
    
    def add_hierarchical(self, small_chunks: List[str], small_embeddings: np.ndarray,
                         large_chunks: List[str], large_embeddings: np.ndarray,
                         metadata: List[Dict], chunk_mapping: Dict[int, int]):
        """Add hierarchical chunks to the store"""
        
        # Store chunks
        base_idx = len(self.small_chunks)
        self.small_chunks.extend(small_chunks)
        self.large_chunks.extend(large_chunks)
        self.chunk_metadata.extend(metadata)
        
        # Update mapping
        for small_idx, large_idx in chunk_mapping.items():
            self.chunk_to_large_mapping[base_idx + small_idx] = large_idx
        
        # Add to indices
        self.small_index.add(small_embeddings)
        self.large_index.add(large_embeddings)
        
        # Reinitialize retriever
        self.enhanced_retriever = EnhancedRetriever(self.small_chunks, self.chunk_metadata)
        
        logger.info(f"Added {len(small_chunks)} small and {len(large_chunks)} large chunks")
    
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
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.settings = settings
        
        # Initialize components
        self.universal_parser = UniversalDocumentParser()
        self.question_analyzer = QuestionAnalyzer()
        self.answer_validator = AnswerValidator()
        
        # Configure Gemini
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise
    
    async def download_document(self, url: str) -> bytes:
        """Enhanced download with better error handling"""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        try:
            # Async download with retries
            timeout = aiohttp.ClientTimeout(total=60)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    
                    # Stream large files
                    chunks = []
                    total_size = 0
                    max_size = settings.MAX_DOCUMENT_SIZE_MB * 1024 * 1024
                    
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
        
        # Download and parse document
        content = await self.download_document(url)
        text, metadata = self.universal_parser.parse_any_document(content, url)
        
        if not text or len(text) < 10:
            logger.error(f"Document parsing failed or empty: {url}")
            text = "Document could not be parsed or is empty."
            metadata = [{'type': 'error'}]
        
        # Hierarchical chunking
        small_chunks, small_metadata = SmartChunker.chunk_document(
            text, metadata,
            chunk_size=settings.CHUNK_SIZE_CHARS,
            overlap=settings.CHUNK_OVERLAP_CHARS
        )
        
        large_chunks, large_metadata = SmartChunker.chunk_document(
            text, metadata,
            chunk_size=settings.LARGE_CHUNK_SIZE_CHARS,
            overlap=settings.LARGE_CHUNK_OVERLAP_CHARS
        )
        
        # Create chunk mapping
        chunk_mapping = self._create_chunk_mapping(small_chunks, large_chunks, text)
        
        logger.info(f"Created {len(small_chunks)} small and {len(large_chunks)} large chunks")
        
        # Generate embeddings
        small_embeddings = await self._generate_embeddings(small_chunks)
        large_embeddings = await self._generate_embeddings(large_chunks)
        
        # Create vector store
        dimension = small_embeddings.shape[1]
        vector_store = AdvancedVectorStore(
            self.embedding_model,
            self.reranker_model,
            dimension
        )
        
        # Add hierarchical data
        vector_store.add_hierarchical(
            small_chunks, small_embeddings,
            large_chunks, large_embeddings,
            small_metadata, chunk_mapping
        )
        
        # Cache the store
        await cache.set(cache_key, vector_store, ttl=settings.CACHE_TTL_SECONDS)
        
        return vector_store
    
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
    
    async def _generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """Generate embeddings in batches"""
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            embeddings = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.embedding_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            )
            all_embeddings.append(embeddings)
        
        return np.vstack(all_embeddings).astype('float32')
    
    async def answer_question(self, question: str, vector_store: AdvancedVectorStore) -> Dict[str, Any]:
        """Generate answer with multi-step reasoning and validation"""
        
        # 1. Analyze question type
        question_info = self.question_analyzer.analyze(question)
        
        # 2. Query expansion
        expanded_queries = self._expand_query(question, question_info)
        
        # 3. Retrieve relevant chunks for all queries
        all_chunks = []
        all_scores = []
        
        for query in expanded_queries:
            search_results = vector_store.search_with_reranking(
                query,
                k=settings.MAX_CHUNKS_PER_QUERY
            )
            
            for chunk, score, meta in search_results:
                all_chunks.append(chunk)
                all_scores.append(score)
        
        # 4. Deduplicate and rank chunks
        unique_chunks = self._deduplicate_chunks(all_chunks, all_scores)
        
        # 5. Multi-step reasoning if needed
        if question_info['requires_multi_step']:
            answer = await self._multi_step_reasoning(question, unique_chunks, question_info)
        else:
            answer = await self._generate_answer(question, unique_chunks, question_info)
        
        # 6. Validate answer
        validated_answer = self.answer_validator.validate(
            question, answer, question_info, unique_chunks
        )
        
        # 7. Calculate confidence
        confidence = self._calculate_confidence(validated_answer, all_scores)
        
        return {
            'answer': validated_answer['answer'],
            'confidence': confidence,
            'question_type': question_info['type'],
            'sources': validated_answer.get('sources', []),
            'validation_notes': validated_answer.get('notes', '')
        }
    
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
    
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Process multiple questions with parallel execution"""
        
        start_time = time.time()
        logger.info(f"Processing {len(questions)} questions for {document_url}")
        
        try:
            # Get vector store
            vector_store = await self.get_or_create_vector_store(document_url)
            
            # Process questions in parallel
            semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_QUESTIONS)
            
            async def process_question(q):
                async with semaphore:
                    try:
                        result = await self.answer_question(q, vector_store)
                        return result['answer']
                    except Exception as e:
                        logger.error(f"Error processing question: {e}")
                        return "Error processing this question."
            
            # Create tasks
            tasks = [process_question(q) for q in questions]
            
            # Execute with timeout
            answers = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=settings.TOTAL_TIMEOUT_SECONDS
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions in {elapsed:.2f}s")
            
            return answers
            
        except asyncio.TimeoutError:
            logger.error("Overall processing timeout")
            return ["Processing timeout. Please try again."] * len(questions)
        except Exception as e:
            logger.error(f"Critical error: {e}", exc_info=True)
            return ["Document processing error."] * len(questions)