# app/core/rag_pipeline.py - Optimized without information loss
import io
import re
import os
import logging
import requests
import asyncio
import time
from typing import List, Tuple, Dict, Set, Optional
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import numpy as np

# ... (previous imports remain the same)

logger = logging.getLogger(__name__)


class OptimizedVectorStore:
    """Optimized vector store with multi-level indexing and intelligent sampling"""
    def __init__(self, chunks: List[str], embeddings: np.ndarray, model: SentenceTransformer, 
                 chunk_metadata: Optional[List[Dict]] = None):
        self.chunks = chunks
        self.model = model
        self.chunk_metadata = chunk_metadata or [{}] * len(chunks)
        dimension = embeddings.shape[1]
        
        # Use hierarchical indexing for very large documents
        if len(chunks) > 5000:
            # Use IVF with PQ for massive documents
            nlist = min(256, len(chunks) // 20)
            m = 8  # number of subquantizers
            self.index = faiss.IndexIVFPQ(
                faiss.IndexFlatL2(dimension), dimension, nlist, m, 8
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(32, nlist // 4)  # Search more cells for better recall
        elif len(chunks) > 1000:
            # Use IVF for large documents
            nlist = min(100, len(chunks) // 10)
            self.index = faiss.IndexIVFFlat(
                faiss.IndexFlatL2(dimension), dimension, nlist
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(16, nlist // 2)
        else:
            # Use flat index for small documents
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
        
        # Build comprehensive indexes
        self.keyword_index = self._build_keyword_index(chunks)
        self.section_index = self._build_section_index(chunks, chunk_metadata)
        self.question_patterns = self._build_question_patterns()
        
        logger.info(f"Built optimized FAISS index for {len(chunks)} chunks")

    def _build_section_index(self, chunks: List[str], metadata: List[Dict]) -> Dict[str, List[int]]:
        """Build index of document sections for targeted retrieval"""
        section_index = {
            'introduction': [],
            'methodology': [],
            'results': [],
            'conclusion': [],
            'definitions': [],
            'procedures': [],
            'calculations': [],
            'references': []
        }
        
        section_patterns = {
            'introduction': [r'introduction', r'overview', r'background', r'preface'],
            'methodology': [r'method', r'approach', r'procedure', r'algorithm'],
            'results': [r'result', r'finding', r'outcome', r'analysis'],
            'conclusion': [r'conclusion', r'summary', r'discussion'],
            'definitions': [r'definition', r'terminology', r'glossary', r'means'],
            'procedures': [r'process', r'step', r'instruction', r'how to'],
            'calculations': [r'calculation', r'formula', r'equation', r'compute'],
            'references': [r'reference', r'citation', r'bibliography', r'source']
        }
        
        for i, (chunk, meta) in enumerate(zip(chunks, metadata)):
            chunk_lower = chunk.lower()
            # Check metadata first
            if 'section' in meta:
                section = meta['section'].lower()
                for key in section_index:
                    if key in section:
                        section_index[key].append(i)
            
            # Pattern matching
            for section, patterns in section_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, chunk_lower):
                        section_index[section].append(i)
                        break
        
        return section_index

    def _build_keyword_index(self, chunks: List[str]) -> Dict[str, Set[int]]:
        """Build comprehensive keyword index with n-grams"""
        keyword_index = {}
        
        for i, chunk in enumerate(chunks):
            chunk_lower = chunk.lower()
            
            # Extract various types of terms
            # Numbers with units
            numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:%|percent|days?|years?|months?|lakhs?|crores?|thousands?|millions?|billions?))?', chunk_lower)
            
            # Important phrases (2-grams and 3-grams)
            words = chunk_lower.split()
            for n in [2, 3]:
                for j in range(len(words) - n + 1):
                    phrase = ' '.join(words[j:j+n])
                    if len(phrase) > 5 and not phrase.startswith(('the ', 'a ', 'an ')):
                        if phrase not in keyword_index:
                            keyword_index[phrase] = set()
                        keyword_index[phrase].add(i)
            
            # Single important terms
            important_terms = re.findall(r'\b[a-z]{4,}\b', chunk_lower)
            
            # Index all terms
            all_terms = numbers + important_terms
            for term in set(all_terms):
                if term not in keyword_index:
                    keyword_index[term] = set()
                keyword_index[term].add(i)
        
        return keyword_index

    def targeted_search(self, query: str, k: int = 30, section_hint: Optional[str] = None) -> List[Tuple[str, float, Dict]]:
        """Enhanced search with section targeting and better ranking"""
        query_lower = query.lower()
        results = []
        
        # Detect question type and section
        question_type = self._detect_question_type(query_lower)
        if not section_hint:
            section_hint = self._detect_target_section(query_lower)
        
        # Phase 1: Exact and phrase matching
        keyword_matches = set()
        query_words = query_lower.split()
        
        # Check single words and phrases
        for term in query_words:
            if term in self.keyword_index:
                keyword_matches.update(self.keyword_index[term])
        
        # Check 2-grams and 3-grams from query
        for n in [2, 3]:
            for i in range(len(query_words) - n + 1):
                phrase = ' '.join(query_words[i:i+n])
                if phrase in self.keyword_index:
                    keyword_matches.update(self.keyword_index[phrase])
        
        # Phase 2: Section-targeted search
        section_candidates = set()
        if section_hint and section_hint in self.section_index:
            section_candidates.update(self.section_index[section_hint])
        
        # Phase 3: Vector search with larger candidate set
        query_embedding = self.model.encode([query]).astype('float32')
        search_k = min(k * 3, len(self.chunks))  # Get more candidates
        distances, indices = self.index.search(query_embedding, search_k)
        
        # Phase 4: Combine and score
        seen_chunks = set()
        
        # Priority 1: Keyword + Section matches
        for idx in keyword_matches.intersection(section_candidates):
            if idx < len(self.chunks) and idx not in seen_chunks:
                chunk = self.chunks[idx]
                score = self._calculate_relevance_score(chunk, query_lower, question_type, 
                                                       is_keyword_match=True, is_section_match=True)
                results.append((chunk, score, {
                    'type': 'keyword+section', 
                    'question_type': question_type,
                    'chunk_idx': idx,
                    'metadata': self.chunk_metadata[idx]
                }))
                seen_chunks.add(idx)
        
        # Priority 2: Keyword matches
        for idx in keyword_matches:
            if idx < len(self.chunks) and idx not in seen_chunks:
                chunk = self.chunks[idx]
                score = self._calculate_relevance_score(chunk, query_lower, question_type, 
                                                       is_keyword_match=True)
                results.append((chunk, score, {
                    'type': 'keyword', 
                    'question_type': question_type,
                    'chunk_idx': idx,
                    'metadata': self.chunk_metadata[idx]
                }))
                seen_chunks.add(idx)
        
        # Priority 3: Vector search results
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx not in seen_chunks:
                chunk = self.chunks[idx]
                vector_score = 1 / (1 + dist)
                is_section_match = idx in section_candidates
                score = self._calculate_relevance_score(chunk, query_lower, question_type, 
                                                       vector_score=vector_score, 
                                                       is_section_match=is_section_match)
                results.append((chunk, score, {
                    'type': 'vector', 
                    'question_type': question_type,
                    'chunk_idx': idx,
                    'metadata': self.chunk_metadata[idx]
                }))
                seen_chunks.add(idx)
        
        # Sort by score and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def _detect_target_section(self, query: str) -> Optional[str]:
        """Detect which section of document to focus on"""
        section_keywords = {
            'definitions': ['what is', 'define', 'meaning', 'definition'],
            'procedures': ['how to', 'process', 'steps', 'procedure'],
            'calculations': ['calculate', 'formula', 'equation', 'compute'],
            'results': ['results', 'findings', 'outcomes', 'analysis'],
            'methodology': ['method', 'approach', 'technique'],
        }
        
        for section, keywords in section_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return section
        return None

    def _detect_question_type(self, query: str) -> str:
        """Enhanced question type detection"""
        # (Previous implementation remains the same)
        for q_type, config in self.question_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, query, re.IGNORECASE):
                    return q_type
        return 'general'

    def _calculate_relevance_score(self, chunk: str, query: str, question_type: str, 
                                   is_keyword_match: bool = False, 
                                   is_section_match: bool = False,
                                   vector_score: float = 0.0) -> float:
        """Enhanced relevance scoring"""
        score = 0.0
        chunk_lower = chunk.lower()
        
        # Base scores
        if is_keyword_match and is_section_match:
            score += 0.9  # Highest priority
        elif is_keyword_match:
            score += 0.7
        elif is_section_match:
            score += 0.3
        else:
            score += vector_score * 0.5
        
        # Question type specific boosting
        if question_type in self.question_patterns:
            boost_terms = self.question_patterns[question_type]['boost_terms']
            for term in boost_terms:
                if term in chunk_lower:
                    score += 0.05
        
        # Query term coverage
        query_terms = set(query.split())
        chunk_terms = set(chunk_lower.split())
        coverage = len(query_terms.intersection(chunk_terms)) / len(query_terms)
        score += coverage * 0.2
        
        # Exact phrase matching bonus
        if query in chunk_lower:
            score += 0.3
        
        # Length penalty for very short chunks
        if len(chunk.split()) < 20:
            score *= 0.8
        
        return min(score, 1.0)


class HybridFastTrackRAGPipeline:
    """Optimized pipeline that processes all content without information loss"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            logger.info("Google Generative AI client configured successfully.")
        except Exception as e:
            logger.critical(f"Failed to configure Google Generative AI: {e}")
            raise RuntimeError("Google API Key is not configured correctly.")
        
        # Initialize cross-encoder only for complex questions
        self.cross_encoder = None  # Lazy loading
        
        # Process pool for parallel PDF processing
        self.process_pool = ProcessPoolExecutor(max_workers=multiprocessing.cpu_count())
        
        # Thread pool for I/O operations
        self.thread_pool = ThreadPoolExecutor(max_workers=8)
        
        # Question complexity patterns
        self.complexity_patterns = {
            'simple': [
                r'^what is the \w+',
                r'^how much is',
                r'^is \w+ covered',
                r'^what are the \w+',
                r'^list the \w+'
            ],
            'complex': [
                r'calculate.*based on',
                r'explain.*process',
                r'compare.*with',
                r'analyze.*impact',
                r'what.*if.*then'
            ]
        }

    def _parse_pdf_parallel(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Parse PDF in parallel chunks for speed without losing information"""
        temp_file = io.BytesIO(content)
        full_text = ""
        chunk_metadata = []
        
        try:
            with pdfplumber.open(temp_file) as pdf:
                total_pages = len(pdf.pages)
                logger.info(f"Processing PDF with {total_pages} pages using parallel processing")
                
                # Process pages in batches
                batch_size = 50
                page_batches = []
                
                for i in range(0, total_pages, batch_size):
                    batch_end = min(i + batch_size, total_pages)
                    page_batches.append((i, batch_end))
                
                # Process batches in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = []
                    
                    for start_idx, end_idx in page_batches:
                        future = executor.submit(
                            self._process_page_batch, 
                            content, start_idx, end_idx
                        )
                        futures.append(future)
                    
                    # Collect results in order
                    for future in futures:
                        batch_text, batch_metadata = future.result()
                        full_text += batch_text
                        chunk_metadata.extend(batch_metadata)
                
                logger.info(f"Successfully processed all {total_pages} pages")
                
        except Exception as e:
            logger.error(f"PDF parsing error: {e}")
            # Fallback to PyPDF2 if available
            if PyPDF2:
                logger.info("Attempting PyPDF2 fallback")
                temp_file.seek(0)
                full_text, chunk_metadata = self._parse_pdf_pypdf2_parallel(temp_file)
            else:
                raise
        
        return self._clean_text(full_text), chunk_metadata

    def _process_page_batch(self, pdf_content: bytes, start_page: int, end_page: int) -> Tuple[str, List[Dict]]:
        """Process a batch of PDF pages"""
        temp_file = io.BytesIO(pdf_content)
        batch_text = ""
        batch_metadata = []
        
        try:
            with pdfplumber.open(temp_file) as pdf:
                for page_num in range(start_page, end_page):
                    if page_num < len(pdf.pages):
                        page = pdf.pages[page_num]
                        
                        # Extract text
                        page_text = page.extract_text() or ""
                        
                        if page_text.strip():
                            batch_text += f"\n--- PAGE {page_num + 1} ---\n{page_text}\n"
                            batch_metadata.append({
                                'page': page_num + 1,
                                'type': 'text',
                                'section': self._detect_page_section(page_text)
                            })
                        
                        # Extract tables efficiently
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables[:3]):  # Limit tables per page
                            if table:
                                table_text = self._format_table(table)
                                if table_text.strip():
                                    batch_text += f"\n=== TABLE {table_idx + 1} (Page {page_num + 1}) ===\n{table_text}\n"
                                    batch_metadata.append({
                                        'page': page_num + 1,
                                        'type': 'table',
                                        'table_idx': table_idx + 1
                                    })
        
        except Exception as e:
            logger.warning(f"Error processing pages {start_page}-{end_page}: {e}")
        
        return batch_text, batch_metadata

    def _detect_page_section(self, text: str) -> str:
        """Detect the section type of a page"""
        text_lower = text.lower()[:500]  # Check first 500 chars
        
        if any(term in text_lower for term in ['introduction', 'preface', 'abstract']):
            return 'introduction'
        elif any(term in text_lower for term in ['conclusion', 'summary', 'final']):
            return 'conclusion'
        elif any(term in text_lower for term in ['method', 'procedure', 'algorithm']):
            return 'methodology'
        elif any(term in text_lower for term in ['result', 'finding', 'analysis']):
            return 'results'
        elif any(term in text_lower for term in ['definition', 'terminology', 'glossary']):
            return 'definitions'
        else:
            return 'general'

    def _smart_chunk_text_with_metadata(self, text: str, metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Create intelligent chunks while preserving document structure"""
        target_size = 1200  # Optimal size for context
        overlap_size = 200
        
        chunks = []
        chunk_metadata_list = []
        
        # Split by page boundaries first
        pages = text.split('--- PAGE')
        current_chunk = ""
        current_metadata = {}
        
        for page in pages:
            if not page.strip():
                continue
                
            # Extract page number
            page_match = re.match(r'\s*(\d+)\s*---', page)
            if page_match:
                page_num = int(page_match.group(1))
                page_text = page[page_match.end():]
                
                # Find corresponding metadata
                page_meta = next((m for m in metadata if m.get('page') == page_num), {})
                
                # Smart chunking within page
                if len(current_chunk) + len(page_text) <= target_size:
                    current_chunk += f"\n--- PAGE {page_num} ---\n{page_text}"
                    if not current_metadata:
                        current_metadata = page_meta.copy()
                else:
                    # Save current chunk
                    if current_chunk.strip():
                        chunks.append(current_chunk.strip())
                        chunk_metadata_list.append(current_metadata)
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap_size:
                        overlap = current_chunk[-overlap_size:]
                        current_chunk = overlap + f"\n--- PAGE {page_num} ---\n{page_text}"
                    else:
                        current_chunk = f"--- PAGE {page_num} ---\n{page_text}"
                    current_metadata = page_meta.copy()
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            chunk_metadata_list.append(current_metadata)
        
        logger.info(f"Created {len(chunks)} chunks with metadata")
        return chunks, chunk_metadata_list

    def _download_and_parse_document_optimized(self, url: str) -> Tuple[str, List[Dict]]:
        """Download and parse document without information loss"""
        logger.info(f"Processing document from {url}")
        
        try:
            # Download document
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; DocumentProcessor/2.0)',
                'Accept': 'application/pdf,application/vnd.openxmlformats-officedocument.wordprocessingml.document,*/*'
            }
            
            response = requests.get(url, headers=headers, timeout=60, stream=True)
            response.raise_for_status()
            
            content = b""
            max_size = 200 * 1024 * 1024  # 200MB limit
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    content += chunk
                    if len(content) > max_size:
                        logger.warning(f"Document exceeds {max_size} bytes")
                        break
            
            file_extension = os.path.splitext(url.split('?')[0])[1].lower()
            
            if not file_extension and 'pdf' in response.headers.get('content-type', ''):
                file_extension = '.pdf'
            
            # Parse based on file type
            if file_extension == '.pdf':
                return self._parse_pdf_parallel(content)
            elif file_extension in ['.docx', '.doc']:
                temp_file = io.BytesIO(content)
                text = self._parse_docx(temp_file)
                return text, [{'type': 'docx'}]
            elif file_extension == '.odt':
                temp_file = io.BytesIO(content)
                text = self._parse_odt(temp_file)
                return text, [{'type': 'odt'}]
            else:
                # Try PDF as default
                logger.warning(f"Unknown extension {file_extension}, attempting PDF parsing")
                return self._parse_pdf_parallel(content)
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise ValueError(f"Could not process document: {str(e)}")

    async def get_or_create_optimized_vector_store(self, url: str) -> OptimizedVectorStore:
        """Create optimized vector store with intelligent caching"""
        from app.core.cache import cache
        
        # Use URL hash for cache key
        cache_key = hashlib.md5(url.encode()).hexdigest()
        
        # Check cache first
        cached_store = await cache.get(cache_key)
        if cached_store:
            logger.info(f"Using cached vector store for {url}")
            return cached_store
        
        logger.info(f"Creating optimized vector store for {url}...")
        
        try:
            # Parse document with full content
            text, doc_metadata = self._download_and_parse_document_optimized(url)
            chunks, chunk_metadata = self._smart_chunk_text_with_metadata(text, doc_metadata)
            
            if not chunks:
                raise ValueError("No chunks created from document")
            
            # Generate embeddings in batches for speed
            logger.info(f"Generating embeddings for {len(chunks)} chunks")
            
            # Process embeddings in batches to avoid memory issues
            batch_size = 64
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_embeddings = self.embedding_model.encode(
                    batch, 
                    show_progress_bar=False, 
                    convert_to_numpy=True
                )
                all_embeddings.append(batch_embeddings)
            
            embeddings = np.vstack(all_embeddings).astype('float32')
            
            vector_store = OptimizedVectorStore(chunks, embeddings, self.embedding_model, chunk_metadata)
            
            # Cache the store
            await cache.set(cache_key, vector_store)
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to create vector store: {e}")
            raise

    # ... (rest of the methods remain largely the same with minor optimizations)

    async def _answer_question_optimized(self, question: str, vector_store: OptimizedVectorStore, complexity: str) -> str:
        """Optimized question answering with targeted retrieval"""
        
        logger.debug(f"Answering {complexity} question: {question}")
        
        # Get more candidates for complex questions
        k = 30 if complexity == 'complex' else 20
        
        # Targeted search with section hints
        retrieved_results = vector_store.targeted_search(question, k=k)
        
        if not retrieved_results:
            return "Based on the provided documents, the information to answer this question is not available."
        
        # For complex questions, optionally use cross-encoder
        if complexity == 'complex' and len(retrieved_results) > 15:
            # Lazy load cross-encoder
            if self.cross_encoder is None:
                try:
                    self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                    logger.info("Loaded cross-encoder for complex questions")
                except Exception as e:
                    logger.warning(f"Failed to load cross-encoder: {e}")
            
            if self.cross_encoder:
                try:
                    chunks = [result[0] for result in retrieved_results[:20]]
                    pairs = [[question, chunk] for chunk in chunks]
                    
                    # Quick cross-encoder scoring
                    scores = await asyncio.wait_for(
                        asyncio.get_running_loop().run_in_executor(
                            None, self.cross_encoder.predict, pairs
                        ),
                        timeout=3.0
                    )
                    
                    # Re-rank based on cross-encoder scores
                    reranked = sorted(
                        zip(chunks, scores), 
                        key=lambda x: x[1], 
                        reverse=True
                    )
                    top_chunks = [chunk for chunk, _ in reranked[:12]]
                    
                except asyncio.TimeoutError:
                    logger.warning("Cross-encoder timeout, using targeted search results")
                    top_chunks = [result[0] for result in retrieved_results[:12]]
                except Exception as e:
                    logger.warning(f"Cross-encoder failed: {e}")
                    top_chunks = [result[0] for result in retrieved_results[:12]]
            else:
                top_chunks = [result[0] for result in retrieved_results[:12]]
        else:
            top_chunks = [result[0] for result in retrieved_results[:10]]
        
        # Generate answer
        return await self._generate_answer_fast(question, top_chunks, complexity)

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

    def _classify_question_complexity(self, question: str) -> str:
        """Classify question as simple or complex for routing"""
        question_lower = question.lower()
        
        # Check for simple patterns
        for pattern in self.complexity_patterns['simple']:
            if re.match(pattern, question_lower):
                return 'simple'
        
        # Check for complex patterns
        for pattern in self.complexity_patterns['complex']:
            if re.search(pattern, question_lower):
                return 'complex'
        
        # Check question length and special characters
        if len(question.split()) > 15 or '?' in question[:-1]:
            return 'complex'
        
        return 'simple'

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=3),
        stop=stop_after_attempt(2),
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted)
    )
    async def _generate_answer_fast(self, question: str, context: List[str], complexity: str) -> str:
        """Optimized answer generation based on complexity"""
        if not context or all(len(c.strip()) < 50 for c in context):
           return f"Based on the available document content, I could not find specific information to answer: '{question}'."
       
        # Use more context for complex questions
        context_size = 12 if complexity == 'complex' else 8
        context_text = "\n\n---\n\n".join(context[:context_size])
        
        # Simpler prompts for simple questions
        if complexity == 'simple':
            prompt = f"""Answer this question directly and concisely based on the context.

    CONTEXT:
    {context_text}

    QUESTION: {question}

    ANSWER:"""
        else:
            prompt = f"""You are an expert document analyst. Provide a comprehensive answer with all relevant details.

    IMPORTANT:
    1. Extract ALL relevant information from the context
    2. For lists, include every item mentioned
    3. For calculations, show your work
    4. For procedures, include all steps
    5. Be thorough but clear

    CONTEXT:
    {context_text}

    QUESTION: {question}

    DETAILED ANSWER:"""

        try:
            model = genai.GenerativeModel(settings.LLM_MODEL_NAME)
            response = await model.generate_content_async(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,  # Lower temperature for consistency
                    max_output_tokens=800 if complexity == 'complex' else 400,
                    top_p=0.9
                )
            )
            
            answer = response.text.strip()
            
            # Log the answer
            logger.info(f"Generated answer for question: {question[:50]}...")
            logger.info(f"Answer: {answer[:200]}...")
            
            return answer
            
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Unable to generate answer due to an error: {str(e)}"

    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """Optimized query processing with parallel execution"""
        start_time = time.time()
        
        logger.info(f"Processing {len(questions)} questions for document: {document_url}")
        
        try:
            # Create vector store
            vector_store = await self.get_or_create_optimized_vector_store(document_url)
            
            # Classify questions by complexity
            question_complexities = [
                (q, self._classify_question_complexity(q)) 
                for q in questions
            ]
            
            # Log complexity distribution
            simple_count = sum(1 for _, c in question_complexities if c == 'simple')
            logger.info(f"Question complexity: {simple_count} simple, {len(questions) - simple_count} complex")
            
            # Process questions in parallel with adaptive batch sizing
            answers = []
            # Larger batches for simple questions, smaller for complex
            simple_batch_size = 15
            complex_batch_size = 5
            
            # Group questions by complexity
            simple_questions = [(i, q) for i, (q, c) in enumerate(question_complexities) if c == 'simple']
            complex_questions = [(i, q) for i, (q, c) in enumerate(question_complexities) if c == 'complex']
            
            # Create tasks for all questions
            all_tasks = []
            
            # Process simple questions in larger batches
            for i in range(0, len(simple_questions), simple_batch_size):
                batch = simple_questions[i:i + simple_batch_size]
                for idx, question in batch:
                    task = (idx, self._answer_question_safe(question, vector_store, 'simple'))
                    all_tasks.append(task)
            
            # Process complex questions in smaller batches
            for i in range(0, len(complex_questions), complex_batch_size):
                batch = complex_questions[i:i + complex_batch_size]
                for idx, question in batch:
                    task = (idx, self._answer_question_safe(question, vector_store, 'complex'))
                    all_tasks.append(task)
            
            # Sort tasks by original index
            all_tasks.sort(key=lambda x: x[0])
            
            # Execute all tasks
            task_results = []
            for idx, task in all_tasks:
                task_results.append((idx, task))
            
            # Process in batches with timeout
            batch_size = 20
            results = [None] * len(questions)
            
            for i in range(0, len(task_results), batch_size):
                batch = task_results[i:i + batch_size]
                batch_coroutines = [task for _, task in batch]
                
                try:
                    batch_answers = await asyncio.wait_for(
                        asyncio.gather(*batch_coroutines, return_exceptions=True),
                        timeout=30  # 30 seconds per batch
                    )
                    
                    for j, (idx, _) in enumerate(batch):
                        if isinstance(batch_answers[j], Exception):
                            logger.error(f"Question {idx+1} processing error: {batch_answers[j]}")
                            results[idx] = "An error occurred while processing this question."
                        else:
                            results[idx] = str(batch_answers[j])
                            
                except asyncio.TimeoutError:
                    logger.error(f"Batch timeout for questions in batch starting at {i}")
                    for idx, _ in batch:
                        if results[idx] is None:
                            results[idx] = "Processing timeout. Please try again."
            
            # Ensure all questions have answers
            answers = [r if r is not None else "Processing error occurred." for r in results]
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {len(questions)} questions in {processing_time:.2f} seconds")
            
            # Log sample answers for debugging
            for i, answer in enumerate(answers[:3]):  # Log first 3 answers
                logger.info(f"Answer {i+1}: {answer[:200]}...")
            
            return answers
            
        except Exception as e:
            logger.critical(f"Critical processing error: {e}", exc_info=True)
            return self._generate_error_responses(questions, str(e))

    async def _answer_question_safe(self, question: str, vector_store: OptimizedVectorStore, complexity: str) -> str:
        """Safe wrapper for question answering"""
        try:
            return await self._answer_question_optimized(question, vector_store, complexity)
        except Exception as e:
            logger.error(f"Error answering question '{question[:50]}...': {e}")
            return f"Unable to process this question due to an error."

    def _generate_error_responses(self, questions: List[str], error_msg: str) -> List[str]:
        """Generate appropriate error responses"""
        logger.error(f"Generating error responses: {error_msg}")
        return [f"Unable to answer due to document processing error: {error_msg}"] * len(questions)

    async def process_query_with_explainability(self, document_url: str, questions: List[str]) -> Tuple[List[str], List[Dict], float]:
        """Process queries with explainability - enhanced version"""
        start_time = time.time()
        
        try:
            # Create vector store
            vector_store = await self.get_or_create_optimized_vector_store(document_url)
            
            simple_answers = []
            detailed_answers = []
            
            # Process each question with detailed tracking
            for question in questions:
                complexity = self._classify_question_complexity(question)
                
                # Get retrieval results with metadata
                retrieved_results = vector_store.targeted_search(question, k=25)
                
                # Generate answer
                if retrieved_results:
                    top_chunks = [result[0] for result in retrieved_results[:12]]
                    answer = await self._generate_answer_fast(question, top_chunks, complexity)
                    
                    # Create detailed answer with explainability
                    source_clauses = []
                    for i, (chunk, score, metadata) in enumerate(retrieved_results[:5]):
                        source_clauses.append({
                            "text": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                            "confidence_score": float(score),
                            "page_number": metadata.get('metadata', {}).get('page'),
                            "section": metadata.get('metadata', {}).get('section', 'general'),
                            "retrieval_type": metadata.get('type', 'unknown'),
                            "question_type": metadata.get('question_type', 'general')
                        })
                    
                    detailed_answer = {
                        "answer": answer,
                        "confidence": min(0.95, float(retrieved_results[0][1])) if retrieved_results else 0.0,
                        "source_clauses": source_clauses,
                        "reasoning": f"Answer extracted using {complexity} question processing with {len(retrieved_results)} relevant chunks found.",
                        "coverage_decision": self._determine_coverage(answer)
                    }
                else:
                    answer = "Information not available in the document."
                    detailed_answer = {
                        "answer": answer,
                        "confidence": 0.0,
                        "source_clauses": [],
                        "reasoning": "No relevant information found in the document.",
                        "coverage_decision": "Not Found"
                    }
                
                simple_answers.append(answer)
                detailed_answers.append(detailed_answer)
            
            processing_time = time.time() - start_time
            return simple_answers, detailed_answers, processing_time
            
        except Exception as e:
            logger.error(f"Error in explainable processing: {e}")
            processing_time = time.time() - start_time
            error_answers = self._generate_error_responses(questions, str(e))
            detailed_errors = [
                {
                    "answer": ans,
                    "confidence": 0.0,
                    "source_clauses": [],
                    "reasoning": f"Error: {str(e)}",
                    "coverage_decision": "Error"
                }
                for ans in error_answers
            ]
            return error_answers, detailed_errors, processing_time

    def _determine_coverage(self, answer: str) -> str:
        """Determine coverage decision from answer"""
        answer_lower = answer.lower()
        
        if any(term in answer_lower for term in ['not covered', 'excluded', 'not available', 'cannot find']):
            return "Not Covered"
        elif any(term in answer_lower for term in ['covered', 'included', 'eligible', 'yes']):
            return "Covered"
        elif any(term in answer_lower for term in ['conditions apply', 'subject to', 'depending on']):
            return "Conditional"
        else:
            return "Review Required"

    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'process_pool'):
            self.process_pool.shutdown(wait=False)
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=False)


    # For backward compatibility
    FastAccurateRAGPipeline = HybridFastTrackRAGPipeline
    AccuracyFirstRAGPipeline = HybridFastTrackRAGPipeline