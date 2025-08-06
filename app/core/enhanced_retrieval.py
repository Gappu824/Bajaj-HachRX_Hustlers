# app/core/enhanced_retrieval.py
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class EnhancedRetriever:
    """Multi-modal retriever combining BM25, TF-IDF, and semantic search"""
    
    def __init__(self, chunks: List[str], chunk_metadata: List[Dict]):
        self.chunks = chunks
        self.chunk_metadata = chunk_metadata
        
        # Initialize stop words
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Initialize BM25
        self._init_bm25()
        
        # Initialize TF-IDF with safeguards
        self._init_tfidf()
        
        # Build specialized indexes
        self._build_specialized_indexes()
        
        logger.info(f"Enhanced retriever initialized with {len(chunks)} chunks")
    
    def _init_bm25(self):
        """Initialize BM25 index"""
        if len(self.chunks) < 1:
            self.bm25 = None
            return
            
        tokenized_chunks = []
        for chunk in self.chunks:
            try:
                tokens = word_tokenize(chunk.lower())
                tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
                tokenized_chunks.append(tokens if tokens else ['empty'])
            except:
                tokenized_chunks.append(['empty'])
        
        try:
            self.bm25 = BM25Okapi(tokenized_chunks)
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}")
            self.bm25 = None
    
    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer with safeguards for small documents"""
        n_chunks = len(self.chunks)
        
        if n_chunks < 2:
            # Skip TF-IDF for single chunk documents
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return
        
        # Dynamic parameters based on document size
        min_df = 1
        max_df = min(0.95, max(0.5, (n_chunks - 1) / n_chunks))
        max_features = min(5000, n_chunks * 10)
        
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2) if n_chunks < 50 else (1, 3),
                stop_words='english',
                min_df=min_df,
                max_df=max_df
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
        except Exception as e:
            logger.warning(f"TF-IDF initialization failed: {e}, skipping TF-IDF")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def _build_specialized_indexes(self):
        """Build indexes for specific types of information"""
        self.indexes = {
            'definitions': {},
            'numbers': {},
            'procedures': {},
            'lists': {},
            'tables': {},
            'formulas': {}
        }
        
        for idx, (chunk, metadata) in enumerate(zip(self.chunks, self.chunk_metadata)):
            chunk_lower = chunk.lower()
            
            # Definition patterns
            definition_patterns = [
                r'(\w+)\s+(?:is|are|means|refers to|defined as)',
                r'definition of\s+(\w+)',
                r'"([^"]+)"\s+(?:means|is|refers to)'
            ]
            for pattern in definition_patterns:
                try:
                    matches = re.findall(pattern, chunk_lower)
                    for match in matches:
                        term = match if isinstance(match, str) else match[0]
                        if term not in self.indexes['definitions']:
                            self.indexes['definitions'][term] = []
                        self.indexes['definitions'][term].append(idx)
                except:
                    pass
            
            # Number patterns
            try:
                numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', chunk)
                for num in numbers:
                    if num not in self.indexes['numbers']:
                        self.indexes['numbers'][num] = []
                    self.indexes['numbers'][num].append(idx)
            except:
                pass
            
            # Table detection
            if metadata.get('type') == 'table' or '|' in chunk:
                if 'all' not in self.indexes['tables']:
                    self.indexes['tables']['all'] = []
                self.indexes['tables']['all'].append(idx)
    
    def retrieve(self, query: str, k: int = 30, question_type: str = 'general') -> List[Tuple[int, float]]:
        """Multi-stage retrieval combining different methods"""
        if not self.chunks:
            return []
            
        query_lower = query.lower()
        scores = np.zeros(len(self.chunks))
        
        # Stage 1: Exact phrase matching (highest priority)
        for idx, chunk in enumerate(self.chunks):
            if query_lower in chunk.lower():
                scores[idx] += 5.0
        
        # Stage 2: BM25 retrieval
        if self.bm25:
            try:
                query_tokens = word_tokenize(query_lower)
                query_tokens = [t for t in query_tokens if t.isalnum() and t not in self.stop_words]
                if query_tokens:
                    bm25_scores = self.bm25.get_scores(query_tokens)
                    scores += np.array(bm25_scores) * 2.0
            except Exception as e:
                logger.warning(f"BM25 scoring failed: {e}")
        
        # Stage 3: TF-IDF similarity
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                query_vec = self.tfidf_vectorizer.transform([query])
                tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
                scores += tfidf_scores * 1.5
            except Exception as e:
                logger.warning(f"TF-IDF scoring failed: {e}")
        
        # Stage 4: Specialized index boosting
        if question_type == 'definitional':
            for term in query_lower.split():
                if term in self.indexes['definitions']:
                    for idx in self.indexes['definitions'][term]:
                        scores[idx] += 3.0
        
        elif question_type == 'computational':
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', query)
            for num in numbers:
                if num in self.indexes['numbers']:
                    for idx in self.indexes['numbers'][num]:
                        scores[idx] += 2.0
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        # If no results, return all chunks with equal score
        if not results and self.chunks:
            results = [(i, 1.0) for i in range(min(k, len(self.chunks)))]
        
        return results