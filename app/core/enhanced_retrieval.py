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
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize BM25
        self._init_bm25()
        
        # Initialize TF-IDF
        self._init_tfidf()
        
        # Build specialized indexes
        self._build_specialized_indexes()
        
        logger.info(f"Enhanced retriever initialized with {len(chunks)} chunks")
    
    def _init_bm25(self):
        """Initialize BM25 index"""
        tokenized_chunks = []
        for chunk in self.chunks:
            tokens = word_tokenize(chunk.lower())
            tokens = [t for t in tokens if t.isalnum() and t not in self.stop_words]
            tokenized_chunks.append(tokens)
        
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def _init_tfidf(self):
        """Initialize TF-IDF vectorizer"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            stop_words='english',
            min_df=1,
            max_df=0.95
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
    
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
                matches = re.findall(pattern, chunk_lower)
                for match in matches:
                    term = match if isinstance(match, str) else match[0]
                    if term not in self.indexes['definitions']:
                        self.indexes['definitions'][term] = []
                    self.indexes['definitions'][term].append(idx)
            
            # Number patterns with context
            number_patterns = [
                r'(\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:%|percent|km|kg|lakh|crore|days?|years?|months?)',
                r'(?:Rs\.?|INR|₹)\s*(\d+(?:,\d{3})*(?:\.\d+)?)',
                r'(\d+(?:\.\d+)?)\s*(?:to|and|-)\s*(\d+(?:\.\d+)?)'
            ]
            for pattern in number_patterns:
                matches = re.findall(pattern, chunk)
                for match in matches:
                    if match not in self.indexes['numbers']:
                        self.indexes['numbers'][match] = []
                    self.indexes['numbers'][match].append(idx)
            
            # Table detection
            if metadata.get('type') == 'table' or '|' in chunk:
                if 'tables' not in self.indexes['tables']:
                    self.indexes['tables']['all'] = []
                self.indexes['tables']['all'].append(idx)
            
            # Procedure/steps detection
            if any(marker in chunk_lower for marker in ['step 1', 'step 2', 'firstly', 'secondly', 'procedure:', 'process:']):
                if 'procedures' not in self.indexes['procedures']:
                    self.indexes['procedures']['all'] = []
                self.indexes['procedures']['all'].append(idx)
            
            # List detection
            if any(marker in chunk for marker in ['•', '1.', '2.', '- ', '* ', 'following:', 'includes:']):
                if 'lists' not in self.indexes['lists']:
                    self.indexes['lists']['all'] = []
                self.indexes['lists']['all'].append(idx)
    
    def retrieve(self, query: str, k: int = 30, question_type: str = 'general') -> List[Tuple[int, float]]:
        """Multi-stage retrieval combining different methods"""
        query_lower = query.lower()
        scores = np.zeros(len(self.chunks))
        
        # Stage 1: Exact phrase matching
        for idx, chunk in enumerate(self.chunks):
            if query_lower in chunk.lower():
                scores[idx] += 5.0
        
        # Stage 2: BM25 retrieval
        query_tokens = word_tokenize(query_lower)
        query_tokens = [t for t in query_tokens if t.isalnum() and t not in self.stop_words]
        bm25_scores = self.bm25.get_scores(query_tokens)
        scores += np.array(bm25_scores) * 2.0
        
        # Stage 3: TF-IDF similarity
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        scores += tfidf_scores * 1.5
        
        # Stage 4: Specialized index boosting
        if question_type == 'definitional':
            for term in query_lower.split():
                if term in self.indexes['definitions']:
                    for idx in self.indexes['definitions'][term]:
                        scores[idx] += 3.0
        
        elif question_type == 'computational':
            # Extract numbers from query
            numbers = re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', query)
            for num in numbers:
                for key in self.indexes['numbers']:
                    if num in str(key):
                        for idx in self.indexes['numbers'][key]:
                            scores[idx] += 2.0
        
        elif question_type == 'list_based':
            if 'all' in self.indexes['lists']:
                for idx in self.indexes['lists']['all']:
                    scores[idx] += 1.5
        
        elif question_type == 'procedural':
            if 'all' in self.indexes['procedures']:
                for idx in self.indexes['procedures']['all']:
                    scores[idx] += 1.5
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        return results