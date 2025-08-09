# app/core/enhanced_retrieval.py - Better keyword and semantic matching
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

# CHANGED: Avoid runtime downloads; provide safe fallbacks
try:
    _NLTK_AVAILABLE = True
    # Probe for resources without triggering downloads
    nltk.data.find('tokenizers/punkt')
except Exception:
    _NLTK_AVAILABLE = False


def _fast_tokenize(text: str) -> list:
    """Fast, dependency-light tokenizer as a fallback."""
    # Alphanum tokens only, lowercased
    return re.findall(r"[A-Za-z0-9]+", text.lower())


class EnhancedRetriever:
    """Advanced retrieval with multiple strategies"""
    
    def __init__(self, chunks: List[str], chunk_metadata: List[Dict]):
        self.chunks = chunks
        self.chunk_metadata = chunk_metadata
        
        # Initialize components
        self._init_keyword_index()
        self._init_bm25()
        self._init_tfidf()
        
        logger.info(f"Enhanced retriever initialized with {len(chunks)} chunks")
    
    def _init_keyword_index(self):
        """Build keyword index for exact matching"""
        self.keyword_index = {}
        
        for idx, chunk in enumerate(self.chunks):
            # Extract important terms
            chunk_lower = chunk.lower()
            
            # Numbers with units
            numbers = re.findall(
                r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:%|percent|lakh|crore|million|billion))?\b',
                chunk_lower
            )
            
            # Important phrases
            words = chunk_lower.split()
            for n in [2, 3]:  # 2-grams and 3-grams
                for i in range(len(words) - n + 1):
                    phrase = ' '.join(words[i:i+n])
                    if len(phrase) > 5:
                        if phrase not in self.keyword_index:
                            self.keyword_index[phrase] = set()
                        self.keyword_index[phrase].add(idx)
            
            # Add numbers to index
            for num in numbers:
                if num not in self.keyword_index:
                    self.keyword_index[num] = set()
                self.keyword_index[num].add(idx)
    
    def _init_bm25(self):
        """Initialize BM25 for probabilistic retrieval"""
        tokenized_chunks = []
        for chunk in self.chunks:
            if _NLTK_AVAILABLE:
                try:
                    tokens = word_tokenize(chunk.lower())
                    tokens = [t for t in tokens if t.isalnum()]
                except Exception:
                    tokens = _fast_tokenize(chunk)
            else:
                tokens = _fast_tokenize(chunk)
            tokenized_chunks.append(tokens if tokens else ['empty'])
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def _init_tfidf(self):
        """Initialize TF-IDF for term frequency analysis with robust fallbacks."""
        if len(self.chunks) < 2:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return
        
        try:
            use_stop_words = 'english' if len(self.chunks) >= 20 else None
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=min(5000, len(self.chunks) * 10),
                ngram_range=(1, 2),
                stop_words=use_stop_words,
                min_df=1,
                max_df=1.0
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
        except Exception as e:
            logger.warning(f"TF-IDF initialization failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def retrieve(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
        """Multi-strategy retrieval"""
        query_lower = query.lower()
        scores = np.zeros(len(self.chunks))
        
        # 1. Exact phrase matching
        for idx, chunk in enumerate(self.chunks):
            if query_lower in chunk.lower():
                scores[idx] += 10.0  # High weight for exact match
        
        # 2. Keyword matching
        query_words = query_lower.split()
        for word in query_words:
            if word in self.keyword_index:
                for idx in self.keyword_index[word]:
                    scores[idx] += 2.0
        
        # Check bigrams
        for i in range(len(query_words) - 1):
            bigram = ' '.join(query_words[i:i+2])
            if bigram in self.keyword_index:
                for idx in self.keyword_index[bigram]:
                    scores[idx] += 3.0
        
        # 3. BM25 scoring
        if _NLTK_AVAILABLE:
            try:
                query_tokens = [t for t in word_tokenize(query_lower) if t.isalnum()]
            except Exception:
                query_tokens = _fast_tokenize(query_lower)
        else:
            query_tokens = _fast_tokenize(query_lower)
        
        if query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            scores += np.array(bm25_scores) * 2.0
        
        # 4. TF-IDF scoring
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                query_vec = self.tfidf_vectorizer.transform([query])
                tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
                scores += tfidf_scores * 1.5
            except Exception:
                pass
        
        # Get top-k results
        top_indices = np.argsort(scores)[-k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        # If no results, return top chunks
        if not results:
            results = [(i, 1.0) for i in range(min(k, len(self.chunks)))]
        
        return results