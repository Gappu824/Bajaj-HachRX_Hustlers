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

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

class EnhancedRetriever:
    """Advanced retrieval with multiple strategies"""
    
    def __init__(self, chunks: List[str], chunk_metadata: List[Dict]):
        self.chunks = chunks
        self.chunk_metadata = chunk_metadata or []
        
        # Initialize components
        self._init_keyword_index()
        self._init_bm25()
        self._init_tfidf()
        
        logger.info(f"Enhanced retriever initialized with {len(chunks)} chunks")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25"""
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        tokens = word_tokenize(text.lower())
        tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
        return tokens if tokens else ['empty']    
    
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
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        tokenized_chunks = []
        for chunk in self.chunks:
            tokens = word_tokenize(chunk.lower())
            tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
            tokenized_chunks.append(tokens if tokens else ['empty'])
        
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    # def _init_tfidf(self):
    #     """Initialize TF-IDF for term frequency analysis"""
    #     if len(self.chunks) < 2:
    #         self.tfidf_vectorizer = None
    #         self.tfidf_matrix = None
    #         return
        
    #     try:
    #         # self.tfidf_vectorizer = TfidfVectorizer(
    #         #     max_features=min(5000, len(self.chunks) * 10),
    #         #     ngram_range=(1, 2),
    #         #     stop_words='english',
    #         #     min_df=1,
    #         #     max_df=0.95
    #         # )
    #         # Adjust parameters based on collection size
    #         if len(self.chunks) < 10:
    #             # Very small collections - be more permissive
    #             self.tfidf_vectorizer = TfidfVectorizer(
    #                 max_features=min(100, len(self.chunks) * 20),
    #                 ngram_range=(1, 1),  # Only unigrams for small collections
    #                 stop_words=None,     # Don't remove stop words for small collections
    #                 min_df=1,
    #                 max_df=1.0           # Allow all terms
    #             )
    #         else:
    #             # Larger collections - use original settings
    #             self.tfidf_vectorizer = TfidfVectorizer(
    #                 max_features=min(5000, len(self.chunks) * 10),
    #                 ngram_range=(1, 2),
    #                 stop_words='english',
    #                 min_df=1,
    #                 max_df=0.95
    #             )
    #         self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
    #     except Exception as e:
    #         logger.warning(f"TF-IDF initialization failed: {e}")
    #         self.tfidf_vectorizer = None
    #         self.tfidf_matrix = None
    def _init_tfidf(self):
        """Initialize TF-IDF with error handling"""
        # OLD: Fails on small collections
        # NEW: Adaptive initialization
        
        if len(self.chunks) < 2:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            logger.warning("Not enough chunks for TF-IDF (need at least 2)")
            return
        
        try:
            # Adjust parameters based on collection size
            if len(self.chunks) < 10:
                # Very small collections
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=min(100, len(self.chunks) * 20),
                    ngram_range=(1, 1),  # Only unigrams
                    stop_words=None,     # Keep all words
                    min_df=1,
                    max_df=1.0,
                    token_pattern=r'(?u)\b\w+\b'  # Single character tokens allowed
                )
            else:
                # Normal collections
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=min(5000, len(self.chunks) * 10),
                    ngram_range=(1, 2),
                    stop_words='english',
                    min_df=1,
                    max_df=0.95
                )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunks)
            logger.info(f"TF-IDF initialized with {self.tfidf_matrix.shape[0]} documents, {self.tfidf_matrix.shape[1]} features")
            
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
        try:
            stop_words = set(stopwords.words('english'))
        except:
            stop_words = set()
        
        query_tokens = word_tokenize(query_lower)
        query_tokens = [t for t in query_tokens if t.isalnum() and t not in stop_words]
        
        if query_tokens:
            bm25_scores = self.bm25.get_scores(query_tokens)
            scores += np.array(bm25_scores) * 2.0
        
        # 4. TF-IDF scoring
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                query_vec = self.tfidf_vectorizer.transform([query])
                tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
                scores += tfidf_scores * 1.5
            except:
                pass
        
        # Get top-k results
        top_indices = np.argsort(scores)[-k:][::-1]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]
        
        # If no results, return top chunks
        if not results:
            results = [(i, 1.0) for i in range(min(k, len(self.chunks)))]
        
        return results