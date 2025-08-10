# app/core/enhanced_retrieval.py - Better keyword and semantic matching with fixes
import re
import numpy as np
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize
import logging

logger = logging.getLogger(__name__)

# Download NLTK data with error handling
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)  # New requirement
except Exception as e:
    logger.warning(f"Failed to download NLTK data: {e}")

class EnhancedRetriever:
    """Advanced retrieval with multiple strategies and robust error handling"""
    
    def __init__(self, chunks: List[str], chunk_metadata: List[Dict]):
        self.chunks = chunks
        self.chunk_metadata = chunk_metadata
        
        # Initialize components with error handling
        self._init_keyword_index()
        self._init_bm25()
        self._init_tfidf()
        
        logger.info(f"Enhanced retriever initialized with {len(chunks)} chunks")
    
    def _init_keyword_index(self):
        """Build keyword index for exact matching"""
        self.keyword_index = {}
        
        try:
            for idx, chunk in enumerate(self.chunks):
                if not chunk:
                    continue
                    
                chunk_lower = chunk.lower()
                
                # Extract important numbers with units
                numbers = re.findall(
                    r'\b\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:%|percent|lakh|crore|million|billion|k|mb|gb))?\b',
                    chunk_lower
                )
                
                # Extract important phrases (2-grams and 3-grams)
                words = chunk_lower.split()
                for n in [2, 3]:
                    for i in range(len(words) - n + 1):
                        phrase = ' '.join(words[i:i+n])
                        if len(phrase) > 5 and phrase.isascii():  # Filter non-ASCII
                            if phrase not in self.keyword_index:
                                self.keyword_index[phrase] = set()
                            self.keyword_index[phrase].add(idx)
                
                # Add numbers to index
                for num in numbers:
                    if num not in self.keyword_index:
                        self.keyword_index[num] = set()
                    self.keyword_index[num].add(idx)
                    
        except Exception as e:
            logger.warning(f"Keyword index initialization failed: {e}")
            self.keyword_index = {}
    
    def _init_bm25(self):
        """Initialize BM25 for probabilistic retrieval with robust tokenization"""
        try:
            tokenized_chunks = []
            
            for chunk in self.chunks:
                if not chunk:
                    tokenized_chunks.append(['empty'])
                    continue
                
                try:
                    # Use NLTK tokenizer with fallback
                    tokens = word_tokenize(chunk.lower())
                except:
                    # Fallback to simple split if NLTK fails
                    tokens = chunk.lower().split()
                
                # Filter tokens - keep alphanumeric only
                filtered_tokens = [t for t in tokens if t.isalnum() and len(t) > 1]
                tokenized_chunks.append(filtered_tokens if filtered_tokens else ['empty'])
            
            self.bm25 = BM25Okapi(tokenized_chunks)
            
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}")
            # Create a dummy BM25 that returns zeros
            self.bm25 = None
    
    def _init_tfidf(self):
        """Initialize TF-IDF with robust fallbacks"""
        if len(self.chunks) < 2:
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
            return
        
        try:
            # Filter out empty chunks
            valid_chunks = [chunk for chunk in self.chunks if chunk and chunk.strip()]
            
            if len(valid_chunks) < 2:
                self.tfidf_vectorizer = None
                self.tfidf_matrix = None
                return
            
            # Adjust parameters based on collection size
            use_stop_words = None  # Don't use stopwords to avoid language issues
            max_features = min(5000, len(valid_chunks) * 10)
            
            # Different settings for small vs large collections
            if len(valid_chunks) < 20:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 1),  # Only unigrams for small collections
                    stop_words=use_stop_words,
                    min_df=1,
                    max_df=1.0,
                    token_pattern=r'\b[a-zA-Z0-9][a-zA-Z0-9]+\b'  # Better token pattern
                )
            else:
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(1, 2),
                    stop_words=use_stop_words,
                    min_df=1,
                    max_df=0.95,
                    token_pattern=r'\b[a-zA-Z0-9][a-zA-Z0-9]+\b'
                )

            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(valid_chunks)
            
            # If we filtered chunks, create mapping
            self.valid_chunk_indices = [i for i, chunk in enumerate(self.chunks) if chunk and chunk.strip()]
            
        except Exception as e:
            logger.warning(f"TF-IDF initialization failed: {e}")
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
    
    def retrieve(self, query: str, k: int = 30) -> List[Tuple[int, float]]:
        """Multi-strategy retrieval with enhanced error handling"""
        if not self.chunks or not query.strip():
            return []
        
        query_lower = query.lower()
        scores = np.zeros(len(self.chunks))
        
        try:
            # 1. Exact phrase matching (highest weight)
            for idx, chunk in enumerate(self.chunks):
                if chunk and query_lower in chunk.lower():
                    scores[idx] += 10.0
            
            # 2. Keyword matching
            query_words = query_lower.split()
            for word in query_words:
                if word in self.keyword_index:
                    for idx in self.keyword_index[word]:
                        if idx < len(scores):
                            scores[idx] += 2.0
            
            # Check bigrams
            for i in range(len(query_words) - 1):
                bigram = ' '.join(query_words[i:i+2])
                if bigram in self.keyword_index:
                    for idx in self.keyword_index[bigram]:
                        if idx < len(scores):
                            scores[idx] += 3.0
            
            # 3. BM25 scoring
            if self.bm25:
                try:
                    # Tokenize query
                    try:
                        query_tokens = word_tokenize(query_lower)
                    except:
                        query_tokens = query_lower.split()
                    
                    query_tokens = [t for t in query_tokens if t.isalnum() and len(t) > 1]
                    
                    if query_tokens:
                        bm25_scores = self.bm25.get_scores(query_tokens)
                        if len(bm25_scores) == len(scores):
                            scores += np.array(bm25_scores) * 2.0
                            
                except Exception as e:
                    logger.warning(f"BM25 scoring failed: {e}")
            
            # 4. TF-IDF scoring
            if self.tfidf_vectorizer and self.tfidf_matrix is not None:
                try:
                    query_vec = self.tfidf_vectorizer.transform([query])
                    tfidf_scores = (self.tfidf_matrix * query_vec.T).toarray().flatten()
                    
                    # Map scores back to original indices if we filtered chunks
                    if hasattr(self, 'valid_chunk_indices'):
                        for i, original_idx in enumerate(self.valid_chunk_indices):
                            if i < len(tfidf_scores) and original_idx < len(scores):
                                scores[original_idx] += tfidf_scores[i] * 1.5
                    else:
                        if len(tfidf_scores) == len(scores):
                            scores += tfidf_scores * 1.5
                            
                except Exception as e:
                    logger.warning(f"TF-IDF scoring failed: {e}")
            
            # Get top-k results
            if np.max(scores) > 0:
                top_indices = np.argsort(scores)[-k:][::-1]
                results = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
            else:
                # Fallback: return first k chunks if no scoring worked
                results = [(i, 1.0) for i in range(min(k, len(self.chunks)))]
            
            return results
            
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            # Ultimate fallback
            return [(i, 1.0) for i in range(min(k, len(self.chunks)))]