# app/core/contextual_embeddings.py
import numpy as np
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class ContextualRetrieval:
    """Fast contextual retrieval using document structure"""
    
    @staticmethod
    def add_context_to_chunks(chunks: List[str], chunk_metadata: List[dict]) -> List[str]:
        """Add contextual information to chunks for better retrieval"""
        if not chunks:
            return chunks
            
        contextualized_chunks = []
        
        for i, (chunk, metadata) in enumerate(zip(chunks, chunk_metadata)):
            # Build context from metadata and surrounding chunks
            context_parts = []
            
            # Add document section if available
            if metadata.get('section'):
                context_parts.append(f"Section: {metadata['section']}")
            
            # Add page info
            if metadata.get('page'):
                context_parts.append(f"Page {metadata['page']}")
            
            # Add document type
            if metadata.get('type'):
                context_parts.append(f"Type: {metadata['type']}")
            
            # Add brief summary of previous chunk for continuity
            if i > 0 and len(chunks[i-1]) > 100:
                # Extract first sentence or first 100 chars
                prev_text = chunks[i-1]
                if '.' in prev_text[:100]:
                    prev_summary = prev_text[:prev_text.index('.')+1]
                else:
                    prev_summary = prev_text[:100].strip() + "..."
                context_parts.append(f"Previous: {prev_summary}")
            
            # Combine context with chunk
            if context_parts:
                context_str = " | ".join(context_parts)
                contextualized_chunk = f"[Context: {context_str}]\n{chunk}"
            else:
                contextualized_chunk = chunk
            
            contextualized_chunks.append(contextualized_chunk)
        
        return contextualized_chunks
    
    @staticmethod
    def extract_key_sentences(text: str, max_sentences: int = 3) -> str:
        """Extract key sentences from text for faster processing"""
        if not text:
            return text
            
        sentences = text.split('.')
        if len(sentences) <= max_sentences:
            return text
        
        # Prioritize sentences with numbers, definitions, or key terms
        scored_sentences = []
        for sent in sentences:
            score = 0
            sent_lower = sent.lower()
            
            # Score based on content
            if any(term in sent_lower for term in ['means', 'defined', 'includes', 'excludes']):
                score += 3
            if any(char.isdigit() for char in sent):
                score += 2
            if len(sent.split()) > 5:  # Non-trivial sentences
                score += 1
            
            scored_sentences.append((sent, score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        key_sentences = [sent for sent, _ in scored_sentences[:max_sentences]]
        
        return '. '.join(key_sentences) + '.'