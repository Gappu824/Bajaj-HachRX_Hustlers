# app/core/smart_chunker.py - Intelligent chunking
import re
from typing import List, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

class SmartChunker:
    """Context-aware document chunking"""
    
    @staticmethod
    def chunk_document(text: str, metadata: List[Dict], 
                      chunk_size: int = 1000, 
                      overlap: int = 200) -> Tuple[List[str], List[Dict]]:
        """Smart chunking based on document structure"""
        doc_length = len(text)
    
    # Adjust chunk size for large documents
        if doc_length > 100000:  # > 100k chars
            chunk_size = min(2000, chunk_size * 2)  # Double chunk size
            overlap = min(400, overlap * 2)
            logger.info(f"Using larger chunks for document with {doc_length} chars")
        
        # Limit total chunks
        max_chunks = settings.MAX_TOTAL_CHUNKS
        min_chunk_size = max(chunk_size, doc_length // max_chunks)
        
        if min_chunk_size > chunk_size:
            chunk_size = min_chunk_size
            logger.warning(f"Increased chunk size to {chunk_size} to stay under {max_chunks} chunks")
        
        # Detect document type from metadata
        doc_type = SmartChunker._detect_document_type(metadata)
        
        if doc_type == 'spreadsheet':
            return SmartChunker._chunk_spreadsheet(text, metadata)
        elif doc_type == 'presentation':
            return SmartChunker._chunk_presentation(text, metadata)
        else:
            return SmartChunker._chunk_text_semantic(text, metadata, chunk_size, overlap)
    
    @staticmethod
    def _detect_document_type(metadata: List[Dict]) -> str:
        """Detect document type from metadata"""
        if any(m.get('type') == 'spreadsheet' for m in metadata):
            return 'spreadsheet'
        elif any(m.get('type') == 'slide' for m in metadata):
            return 'presentation'
        else:
            return 'text'
    
    @staticmethod
    def _chunk_spreadsheet(text: str, metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Special chunking for spreadsheets"""
        chunks = []
        chunk_metadata = []
        
        lines = text.split('\n')
        
        # Extract headers and summary
        header_lines = []
        data_lines = []
        current_section = 'header'
        
        for line in lines:
            if '=== DATA ROWS ===' in line:
                current_section = 'data'
            elif current_section == 'header' or line.startswith('Column Headers:'):
                header_lines.append(line)
            else:
                data_lines.append(line)
        
        # Create header context
        header_context = '\n'.join(header_lines)
        
        # Chunk data rows with header context
        chunk_size = 20  # rows per chunk
        current_chunk = []
        
        for i, line in enumerate(data_lines):
            if line.startswith('Row '):
                current_chunk.append(line)
                
                if len(current_chunk) >= chunk_size:
                    # Create chunk with header
                    chunk_text = header_context + '\n\n' + '\n'.join(current_chunk)
                    chunks.append(chunk_text)
                    chunk_metadata.append({
                        'type': 'spreadsheet_chunk',
                        'rows': len(current_chunk)
                    })
                    current_chunk = []
        
        # Add remaining
        if current_chunk:
            chunk_text = header_context + '\n\n' + '\n'.join(current_chunk)
            chunks.append(chunk_text)
            chunk_metadata.append({
                'type': 'spreadsheet_chunk',
                'rows': len(current_chunk)
            })
        
        return chunks, chunk_metadata
    
    @staticmethod
    def _chunk_presentation(text: str, metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Chunk presentation by slides"""
        chunks = []
        chunk_metadata = []
        
        # Split by slide markers
        slides = re.split(r'=== SLIDE \d+ ===', text)
        
        for i, slide_content in enumerate(slides[1:], 1):  # Skip empty first split
            if slide_content.strip():
                chunks.append(f"=== SLIDE {i} ===\n{slide_content.strip()}")
                chunk_metadata.append({'type': 'slide', 'slide_num': i})
        
        return chunks, chunk_metadata
    
    @staticmethod
    def _chunk_text_semantic(text: str, metadata: List[Dict], 
                            chunk_size: int, overlap: int) -> Tuple[List[str], List[Dict]]:
        """Semantic chunking for text documents"""
        chunks = []
        chunk_metadata = []
        
        # Split by major sections
        sections = re.split(r'\n(?:={3,}|#{1,3})\s*([^\n]+)', text)
        
        current_chunk = ""
        current_meta = {'type': 'text'}
        
        for i, section in enumerate(sections):
            if not section.strip():
                continue
            
            # Check if this is a section header
            if i % 2 == 1 and i > 0:  # Headers are at odd indices
                section = f"### {section}"  # Mark as header
                current_meta['section'] = section.strip()
            
            # Smart paragraph splitting
            paragraphs = re.split(r'\n\n+', section)
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                
                # Check chunk size
                if len(current_chunk) + len(para) > chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append(current_chunk.strip())
                    chunk_metadata.append(current_meta.copy())
                    
                    # Start new chunk with overlap
                    if len(current_chunk) > overlap:
                        # Take last part as overlap
                        overlap_text = ' '.join(current_chunk.split()[-overlap//4:])
                        current_chunk = overlap_text + "\n\n" + para
                    else:
                        current_chunk = para
                    
                    # Update metadata
                    current_meta = {'type': 'text'}
                    if 'section' in current_meta:
                        current_meta['section'] = current_meta.get('section', '')
                else:
                    if current_chunk:
                        current_chunk += "\n\n" + para
                    else:
                        current_chunk = para
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            chunk_metadata.append(current_meta)
        
        # If no chunks created, use fallback
        if not chunks:
            # Simple splitting
            # Simple splitting fallback
            words = text.split()
            for i in range(0, len(words), chunk_size // 5):  # Assuming ~5 chars per word
                chunk_words = words[i:i + chunk_size // 5]
                if chunk_words:
                    chunks.append(' '.join(chunk_words))
                    chunk_metadata.append({'type': 'text', 'fallback': True})
        
        return chunks, chunk_metadata