# app/core/universal_parser.py - Universal document parsing with Tika
import io
import os
import re
import logging
import tempfile
import zipfile
import tarfile
import json
from typing import Tuple, List, Dict, Optional, Any
from tika import parser as tika_parser
from tika import config as tika_config
import pandas as pd
import camelot
import tabula
from PIL import Image
import pytesseract
import easyocr
import chardet
from app.core.document_parser import DocumentParser
from app.core.config import settings
# In universal_parser.py, add after imports:
import os
os.environ['JAVA_HOME'] = '/usr/lib/jvm/default-java'  # Set Java home for Tika

# Initialize Tika properly
# from tika import tika
# tika.initVM()
from tika import parser as tika_parser, config as tika_config
logger = logging.getLogger(__name__)

# Initialize Tika
tika_config.getParsers()

class UniversalDocumentParser:
    """Universal parser that handles any document format"""
    
    def __init__(self):
        self.fallback_parser = DocumentParser()
        self.easyocr_reader = None  # Lazy load
        
    # def parse_any_document(self, content: bytes, url: str) -> Tuple[str, List[Dict]]:
    #     """Parse any document format using multiple strategies"""
        
    #     # Detect file extension
    #     file_extension = self._get_extension(url)
        
    #     # Try specialized parsers first for better quality
    #     if file_extension in ['.xlsx', '.xls', '.csv']:
    #         return self._parse_spreadsheet_advanced(content, file_extension)
    #     elif file_extension == '.pdf':
    #         return self._parse_pdf_advanced(content)
    #     elif file_extension in ['.zip', '.tar', '.gz', '.rar', '.7z']:
    #         return self._parse_archive_advanced(content, file_extension)
    #     elif file_extension in ['.bin', '.dat', '.raw']:
    #         return self._parse_binary_file(content)
        
    #     # Try Tika for universal parsing
    #     try:
    #         text, metadata = self._parse_with_tika(content)
    #         if text and len(text) > 50:
    #             return text, metadata
    #     except Exception as e:
    #         logger.warning(f"Tika parsing failed: {e}")
        
    #     # Fallback to original parser
    #     try:
    #         return self.fallback_parser.parse_document(content, file_extension)
    #     except Exception as e:
    #         logger.error(f"All parsing methods failed: {e}")
    #         return self._extract_raw_text(content)
    # Replace your existing parse_any_document method with this one
    def parse_any_document(self, content: bytes, url: str) -> Tuple[str, List[Dict]]:
        """
        Parses any document by writing its in-memory content to a temporary
        file and then using the disk-based parsing workflow.

        Args:
            content: The document content as bytes.
            url: The original URL of the document.

        Returns:
            A tuple containing the extracted text and metadata.
        """
        temp_file_path = None
        try:
            # Create a temporary file to store the content
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_extension(url)) as tmp:
                tmp.write(content)
                temp_file_path = tmp.name

            # Delegate to the new disk-based parsing method
            return self.parse_document_from_path(temp_file_path, url)

        finally:
            # Ensure the temporary file is always cleaned up
            if temp_file_path and os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
    # Add this new method to your UniversalDocumentParser class
    def parse_document_from_path(self, file_path: str, url: str) -> Tuple[str, List[Dict]]:
        """
        Parses a document directly from a file path. This is the preferred method
        for large files to avoid loading them into memory.

        Args:
            file_path: The local path to the document.
            url: The original URL of the document, used to infer the file type.

        Returns:
            A tuple containing the extracted text and a list of metadata dictionaries.
        """
        file_extension = self._get_extension(url)
        logger.info(f"Parsing document from path: {file_path} with extension: {file_extension}")

        try:
            # Route to a specific disk-based parser if available
            if file_extension in ['.bin', '.dat', '.raw']:
                return self._parse_binary_file_from_path(file_path)

            # Fallback to Tika for universal parsing directly from a file path
            # Tika is excellent at handling various formats from disk
            if settings.USE_TIKA:
                logger.info(f"Using Apache Tika for {file_path}")
                parsed = tika_parser.from_file(file_path)
                text = parsed.get('content', '').strip()
                metadata = [{'type': 'tika_from_path', 'metadata': parsed.get('metadata', {})}]
                return text, metadata

            # If Tika is not used, read the content and use the in-memory parser
            with open(file_path, 'rb') as f:
                content = f.read()
            return self.fallback_parser.parse_document(content, file_extension)

        except Exception as e:
            logger.error(f"Failed to parse from path {file_path}: {e}", exc_info=True)
            return f"Error parsing file from disk: {e}", [{'type': 'error'}]  
    # Add this helper method to your UniversalDocumentParser class
    def _parse_binary_file_from_path(self, file_path: str) -> Tuple[str, List[Dict]]:
        """
        Parses a large binary file from a path by streaming it and extracting
        readable ASCII strings.

        Args:
            file_path: The local path to the binary file.

        Returns:
            A tuple containing the extracted strings and metadata.
        """
        logger.info(f"Parsing binary file from path: {file_path}")
        text_parts = ["=== EXTRACTED STRINGS FROM BINARY FILE ==="]
        metadata = [{'type': 'binary_stream'}]
        max_output_chars = 5_000_000  # Limit output to ~5MB of text to not overload the LLM
        current_chars = 0

        try:
            with open(file_path, 'rb') as f:
                while True:
                    # Read the file in 1MB chunks
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break  # End of file

                    strings = self._extract_strings(chunk)
                    for s in strings:
                        if current_chars + len(s) > max_output_chars:
                            logger.warning("Reached max character limit for binary file string extraction.")
                            return "\n".join(text_parts), metadata
                        text_parts.append(s)
                        current_chars += len(s)

            return "\n".join(text_parts), metadata
        except Exception as e:
            logger.error(f"Could not process binary file {file_path}: {e}")
            return f"Error processing binary file: {e}", [{'type': 'error'}]      
    
    def _parse_with_tika(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Parse using Apache Tika"""
        try:
            # Write to temp file for Tika
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Parse with Tika
            parsed = tika_parser.from_file(tmp_path)
            
            # Extract text and metadata
            text = parsed.get('content', '').strip()
            tika_metadata = parsed.get('metadata', {})
            
            # Convert metadata
            metadata = [{
                'type': 'tika',
                'format': tika_metadata.get('Content-Type', 'unknown'),
                'pages': tika_metadata.get('xmpTPg:NPages', 0),
                'author': tika_metadata.get('Author', ''),
                'title': tika_metadata.get('title', '')
            }]
            
            # Clean up
            os.unlink(tmp_path)
            
            return text, metadata
            
        except Exception as e:
            logger.error(f"Tika parsing error: {e}")
            raise
    
    def _parse_spreadsheet_advanced(self, content: bytes, extension: str) -> Tuple[str, List[Dict]]:
        """Advanced spreadsheet parsing with better structure preservation"""
        try:
            # Read spreadsheet
            if extension == '.csv':
                # Detect encoding
                encoding = chardet.detect(content)['encoding'] or 'utf-8'
                df = pd.read_csv(io.BytesIO(content), encoding=encoding, error_bad_lines=False)
            else:
                excel_file = pd.ExcelFile(io.BytesIO(content))
                all_text = []
                all_metadata = []
                
                # Process each sheet
                for sheet_name in excel_file.sheet_names:
                    df = excel_file.parse(sheet_name)
                    sheet_text, sheet_meta = self._process_dataframe(df, sheet_name)
                    all_text.append(f"\n=== SHEET: {sheet_name} ===\n{sheet_text}")
                    all_metadata.extend(sheet_meta)
                
                return "\n".join(all_text), all_metadata
            
            # Single sheet processing
            return self._process_dataframe(df, "main")
            
        except Exception as e:
            logger.error(f"Advanced spreadsheet parsing failed: {e}")
            return self.fallback_parser.parse_spreadsheet(content, extension)
    
    def _process_dataframe(self, df: pd.DataFrame, sheet_name: str) -> Tuple[str, List[Dict]]:
        """Process a dataframe with intelligent structure detection"""
        text_parts = []
        
        # 1. Summary
        text_parts.append(f"Sheet: {sheet_name}")
        text_parts.append(f"Dimensions: {len(df)} rows × {len(df.columns)} columns")
        text_parts.append(f"Columns: {', '.join(df.columns)}")
        
        # 2. Detect and extract key patterns
        # Look for totals, summaries, etc.
        for col in df.columns:
            if 'total' in col.lower() or 'sum' in col.lower():
                if pd.api.types.is_numeric_dtype(df[col]):
                    text_parts.append(f"{col}: {df[col].sum()}")
        
        # 3. Statistical summary for numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        if not numeric_df.empty:
            text_parts.append("\n=== STATISTICS ===")
            text_parts.append(numeric_df.describe().to_string())
        
        # 4. Full data (intelligently sampled for large datasets)
        text_parts.append("\n=== DATA ===")
        if len(df) <= 500:
            text_parts.append(df.to_string())
        else:
            # Include head, tail, and sample
            text_parts.append("First 100 rows:")
            text_parts.append(df.head(100).to_string())
            text_parts.append("\nLast 100 rows:")
            text_parts.append(df.tail(100).to_string())
            text_parts.append(f"\n({len(df) - 200} rows omitted)")
        
        metadata = [{
            'type': 'spreadsheet',
            'sheet': sheet_name,
            'rows': len(df),
            'columns': len(df.columns)
        }]
        
        return "\n".join(text_parts), metadata
    
    def _parse_pdf_advanced(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Advanced PDF parsing with table extraction"""
        all_text = []
        all_metadata = []
        
        try:
            # Save to temp file for advanced processing
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # 1. Extract text with pdfplumber (best quality)
            text, metadata = self.fallback_parser.parse_pdf(content)
            all_text.append(text)
            all_metadata.extend(metadata)
            
            # 2. Extract tables with Camelot (best for tables)
            try:
                tables = camelot.read_pdf(tmp_path, pages='all', flavor='lattice')
                if not tables:
                    tables = camelot.read_pdf(tmp_path, pages='all', flavor='stream')
                
                for i, table in enumerate(tables):
                    table_text = f"\n=== EXTRACTED TABLE {i+1} (Page {table.page}) ===\n"
                    table_text += table.df.to_string()
                    all_text.append(table_text)
                    all_metadata.append({'type': 'extracted_table', 'page': table.page})
                    
            except Exception as e:
                logger.warning(f"Camelot table extraction failed: {e}")
                
                # Try Tabula as fallback
                try:
                    tables = tabula.read_pdf(tmp_path, pages='all', multiple_tables=True)
                    for i, table in enumerate(tables):
                        table_text = f"\n=== EXTRACTED TABLE {i+1} ===\n"
                        table_text += table.to_string()
                        all_text.append(table_text)
                        all_metadata.append({'type': 'extracted_table_tabula'})
                except:
                    pass
            
            # Clean up
            os.unlink(tmp_path)
            
            return "\n".join(all_text), all_metadata
            
        except Exception as e:
            logger.error(f"Advanced PDF parsing failed: {e}")
            return self.fallback_parser.parse_pdf(content)
    
    def _parse_archive_advanced(self, content: bytes, extension: str) -> Tuple[str, List[Dict]]:
        """Parse archive files with streaming to avoid memory issues"""
        text_parts = []
        metadata = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Save archive to temp file
            archive_path = os.path.join(temp_dir, f'archive{extension}')
            with open(archive_path, 'wb') as f:
                f.write(content)
            
            # Extract based on type
            if extension in ['.zip']:
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    # Process files in batches to avoid memory issues
                    for file_info in zf.filelist:  # Limit to first 100 files
                        if file_info.is_dir() or file_info.file_size > 50 * 1024 * 1024:  # Skip large files
                            continue
                        
                        try:
                            file_content = zf.read(file_info.filename)
                            file_ext = os.path.splitext(file_info.filename)[1]
                            
                            # Parse the file
                            file_text, file_meta = self.parse_any_document(file_content, file_info.filename)
                            
                            text_parts.append(f"\n=== FILE: {file_info.filename} ===")
                            text_parts.append(file_text[:5000])  # Limit text per file
                            metadata.append({'file': file_info.filename, 'size': file_info.file_size})
                            
                        except Exception as e:
                            logger.warning(f"Failed to process {file_info.filename}: {e}")
                            
            elif extension in ['.tar', '.gz', '.tgz']:
                with tarfile.open(archive_path, 'r:*') as tf:
                    for member in tf.getmembers():  # Limit
                        if member.isfile() and member.size < 50 * 1024 * 1024:
                            try:
                                file_content = tf.extractfile(member).read()
                                file_text, file_meta = self.parse_any_document(file_content, member.name)
                                
                                text_parts.append(f"\n=== FILE: {member.name} ===")
                                text_parts.append(file_text[:5000])
                                metadata.append({'file': member.name, 'size': member.size})
                                
                            except Exception as e:
                                logger.warning(f"Failed to process {member.name}: {e}")
            
            return "\n".join(text_parts), metadata
            
        except Exception as e:
            logger.error(f"Archive parsing failed: {e}")
            return f"Archive parsing error: {str(e)}", [{'type': 'error'}]
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
    
    def _parse_binary_file(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Parse binary files by attempting multiple strategies"""
        text_parts = []
        metadata = [{'type': 'binary'}]
        
        # 1. Try to detect if it's text-based binary
        try:
            # Check if it's actually text
            text = content.decode('utf-8', errors='ignore')
            if self._is_mostly_text(text):
                return text, metadata
        except:
            pass
        
        # 2. Try to extract strings
        strings = self._extract_strings(content)
        if strings:
            text_parts.append("=== EXTRACTED STRINGS ===")
            text_parts.extend(strings[:1000])  # Limit output
        
        # 3. Hex dump for analysis (first 1KB)
        text_parts.append("\n=== HEX DUMP (First 1KB) ===")
        hex_dump = self._create_hex_dump(content[:1024])
        text_parts.append(hex_dump)
        
        # 4. File analysis
        text_parts.append("\n=== FILE ANALYSIS ===")
        text_parts.append(f"File size: {len(content)} bytes")
        text_parts.append(f"Entropy: {self._calculate_entropy(content):.2f}")
        
        # Detect file type using magic bytes
        file_type = self._detect_file_type(content)
        text_parts.append(f"Detected type: {file_type}")
        
        return "\n".join(text_parts), metadata
    
    def _extract_strings(self, content: bytes, min_length: int = 4) -> List[str]:
        """Extract ASCII strings from binary content"""
        strings = []
        current = []
        
        for byte in content:
            if 32 <= byte <= 126:  # Printable ASCII
                current.append(chr(byte))
            else:
                if len(current) >= min_length:
                    strings.append(''.join(current))
                current = []
        
        if len(current) >= min_length:
            strings.append(''.join(current))
        
        return strings
    
    def _create_hex_dump(self, content: bytes) -> str:
        """Create a hex dump of binary content"""
        lines = []
        for i in range(0, len(content), 16):
            chunk = content[i:i+16]
            hex_str = ' '.join(f'{b:02x}' for b in chunk)
            ascii_str = ''.join(chr(b) if 32 <= b <= 126 else '.' for b in chunk)
            lines.append(f'{i:08x}  {hex_str:<48}  {ascii_str}')
        return '\n'.join(lines)
    
    def _calculate_entropy(self, content: bytes) -> float:
        """Calculate Shannon entropy of content"""
        import math
        from collections import Counter
        
        if not content:
            return 0
        
        counter = Counter(content)
        entropy = 0
        total = len(content)
        
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        return entropy
    
    def _detect_file_type(self, content: bytes) -> str:
        """Detect file type from magic bytes"""
        magic_bytes = {
            b'\x89PNG': 'PNG Image',
            b'\xFF\xD8\xFF': 'JPEG Image',
            b'GIF8': 'GIF Image',
            b'%PDF': 'PDF Document',
            b'PK\x03\x04': 'ZIP Archive',
            b'\x1F\x8B': 'GZIP Archive',
            b'Rar!': 'RAR Archive',
            b'\x42\x4D': 'BMP Image',
            b'\x49\x49\x2A\x00': 'TIFF Image',
            b'\x4D\x4D\x00\x2A': 'TIFF Image',
            b'\x50\x4B\x03\x04': 'Office Document',
            b'\xD0\xCF\x11\xE0': 'MS Office Document',
            b'\x7F\x45\x4C\x46': 'ELF Executable',
            b'\x4D\x5A': 'PE Executable',
        }
        
        for magic, file_type in magic_bytes.items():
            if content.startswith(magic):
                return file_type
        
        return 'Unknown'
    
    def _is_mostly_text(self, text: str) -> bool:
        """Check if content is mostly readable text"""
        if not text:
            return False
        
        printable = sum(1 for c in text if c.isprintable() or c.isspace())
        return (printable / len(text)) > 0.85
    
    def _extract_raw_text(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Last resort: extract any readable text"""
        try:
            # Try different encodings
            for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
                try:
                    text = content.decode(encoding, errors='ignore')
                    if text and self._is_mostly_text(text):
                        return text, [{'type': 'raw_text', 'encoding': encoding}]
                except:
                    continue
            
            # Extract strings if all else fails
            strings = self._extract_strings(content)
            return "\n".join(strings), [{'type': 'extracted_strings'}]
            
        except Exception as e:
            return f"Unable to extract text: {str(e)}", [{'type': 'error'}]
    
    def _get_extension(self, url: str) -> str:
        """Get file extension from URL"""
        import urllib.parse
        
        # Parse URL to get path
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        
        # Get extension
        ext = os.path.splitext(path)[1].lower()
        
        # Handle common cases
        if not ext:
            # Try to extract from query parameter
            if 'format=' in url:
                format_param = urllib.parse.parse_qs(parsed.query).get('format', [''])[0]
                if format_param:
                    ext = f'.{format_param}'
            # Check for common patterns in URL
            elif '.pdf' in url:
                ext = '.pdf'
            elif '.xlsx' in url or '.xls' in url:
                ext = '.xlsx'
            elif '.docx' in url or '.doc' in url:
                ext = '.docx'
            elif '.zip' in url:
                ext = '.zip'
        
        return ext or '.unknown'