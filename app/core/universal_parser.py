# app/core/universal_parser.py - Universal document parsing with enhanced error handling
import io
import os
import re
import logging
import tempfile
import zipfile
import tarfile
import json
import subprocess
from typing import Tuple, List, Dict, Optional, Any
import pandas as pd
from PIL import Image
import pytesseract
import chardet
from app.core.config import settings

logger = logging.getLogger(__name__)

# Set Java environment for Tika
os.environ.setdefault('JAVA_HOME', '/usr/lib/jvm/default-java')

class UniversalDocumentParser:
    """Universal parser that handles any document format with robust fallbacks"""
    
    def __init__(self):
        # Check Java availability for Tika
        self.tika_available = self._check_tika_availability()
        self.fallback_parser = None
        self._easyocr_reader = None
        self._camelot_available = self._check_camelot_availability()
        self._tabula_available = self._check_tabula_availability()
    
    def _check_tika_availability(self) -> bool:
        """Check if Tika/Java is available"""
        try:
            subprocess.run(['java', '-version'], capture_output=True, check=True, timeout=10)
            
            # Try importing Tika
            from tika import parser as tika_parser
            from tika import config as tika_config
            
            # Initialize Tika
            tika_config.getParsers()
            return True
            
        except (subprocess.CalledProcessError, FileNotFoundError, ImportError, Exception) as e:
            logger.warning(f"Tika not available: {e}")
            return False
    
    def _check_camelot_availability(self) -> bool:
        """Check if Camelot is available"""
        try:
            import camelot
            return True
        except ImportError:
            logger.warning("Camelot not available for table extraction")
            return False
    
    def _check_tabula_availability(self) -> bool:
        """Check if Tabula is available"""
        try:
            import tabula
            return True
        except ImportError:
            logger.warning("Tabula not available for table extraction")
            return False
    
    @property
    def easyocr_reader(self):
        """Lazy load EasyOCR"""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                logger.info("EasyOCR initialized successfully")
            except Exception as e:
                logger.warning(f"EasyOCR initialization failed: {e}")
                self._easyocr_reader = False
        return self._easyocr_reader if self._easyocr_reader else None
    
    @property  
    def fallback_parser(self):
        """Lazy load fallback parser to avoid circular imports"""
        if self._fallback_parser is None:
            try:
                from app.core.document_parser import DocumentParser
                self._fallback_parser = DocumentParser()
            except ImportError:
                logger.warning("Fallback parser not available")
                self._fallback_parser = False
        return self._fallback_parser if self._fallback_parser else None
        
    def parse_any_document(self, content: bytes, url: str) -> Tuple[str, List[Dict]]:
        """Parse any document format using multiple strategies"""
        
        if not content:
            return "Empty document content", [{'type': 'empty'}]
        
        # Detect file extension
        file_extension = self._get_extension(url)
        
        # Try specialized parsers first for better quality
        try:
            if file_extension in ['.xlsx', '.xls', '.csv']:
                return self._parse_spreadsheet_advanced(content, file_extension)
            elif file_extension == '.pdf':
                return self._parse_pdf_advanced(content)
            elif file_extension in ['.zip', '.tar', '.gz', '.rar', '.7z']:
                return self._parse_archive_advanced(content, file_extension)
            elif file_extension in ['.bin', '.dat', '.raw']:
                return self._parse_binary_file(content)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif']:
                return self._parse_image_advanced(content)
        except Exception as e:
            logger.warning(f"Specialized parser for {file_extension} failed: {e}")
        
        # Try Tika for universal parsing
        if self.tika_available:
            try:
                text, metadata = self._parse_with_tika(content)
                if text and len(text.strip()) > 50:
                    return text, metadata
            except Exception as e:
                logger.warning(f"Tika parsing failed: {e}")
        
        # Fallback to original parser
        if self.fallback_parser:
            try:
                return self.fallback_parser.parse_document(content, file_extension)
            except Exception as e:
                logger.warning(f"Fallback parser failed: {e}")
        
        # Final fallback: raw text extraction
        return self._extract_raw_text(content)
    
    def _parse_with_tika(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Parse using Apache Tika with proper error handling"""
        try:
            from tika import parser as tika_parser
            
            # Write to temp file for Tika
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            try:
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
                
                return text, metadata
                
            finally:
                # Clean up temp file
                try:
                    os.unlink(tmp_path)
                except:
                    pass
            
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
                df = pd.read_csv(io.BytesIO(content), encoding=encoding, on_bad_lines='skip')
                return self._process_dataframe(df, "main")
            else:
                # Handle Excel files
                excel_file = pd.ExcelFile(io.BytesIO(content), engine='openpyxl')
                all_text = []
                all_metadata = []
                
                # Process each sheet (limit to 10 sheets)
                for i, sheet_name in enumerate(excel_file.sheet_names[:10]):
                    try:
                        df = excel_file.parse(sheet_name)
                        sheet_text, sheet_meta = self._process_dataframe(df, sheet_name)
                        all_text.append(f"\n=== SHEET: {sheet_name} ===\n{sheet_text}")
                        all_metadata.extend(sheet_meta)
                    except Exception as e:
                        logger.warning(f"Failed to process sheet {sheet_name}: {e}")
                        continue
                
                return "\n".join(all_text), all_metadata
            
        except Exception as e:
            logger.error(f"Advanced spreadsheet parsing failed: {e}")
            # Simple fallback
            if self.fallback_parser:
                return self.fallback_parser.parse_spreadsheet(content, extension)
            else:
                return f"Spreadsheet parsing error: {str(e)}", [{'type': 'error'}]
    
    def _process_dataframe(self, df: pd.DataFrame, sheet_name: str) -> Tuple[str, List[Dict]]:
        """Process a dataframe with intelligent structure detection"""
        text_parts = []
        
        try:
            # 1. Summary
            text_parts.append(f"Sheet: {sheet_name}")
            text_parts.append(f"Dimensions: {len(df)} rows × {len(df.columns)} columns")
            text_parts.append(f"Columns: {', '.join(str(col) for col in df.columns)}")
            
            # 2. Quick statistics for numeric columns
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty and len(numeric_df.columns) <= 10:
                text_parts.append("\n=== NUMERIC SUMMARY ===")
                try:
                    summary = numeric_df.describe()
                    text_parts.append(summary.to_string())
                except:
                    pass
            
            # 3. Categorical summary
            cat_df = df.select_dtypes(include=['object'])
            if not cat_df.empty and len(cat_df.columns) <= 10:
                text_parts.append("\n=== CATEGORICAL SUMMARY ===")
                for col in cat_df.columns:
                    try:
                        unique_count = cat_df[col].nunique()
                        text_parts.append(f"{col}: {unique_count} unique values")
                        if unique_count <= 10:
                            values = cat_df[col].dropna().unique()[:10]
                            text_parts.append(f"  Values: {', '.join(str(v) for v in values)}")
                    except:
                        continue
            
            # 4. Data sample (smart sampling)
            text_parts.append("\n=== DATA SAMPLE ===")
            text_parts.append(" | ".join(str(col) for col in df.columns))
            
            if len(df) <= 50:
                # Small dataset: show all
                for idx, row in df.iterrows():
                    row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row.values)
                    text_parts.append(f"Row {idx+1}: {row_text}")
            else:
                # Large dataset: show head and tail
                for idx, row in df.head(25).iterrows():
                    row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row.values)
                    text_parts.append(f"Row {idx+1}: {row_text}")
                
                text_parts.append(f"\n... ({len(df) - 50} rows omitted) ...\n")
                
                for idx, row in df.tail(25).iterrows():
                    row_text = " | ".join(str(val) if pd.notna(val) else "" for val in row.values)
                    text_parts.append(f"Row {idx+1}: {row_text}")
            
        except Exception as e:
            logger.error(f"DataFrame processing error: {e}")
            text_parts = [f"DataFrame processing error: {str(e)}"]
        
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
            # First try fallback parser for basic text extraction
            if self.fallback_parser:
                try:
                    text, metadata = self.fallback_parser.parse_pdf(content)
                    all_text.append(text)
                    all_metadata.extend(metadata)
                except Exception as e:
                    logger.warning(f"Fallback PDF parsing failed: {e}")
            
            # Try advanced table extraction if available
            if self._camelot_available or self._tabula_available:
                try:
                    # Save to temp file for table extraction
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                        tmp.write(content)
                        tmp_path = tmp.name
                    
                    try:
                        # Try Camelot first
                        if self._camelot_available:
                            import camelot
                            tables = camelot.read_pdf(tmp_path, pages='all', flavor='lattice')
                            if not tables:
                                tables = camelot.read_pdf(tmp_path, pages='all', flavor='stream')
                            
                            for i, table in enumerate(tables):
                                table_text = f"\n=== CAMELOT TABLE {i+1} (Page {table.page}) ===\n"
                                table_text += table.df.to_string()
                                all_text.append(table_text)
                                all_metadata.append({'type': 'extracted_table', 'page': table.page})
                        
                        # Try Tabula as additional option
                        elif self._tabula_available:
                            import tabula
                            tables = tabula.read_pdf(tmp_path, pages='all', multiple_tables=True)
                            for i, table in enumerate(tables):
                                table_text = f"\n=== TABULA TABLE {i+1} ===\n"
                                table_text += table.to_string()
                                all_text.append(table_text)
                                all_metadata.append({'type': 'extracted_table_tabula'})
                    
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                
                except Exception as e:
                    logger.warning(f"Table extraction failed: {e}")
            
            result_text = "\n".join(all_text) if all_text else "No text extracted from PDF"
            return result_text, all_metadata
            
        except Exception as e:
            logger.error(f"Advanced PDF parsing failed: {e}")
            return f"PDF parsing error: {str(e)}", [{'type': 'error'}]
    
    def _parse_image_advanced(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Advanced image parsing with multiple OCR engines"""
        try:
            image = Image.open(io.BytesIO(content))
            
            # Try Tesseract first
            text = ""
            try:
                text = pytesseract.image_to_string(image)
            except Exception as e:
                logger.warning(f"Tesseract OCR failed: {e}")
            
            # Try EasyOCR if Tesseract fails or gives poor results
            if (not text or len(text.strip()) < 10) and self.easyocr_reader:
                try:
                    # Convert PIL to numpy array for EasyOCR
                    import numpy as np
                    img_array = np.array(image)
                    results = self.easyocr_reader.readtext(img_array)
                    text = " ".join([result[1] for result in results])
                except Exception as e:
                    logger.warning(f"EasyOCR failed: {e}")
            
            # Image metadata
            metadata = [{
                'type': 'image',
                'format': getattr(image, 'format', 'unknown'),
                'size': getattr(image, 'size', (0, 0)),
                'mode': getattr(image, 'mode', 'unknown')
            }]
            
            return text.strip() if text.strip() else "No text found in image", metadata
                
        except Exception as e:
            logger.error(f"Image parsing failed: {e}")
            return "Unable to process image", [{'type': 'error'}]
    
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
                    processed = 0
                    for file_info in zf.filelist[:20]:  # Limit to first 20 files
                        if file_info.is_dir() or file_info.file_size > 10 * 1024 * 1024:  # Skip large files
                            continue
                        
                        try:
                            file_content = zf.read(file_info.filename)
                            file_ext = os.path.splitext(file_info.filename)[1]
                            
                            # Parse the file
                            file_text, file_meta = self.parse_any_document(file_content, file_info.filename)
                            
                            text_parts.append(f"\n=== FILE: {file_info.filename} ===")
                            text_parts.append(file_text[:2000])  # Limit text per file
                            metadata.append({'file': file_info.filename, 'size': file_info.file_size})
                            
                            processed += 1
                            if processed >= 20:
                                break
                            
                        except Exception as e:
                            logger.warning(f"Failed to process {file_info.filename}: {e}")
                            continue
            
            return "\n".join(text_parts), metadata
            
        except Exception as e:
            logger.error(f"Archive parsing failed: {e}")
            return f"Archive parsing error: {str(e)}", [{'type': 'error'}]
            
        finally:
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def _parse_binary_file(self, content: bytes) -> Tuple[str, List[Dict]]:
        """Parse binary files by attempting multiple strategies"""
        text_parts = []
        metadata = [{'type': 'binary', 'size': len(content)}]
        
        # 1. Try to detect if it's text-based binary
        try:
            # Check if it's actually text
            text = content.decode('utf-8', errors='ignore')
            if self._is_mostly_text(text):
                return text, metadata
        except:
            pass
        
        # 2. Extract strings
        strings = self._extract_strings(content)
        if strings:
            text_parts.append("=== EXTRACTED STRINGS ===")
            text_parts.extend(strings[:500])  # Limit output
        
        # 3. File analysis
        text_parts.append("\n=== FILE ANALYSIS ===")
        text_parts.append(f"File size: {len(content)} bytes")
        
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
        """Get file extension from URL with better detection"""
        import urllib.parse
        
        # Parse URL to get path
        parsed = urllib.parse.urlparse(url)
        path = parsed.path
        
        # Get extension
        ext = os.path.splitext(path)[1].lower()
        
        # Handle common cases where extension might be missing
        if not ext:
            # Try to extract from query parameter
            query_params = urllib.parse.parse_qs(parsed.query)
            if 'format' in query_params:
                format_param = query_params['format'][0]
                if format_param:
                    ext = f'.{format_param}'
            # Check for common patterns in URL
            elif any(pattern in url.lower() for pattern in ['.pdf', '.xlsx', '.xls', '.docx', '.doc', '.zip']):
                for pattern in ['.pdf', '.xlsx', '.xls', '.docx', '.doc', '.zip']:
                    if pattern in url.lower():
                        ext = pattern
                        break
        
        return ext or '.unknown'