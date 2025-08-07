# app/core/document_parser.py - Comprehensive document parsing
import io
import re
import logging
import zipfile
import tempfile
import shutil
import os
from typing import Tuple, List, Dict, Optional
import pandas as pd
import numpy as np
from PIL import Image
import pytesseract
import pdfplumber
import pypdf
from docx import Document
from pptx import Presentation
from odf.text import P
from odf.opendocument import load
from app.core.smart_chunker import SmartChunker
from app.core.config import settings
import asyncio
# In document_parser.py

# --- (7) ADD THIS IMPORT AT THE TOP ---
import aiofiles

logger = logging.getLogger(__name__)

class DocumentParser:
    """Unified document parser with format-specific optimizations"""
    
    @staticmethod
    def parse_document(content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
        """Route to appropriate parser based on file type"""
        
        # Normalize extension
        file_extension = file_extension.lower()
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension

        try:    
        
        # Route to appropriate parser
            if file_extension == '.pdf':
                return DocumentParser.parse_pdf(content)
            elif file_extension in ['.xlsx', '.xls', '.csv']:
                return DocumentParser.parse_spreadsheet(content, file_extension)
            elif file_extension in ['.docx', '.doc']:
                return DocumentParser.parse_word(content)
            elif file_extension in ['.pptx', '.ppt']:
                return DocumentParser.parse_powerpoint(content)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                return DocumentParser.parse_image(content)
            # elif file_extension == '.zip':
            #     return DocumentParser.parse_zip(content)
            elif file_extension == '.odt':
                return DocumentParser.parse_odt(content)
            else:
                # Try text extraction
                return DocumentParser.parse_text(content)
        except Exception as e:
            logger.warning(f"Primary parser failed for {file_extension}: {e}")
            
            # Try Tika as fallback
            try:
                return DocumentParser._parse_with_tika_fallback(content)
            except:
                pass
            
            # Last resort: extract any text
            return DocumentParser._extract_any_text(content)
        
    @staticmethod
    def _parse_with_tika_fallback(content: bytes) -> Tuple[str, List[Dict]]:
        """Fallback to Tika parser"""
        # NEW: Tika fallback
        
        try:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            
            # Try Tika
            from tika import parser
            parsed = parser.from_file(tmp_path)
            text = parsed.get('content', '').strip()
            
            os.unlink(tmp_path)
            
            if text:
                return text, [{'type': 'tika_fallback'}]
        except:
            pass
        
        return "", [{'type': 'error'}]    
    @staticmethod
    def _extract_any_text(content: bytes) -> Tuple[str, List[Dict]]:
        """Extract whatever text possible"""
        # NEW: Last resort text extraction
        
        text_parts = []
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'ascii']:
            try:
                text = content.decode(encoding, errors='ignore')
                if len(text) > len(''.join(text_parts)):
                    text_parts = [text]
            except:
                continue
        
        # Extract ASCII strings from binary
        if not text_parts:
            ascii_strings = []
            current = []
            for byte in content:
                if 32 <= byte <= 126:
                    current.append(chr(byte))
                else:
                    if len(current) >= 4:
                        ascii_strings.append(''.join(current))
                    current = []
            
            if ascii_strings:
                text_parts = ascii_strings[:1000]  # Limit output
        
        if text_parts:
            return '\n'.join(text_parts), [{'type': 'extracted_text'}]
        else:
            return "Unable to extract any text from document", [{'type': 'error'}]

    
    # @staticmethod
    # def parse_pdf(content: bytes) -> Tuple[str, List[Dict]]:
    #     """Enhanced PDF parsing with table extraction"""
    #     full_text = []
    #     metadata = []
        
    #     try:
    #         with pdfplumber.open(io.BytesIO(content)) as pdf:
    #             for i, page in enumerate(pdf.pages):
    #                 # Extract text
    #                 page_text = page.extract_text() or ""
    #                 if page_text.strip():
    #                     full_text.append(f"\n=== PAGE {i+1} ===\n{page_text}")
    #                     metadata.append({'page': i+1, 'type': 'text'})
                    
    #                 # Extract tables
    #                 tables = page.extract_tables()
    #                 for j, table in enumerate(tables):
    #                     if table and len(table) > 1:  # Valid table
    #                         table_text = DocumentParser._format_table(table)
    #                         full_text.append(f"\n=== TABLE {j+1} (Page {i+1}) ===\n{table_text}")
    #                         metadata.append({'page': i+1, 'type': 'table', 'table_id': j+1})
        
    #     except Exception as e:
    #         logger.warning(f"pdfplumber failed: {e}, trying pypdf")
    #         # Fallback to pypdf
    #         try:
    #             reader = pypdf.PdfReader(io.BytesIO(content))
    #             for i, page in enumerate(reader.pages):
    #                 page_text = page.extract_text()
    #                 if page_text:
    #                     full_text.append(f"\n=== PAGE {i+1} ===\n{page_text}")
    #                     metadata.append({'page': i+1, 'type': 'text_fallback'})
    #         except Exception as e2:
    #             logger.error(f"PDF parsing failed: {e2}")
    #             return "Unable to parse PDF document", [{'type': 'error'}]
        
    #     return "\n".join(full_text), metadata
    @staticmethod
    def parse_pdf(content: bytes) -> Tuple[str, List[Dict]]:
        """Enhanced PDF parsing with page-by-page processing"""
        # OLD: Load entire PDF at once
        # NEW: Process page by page with memory limits
        
        full_text = []
        metadata = []
        MAX_PAGES = 100  # Process max 100 pages initially
        
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages[:MAX_PAGES]):  # Limit pages
                    # Process each page individually
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        full_text.append(f"\n=== PAGE {i+1} ===\n{page_text}")
                        metadata.append({'page': i+1, 'type': 'text'})
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        if table and len(table) > 1:
                            table_text = DocumentParser._format_table(table)
                            full_text.append(f"\n=== TABLE {j+1} (Page {i+1}) ===\n{table_text}")
                            metadata.append({'page': i+1, 'type': 'table', 'table_id': j+1})
                    
                    # Clear page from memory
                    page.close()
                    
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}, trying pypdf")
            # Fallback to pypdf
            try:
                reader = pypdf.PdfReader(io.BytesIO(content))
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text.append(f"\n=== PAGE {i+1} ===\n{page_text}")
                        metadata.append({'page': i+1, 'type': 'text_fallback'})
            except Exception as e2:
                logger.error(f"PDF parsing failed: {e2}")
                return "Unable to parse PDF document", [{'type': 'error'}]
        
        return "\n".join(full_text), metadata
    
    @staticmethod
    def parse_spreadsheet(content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
        """Enhanced spreadsheet parsing with structure preservation"""
        try:
            # Read the spreadsheet
            if file_extension == '.csv':
                # df = pd.read_csv(io.BytesIO(content))
                chunks = []
                for chunk_df in pd.read_csv(io.BytesIO(content), chunksize=1000):
                    chunks.append(chunk_df)
                df = pd.concat(chunks, ignore_index=True)
            
            else:
                df = pd.read_excel(io.BytesIO(content), engine='openpyxl' if file_extension == '.xlsx' else None)
            
            # Build structured text
            text_parts = []
            
            # 1. Summary information
            text_parts.append("=== SPREADSHEET SUMMARY ===")
            text_parts.append(f"Total Rows: {len(df)}")
            text_parts.append(f"Total Columns: {len(df.columns)}")
            text_parts.append(f"Columns: {', '.join(df.columns)}")
            
            # 2. Statistical summary for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                text_parts.append("\n=== NUMERIC COLUMN STATISTICS ===")
                for col in numeric_cols:
                    text_parts.append(f"{col}:")
                    text_parts.append(f"  Min: {df[col].min()}, Max: {df[col].max()}")
                    text_parts.append(f"  Mean: {df[col].mean():.2f}, Median: {df[col].median()}")
                    text_parts.append(f"  Non-null count: {df[col].notna().sum()}")
            
            # 3. Categorical column information
            cat_cols = df.select_dtypes(include=['object']).columns
            if len(cat_cols) > 0:
                text_parts.append("\n=== CATEGORICAL COLUMN INFORMATION ===")
                for col in cat_cols:
                    unique_vals = df[col].nunique()
                    text_parts.append(f"{col}: {unique_vals} unique values")
                    if unique_vals <= 20:  # List values if not too many
                        values = df[col].dropna().unique()[:20]
                        text_parts.append(f"  Values: {', '.join(map(str, values))}")
            
            # 4. Data rows (with smart sampling for large datasets)
            text_parts.append("\n=== DATA ROWS ===")
            text_parts.append(f"Column Headers: {' | '.join(df.columns)}")
            
            if len(df) <= 1000:
                # Include all rows for small datasets
                for idx, row in df.iterrows():
                    row_text = " | ".join([f"{col}={val}" for col, val in row.items() if pd.notna(val)])
                    text_parts.append(f"Row {idx+1}: {row_text}")
            else:
                # Sample for large datasets
                # First 100 rows
                for idx, row in df.head(100).iterrows():
                    row_text = " | ".join([f"{col}={val}" for col, val in row.items() if pd.notna(val)])
                    text_parts.append(f"Row {idx+1}: {row_text}")
                
                text_parts.append(f"\n... ({len(df) - 200} rows omitted) ...\n")
                
                # Last 100 rows
                for idx, row in df.tail(100).iterrows():
                    row_text = " | ".join([f"{col}={val}" for col, val in row.items() if pd.notna(val)])
                    text_parts.append(f"Row {idx+1}: {row_text}")
            
            metadata = [{'type': 'spreadsheet', 'rows': len(df), 'columns': len(df.columns)}]
            return "\n".join(text_parts), metadata
            
        except Exception as e:
            logger.error(f"Spreadsheet parsing failed: {e}")
            return f"Error parsing spreadsheet: {str(e)}", [{'type': 'error'}]
    
    

    
    # @staticmethod
    # def parse_word(content: bytes) -> Tuple[str, List[Dict]]:
    #     """Enhanced Word document parsing"""
    #     try:
    #         doc = Document(io.BytesIO(content))
    #         text_parts = []
    #         metadata = []
            
    #         # Document properties
    #         if doc.core_properties.title:
    #             text_parts.append(f"Title: {doc.core_properties.title}")
    #         if doc.core_properties.subject:
    #             text_parts.append(f"Subject: {doc.core_properties.subject}")
            
    #         # Process paragraphs with style preservation
    #         for para in doc.paragraphs:
    #             if para.text.strip():
    #                 if para.style.name.startswith('Heading'):
    #                     level = para.style.name[-1] if para.style.name[-1].isdigit() else '1'
    #                     text_parts.append(f"\n{'#' * int(level)} {para.text}")
    #                     metadata.append({'type': 'heading', 'level': int(level)})
    #                 else:
    #                     text_parts.append(para.text)
    #                     metadata.append({'type': 'paragraph'})
            
    #         # Process tables
    #         for i, table in enumerate(doc.tables):
    #             text_parts.append(f"\n=== TABLE {i+1} ===")
    #             table_data = []
                
    #             for row in table.rows:
    #                 row_text = [cell.text.strip() for cell in row.cells]
    #                 if any(row_text):
    #                     table_data.append(" | ".join(row_text))
                
    #             text_parts.extend(table_data)
    #             text_parts.append("=== END TABLE ===\n")
    #             metadata.append({'type': 'table', 'table_id': i+1})
    #         if text_parts:
    #             return "\n".join(text_parts), metadata
    #         else:
    #             raise ValueError("No text extracted with python-docx, trying fallback.")    
            
    #         # return "\n".join(text_parts), metadata
            
    #     except Exception as e:
    #     # --- MODIFIED: This block now triggers the fallback instead of returning an error ---
    #         logger.warning(f"python-docx failed: {e}. Trying pypdf fallback.")
            
    #         # --- ADDED: The entire fallback block using pypdf ---
    #         try:
    #             reader = pypdf.PdfReader(io.BytesIO(content))
    #             text_parts = []
    #             metadata = []
    #             for i, page in enumerate(reader.pages):
    #                 page_text = page.extract_text()
    #                 if page_text:
    #                     text_parts.append(page_text)
    #                     metadata.append({'page': i+1, 'type': 'text_fallback'})
                
    #             if text_parts:
    #                 return "\n\n".join(text_parts), metadata
    #             else:
    #                 return "Unable to parse Word document with any method.", [{'type': 'error'}]
                    
    #         except Exception as e2:
    #             logger.error(f"Word parsing failed with all methods: {e2}")
    #             return "Error parsing Word document.", [{'type': 'error'}]
    # +++ New, Correct Version +++
    @staticmethod
    def parse_word(content: bytes) -> Tuple[str, List[Dict]]:
        """Enhanced Word document parsing with pypdf fallback"""
        try:
            doc = Document(io.BytesIO(content))
            text_parts = []
            metadata = []
            
            if doc.core_properties.title:
                text_parts.append(f"Title: {doc.core_properties.title}")
            if doc.core_properties.subject:
                text_parts.append(f"Subject: {doc.core_properties.subject}")
            
            for para in doc.paragraphs:
                if para.text.strip():
                    if para.style.name.startswith('Heading'):
                        level = para.style.name[-1] if para.style.name[-1].isdigit() else '1'
                        text_parts.append(f"\n{'#' * int(level)} {para.text}")
                        metadata.append({'type': 'heading', 'level': int(level)})
                    else:
                        text_parts.append(para.text)
                        metadata.append({'type': 'paragraph'})
            
            for i, table in enumerate(doc.tables):
                text_parts.append(f"\n=== TABLE {i+1} ===")
                table_data = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    if any(row_text):
                        table_data.append(" | ".join(row_text))
                
                text_parts.extend(table_data)
                text_parts.append("=== END TABLE ===\n")
                metadata.append({'type': 'table', 'table_id': i+1})
            if text_parts:
                return "\n".join(text_parts), metadata
            else:
                # Return error if python-docx fails
                return "Unable to parse Word document", [{'type': 'error'}]
                
        except Exception as e:
            logger.error(f"Word parsing failed: {e}")
            return "Error parsing Word document", [{'type': 'error'}]    

        #     if text_parts:
        #         return "\n".join(text_parts), metadata
        #     else:
        #         raise ValueError("No text extracted with python-docx, trying fallback.")
                
        # except Exception as e:
        #     logger.warning(f"python-docx failed: {e}. Trying pypdf fallback.")
        #     try:
        #         reader = pypdf.PdfReader(io.BytesIO(content))
        #         text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
        #         if text_parts:
        #             return "\n\n".join(text_parts), [{'type': 'text_fallback'}] * len(text_parts)
        #         else:
        #             return "Unable to parse Word document with any method.", [{'type': 'error'}]
        #     except Exception as e2:
        #         logger.error(f"Word parsing failed with all methods: {e2}")
        #         return "Error parsing Word document.", [{'type': 'error'}]
    # @staticmethod
    # def parse_word(content: bytes) -> Tuple[str, List[Dict]]:
    #     """Enhanced Word document parsing"""
    #     try:
    #         doc = Document(io.BytesIO(content))
    #         text_parts = []
    #         metadata = []
            
    #         # Process content...
            
    #         if text_parts:
    #             return "\n".join(text_parts), metadata
    #         else:
    #             # Return error if python-docx fails
    #             return "Unable to parse Word document", [{'type': 'error'}]
                
    #     except Exception as e:
    #         logger.error(f"Word parsing failed: {e}")
    #         return "Error parsing Word document", [{'type': 'error'}]
    @staticmethod
    def parse_powerpoint(content: bytes) -> Tuple[str, List[Dict]]:
        """Enhanced PowerPoint parsing"""
        try:
            prs = Presentation(io.BytesIO(content))
            text_parts = []
            metadata = []
            
            # Presentation title
            if prs.core_properties.title:
                text_parts.append(f"Presentation: {prs.core_properties.title}\n")
            
            for slide_num, slide in enumerate(prs.slides, 1):
                text_parts.append(f"\n=== SLIDE {slide_num} ===")
                slide_content = []
                
                # Title
                if slide.shapes.title:
                    slide_content.append(f"Title: {slide.shapes.title.text}")
                
                # Process all shapes
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text and shape != slide.shapes.title:
                        slide_content.append(shape.text)
                    
                    # Tables
                    if shape.has_table:
                        slide_content.append("Table:")
                        for row in shape.table.rows:
                            row_text = " | ".join([cell.text for cell in row.cells])
                            slide_content.append(f"  {row_text}")
                
                text_parts.extend(slide_content)
                metadata.append({'type': 'slide', 'slide_num': slide_num})
            
            return "\n".join(text_parts), metadata
            
        except Exception as e:
            logger.error(f"PowerPoint parsing failed: {e}")
            return f"Error parsing PowerPoint: {str(e)}", [{'type': 'error'}]
    
    @staticmethod
    def parse_image(content: bytes) -> Tuple[str, List[Dict]]:
        """Parse image using OCR"""
        try:
            image = Image.open(io.BytesIO(content))
            
            # OCR text extraction
            text = pytesseract.image_to_string(image)
            
            # Image metadata
            metadata = [{
                'type': 'image',
                'format': image.format,
                'size': image.size,
                'mode': image.mode
            }]
            
            if text.strip():
                return text, metadata
            else:
                return "No text found in image", metadata
                
        except Exception as e:
            logger.error(f"Image parsing failed: {e}")
            return "Unable to process image", [{'type': 'error'}]
    
    # @staticmethod
    # def parse_zip(content: bytes) -> Tuple[str, List[Dict]]:
    #     """Parse ZIP archive"""
    #     text_parts = []
    #     metadata = []
    #     temp_dir = tempfile.mkdtemp()
        
    #     try:
    #         zip_path = os.path.join(temp_dir, 'source.zip')
    #         with open(zip_path, 'wb') as f:
    #             f.write(content)

    #         # --- ADDED: Block to extract the on-disk zip to a separate directory ---
    #         extract_dir = os.path.join(temp_dir, 'extracted')
    #         os.makedirs(extract_dir)
    #         with zipfile.ZipFile(io.BytesIO(content)) as zf:
    #             for file_info in zf.filelist:
    #                 if file_info.filename.startswith('__MACOSX') or file_info.is_dir():
    #                     continue
                    
    #                 # Extract and parse each file
    #                 file_content = zf.read(file_info.filename)
    #                 file_ext = '.' + file_info.filename.split('.')[-1] if '.' in file_info.filename else ''
                    
    #                 text_parts.append(f"\n=== FILE: {file_info.filename} ===")
                    
    #                 # Recursively parse
    #                 try:
    #                     file_text, file_meta = DocumentParser.parse_document(file_content, file_ext)
    #                     text_parts.append(file_text)
    #                     metadata.extend(file_meta)
    #                 except:
    #                     text_parts.append(f"Could not parse {file_info.filename}")
            
    #         return "\n".join(text_parts), metadata
            
    #     except Exception as e:
    #         logger.error(f"ZIP parsing failed: {e}")
    #         return f"Error parsing ZIP: {str(e)}", [{'type': 'error'}]
    # @staticmethod
    # def parse_zip(content: bytes) -> Tuple[str, List[Dict]]:
    #     """Parse ZIP by extracting to a temporary directory to save memory.""" # --- MODIFIED ---
    #     text_parts = []
    #     metadata = []
    #     temp_dir = tempfile.mkdtemp() # --- ADDED: Create a temporary directory to work in.

    #     try:
    #         # --- ADDED: Block to write the zip to a temp file on disk ---
    #         zip_path = os.path.join(temp_dir, 'source.zip')
    #         with open(zip_path, 'wb') as f:
    #             f.write(content)

    #         # --- ADDED: Block to extract the on-disk zip to a separate directory ---
    #         extract_dir = os.path.join(temp_dir, 'extracted')
    #         os.makedirs(extract_dir)
    #         with zipfile.ZipFile(zip_path, 'r') as zf:
    #             zf.extractall(extract_dir)

    #         # --- MODIFIED: Loop through the extracted files on disk instead of in-memory ---
    #         for root, _, files in os.walk(extract_dir):
    #             for filename in files:
    #                 # --- ADDED: Skip common junk files ---
    #                 if filename.startswith('__MACOSX') or filename.startswith('.'):
    #                     continue
                    
    #                 file_path = os.path.join(root, filename)
    #                 file_ext = os.path.splitext(filename)[1].lower()

    #                 text_parts.append(f"\n=== FILE: {filename} ===")
                    
    #                 try:
    #                     # --- MODIFIED: Open and read each file from disk one by one ---
    #                     with open(file_path, 'rb') as file_content:
    #                         # Recursively parse the file content
    #                         file_text, file_meta = DocumentParser.parse_document(file_content.read(), file_ext)
    #                         text_parts.append(file_text)
    #                         metadata.extend(file_meta)
    #                 except Exception as parse_error:
    #                     logger.warning(f"Could not parse {filename}: {parse_error}")
    #                     text_parts.append(f"Could not parse {filename}")

    #         return "\n".join(text_parts), metadata

    #     except Exception as e:
    #         logger.error(f"ZIP parsing failed: {e}")
    #         return f"Error parsing ZIP: {str(e)}", [{'type': 'error'}]
    #     finally:
    #         # --- ADDED: CRITICAL step to always clean up the temporary directory and its contents ---
    #         shutil.rmtree(temp_dir)
    # Line 278 in document_parser.py
    # @staticmethod
    # async def parse_zip_incrementally(zip_path: str, vector_store, pipeline):
    #     """
    #     Extracts a zip and processes its contents one-by-one, adding them
    #     to the vector store incrementally to save memory.
    #     """
    #     temp_dir = tempfile.mkdtemp()
    #     try:
    #         with zipfile.ZipFile(zip_path, 'r') as zf:
    #             zf.extractall(temp_dir)

    #         # Walk through the extracted files
    #         for root, _, files in os.walk(temp_dir):
    #             for filename in files:
    #                 if filename.startswith('.') or filename.startswith('__MACOSX'):
    #                     continue
                    
    #                 file_path = os.path.join(root, filename)
    #                 file_ext = os.path.splitext(filename)[1].lower()
                    
    #                 logger.info(f"Incrementally processing: {filename}")
    #                 try:
    #                     with open(file_path, 'rb') as f:
    #                         content = f.read()

    #                     # INSTEAD of appending to a list, we process immediately:
                        
    #                     # 1. Parse the single file's content
    #                     text, metadata = DocumentParser.parse_document(content, file_ext)
    #                     if not text.strip():
    #                         continue
                        
    #                     # 2. Chunk the text from just this one file
    #                     # chunks, chunk_meta = SmartChunker.chunk_document(
    #                     #     text, metadata, 
    #                     #     chunk_size=pipeline.settings.CHUNK_SIZE_CHARS, 
    #                     #     overlap=pipeline.settings.CHUNK_OVERLAP_CHARS
    #                     # )
    #                     chunks, chunk_meta = SmartChunker.chunk_document(
    #                     text, metadata, 
    #                     chunk_size=settings.CHUNK_SIZE_CHARS, 
    #                     overlap=settings.CHUNK_OVERLAP_CHARS
    #                     )
                        
    #                     # 3. Embed and add the chunks to the vector store immediately
    #                     # if chunks:
    #                     #     # loop = asyncio.get_event_loop()
    #                     #     # embeddings = loop.run_until_complete(pipeline._generate_embeddings(chunks))
    #                     #     # Line 296 in document_parser.py
    #                     #     try:
    #                     #         loop = asyncio.get_event_loop()
    #                     #         embeddings = loop.run_until_complete(pipeline._generate_embeddings(chunks))
    #                     #     except RuntimeError:
    #                     #         # Already in async context
    #                     #         embeddings = await pipeline._generate_embeddings(chunks)
    #                     #     vector_store.add(chunks, embeddings, chunk_meta)
    #                     # 3. Embed and add the chunks to the vector store immediately
    #                     # if chunks:
    #                     #     try:
    #                     #         loop = asyncio.get_event_loop()
    #                     #         if loop.is_running():
    #                     #             # We're already in an async context, create a task
    #                     #             import concurrent.futures
    #                     #             with concurrent.futures.ThreadPoolExecutor() as executor:
    #                     #                 future = executor.submit(
    #                     #                     lambda: asyncio.run(pipeline._generate_embeddings(chunks))
    #                     #                 )
    #                     #                 embeddings = future.result()
    #                     #         else:
    #                     #             embeddings = loop.run_until_complete(pipeline._generate_embeddings(chunks))
    #                     #     except RuntimeError:
    #                     #         # Fallback: run in new event loop
    #                     #         import concurrent.futures
    #                     #         with concurrent.futures.ThreadPoolExecutor() as executor:
    #                     #             future = executor.submit(
    #                     #                 lambda: asyncio.run(pipeline._generate_embeddings(chunks))
    #                     #             )
    #                     #             embeddings = future.result()
                            
    #                     #     vector_store.add(chunks, embeddings, chunk_meta)
    #                     if chunks:
    #                         embeddings = await pipeline._generate_embeddings(chunks)
    #                         vector_store.add(chunks, embeddings, chunk_meta)

    #                 except Exception as e:
    #                     logger.error(f"Failed to process '{filename}' in zip: {e}")
    #     finally:
    #         # Always clean up the temporary directory
    #         shutil.rmtree(temp_dir)
    # @staticmethod
    # async def parse_zip_incrementally(zip_path: str, vector_store, pipeline):
    #     """
    #     Extracts a zip and processes its contents iteratively to save memory
    #     and handle arbitrarily deep nested archives.
    #     """
    #     processing_queue = [zip_path]
        
    #     while processing_queue:
    #         current_zip_path = processing_queue.pop(0)
    #         temp_dir = tempfile.mkdtemp()

    #         try:
    #             with zipfile.ZipFile(current_zip_path, 'r') as zf:
    #                 zf.extractall(temp_dir)
                
    #             for root, _, files in os.walk(temp_dir):
    #                 for filename in files:
    #                     if filename.startswith('.') or filename.startswith('__MACOSX'):
    #                         continue
                        
    #                     file_path = os.path.join(root, filename)
    #                     file_ext = os.path.splitext(filename)[1].lower()

    #                     if file_ext == '.zip':
    #                         # Add nested zips to the queue for later processing
    #                         processing_queue.append(file_path)
    #                         continue

    #                     logger.info(f"Incrementally processing: {filename}")
    #                     try:
    #                         with open(file_path, 'rb') as f:
    #                             content = f.read()

    #                         # 1. Parse the file content
    #                         text, metadata = DocumentParser.parse_document(content, file_ext)
    #                         if not text.strip():
    #                             continue
                            
    #                         # 2. Chunk the text
    #                         chunks, chunk_meta = SmartChunker.chunk_document(
    #                             text, metadata,
    #                             chunk_size=settings.CHUNK_SIZE_CHARS,
    #                             overlap=settings.CHUNK_OVERLAP_CHARS
    #                         )
                            
    #                         # 3. Embed and add to the vector store
    #                         if chunks:
    #                             embeddings = await pipeline._generate_embeddings(chunks)
    #                             vector_store.add(chunks, embeddings, chunk_meta)

    #                     except Exception as e:
    #                         logger.error(f"Failed to process '{filename}' in zip: {e}")
    #         finally:
    #             # Clean up the temporary directory for the current zip
    #             shutil.rmtree(temp_dir)

    # @staticmethod
    # async def parse_zip_incrementally(zip_path: str, vector_store, pipeline):
    #     """Fixed incremental ZIP processing"""
    #     # OLD: Broken parse_zip_incrementally
    #     # NEW: Working implementation
        
    #     temp_dir = tempfile.mkdtemp()
        
    #     try:
    #         # Extract ZIP
    #         with zipfile.ZipFile(zip_path, 'r') as zf:
    #             # Only process reasonable number of files
    #             file_list = [f for f in zf.filelist 
    #                         if not f.is_dir() 
    #                         and not f.filename.startswith('__MACOSX')
    #                         and f.file_size < 10 * 1024 * 1024][:50]  # Max 50 files, < 10MB each
                
    #             for file_info in file_list:
    #                 try:
    #                     # Extract single file
    #                     file_path = zf.extract(file_info, temp_dir)
                        
    #                     # Read and parse
    #                     with open(file_path, 'rb') as f:
    #                         content = f.read()
                        
    #                     file_ext = os.path.splitext(file_info.filename)[1]
    #                     text, metadata = DocumentParser.parse_document(content, file_ext)
                        
    #                     if not text.strip():
    #                         continue
                        
    #                     # Chunk and add to vector store
    #                     chunks, chunk_meta = SmartChunker.chunk_document(
    #                         text, metadata,
    #                         chunk_size=settings.CHUNK_SIZE_CHARS,
    #                         overlap=settings.CHUNK_OVERLAP_CHARS
    #                     )
                        
    #                     if chunks:
    #                         # Generate embeddings and add
    #                         embeddings = await pipeline._generate_embeddings(chunks)
    #                         vector_store.add(chunks, embeddings, chunk_meta)
                        
    #                     # Clean up extracted file
    #                     os.unlink(file_path)
                        
    #                 except Exception as e:
    #                     logger.error(f"Failed to process {file_info.filename}: {e}")
                        
    #     finally:
    #         # Clean up
    #         shutil.rmtree(temp_dir)
    @staticmethod
    async def parse_zip_incrementally(zip_path: str, vector_store, pipeline):
        """Fixed async ZIP processing"""
        # OLD: Async context issues
        # NEW: Proper async handling
        
        temp_dir = tempfile.mkdtemp()
        pipeline.temp_dirs.append(temp_dir)  # Track for cleanup
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                file_list = [f for f in zf.filelist 
                            if not f.is_dir() 
                            and not f.filename.startswith('__MACOSX')
                            and not f.filename.startswith('.')
                            and f.file_size < 10 * 1024 * 1024][:50]
                
                for file_info in file_list:
                    try:
                        file_path = zf.extract(file_info, temp_dir)
                        
                        # Read file
                        async with aiofiles.open(file_path, 'rb') as f:
                            content = await f.read()
                        
                        # Parse document
                        file_ext = os.path.splitext(file_info.filename)[1]
                        text, metadata = DocumentParser.parse_document(content, file_ext)
                        
                        if not text.strip():
                            continue
                        
                        # Chunk text
                        chunks, chunk_meta = SmartChunker.chunk_document(
                            text, metadata,
                            chunk_size=settings.CHUNK_SIZE_CHARS,
                            overlap=settings.CHUNK_OVERLAP_CHARS
                        )
                        
                        if chunks:
                            # Generate embeddings properly in async context
                            embeddings = await pipeline._generate_embeddings(chunks)
                            
                            # Add to vector store
                            if hasattr(vector_store, 'add'):
                                vector_store.add(chunks, embeddings, chunk_meta)
                            else:
                                # For AdvancedVectorStore
                                vector_store.add_hierarchical(
                                    chunks, embeddings, 
                                    chunks, embeddings,  # Use same for both small and large
                                    chunk_meta, {}
                                )
                        
                        # Clean up file
                        os.unlink(file_path)
                        
                    except Exception as e:
                        logger.error(f"Failed to process {file_info.filename}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"ZIP processing failed: {e}")
        finally:
            # Clean up temp directory
            try:
                shutil.rmtree(temp_dir)
                pipeline.temp_dirs.remove(temp_dir)
            except:
                pass
    @staticmethod
    def parse_odt(content: bytes) -> Tuple[str, List[Dict]]:
        """Parse ODT files"""
        try:
            doc = load(io.BytesIO(content))
            text_parts = []
            
            for element in doc.getElementsByType(P):
                text = str(element)
                if text.strip():
                    text_parts.append(text)
            
            return "\n\n".join(text_parts), [{'type': 'odt'}]
            
        except Exception as e:
            logger.error(f"ODT parsing failed: {e}")
            return f"Error parsing ODT: {str(e)}", [{'type': 'error'}]
    
    @staticmethod
    def parse_text(content: bytes) -> Tuple[str, List[Dict]]:
        """Fallback text extraction"""
        try:
            text = content.decode('utf-8', errors='ignore')
            return text, [{'type': 'text'}]
        except Exception as e:
            return "Unable to extract text from file", [{'type': 'error'}]
    
    @staticmethod
    def _format_table(table: List[List]) -> str:
        """Format table data nicely"""
        if not table:
            return ""
        
        formatted = []
        
        # Headers
        if len(table) > 0:
            headers = [str(cell) if cell else "" for cell in table[0]]
            formatted.append(" | ".join(headers))
            formatted.append("-" * min(100, sum(len(h) + 3 for h in headers)))
        
        # Data rows
        for row in table[1:]:
            if row and any(cell for cell in row):
                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                formatted.append(row_text)
        
        return "\n".join(formatted)