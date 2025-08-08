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
# from app.core.universal_parser import universal_parser
import asyncio

logger = logging.getLogger(__name__)

class DocumentParser:
    """Unified document parser with format-specific optimizations"""
    # @staticmethod
    # async def parse_bin_incrementally(file_path: str, vector_store, pipeline):
    #     """
    #     Reads a large binary file in chunks from disk, extracts text,
    #     and adds it to the vector store incrementally.
    #     """
    #     READ_CHUNK_SIZE = 1024 * 1024  # Read 1MB at a time
        
    #     try:
    #         with open(file_path, 'rb') as f:
    #             while True:
    #                 byte_chunk = f.read(READ_CHUNK_SIZE)
    #                 if not byte_chunk:
    #                     break  # End of file

    #                 # 1. Use existing binary parsing logic on the small chunk of bytes
    #                 text, _ = DocumentParser.parse_binary(byte_chunk)
    #                 if not text.strip():
    #                     continue
                    
    #                 # 2. Chunk the extracted text from this small piece
    #                 # The metadata is generic since we're streaming
    #                 chunks, chunk_meta = SmartChunker.chunk_document(
    #                     text, 
    #                     [{'type': 'binary_stream'}],
    #                     chunk_size=settings.CHUNK_SIZE_CHARS,
    #                     overlap=settings.CHUNK_OVERLAP_CHARS
    #                 )
                    
    #                 # 3. Embed and add the resulting smaller chunks to the vector store
    #                 if chunks:
    #                     embeddings = await pipeline._generate_embeddings(chunks)
    #                     vector_store.add(chunks, embeddings, chunk_meta)

    #     except Exception as e:
    #         logger.error(f"Failed to process binary file stream at '{file_path}': {e}")
    
    # @staticmethod
    # def parse_document(content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
    #     """Route to appropriate parser based on file type"""
        
    #     # Normalize extension
    #     file_extension = file_extension.lower()
    #     if not file_extension.startswith('.'):
    #         file_extension = '.' + file_extension
        
    #     # Route to appropriate parser
    #     if file_extension == '.pdf':
    #         return DocumentParser.parse_pdf(content)
    #     elif file_extension in ['.xlsx', '.xls', '.csv']:
    #         return DocumentParser.parse_spreadsheet(content, file_extension)
    #     elif file_extension in ['.docx', '.doc']:
    #         return DocumentParser.parse_word(content)
    #     elif file_extension in ['.pptx', '.ppt']:
    #         return DocumentParser.parse_powerpoint(content)
    #     elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
    #         return DocumentParser.parse_image(content)
    #     # elif file_extension == '.zip':
    #     #     return DocumentParser.parse_zip(content)
    #     elif file_extension == '.odt':
    #         return DocumentParser.parse_odt(content)
    #     else:
    #         # Try text extraction
    #         return DocumentParser.parse_text(content)
    # @staticmethod
    # def parse_document(content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
    #     """Route to appropriate parser based on file type"""
        
    #     # Normalize extension
    #     file_extension = file_extension.lower()
    #     if not file_extension.startswith('.'):
    #         file_extension = '.' + file_extension
        
    #     # Route to appropriate parser
    #     if file_extension == '.pdf':
    #         return DocumentParser.parse_pdf(content)
    #     elif file_extension in ['.xlsx', '.xls', '.csv']:
    #         return DocumentParser.parse_spreadsheet(content, file_extension)
    #     elif file_extension in ['.docx', '.doc']:
    #         return DocumentParser.parse_word(content)
    #     elif file_extension in ['.pptx', '.ppt']:
    #         return DocumentParser.parse_powerpoint(content)
    #     elif file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
    #         return DocumentParser.parse_image(content)
    #     elif file_extension == '.bin':
    #         return DocumentParser.parse_binary(content)
    #     # elif file_extension == '.zip':
    #     #     return DocumentParser.parse_zip(content)
    #     elif file_extension == '.odt':
    #         return DocumentParser.parse_odt(content)
    #     else:
    #         # Try text extraction
    #         return DocumentParser.parse_text(content)
    # @staticmethod
    # def parse_document(content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
    #     """
    #     Route to appropriate parser, with an agentic fallback to a universal
    #     parser if the specialized method fails or returns poor content.
    #     """
    #     # Normalize extension (same as before)
    #     file_extension = file_extension.lower()
    #     if not file_extension.startswith('.'):
    #         file_extension = '.' + file_extension
        
    #     # --- AGENTIC PARSING LOGIC ---
        
    #     # Define the primary parser to try first
    #     parser_map = {
    #         '.pdf': DocumentParser.parse_pdf,
    #         '.xlsx': DocumentParser.parse_spreadsheet,
    #         '.xls': DocumentParser.parse_spreadsheet,
    #         '.csv': DocumentParser.parse_spreadsheet,
    #         '.docx': DocumentParser.parse_word,
    #         '.doc': DocumentParser.parse_word,
    #         '.pptx': DocumentParser.parse_powerpoint,
    #         '.ppt': DocumentParser.parse_powerpoint,
    #         '.png': DocumentParser.parse_image,
    #         '.jpg': DocumentParser.parse_image,
    #         '.jpeg': DocumentParser.parse_image,
    #         '.tiff': DocumentParser.parse_image,
    #         '.bmp': DocumentParser.parse_image,
    #         '.odt': DocumentParser.parse_odt,
    #         '.bin': DocumentParser.parse_binary,
    #     }
        
    #     # Default to the basic text parser if the extension is unknown
    #     primary_parser = parser_map.get(file_extension, DocumentParser.parse_text)
        
    #     try:
    #         # Attempt to parse with the specialized tool
    #         if primary_parser in [DocumentParser.parse_spreadsheet]:
    #              text, metadata = primary_parser(content, file_extension)
    #         else:
    #              text, metadata = primary_parser(content)

    #         # Check if the primary parser failed or returned weak results
    #         if "unable to parse" in text.lower() or "error parsing" in text.lower() or len(text.strip()) < 20:
    #             logger.warning(f"Specialized parser for '{file_extension}' failed or returned poor content. Triggering Universal Parser fallback.")
    #             raise ValueError("Primary parser failed.")

    #         return text, metadata

    #     except Exception as e:
    #         # If the primary parser fails, the agent tries a different tool.
    #         logger.info(f"Fallback initiated due to: {e}. Using universal_parser.")
    #         try:
    #             # The UniversalDocumentParser can handle almost any format.
    #             # We pass a dummy URL since it's required by the method signature.
    #             from app.core.universal_parser import UniversalDocumentParser
    #             universal_parser = universal_parser()
    #             return universal_parser.parse_any_document(content, f"file{file_extension}")
    #         except Exception as universal_e:
    #             logger.error(f"FATAL: All parsing methods failed for {file_extension}: {universal_e}", exc_info=True)
    #             return f"Unable to process the document. The format '{file_extension}' is not supported by any available parser.", [{'type': 'fatal_error'}]
    @staticmethod
    def parse_document(content: bytes, file_extension: str) -> Tuple[str, List[Dict]]:
        """
        Route to appropriate parser, with an agentic fallback to a universal
        parser if the specialized method fails or returns poor content.
        """
        # Normalize extension
        file_extension = file_extension.lower()
        if not file_extension.startswith('.'):
            file_extension = '.' + file_extension
        
        # Define the primary parser to try first
        parser_map = {
            '.pdf': DocumentParser.parse_pdf,
            '.xlsx': DocumentParser.parse_spreadsheet,
            '.xls': DocumentParser.parse_spreadsheet,
            '.csv': DocumentParser.parse_spreadsheet,
            '.docx': DocumentParser.parse_word,
            '.doc': DocumentParser.parse_word,
            '.pptx': DocumentParser.parse_powerpoint,
            '.ppt': DocumentParser.parse_powerpoint,
            '.png': DocumentParser.parse_image,
            '.jpg': DocumentParser.parse_image,
            '.jpeg': DocumentParser.parse_image,
            '.tiff': DocumentParser.parse_image,
            '.bmp': DocumentParser.parse_image,
            '.odt': DocumentParser.parse_odt,
            '.bin': DocumentParser.parse_binary,
        }
        
        # Default to the basic text parser if the extension is unknown
        primary_parser = parser_map.get(file_extension, DocumentParser.parse_text)
        
        try:
            # Attempt to parse with the specialized tool
            if primary_parser is DocumentParser.parse_spreadsheet:
                 text, metadata = primary_parser(content, file_extension)
            else:
                 text, metadata = primary_parser(content)

            # Check if the primary parser failed or returned weak results
            if "unable to parse" in text.lower() or "error parsing" in text.lower() or len(text.strip()) < 20:
                logger.warning(f"Specialized parser for '{file_extension}' returned poor content. Triggering Universal Parser fallback.")
                raise ValueError("Primary parser yielded insufficient content.")

            return text, metadata

        except Exception as e:
            # If the primary parser fails, the agent tries a different tool.
            logger.info(f"Fallback initiated due to: {e}. Using UniversalDocumentParser.")
            try:
                # --- FIX: Import is moved inside the function to prevent the circular dependency ---
                from app.core.universal_parser import UniversalDocumentParser
                
                # The UniversalDocumentParser can handle almost any format.
                universal_parser = UniversalDocumentParser()
                return universal_parser.parse_any_document(content, f"file{file_extension}")
            except Exception as universal_e:
                logger.error(f"FATAL: All parsing methods failed for {file_extension}: {universal_e}", exc_info=True)
                return f"Unable to process the document. The format '{file_extension}' is not supported by any available parser.", [{'type': 'fatal_error'}]
    
    @staticmethod
    def parse_binary(content: bytes) -> Tuple[str, List[Dict]]:
        """Extracts printable strings from a binary file."""
        try:
            # Decode with error handling, replacing non-UTF-8 sequences
            text = content.decode('utf-8', errors='replace')
            
            # Further clean up and extract sequences of printable characters
            # This regex finds sequences of 4 or more "word" characters
            printable_strings = re.findall(r'\w{4,}', text)
            
            if printable_strings:
                extracted_text = " ".join(printable_strings)
                logger.info(f"Extracted {len(extracted_text)} characters of text from binary file.")
                return extracted_text, [{'type': 'binary_extract'}]
            else:
                return "No printable text found in the binary file.", [{'type': 'binary_extract'}]

        except Exception as e:
            logger.error(f"Binary file parsing failed: {e}")
            return "Unable to process binary file.", [{'type': 'error'}]    
    
    @staticmethod
    def parse_pdf(content: bytes) -> Tuple[str, List[Dict]]:
        """Enhanced PDF parsing with table extraction"""
        full_text = []
        metadata = []
        
        try:
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract text
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        full_text.append(f"\n=== PAGE {i+1} ===\n{page_text}")
                        metadata.append({'page': i+1, 'type': 'text'})
                    
                    # Extract tables
                    tables = page.extract_tables()
                    for j, table in enumerate(tables):
                        if table and len(table) > 1:  # Valid table
                            table_text = DocumentParser._format_table(table)
                            full_text.append(f"\n=== TABLE {j+1} (Page {i+1}) ===\n{table_text}")
                            metadata.append({'page': i+1, 'type': 'table', 'table_id': j+1})
        
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
                df = pd.read_csv(io.BytesIO(content))
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
                raise ValueError("No text extracted with python-docx, trying fallback.")
                
        except Exception as e:
            logger.warning(f"python-docx failed: {e}. Trying pypdf fallback.")
            try:
                reader = pypdf.PdfReader(io.BytesIO(content))
                text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
                if text_parts:
                    return "\n\n".join(text_parts), [{'type': 'text_fallback'}] * len(text_parts)
                else:
                    return "Unable to parse Word document with any method.", [{'type': 'error'}]
            except Exception as e2:
                logger.error(f"Word parsing failed with all methods: {e2}")
                return "Error parsing Word document.", [{'type': 'error'}]
    
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
    @staticmethod
    async def parse_zip_incrementally(zip_path: str, vector_store, pipeline):
        """
        Extracts a zip and processes its contents one-by-one, adding them
        to the vector store incrementally to save memory.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(temp_dir)

            # Walk through the extracted files
            for root, _, files in os.walk(temp_dir):
                for filename in files:
                    if filename.startswith('.') or filename.startswith('__MACOSX'):
                        continue
                    
                    file_path = os.path.join(root, filename)
                    file_ext = os.path.splitext(filename)[1].lower()
                    
                    logger.info(f"Incrementally processing: {filename}")
                    try:
                        with open(file_path, 'rb') as f:
                            content = f.read()

                        # 1. Parse the single file's content
                        text, metadata = DocumentParser.parse_document(content, file_ext)
                        if not text.strip():
                            continue
                        
                        # 2. Chunk the text from just this one file
                        chunks, chunk_meta = SmartChunker.chunk_document(
                            text, metadata, 
                            chunk_size=settings.CHUNK_SIZE_CHARS, 
                            overlap=settings.CHUNK_OVERLAP_CHARS
                        )
                        
                        # 3. Embed and add the chunks to the vector store immediately
                        if chunks:
                            # This await call is now valid because the function is async
                            embeddings = await pipeline._generate_embeddings(chunks)
                            vector_store.add(chunks, embeddings, chunk_meta)

                    except Exception as e:
                        logger.error(f"Failed to process '{filename}' in zip: {e}")
        finally:
            # Always clean up the temporary directory
            shutil.rmtree(temp_dir)
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