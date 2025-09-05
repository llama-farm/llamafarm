"""LlamaIndex-based PDF Parser with multiple fallback strategies."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_parser import BaseLlamaIndexParser, ParserFactory

logger = logging.getLogger(__name__)

try:
    from llama_index.readers.file import PDFReader
    LLAMA_PDF_AVAILABLE = True
except ImportError:
    LLAMA_PDF_AVAILABLE = False

try:
    from llama_index.readers.file import PyMuPDFReader
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# Fallback imports for manual PDF parsing
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_DIRECT_AVAILABLE = True
except ImportError:
    PYMUPDF_DIRECT_AVAILABLE = False


class LlamaIndexPDFParser(BaseLlamaIndexParser):
    """LlamaIndex-based PDF parser with multiple fallback strategies."""
    
    def __init__(self, name: str = "LlamaIndexPDFParser", config: Optional[Dict[str, Any]] = None):
        """
        Initialize LlamaIndex PDF parser.
        
        Args:
            name: Parser name
            config: Parser configuration
        """
        super().__init__(name=name, config=config or {})
        
        # PDF-specific configuration
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_page_structure = self.config.get("extract_page_structure", True)
        self.combine_pages = self.config.get("combine_pages", True)
        self.page_separator = self.config.get("page_separator", "\n\n--- Page Break ---\n\n")
        self.min_text_length = self.config.get("min_text_length", 10)
        self.include_page_numbers = self.config.get("include_page_numbers", True)
        self.extract_outline = self.config.get("extract_outline", True)
        
        # Fallback strategy configuration
        self.fallback_strategies = self.config.get("fallback_strategies", [
            "llama_pdf_reader",  # Primary LlamaIndex PDF reader
            "llama_pymupdf_reader",  # LlamaIndex PyMuPDF reader
            "direct_pymupdf",  # Direct PyMuPDF usage
            "pypdf2_fallback"  # PyPDF2 fallback
        ])
        
        self.reader_type = None  # Will be set based on available readers
    
    def _get_reader(self):
        """Get the best available LlamaIndex PDF reader."""
        # Try readers in order of preference
        if LLAMA_PDF_AVAILABLE:
            try:
                reader = PDFReader()
                self.reader_type = "llama_pdf_reader"
                logger.debug("Using LlamaIndex PDFReader")
                return reader
            except Exception as e:
                logger.warning(f"Failed to initialize LlamaIndex PDFReader: {e}")
        
        if PYMUPDF_AVAILABLE:
            try:
                reader = PyMuPDFReader()
                self.reader_type = "llama_pymupdf_reader"
                logger.debug("Using LlamaIndex PyMuPDFReader")
                return reader
            except Exception as e:
                logger.warning(f"Failed to initialize LlamaIndex PyMuPDFReader: {e}")
        
        # If no LlamaIndex PDF readers available, we'll handle this in parse()
        logger.warning("No LlamaIndex PDF readers available, will use manual fallbacks")
        return None
    
    def parse(self, file_path: str, **kwargs) -> "ProcessingResult":
        """
        Parse a PDF file using LlamaIndex with fallbacks.
        
        Args:
            file_path: Path to the PDF file
            **kwargs: Additional parsing options
            
        Returns:
            ProcessingResult containing parsed documents
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {file_path}", "source": str(file_path)}]
            )
        
        # Try different parsing strategies
        for strategy in self.fallback_strategies:
            try:
                result = self._try_parsing_strategy(file_path, strategy, **kwargs)
                if result.documents or not result.errors:
                    # Add strategy metadata
                    for doc in result.documents:
                        doc.metadata.update({
                            "pdf_parsing_strategy": strategy,
                            "reader_type": self.reader_type or strategy
                        })
                    
                    result.metrics.update({
                        "pdf_parsing_strategy": strategy,
                        "reader_type": self.reader_type or strategy
                    })
                    
                    return result
                else:
                    logger.warning(f"Strategy {strategy} failed: {result.errors}")
                    
            except Exception as e:
                logger.warning(f"PDF parsing strategy {strategy} failed: {e}")
                continue
        
        # All strategies failed
        from core.base import ProcessingResult
        return ProcessingResult(
            documents=[],
            errors=[{
                "error": "All PDF parsing strategies failed",
                "source": str(file_path),
                "tried_strategies": self.fallback_strategies
            }]
        )
    
    def _try_parsing_strategy(self, file_path: Path, strategy: str, **kwargs) -> "ProcessingResult":
        """Try a specific PDF parsing strategy."""
        if strategy == "llama_pdf_reader" and self._reader and self.reader_type == "llama_pdf_reader":
            return self._parse_with_llamaindex_reader(file_path, **kwargs)
        
        elif strategy == "llama_pymupdf_reader" and self._reader and self.reader_type == "llama_pymupdf_reader":
            return self._parse_with_llamaindex_reader(file_path, **kwargs)
        
        elif strategy == "direct_pymupdf" and PYMUPDF_DIRECT_AVAILABLE:
            return self._parse_with_direct_pymupdf(file_path, **kwargs)
        
        elif strategy == "pypdf2_fallback" and PYPDF2_AVAILABLE:
            return self._parse_with_pypdf2(file_path, **kwargs)
        
        else:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Strategy {strategy} not available",
                    "source": str(file_path)
                }]
            )
    
    def _parse_with_llamaindex_reader(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse PDF using LlamaIndex reader."""
        try:
            # Load documents using LlamaIndex
            documents = self._reader.load_data(file=str(file_path))
            
            if not documents:
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No content extracted by LlamaIndex", "source": str(file_path)}]
                )
            
            # Process documents
            return self._process_llamaindex_documents(documents, file_path)
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"LlamaIndex PDF parsing failed: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _parse_with_direct_pymupdf(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse PDF using direct PyMuPDF."""
        try:
            import fitz
            
            doc = fitz.open(str(file_path))
            
            # Extract document metadata
            pdf_metadata = self._extract_pymupdf_metadata(doc, file_path)
            
            # Extract text from pages
            page_texts = []
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text()
                
                if len(page_text.strip()) >= self.min_text_length:
                    if self.include_page_numbers:
                        page_text = f"[Page {page_num + 1}]\n\n{page_text}"
                    
                    page_texts.append({
                        "text": page_text,
                        "page_num": page_num + 1,
                        "metadata": {
                            "page_number": page_num + 1,
                            "page_width": page.rect.width,
                            "page_height": page.rect.height
                        }
                    })
            
            doc.close()
            
            if not page_texts:
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No text content found in PDF", "source": str(file_path)}]
                )
            
            # Convert to our document format
            return self._create_documents_from_pages(page_texts, pdf_metadata, file_path)
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"PyMuPDF parsing failed: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _parse_with_pypdf2(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse PDF using PyPDF2 as fallback."""
        try:
            import PyPDF2
            
            with open(file_path, "rb") as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract document metadata
                pdf_metadata = self._extract_pypdf2_metadata(reader, file_path)
                
                # Extract text from pages
                page_texts = []
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        
                        if len(page_text.strip()) >= self.min_text_length:
                            if self.include_page_numbers:
                                page_text = f"[Page {page_num}]\n\n{page_text}"
                            
                            page_texts.append({
                                "text": page_text,
                                "page_num": page_num,
                                "metadata": {
                                    "page_number": page_num
                                }
                            })
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
                
                if not page_texts:
                    from core.base import ProcessingResult
                    return ProcessingResult(
                        documents=[],
                        errors=[{"error": "No text content found in PDF", "source": str(file_path)}]
                    )
                
                return self._create_documents_from_pages(page_texts, pdf_metadata, file_path)
                
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"PyPDF2 parsing failed: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _process_llamaindex_documents(self, documents, file_path: Path) -> "ProcessingResult":
        """Process LlamaIndex documents into our format."""
        from core.base import ProcessingResult, Document
        from utils.hash_utils import generate_document_metadata, generate_chunk_metadata
        
        result_documents = []
        errors = []
        
        # If combining pages, merge all content
        if self.combine_pages and len(documents) > 1:
            combined_content = self.page_separator.join(doc.text for doc in documents if doc.text)
            
            if not combined_content.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No content after combining pages", "source": str(file_path)}]
                )
            
            # Generate metadata
            base_metadata = generate_document_metadata(str(file_path), combined_content)
            base_metadata.update({
                "parser_type": self.name,
                "total_pages": len(documents),
                "combined_pages": True,
                "reader_type": self.reader_type
            })
            
            # Add LlamaIndex metadata from first document
            if documents[0].metadata:
                base_metadata.update(documents[0].metadata)
            
            # Handle chunking
            if self._node_parser and self.chunk_size:
                chunked_docs = self._create_chunked_documents(
                    combined_content, base_metadata, documents[0]
                )
                result_documents.extend(chunked_docs)
            else:
                document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
                doc = Document(
                    content=combined_content,
                    metadata=base_metadata,
                    id=document_id,
                    source=str(file_path)
                )
                result_documents.append(doc)
        
        else:
            # Process each page separately
            for i, llama_doc in enumerate(documents):
                try:
                    content = llama_doc.text or ""
                    
                    if len(content.strip()) < self.min_text_length:
                        continue
                    
                    base_metadata = generate_document_metadata(str(file_path), content)
                    base_metadata.update({
                        "parser_type": self.name,
                        "page_number": i + 1,
                        "total_pages": len(documents),
                        "combined_pages": False,
                        "reader_type": self.reader_type,
                        "llama_doc_id": llama_doc.id_
                    })
                    
                    if llama_doc.metadata:
                        base_metadata.update(llama_doc.metadata)
                    
                    document_id = f"doc_{base_metadata['document_hash'][:12]}_page_{i+1}"
                    doc = Document(
                        content=content,
                        metadata=base_metadata,
                        id=document_id,
                        source=str(file_path)
                    )
                    result_documents.append(doc)
                    
                except Exception as e:
                    errors.append({
                        "error": f"Failed to process page {i+1}: {str(e)}",
                        "source": str(file_path)
                    })
        
        return ProcessingResult(
            documents=result_documents,
            errors=errors,
            metrics={
                "total_documents": len(result_documents),
                "total_errors": len(errors),
                "file_processed": str(file_path),
                "parser_type": self.name,
                "reader_type": self.reader_type
            }
        )
    
    def _create_documents_from_pages(self, page_texts, pdf_metadata, file_path: Path) -> "ProcessingResult":
        """Create documents from extracted page texts."""
        from core.base import ProcessingResult, Document
        from utils.hash_utils import generate_document_metadata, generate_chunk_metadata
        
        result_documents = []
        
        if self.combine_pages and len(page_texts) > 1:
            # Combine all pages
            combined_content = self.page_separator.join(page["text"] for page in page_texts)
            
            base_metadata = generate_document_metadata(str(file_path), combined_content)
            base_metadata.update(pdf_metadata)
            base_metadata.update({
                "total_pages": len(page_texts),
                "combined_pages": True
            })
            
            # Handle chunking
            if self._node_parser and self.chunk_size:
                # Create mock LlamaIndex document for chunking
                from llama_index.core.schema import Document as LlamaDocument
                llama_doc = LlamaDocument(text=combined_content, metadata=base_metadata)
                
                chunked_docs = self._create_chunked_documents(
                    combined_content, base_metadata, llama_doc
                )
                result_documents.extend(chunked_docs)
            else:
                document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
                doc = Document(
                    content=combined_content,
                    metadata=base_metadata,
                    id=document_id,
                    source=str(file_path)
                )
                result_documents.append(doc)
        
        else:
            # Create separate documents for each page
            for page_data in page_texts:
                content = page_data["text"]
                page_metadata = generate_document_metadata(str(file_path), content)
                page_metadata.update(pdf_metadata)
                page_metadata.update(page_data["metadata"])
                page_metadata["combined_pages"] = False
                
                document_id = f"doc_{page_metadata['document_hash'][:12]}_page_{page_data['page_num']}"
                doc = Document(
                    content=content,
                    metadata=page_metadata,
                    id=document_id,
                    source=str(file_path)
                )
                result_documents.append(doc)
        
        return ProcessingResult(
            documents=result_documents,
            errors=[],
            metrics={
                "total_documents": len(result_documents),
                "file_processed": str(file_path),
                "parser_type": self.name
            }
        )
    
    def _extract_pymupdf_metadata(self, doc, file_path: Path) -> Dict[str, Any]:
        """Extract metadata using PyMuPDF."""
        metadata = {
            "parser_type": self.name,
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "total_pages": len(doc)
        }
        
        try:
            pdf_metadata = doc.metadata
            if pdf_metadata:
                metadata.update({
                    "title": pdf_metadata.get("title", "").strip(),
                    "author": pdf_metadata.get("author", "").strip(),
                    "subject": pdf_metadata.get("subject", "").strip(),
                    "creator": pdf_metadata.get("creator", "").strip(),
                    "producer": pdf_metadata.get("producer", "").strip(),
                    "creation_date": pdf_metadata.get("creationDate", "").strip(),
                    "modification_date": pdf_metadata.get("modDate", "").strip()
                })
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
        
        return metadata
    
    def _extract_pypdf2_metadata(self, reader, file_path: Path) -> Dict[str, Any]:
        """Extract metadata using PyPDF2."""
        metadata = {
            "parser_type": self.name,
            "file_name": file_path.name,
            "file_size": file_path.stat().st_size,
            "total_pages": len(reader.pages)
        }
        
        try:
            pdf_info = reader.metadata
            if pdf_info:
                metadata.update({
                    "title": str(pdf_info.get("/Title", "")).strip(),
                    "author": str(pdf_info.get("/Author", "")).strip(),
                    "subject": str(pdf_info.get("/Subject", "")).strip(),
                    "creator": str(pdf_info.get("/Creator", "")).strip(),
                    "producer": str(pdf_info.get("/Producer", "")).strip()
                })
                
                # Handle dates
                for date_field, meta_key in [("/CreationDate", "creation_date"), ("/ModDate", "modification_date")]:
                    if pdf_info.get(date_field):
                        try:
                            date_str = str(pdf_info[date_field])
                            if date_str.startswith("D:"):
                                from datetime import datetime
                                date_str = date_str[2:16]  # Extract YYYYMMDDHHMMSS
                                parsed_date = datetime.strptime(date_str, "%Y%m%d%H%M%S")
                                metadata[meta_key] = parsed_date.isoformat()
                        except Exception as e:
                            logger.debug(f"Could not parse date {date_field}: {e}")
        except Exception as e:
            logger.warning(f"Failed to extract PDF metadata: {e}")
        
        return metadata
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    @staticmethod
    def can_parse_mime_type(mime_type: str) -> bool:
        """Check if this parser can handle the given MIME type."""
        return mime_type == 'application/pdf'
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions."""
        return ['.pdf']
    
    @staticmethod
    def get_description() -> str:
        """Get parser description."""
        return "LlamaIndex-based PDF parser with multiple fallback strategies"


# Register the parser
ParserFactory.register_parser("LlamaIndexPDFParser", LlamaIndexPDFParser)