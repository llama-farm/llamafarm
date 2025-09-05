"""PDF parser using LlamaIndex with advanced features."""

from pathlib import Path
from typing import Dict, Any, Optional
import logging
import yaml

from ..base import LlamaIndexParser, ParserConfig

logger = logging.getLogger(__name__)

# Lazy imports
PDF_READER_AVAILABLE = False
try:
    from llama_index.readers.file import PDFReader
    PDF_READER_AVAILABLE = True
except ImportError:
    logger.warning("PDF reader not available. Install with: pip install llama-index-readers-file")


class PDFParser(LlamaIndexParser):
    """PDF parser with OCR and table extraction support."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize PDF parser.
        
        Args:
            config: Parser configuration
        """
        if not PDF_READER_AVAILABLE:
            raise ImportError("PDF reader required. Install with: pip install llama-index-readers-file")
        
        super().__init__(config)
        
        # Initialize PDF reader with configuration
        self.reader = PDFReader(
            return_full_document=self.config.get("return_full_document", False)
        )
        
        # PDF-specific options
        self.extract_tables = self.config.get("extract_tables", False)
        self.extract_images = self.config.get("extract_images", False)
        self.ocr_enabled = self.config.get("ocr_enabled", False)
        self.ocr_language = self.config.get("ocr_language", "eng")
    
    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata from config.yaml.
        
        Returns:
            ParserConfig object with metadata
        """
        config_path = Path(__file__).parent / "config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                return ParserConfig(**data['parser'])
        
        # Fallback configuration
        return ParserConfig(
            name="pdf",
            display_name="PDF Parser",
            version="2.0.0",
            supported_extensions=[".pdf"],
            mime_types=["application/pdf"],
            capabilities=[
                "text_extraction",
                "chunking",
                "table_extraction",
                "ocr",
                "metadata_extraction"
            ],
            dependencies={
                "required": ["llama-index", "llama-index-readers-file"],
                "optional": ["pypdf2", "pdfplumber", "pytesseract", "camelot-py"]
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "sentences",
                "return_full_document": False,
                "extract_tables": False,
                "extract_images": False,
                "ocr_enabled": False,
                "ocr_language": "eng"
            }
        )
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if parser can handle the file
        """
        path = Path(file_path)
        
        # Check extension
        if path.suffix.lower() == '.pdf':
            return True
        
        # Check content (PDF magic bytes)
        try:
            with open(file_path, 'rb') as f:
                header = f.read(4)
                if header == b'%PDF':
                    return True
        except:
            pass
        
        return False
    
    def _extract_advanced_content(self, file_path: str) -> Dict[str, Any]:
        """Extract tables, images, and OCR content if configured.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content
        """
        advanced_content = {}
        
        # Extract tables using camelot or pdfplumber
        if self.extract_tables:
            try:
                import camelot
                tables = camelot.read_pdf(file_path, pages='all')
                advanced_content['tables'] = [table.df.to_dict() for table in tables]
                logger.info(f"Extracted {len(tables)} tables from PDF")
            except ImportError:
                logger.warning("Camelot not available for table extraction")
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
        
        # Perform OCR if enabled
        if self.ocr_enabled:
            try:
                import pytesseract
                from pdf2image import convert_from_path
                
                images = convert_from_path(file_path)
                ocr_text = []
                for i, image in enumerate(images):
                    text = pytesseract.image_to_string(
                        image, 
                        lang=self.ocr_language
                    )
                    ocr_text.append(f"Page {i+1}:\n{text}")
                
                advanced_content['ocr_text'] = "\n\n".join(ocr_text)
                logger.info(f"OCR processed {len(images)} pages")
            except ImportError:
                logger.warning("OCR dependencies not available")
            except Exception as e:
                logger.warning(f"OCR processing failed: {e}")
        
        return advanced_content
    
    def parse(self, source: str):
        """Parse PDF with advanced features.
        
        Args:
            source: Path to PDF file
            
        Returns:
            ProcessingResult with documents
        """
        # Extract advanced content if configured
        advanced_content = {}
        if self.extract_tables or self.ocr_enabled:
            advanced_content = self._extract_advanced_content(source)
        
        # Parse with base LlamaIndex reader
        result = super().parse(source)
        
        # Enhance documents with advanced content
        if advanced_content:
            for doc in result.documents:
                doc.metadata.update(advanced_content)
        
        return result