"""DOCX parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any
import logging
import yaml

from ..base import LlamaIndexParser, ParserConfig

logger = logging.getLogger(__name__)

# Lazy imports
DOCX_READER_AVAILABLE = False
try:
    from llama_index.readers.file import DocxReader
    DOCX_READER_AVAILABLE = True
except ImportError:
    logger.warning("DOCX reader not available. Install with: pip install llama-index-readers-file python-docx")


class DOCXParser(LlamaIndexParser):
    """DOCX parser with style and structure preservation."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize DOCX parser.
        
        Args:
            config: Parser configuration
        """
        if not DOCX_READER_AVAILABLE:
            raise ImportError("DOCX reader required. Install with: pip install llama-index-readers-file python-docx")
        
        super().__init__(config)
        
        # Initialize DOCX reader
        self.reader = DocxReader()
        
        # DOCX-specific options
        self.extract_images = self.config.get("extract_images", False)
        self.preserve_formatting = self.config.get("preserve_formatting", False)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_headers_footers = self.config.get("extract_headers_footers", False)
    
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
        
        return ParserConfig(
            name="docx",
            display_name="DOCX Parser",
            version="2.0.0",
            supported_extensions=[".docx", ".doc", ".odt", ".rtf"],
            mime_types=[
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/msword",
                "application/vnd.oasis.opendocument.text",
                "application/rtf"
            ],
            capabilities=[
                "text_extraction",
                "chunking",
                "table_extraction",
                "image_extraction",
                "metadata_extraction",
                "style_preservation"
            ],
            dependencies={
                "required": ["llama-index", "llama-index-readers-file", "python-docx"],
                "optional": ["mammoth", "python-docx2txt"]
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "paragraphs",
                "extract_images": False,
                "preserve_formatting": False,
                "extract_tables": True,
                "extract_headers_footers": False
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
        return path.suffix.lower() in self.metadata.supported_extensions