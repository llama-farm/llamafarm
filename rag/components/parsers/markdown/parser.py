"""Markdown parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any
import logging
import yaml

from ..base import LlamaIndexParser, ParserConfig

logger = logging.getLogger(__name__)

# Lazy imports
MARKDOWN_READER_AVAILABLE = False
try:
    from llama_index.readers.file import MarkdownReader
    from llama_index.core.node_parser import MarkdownNodeParser
    MARKDOWN_READER_AVAILABLE = True
except ImportError:
    logger.warning("Markdown reader not available. Install with: pip install llama-index-readers-file")


class MarkdownParser(LlamaIndexParser):
    """Markdown parser with structure-aware chunking."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Markdown parser.
        
        Args:
            config: Parser configuration
        """
        if not MARKDOWN_READER_AVAILABLE:
            raise ImportError("Markdown reader required. Install with: pip install llama-index-readers-file")
        
        super().__init__(config)
        
        # Initialize Markdown reader
        self.reader = MarkdownReader()
        
        # Override text splitter with Markdown-specific one
        if self.config.get("chunk_strategy") == "markdown_aware":
            self.text_splitter = MarkdownNodeParser()
    
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
            name="markdown",
            display_name="Markdown Parser",
            version="2.0.0",
            supported_extensions=[".md", ".markdown", ".mdown", ".mkd"],
            mime_types=["text/markdown", "text/x-markdown"],
            capabilities=[
                "text_extraction",
                "chunking",
                "structure_preservation",
                "code_block_extraction",
                "link_extraction"
            ],
            dependencies={
                "required": ["llama-index", "llama-index-readers-file"],
                "optional": ["markdown2", "mistune"]
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "markdown_aware",
                "preserve_structure": True,
                "extract_code_blocks": True
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