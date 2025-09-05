"""Web parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any
import logging
import yaml

from ..base import LlamaIndexParser, ParserConfig

logger = logging.getLogger(__name__)

# Lazy imports
WEB_READER_AVAILABLE = False
try:
    from llama_index.readers.web import SimpleWebPageReader, BeautifulSoupWebReader
    WEB_READER_AVAILABLE = True
except ImportError:
    logger.warning("Web reader not available. Install with: pip install llama-index-readers-web beautifulsoup4")


class WebParser(LlamaIndexParser):
    """Web page parser with content extraction and cleaning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Web parser.
        
        Args:
            config: Parser configuration
        """
        if not WEB_READER_AVAILABLE:
            raise ImportError("Web reader required. Install with: pip install llama-index-readers-web beautifulsoup4")
        
        super().__init__(config)
        
        # Web-specific options
        self.use_beautiful_soup = self.config.get("use_beautiful_soup", True)
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.remove_scripts = self.config.get("remove_scripts", True)
        self.remove_styles = self.config.get("remove_styles", True)
        
        # Initialize appropriate reader
        if self.use_beautiful_soup:
            self.reader = BeautifulSoupWebReader()
        else:
            self.reader = SimpleWebPageReader()
    
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
            name="web",
            display_name="Web Parser",
            version="2.0.0",
            supported_extensions=[".html", ".htm", ".xhtml"],
            mime_types=["text/html", "application/xhtml+xml"],
            capabilities=[
                "text_extraction",
                "chunking",
                "metadata_extraction",
                "link_extraction",
                "structured_data_extraction"
            ],
            dependencies={
                "required": ["llama-index", "llama-index-readers-web", "beautifulsoup4"],
                "optional": ["lxml", "html5lib", "requests", "selenium"]
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "paragraphs",
                "use_beautiful_soup": True,
                "extract_metadata": True,
                "remove_scripts": True,
                "remove_styles": True
            }
        )
    
    def can_parse(self, source: str) -> bool:
        """Check if this parser can handle the source.
        
        Args:
            source: File path or URL
            
        Returns:
            True if parser can handle the source
        """
        # Check if it's a URL
        if source.startswith(('http://', 'https://')):
            return True
        
        # Check if it's an HTML file
        path = Path(source)
        return path.suffix.lower() in self.metadata.supported_extensions
    
    def parse(self, source: str):
        """Parse web page or HTML file.
        
        Args:
            source: URL or path to HTML file
            
        Returns:
            ProcessingResult with documents
        """
        # If it's a local file, convert to file:// URL for reader
        if not source.startswith(('http://', 'https://')):
            path = Path(source).absolute()
            if path.exists():
                source = f"file://{path}"
        
        # Use base implementation with web reader
        return super().parse(source)