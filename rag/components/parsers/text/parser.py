"""Text parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any
import logging
import yaml

from ..base import LlamaIndexParser, ParserConfig

logger = logging.getLogger(__name__)

# Lazy imports
TEXT_READER_AVAILABLE = False
try:
    from llama_index.readers.file import FlatReader
    TEXT_READER_AVAILABLE = True
except ImportError:
    logger.warning("Text reader not available. Install with: pip install llama-index-readers-file")


class TextParser(LlamaIndexParser):
    """Plain text parser with encoding detection."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize text parser.
        
        Args:
            config: Parser configuration
        """
        if not TEXT_READER_AVAILABLE:
            raise ImportError("Text reader required. Install with: pip install llama-index-readers-file")
        
        super().__init__(config)
        
        # Initialize text reader
        self.reader = FlatReader()
        
        # Text-specific options
        self.encoding = self.config.get("encoding", "auto")
        self.clean_text = self.config.get("clean_text", True)
        self.preserve_whitespace = self.config.get("preserve_whitespace", False)
        self.extract_metadata = self.config.get("extract_metadata", True)
    
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
            name="text",
            display_name="Text Parser",
            version="2.0.0",
            supported_extensions=[
                ".txt", ".text", ".log", ".md", ".markdown",
                ".rst", ".tex", ".rtf", ".asc", ".ascii"
            ],
            mime_types=[
                "text/plain",
                "text/x-log",
                "text/markdown",
                "text/x-rst",
                "text/x-tex",
                "text/rtf"
            ],
            capabilities=[
                "text_extraction",
                "chunking",
                "encoding_detection",
                "metadata_extraction"
            ],
            dependencies={
                "required": ["llama-index", "llama-index-readers-file"],
                "optional": ["chardet", "ftfy"]
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "sentences",
                "encoding": "auto",
                "clean_text": True,
                "preserve_whitespace": False,
                "extract_metadata": True
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
        if path.suffix.lower() in self.metadata.supported_extensions:
            return True
        
        # Check if file is text by reading first bytes
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(8192)
                # Try to decode as text
                try:
                    chunk.decode('utf-8')
                    return True
                except UnicodeDecodeError:
                    # Try other encodings
                    for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                        try:
                            chunk.decode(encoding)
                            return True
                        except:
                            continue
        except:
            pass
        
        return False
    
    def _detect_encoding(self, file_path: str) -> str:
        """Detect file encoding.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected encoding
        """
        if self.encoding != "auto":
            return self.encoding
        
        try:
            import chardet
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                return result['encoding'] or 'utf-8'
        except ImportError:
            logger.warning("chardet not available, using utf-8")
            return 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def _clean_text_content(self, text: str) -> str:
        """Clean text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text
        """
        if not self.clean_text:
            return text
        
        # Try to fix text encoding issues
        try:
            import ftfy
            text = ftfy.fix_text(text)
        except ImportError:
            pass
        
        if not self.preserve_whitespace:
            # Normalize whitespace
            lines = text.split('\n')
            lines = [' '.join(line.split()) for line in lines]
            text = '\n'.join(lines)
            
            # Remove multiple blank lines
            while '\n\n\n' in text:
                text = text.replace('\n\n\n', '\n\n')
        
        return text.strip()
    
    def parse(self, source: str):
        """Parse text file with encoding detection.
        
        Args:
            source: Path to text file
            
        Returns:
            ProcessingResult with documents
        """
        # Detect encoding
        encoding = self._detect_encoding(source)
        logger.debug(f"Using encoding {encoding} for {source}")
        
        # Read file with detected encoding
        try:
            with open(source, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to read file {source}: {e}")
            # Try with base reader as fallback
            return super().parse(source)
        
        # Clean text if configured
        content = self._clean_text_content(content)
        
        # Extract metadata if configured
        metadata = {}
        if self.extract_metadata:
            path = Path(source)
            metadata = {
                "file_name": path.name,
                "file_size": path.stat().st_size,
                "encoding": encoding,
                "line_count": content.count('\n') + 1,
                "word_count": len(content.split()),
                "char_count": len(content)
            }
        
        # Create document
        from core.base import Document
        doc = Document(
            content=content,
            metadata=metadata,
            source=source
        )
        
        # Apply chunking if configured
        documents = self._apply_chunking([doc])
        
        # Return result
        from core.base import ProcessingResult
        return ProcessingResult(
            documents=documents,
            errors=[],
            metrics={
                'total_documents': len(documents),
                'encoding': encoding,
                'parser_type': 'TextParser'
            }
        )