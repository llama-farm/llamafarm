"""Abstract base parser class for all RAG parsers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging

# Import from parent directory
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from core.base import Document, ProcessingResult

logger = logging.getLogger(__name__)


@dataclass
class ParserConfig:
    """Configuration for a parser."""
    name: str
    display_name: str
    version: str
    supported_extensions: List[str]
    mime_types: List[str]
    capabilities: List[str]
    dependencies: Dict[str, List[str]]
    default_config: Dict[str, Any]


class BaseParser(ABC):
    """Abstract base class for all parsers."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize parser with configuration.
        
        Args:
            config: Parser configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load parser metadata
        self.metadata = self._load_metadata()
        
        # Validate configuration against schema
        self._validate_config()
    
    @abstractmethod
    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata from config.yaml.
        
        Returns:
            ParserConfig object with metadata
        """
        pass
    
    @abstractmethod
    def parse(self, source: str) -> ProcessingResult:
        """Parse a file or directory.
        
        Args:
            source: Path to file or directory
            
        Returns:
            ProcessingResult with documents and any errors
        """
        pass
    
    @abstractmethod
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if parser can handle the file
        """
        pass
    
    def _validate_config(self):
        """Validate configuration against parser schema."""
        # This will use the individual parser's schema.json
        # Implementation depends on jsonschema library
        pass
    
    def detect_encoding(self, file_path: str) -> str:
        """Detect file encoding.
        
        Args:
            file_path: Path to file
            
        Returns:
            Detected encoding
        """
        try:
            import chardet
            with open(file_path, 'rb') as f:
                result = chardet.detect(f.read(10000))
                return result['encoding'] or 'utf-8'
        except:
            return 'utf-8'
    
    def create_document(
        self, 
        content: str, 
        metadata: Dict[str, Any] = None,
        doc_id: str = None,
        source: str = None
    ) -> Document:
        """Create a Document object.
        
        Args:
            content: Document content
            metadata: Document metadata
            doc_id: Document ID
            source: Source file path
            
        Returns:
            Document object
        """
        return Document(
            content=content,
            metadata=metadata or {},
            id=doc_id,
            source=source
        )