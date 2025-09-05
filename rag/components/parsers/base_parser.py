"""Base LlamaIndex Parser - Abstract base class for all LlamaIndex-based parsers."""

import logging
from abc import ABC, abstractmethod

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type

# Import hash utilities for deduplication
from utils.hash_utils import (
    generate_document_metadata,
    generate_chunk_metadata,
    DeduplicationTracker
)

from core.base import Document, Parser, ProcessingResult

logger = logging.getLogger(__name__)

try:
    from llama_index.core import SimpleDirectoryReader
    from llama_index.core.node_parser import (
        SimpleNodeParser, 
        SentenceSplitter, 
        TokenTextSplitter,
        SemanticSplitterNodeParser
    )
    from llama_index.core.schema import BaseNode, TextNode
    LLAMA_INDEX_AVAILABLE = True
    logger.info("LlamaIndex is available")
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    logger.warning("LlamaIndex not available. Install with: pip install llama-index")


class BaseLlamaIndexParser(Parser):
    """Abstract base class that wraps LlamaIndex readers with unified chunking interface."""
    
    def __init__(self, name: str = "BaseLlamaIndexParser", config: Optional[Dict[str, Any]] = None):
        """
        Initialize base LlamaIndex parser.
        
        Args:
            name: Parser name
            config: Parser configuration
        """
        super().__init__(name=name, config=config or {})
        
        if not LLAMA_INDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex is required for LlamaIndex-based parsing. "
                "Install it with: pip install llama-index"
            )
        
        # Configuration options for chunking
        self.chunk_size = self.config.get("chunk_size", None)
        self.chunk_overlap = self.config.get("chunk_overlap", 0)
        self.chunk_strategy = self.config.get("chunk_strategy", "characters")
        self.respect_sentence_boundaries = self.config.get("respect_sentence_boundaries", True)
        self.respect_paragraph_boundaries = self.config.get("respect_paragraph_boundaries", False)
        self.min_chunk_size = self.config.get("min_chunk_size", 50)
        
        # Initialize the LlamaIndex reader
        self._reader = None
        self._node_parser = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize LlamaIndex components based on configuration."""
        # Initialize the reader (to be implemented by subclasses)
        self._reader = self._get_reader()
        
        # Initialize the node parser for chunking
        if self.chunk_size:
            self._node_parser = self._get_node_parser()
    
    @abstractmethod
    def _get_reader(self):
        """Get the appropriate LlamaIndex reader. To be implemented by subclasses."""
        pass
    
    def _get_node_parser(self):
        """Get the appropriate node parser based on chunk strategy."""
        if self.chunk_strategy == "sentences":
            return SentenceSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                paragraph_separator="\n\n",
                chunking_tokenizer_fn=None,  # Use default
                secondary_chunking_regex=None,
                respect_sentence_boundaries=self.respect_sentence_boundaries
            )
        elif self.chunk_strategy == "tokens":
            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separator=" ",
                backup_separators=["\n"]
            )
        else:  # Default to simple character-based splitting
            return SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                include_metadata=True,
                include_prev_next_rel=False
            )
    
    @staticmethod
    def detect_file_type(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Detect file type using content-based analysis (python-magic).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict containing file type information
        """
        file_path = Path(file_path)
        
        detection_result = {
            "file_path": str(file_path),
            "extension": file_path.suffix.lower(),
            "mime_type": None,
            "encoding": None,
            "description": None,
            "is_text": False,
            "confidence": "low"
        }
        
        if not file_path.exists():
            detection_result["error"] = "File not found"
            return detection_result
        
        try:
            # Use python-magic for MIME type detection
            mime_type = magic.from_file(str(file_path), mime=True)
            description = magic.from_file(str(file_path))
            
            detection_result.update({
                "mime_type": mime_type,
                "description": description,
                "confidence": "high"
            })
            
            # Determine if it's a text-based file
            if mime_type:
                detection_result["is_text"] = (
                    mime_type.startswith('text/') or
                    mime_type in [
                        'application/json',
                        'application/xml',
                        'application/javascript',
                        'application/x-yaml'
                    ]
                )
            
            # Try to detect encoding for text files
            if detection_result["is_text"]:
                try:
                    encoding = magic.from_file(str(file_path), mime=True)
                    detection_result["encoding"] = "utf-8"  # Default assumption
                except:
                    detection_result["encoding"] = "unknown"
                    
        except Exception as e:
            logger.warning(f"Failed to detect file type for {file_path}: {e}")
            detection_result["error"] = str(e)
            # Fallback to extension-based detection
            detection_result["confidence"] = "extension_fallback"
        
        return detection_result
    
    def parse(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Parse a file using LlamaIndex.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional parsing options
            
        Returns:
            ProcessingResult containing parsed documents
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {file_path}", "source": str(file_path)}]
            )
        
        try:
            # Use LlamaIndex to load the document
            documents = self._reader.load_data(file=str(file_path))
            
            if not documents:
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No content extracted", "source": str(file_path)}]
                )
            
            # Convert LlamaIndex documents to our Document format
            result_documents = []
            errors = []
            
            for i, llama_doc in enumerate(documents):
                try:
                    # Extract content
                    content = llama_doc.text or ""
                    
                    if len(content.strip()) < self.min_chunk_size:
                        continue
                    
                    # Generate comprehensive metadata
                    base_metadata = generate_document_metadata(str(file_path), content)
                    base_metadata.update({
                        "parser_type": self.name,
                        "llama_index_doc_id": llama_doc.id_,
                        "doc_index": i
                    })
                    
                    # Add any LlamaIndex metadata
                    if hasattr(llama_doc, 'metadata') and llama_doc.metadata:
                        base_metadata.update(llama_doc.metadata)
                    
                    # Handle chunking if enabled
                    if self._node_parser and self.chunk_size:
                        chunked_documents = self._create_chunked_documents(
                            content, base_metadata, llama_doc
                        )
                        result_documents.extend(chunked_documents)
                    else:
                        # Create single document
                        document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
                        doc = Document(
                            content=content,
                            metadata=base_metadata,
                            id=document_id,
                            source=str(file_path)
                        )
                        result_documents.append(doc)
                        
                except Exception as e:
                    errors.append({
                        "error": f"Failed to process document {i}: {str(e)}",
                        "source": str(file_path),
                        "document_index": i
                    })
            
            return ProcessingResult(
                documents=result_documents,
                errors=errors,
                metrics={
                    "total_documents": len(result_documents),
                    "total_errors": len(errors),
                    "file_processed": str(file_path),
                    "parser_type": self.name,
                    "chunked": bool(self._node_parser and self.chunk_size)
                }
            )
            
        except Exception as e:
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to parse file: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _create_chunked_documents(
        self, 
        content: str, 
        base_metadata: Dict[str, Any], 
        original_llama_doc
    ) -> List[Document]:
        """Create chunked documents using LlamaIndex node parser."""
        documents = []
        
        try:
            # Create a TextNode from the content
            text_node = TextNode(
                text=content,
                metadata=base_metadata.copy(),
                id_=original_llama_doc.id_
            )
            
            # Use LlamaIndex to split into chunks
            nodes = self._node_parser.get_nodes_from_documents([text_node])
            
            total_chunks = len(nodes)
            
            for chunk_idx, node in enumerate(nodes):
                chunk_content = node.text.strip()
                
                if len(chunk_content) < self.min_chunk_size:
                    continue
                
                # Generate chunk metadata with hash utilities
                chunk_metadata = generate_chunk_metadata(
                    base_metadata,
                    chunk_content,
                    chunk_idx,
                    total_chunks
                )
                
                # Add LlamaIndex node metadata
                chunk_metadata.update({
                    "node_id": node.id_,
                    "chunk_strategy": self.chunk_strategy,
                    "has_overlap": self.chunk_overlap > 0 and chunk_idx > 0,
                    "llama_index_chunked": True
                })
                
                # Add any node-specific metadata
                if hasattr(node, 'metadata') and node.metadata:
                    chunk_metadata.update(node.metadata)
                
                doc = Document(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    id=chunk_metadata["chunk_id"],
                    source=base_metadata.get('file_path', '')
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Failed to create chunked documents: {e}")
            # Fallback: create single document
            document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
            doc = Document(
                content=content,
                metadata=base_metadata,
                id=document_id,
                source=base_metadata.get('file_path', '')
            )
            documents.append(doc)
        
        return documents
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the parser can handle this file
        """
        # This should be implemented by subclasses
        # Base implementation checks supported extensions
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    @staticmethod
    @abstractmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions."""
        pass
    
    @staticmethod
    @abstractmethod
    def get_description() -> str:
        """Get parser description."""
        pass


# Factory pattern for easy extension
class ParserFactory:
    """Factory for creating parser instances."""
    
    _parsers = {}
    
    @classmethod
    def register_parser(cls, name: str, parser_class: Type[BaseLlamaIndexParser]):
        """Register a parser class."""
        cls._parsers[name] = parser_class
    
    @classmethod
    def create_parser(cls, name: str, config: Optional[Dict[str, Any]] = None) -> BaseLlamaIndexParser:
        """Create a parser instance."""
        if name not in cls._parsers:
            raise ValueError(f"Unknown parser: {name}")
        
        return cls._parsers[name](config=config)
    
    @classmethod
    def list_parsers(cls) -> Dict[str, Type[BaseLlamaIndexParser]]:
        """List all registered parsers."""
        return cls._parsers.copy()
    
    @classmethod
    def get_parser_for_file(cls, file_path: str) -> Optional[Type[BaseLlamaIndexParser]]:
        """Get the best parser for a file based on file type detection."""
        file_info = BaseLlamaIndexParser.detect_file_type(file_path)
        extension = file_info["extension"]
        mime_type = file_info.get("mime_type", "")
        
        # Try to find a parser that can handle this file
        for parser_class in cls._parsers.values():
            if hasattr(parser_class, 'can_parse_mime_type'):
                if parser_class.can_parse_mime_type(mime_type):
                    return parser_class
            
            # Fallback to extension matching
            if extension in parser_class.get_supported_extensions():
                return parser_class
        
        return None