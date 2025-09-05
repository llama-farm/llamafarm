"""LlamaIndex-based Text Parser for plain text files."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_parser import BaseLlamaIndexParser, ParserFactory

logger = logging.getLogger(__name__)

try:
    from llama_index.readers.file import FlatReader
    FLAT_READER_AVAILABLE = True
except ImportError:
    try:
        from llama_index.core.readers import SimpleDirectoryReader
        FLAT_READER_AVAILABLE = False
    except ImportError:
        raise ImportError("LlamaIndex is required. Install with: pip install llama-index")


class LlamaIndexTextParser(BaseLlamaIndexParser):
    """LlamaIndex-based parser for plain text files."""
    
    def __init__(self, name: str = "LlamaIndexTextParser", config: Optional[Dict[str, Any]] = None):
        """
        Initialize LlamaIndex text parser.
        
        Args:
            name: Parser name
            config: Parser configuration
        """
        super().__init__(name=name, config=config or {})
        
        # Text-specific configuration
        self.encoding = self.config.get("encoding", "auto")
        self.preserve_line_breaks = self.config.get("preserve_line_breaks", True)
        self.strip_empty_lines = self.config.get("strip_empty_lines", True)
        self.detect_structure = self.config.get("detect_structure", True)
    
    def _get_reader(self):
        """Get the appropriate LlamaIndex reader for text files."""
        if FLAT_READER_AVAILABLE:
            return FlatReader()
        else:
            # Fallback to SimpleDirectoryReader with file filter
            return SimpleDirectoryReader(
                input_files=[],  # Will be set per file
                file_extractor={
                    ".txt": FlatReader(),
                    ".log": FlatReader(),
                    ".text": FlatReader()
                } if FLAT_READER_AVAILABLE else None
            )
    
    def parse(self, file_path: str, **kwargs) -> "ProcessingResult":
        """
        Parse a text file using LlamaIndex.
        
        Args:
            file_path: Path to the text file
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
        
        try:
            # Handle encoding detection if needed
            if self.encoding == "auto":
                encoding = self._detect_encoding(file_path)
            else:
                encoding = self.encoding
            
            # Read file content
            try:
                with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
            except Exception as e:
                logger.warning(f"Failed to read with encoding {encoding}, falling back to utf-8: {e}")
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                encoding = 'utf-8'
            
            # Process content
            if self.strip_empty_lines or not self.preserve_line_breaks:
                content = self._process_text_content(content)
            
            if not content.strip():
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No content found in file", "source": str(file_path)}]
                )
            
            # Create a mock LlamaIndex document for processing
            from llama_index.core.schema import Document as LlamaDocument
            
            llama_doc = LlamaDocument(
                text=content,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_size": file_path.stat().st_size,
                    "encoding": encoding
                }
            )
            
            # Use base class processing with the mock document
            from core.base import ProcessingResult
            from utils.hash_utils import generate_document_metadata, generate_chunk_metadata
            
            # Generate comprehensive metadata
            base_metadata = generate_document_metadata(str(file_path), content)
            base_metadata.update({
                "parser_type": self.name,
                "encoding": encoding,
                "preserve_line_breaks": self.preserve_line_breaks,
                "strip_empty_lines": self.strip_empty_lines
            })
            
            # Add structure detection if enabled
            if self.detect_structure:
                structure_info = self._detect_text_structure(content)
                base_metadata.update(structure_info)
            
            # Handle chunking if enabled
            result_documents = []
            if self._node_parser and self.chunk_size:
                chunked_documents = self._create_chunked_documents(
                    content, base_metadata, llama_doc
                )
                result_documents.extend(chunked_documents)
            else:
                # Create single document
                from core.base import Document
                document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
                doc = Document(
                    content=content,
                    metadata=base_metadata,
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
                    "parser_type": self.name,
                    "encoding": encoding,
                    "chunked": bool(self._node_parser and self.chunk_size)
                }
            )
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to parse text file: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            import chardet
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                encoding = result.get('encoding', 'utf-8')
                confidence = result.get('confidence', 0)
                
                if confidence > 0.7:
                    logger.debug(f"Detected encoding: {encoding} (confidence: {confidence})")
                    return encoding
                else:
                    logger.debug(f"Low confidence encoding detection ({confidence}), using utf-8")
                    return 'utf-8'
        except ImportError:
            logger.debug("chardet not available, using utf-8 encoding")
            return 'utf-8'
        except Exception as e:
            logger.warning(f"Encoding detection failed: {e}, using utf-8")
            return 'utf-8'
    
    def _process_text_content(self, content: str) -> str:
        """Process text content based on configuration."""
        if self.strip_empty_lines:
            lines = content.split('\n')
            # Remove trailing whitespace from each line
            lines = [line.rstrip() for line in lines]
            
            # Remove completely empty lines but preserve paragraph breaks
            processed_lines = []
            prev_empty = False
            for line in lines:
                if line.strip():
                    processed_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    processed_lines.append('')
                    prev_empty = True
            
            content = '\n'.join(processed_lines)
        
        if not self.preserve_line_breaks:
            # Convert multiple line breaks to single ones and join paragraphs
            lines = content.split('\n')
            paragraphs = []
            current_paragraph = []
            
            for line in lines:
                if line.strip():
                    current_paragraph.append(line.strip())
                else:
                    if current_paragraph:
                        paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = []
            
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            content = '\n\n'.join(paragraphs)
        
        return content
    
    def _detect_text_structure(self, content: str) -> Dict[str, Any]:
        """Detect structural elements in plain text."""
        lines = content.split('\n')
        structure = {
            "has_headers": False,
            "has_lists": False,
            "has_code_blocks": False,
            "has_urls": False,
            "has_emails": False,
            "headers": [],
            "list_items": 0,
            "code_blocks": 0,
            "urls": 0,
            "emails": 0,
            "line_count": len(lines),
            "paragraph_count": 0,
            "avg_line_length": 0
        }
        
        import re
        
        in_code_block = False
        paragraph_count = 0
        total_line_length = 0
        non_empty_lines = 0
        
        url_pattern = re.compile(r'https?://[^\s]+')
        email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if not stripped:
                continue
            
            non_empty_lines += 1
            total_line_length += len(stripped)
            
            # Detect headers (lines that are all caps, start with #, or followed by underlines)
            if (stripped.isupper() and len(stripped) > 3 and len(stripped) < 80) or \
               stripped.startswith('#') or \
               (line_num < len(lines) and lines[line_num].strip() in ['=' * len(stripped), '-' * len(stripped)]):
                structure["has_headers"] = True
                structure["headers"].append({
                    "line": line_num,
                    "text": stripped[:50],  # Limit length
                    "type": "header"
                })
            
            # Detect lists
            if stripped.startswith(('- ', '* ', '+ ')) or \
               (len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] in ['. ', ') ']):
                structure["has_lists"] = True
                structure["list_items"] += 1
            
            # Detect code blocks
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block:
                    structure["code_blocks"] += 1
                structure["has_code_blocks"] = True
            
            # Detect URLs
            urls_in_line = len(url_pattern.findall(line))
            if urls_in_line > 0:
                structure["has_urls"] = True
                structure["urls"] += urls_in_line
            
            # Detect emails
            emails_in_line = len(email_pattern.findall(line))
            if emails_in_line > 0:
                structure["has_emails"] = True
                structure["emails"] += emails_in_line
        
        # Count paragraphs (sequences of non-empty lines)
        in_paragraph = False
        for line in lines:
            if line.strip():
                if not in_paragraph:
                    paragraph_count += 1
                    in_paragraph = True
            else:
                in_paragraph = False
        
        structure["paragraph_count"] = paragraph_count
        structure["avg_line_length"] = total_line_length / non_empty_lines if non_empty_lines > 0 else 0
        structure["non_empty_line_count"] = non_empty_lines
        
        return structure
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    @staticmethod
    def can_parse_mime_type(mime_type: str) -> bool:
        """Check if this parser can handle the given MIME type."""
        return mime_type in [
            'text/plain',
            'text/x-log',
            'application/x-log'
        ]
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions."""
        return ['.txt', '.log', '.text', '.asc', '.readme']
    
    @staticmethod
    def get_description() -> str:
        """Get parser description."""
        return "LlamaIndex-based parser for plain text files (.txt, .log, etc.)"


# Register the parser
ParserFactory.register_parser("LlamaIndexTextParser", LlamaIndexTextParser)