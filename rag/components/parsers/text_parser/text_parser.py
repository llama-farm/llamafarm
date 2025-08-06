"""
Plain Text Parser for .txt files and other plain text formats.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

from core.base import Document, Parser

logger = logging.getLogger(__name__)


class PlainTextParser(Parser):
    """Parser for plain text files (.txt, .log, etc.)."""
    
    def __init__(self, name: str = "PlainTextParser", config: Optional[Dict[str, Any]] = None):
        """
        Initialize plain text parser.
        
        Args:
            name: Parser name
            config: Parser configuration
        """
        super().__init__(name, config)
        
        # Configuration options
        self.encoding = self.config.get("encoding", "auto")  # auto-detect or specify
        self.chunk_size = self.config.get("chunk_size", None)  # Split into chunks if specified
        self.preserve_line_breaks = self.config.get("preserve_line_breaks", True)
        self.strip_empty_lines = self.config.get("strip_empty_lines", True)
        self.detect_structure = self.config.get("detect_structure", True)  # Detect headers, lists, etc.
    
    def parse(self, file_path: str, **kwargs) -> List[Document]:
        """
        Parse a plain text file.
        
        Args:
            file_path: Path to the text file
            **kwargs: Additional parsing options
            
        Returns:
            List of Document objects
        """
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Detect encoding if needed
            encoding = self.encoding
            if encoding == "auto":
                if CHARDET_AVAILABLE:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read()
                        detected = chardet.detect(raw_data)
                        encoding = detected.get('encoding', 'utf-8')
                        logger.debug(f"Detected encoding: {encoding} (confidence: {detected.get('confidence', 0)})")
                else:
                    encoding = 'utf-8'
                    logger.warning("chardet not available, defaulting to utf-8 encoding")
            
            # Read the file
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Fallback to utf-8 with error handling
                logger.warning(f"Failed to decode with {encoding}, falling back to utf-8 with error handling")
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            
            # Basic metadata
            metadata = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size,
                "parser_type": "PlainTextParser",
                "encoding": encoding
            }
            
            # Process content
            processed_content = self._process_content(content)
            
            # Detect structure if enabled
            if self.detect_structure:
                structure_info = self._detect_structure(processed_content)
                metadata.update(structure_info)
            
            # Add content statistics
            lines = processed_content.split('\n')
            metadata.update({
                "line_count": len(lines),
                "character_count": len(processed_content),
                "word_count": len(processed_content.split()) if processed_content else 0,
                "non_empty_lines": len([line for line in lines if line.strip()])
            })
            
            # Split into chunks if specified
            if self.chunk_size and len(processed_content) > self.chunk_size:
                return self._create_chunked_documents(processed_content, metadata)
            else:
                return [Document(
                    content=processed_content,
                    metadata=metadata,
                    id=f"txt_{file_path.stem}_{hash(processed_content) % 1000000}",
                    source=str(file_path)
                )]
                
        except Exception as e:
            logger.error(f"Error parsing text file {file_path}: {e}")
            raise
    
    def _process_content(self, content: str) -> str:
        """Process the raw content according to configuration."""
        if self.strip_empty_lines:
            lines = content.split('\n')
            lines = [line.rstrip() for line in lines]  # Remove trailing whitespace
            if self.strip_empty_lines:
                # Remove completely empty lines but preserve single empty lines between paragraphs
                processed_lines = []
                prev_empty = False
                for line in lines:
                    if line.strip():
                        processed_lines.append(line)
                        prev_empty = False
                    elif not prev_empty:
                        processed_lines.append(line)
                        prev_empty = True
                content = '\n'.join(processed_lines)
        
        return content
    
    def _detect_structure(self, content: str) -> Dict[str, Any]:
        """Detect structural elements in the text."""
        lines = content.split('\n')
        structure = {
            "has_headers": False,
            "has_lists": False,
            "has_code_blocks": False,
            "headers": [],
            "list_items": 0,
            "code_blocks": 0
        }
        
        in_code_block = False
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Detect headers (lines that are all caps or start with #)
            if stripped and (
                stripped.isupper() and len(stripped) > 3 or
                stripped.startswith('#') or
                (line_num < len(lines) - 1 and lines[line_num].strip() in ['===', '---']) # Next line is underline
            ):
                structure["has_headers"] = True
                structure["headers"].append({
                    "line": line_num,
                    "text": stripped,
                    "type": "header"
                })
            
            # Detect lists
            if stripped.startswith(('- ', '* ', '+ ')) or (
                len(stripped) > 2 and stripped[0].isdigit() and stripped[1:3] in ['. ', ') ']
            ):
                structure["has_lists"] = True
                structure["list_items"] += 1
            
            # Detect code blocks
            if stripped.startswith('```'):
                in_code_block = not in_code_block
                if not in_code_block:
                    structure["code_blocks"] += 1
            elif in_code_block:
                structure["has_code_blocks"] = True
        
        return structure
    
    def _create_chunked_documents(self, content: str, base_metadata: Dict[str, Any]) -> List[Document]:
        """Split content into chunks and create multiple documents."""
        documents = []
        
        # Simple chunking by character count with word boundaries
        words = content.split()
        current_chunk = []
        current_size = 0
        chunk_num = 1
        
        for word in words:
            word_size = len(word) + 1  # +1 for space
            if current_size + word_size > self.chunk_size and current_chunk:
                # Create document for current chunk
                chunk_content = ' '.join(current_chunk)
                chunk_metadata = base_metadata.copy()
                chunk_metadata.update({
                    "chunk_number": chunk_num,
                    "chunk_size": len(chunk_content),
                    "is_chunk": True
                })
                
                documents.append(Document(
                    content=chunk_content,
                    metadata=chunk_metadata,
                    id=f"txt_{Path(base_metadata['file_path']).stem}_chunk_{chunk_num}_{hash(chunk_content) % 1000000}",
                    source=base_metadata['file_path']
                ))
                
                current_chunk = [word]
                current_size = word_size
                chunk_num += 1
            else:
                current_chunk.append(word)
                current_size += word_size
        
        # Add final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunk_metadata = base_metadata.copy()
            chunk_metadata.update({
                "chunk_number": chunk_num,
                "chunk_size": len(chunk_content),
                "is_chunk": True
            })
            
            documents.append(Document(
                content=chunk_content,
                metadata=chunk_metadata,
                id=f"txt_{Path(base_metadata['file_path']).stem}_chunk_{chunk_num}_{hash(chunk_content) % 1000000}",
                source=base_metadata['file_path']
            ))
        
        return documents
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if this parser can handle the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if the parser can handle this file
        """
        return Path(file_path).suffix.lower() in ['.txt', '.log', '.text', '.asc', '.readme']
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions."""
        return ['.txt', '.log', '.text', '.asc', '.readme']
    
    @staticmethod
    def get_description() -> str:
        """Get parser description."""
        return "Parser for plain text files (.txt, .log, etc.)"