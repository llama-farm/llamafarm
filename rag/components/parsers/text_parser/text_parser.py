"""
Plain Text Parser for .txt files and other plain text formats.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Import hash utilities for deduplication
from utils.hash_utils import (
    generate_document_metadata,
    generate_chunk_metadata,
    DeduplicationTracker
)

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

from core.base import Document, Parser, ProcessingResult

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
        super().__init__(name=name, config=config or {})
        
        # Configuration options
        self.encoding = self.config.get("encoding", "auto")  # auto-detect or specify
        self.chunk_size = self.config.get("chunk_size", None)  # Split into chunks if specified
        self.chunk_overlap = self.config.get("chunk_overlap", 0)  # Overlap between chunks
        self.preserve_line_breaks = self.config.get("preserve_line_breaks", True)
        self.strip_empty_lines = self.config.get("strip_empty_lines", True)
        self.detect_structure = self.config.get("detect_structure", True)  # Detect headers, lists, etc.
        
        # New chunking options
        self.chunk_strategy = self.config.get("chunk_strategy", "characters")  # characters, sentences, paragraphs
        self.respect_sentence_boundaries = self.config.get("respect_sentence_boundaries", True)  # Don't break sentences
        self.respect_paragraph_boundaries = self.config.get("respect_paragraph_boundaries", False)  # Don't break paragraphs
        self.min_chunk_size = self.config.get("min_chunk_size", 50)  # Minimum chunk size to avoid tiny chunks
    
    def parse(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Parse a plain text file.
        
        Args:
            file_path: Path to the text file
            **kwargs: Additional parsing options
            
        Returns:
            ProcessingResult containing parsed documents
        """
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
                logger.debug("chardet not available, defaulting to utf-8 encoding")
        
        # Read the file
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
        except UnicodeDecodeError:
            # Fallback to utf-8 with error handling
            logger.warning(f"Failed to decode with {encoding}, falling back to utf-8 with error handling")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
        
        # Process content first
        processed_content = self._process_content(content)
        
        # Generate comprehensive metadata with hash utilities
        base_metadata = generate_document_metadata(str(file_path), processed_content)
        
        # Add parser-specific metadata
        base_metadata.update({
            "parser_type": "PlainTextParser",
            "encoding": encoding
        })
        
        # Detect structure if enabled
        if self.detect_structure:
            structure_info = self._detect_structure(processed_content)
            base_metadata.update(structure_info)
        
        # Add content statistics
        lines = processed_content.split('\n')
        base_metadata.update({
            "line_count": len(lines),
            "character_count": len(processed_content),
            "word_count": len(processed_content.split()) if processed_content else 0,
            "non_empty_lines": len([line for line in lines if line.strip()])
        })
        
        # Split into chunks if specified
        if self.chunk_size and len(processed_content) > self.chunk_size:
            documents = self._create_chunked_documents(processed_content, base_metadata)
        else:
            # Create single document with hash-based ID
            document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
            documents = [Document(
                content=processed_content,
                metadata=base_metadata,
                id=document_id,
                source=str(file_path)
            )]
        
        return ProcessingResult(documents=documents, errors=[])
    
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
        """Split content into chunks with overlap and create multiple documents."""
        documents = []
        
        if self.chunk_overlap >= self.chunk_size:
            logger.warning("chunk_overlap should be smaller than chunk_size")
            self.chunk_overlap = min(self.chunk_overlap, self.chunk_size // 2)
        
        # Choose chunking strategy
        if self.chunk_strategy == "paragraphs":
            chunks = self._chunk_by_paragraphs(content)
        elif self.chunk_strategy == "sentences":
            chunks = self._chunk_by_sentences(content)
        else:  # characters (with boundary protection)
            chunks = self._chunk_by_characters(content)
        
        # Create documents from chunks with hash-based metadata
        total_chunks = len([c for c in chunks if len(c.strip()) >= self.min_chunk_size])
        
        for chunk_num, chunk_content in enumerate(chunks, 1):
            if len(chunk_content.strip()) >= self.min_chunk_size:
                # Generate chunk metadata with hash utilities
                chunk_metadata = generate_chunk_metadata(
                    base_metadata,
                    chunk_content.strip(),
                    chunk_num - 1,  # 0-based index for hash generation
                    total_chunks
                )
                
                # Add parser-specific chunk metadata
                chunk_metadata.update({
                    "chunk_strategy": self.chunk_strategy,
                    "has_overlap": self.chunk_overlap > 0 and chunk_num > 1,
                    "respects_sentences": self.respect_sentence_boundaries,
                    "respects_paragraphs": self.respect_paragraph_boundaries
                })
                
                documents.append(Document(
                    content=chunk_content.strip(),
                    metadata=chunk_metadata,
                    id=chunk_metadata["chunk_id"],
                    source=base_metadata['file_path']
                ))
        
        return documents
    
    def _chunk_by_characters(self, content: str) -> List[str]:
        """Split content by character count with intelligent boundary protection."""
        chunks = []
        start_pos = 0
        
        while start_pos < len(content):
            end_pos = start_pos + self.chunk_size
            
            # If we're at the end, take remaining content
            if end_pos >= len(content):
                chunk_content = content[start_pos:]
                if chunk_content.strip():
                    chunks.append(chunk_content)
                break
            
            # Get initial chunk
            chunk_content = content[start_pos:end_pos]
            
            # Apply boundary protection in order of priority
            if self.respect_paragraph_boundaries:
                chunk_content, actual_end = self._find_paragraph_boundary(content, start_pos, end_pos)
            elif self.respect_sentence_boundaries:
                chunk_content, actual_end = self._find_sentence_boundary(content, start_pos, end_pos)
            else:
                # Just respect word boundaries (original behavior)
                last_space = chunk_content.rfind(' ')
                if last_space > 0 and last_space > len(chunk_content) * 0.8:
                    chunk_content = chunk_content[:last_space]
                    actual_end = start_pos + last_space
                else:
                    actual_end = end_pos
            
            if chunk_content.strip():
                chunks.append(chunk_content)
            
            # Move start position forward (with overlap consideration)
            start_pos = max(start_pos + 1, actual_end - self.chunk_overlap)
        
        return chunks
    
    def _chunk_by_sentences(self, content: str) -> List[str]:
        """Split content by sentences, respecting chunk size limits."""
        # Split into sentences using improved regex
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, content)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size, save current chunk and start new one
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Handle overlap by including last sentence(s) from previous chunk
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = self._get_sentence_overlap(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraphs, combining small ones and splitting large ones."""
        # Split by double newlines (paragraph boundaries)
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If this paragraph alone is larger than chunk size, split it by sentences
            if len(paragraph) > self.chunk_size:
                # Save current chunk if it exists
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # Split large paragraph by sentences
                sentence_chunks = self._chunk_large_paragraph(paragraph)
                chunks.extend(sentence_chunks)
                
            # If adding this paragraph would exceed chunk size, save current chunk
            elif current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                # Handle overlap
                if self.chunk_overlap > 0 and chunks:
                    overlap_text = self._get_paragraph_overlap(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + "\n\n" + paragraph if overlap_text else paragraph
                else:
                    current_chunk = paragraph
            else:
                current_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _find_sentence_boundary(self, content: str, start_pos: int, end_pos: int) -> tuple[str, int]:
        """Find the best sentence boundary before end_pos."""
        chunk_content = content[start_pos:end_pos]
        
        # Look for sentence endings (., !, ?) followed by space and capital letter
        sentence_endings = []
        for match in re.finditer(r'[.!?]\s+(?=[A-Z])', chunk_content):
            sentence_endings.append(match.end())
        
        if sentence_endings:
            # Use the last sentence ending that's not too early in the chunk
            best_end = sentence_endings[-1]
            if best_end > len(chunk_content) * 0.5:  # At least 50% of chunk size
                actual_chunk = chunk_content[:best_end]
                return actual_chunk, start_pos + best_end
        
        # Fallback to word boundary
        last_space = chunk_content.rfind(' ')
        if last_space > 0 and last_space > len(chunk_content) * 0.8:
            return chunk_content[:last_space], start_pos + last_space
        
        return chunk_content, end_pos
    
    def _find_paragraph_boundary(self, content: str, start_pos: int, end_pos: int) -> tuple[str, int]:
        """Find the best paragraph boundary before end_pos."""
        chunk_content = content[start_pos:end_pos]
        
        # Look for paragraph breaks (double newlines)
        last_paragraph_break = chunk_content.rfind('\n\n')
        if last_paragraph_break > 0 and last_paragraph_break > len(chunk_content) * 0.3:
            actual_chunk = chunk_content[:last_paragraph_break + 2]  # Include the newlines
            return actual_chunk, start_pos + last_paragraph_break + 2
        
        # Fallback to sentence boundary
        return self._find_sentence_boundary(content, start_pos, end_pos)
    
    def _get_sentence_overlap(self, text: str, overlap_size: int) -> str:
        """Get the last few sentences from text for overlap."""
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        overlap_text = ""
        
        # Add sentences from the end until we reach overlap size
        for sentence in reversed(sentences):
            if len(overlap_text + sentence) <= overlap_size:
                overlap_text = sentence + " " + overlap_text if overlap_text else sentence
            else:
                break
        
        return overlap_text.strip()
    
    def _get_paragraph_overlap(self, text: str, overlap_size: int) -> str:
        """Get the last paragraph(s) from text for overlap."""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        overlap_text = ""
        
        # Add paragraphs from the end until we reach overlap size
        for paragraph in reversed(paragraphs):
            if len(overlap_text + paragraph) <= overlap_size:
                overlap_text = paragraph + "\n\n" + overlap_text if overlap_text else paragraph
            else:
                break
        
        return overlap_text.strip()
    
    def _chunk_large_paragraph(self, paragraph: str) -> List[str]:
        """Split a large paragraph by sentences."""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, paragraph)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_chunk and len(current_chunk) + len(sentence) + 1 > self.chunk_size:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                current_chunk = current_chunk + " " + sentence if current_chunk else sentence
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
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