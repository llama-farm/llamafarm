"""Markdown document parser implementation."""

import logging
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

from core.base import Parser, Document, ProcessingResult

logger = logging.getLogger(__name__)


class MarkdownParser(Parser):
    """Parser for Markdown documents."""
    
    def __init__(self, name: str = "MarkdownParser", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        config = config or {}
        self.extract_headers = config.get("extract_headers", True)
        self.extract_links = config.get("extract_links", True)
        self.extract_code_blocks = config.get("extract_code_blocks", True)
        self.preserve_structure = config.get("preserve_structure", True)
    
    def validate_config(self) -> bool:
        """Validate parser configuration."""
        return True  # Markdown parser has no external dependencies to validate
    
    def parse(self, content: str, **kwargs) -> ProcessingResult:
        """Parse markdown content into documents."""
        errors = []
        documents = []
        
        try:
            # Handle file path or content string
            if Path(content).exists():
                with open(content, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
            else:
                text_content = str(content)
            
            if self.preserve_structure:
                # Split by headers to create separate documents
                documents = self._parse_by_sections(text_content, **kwargs)
            else:
                # Create single document
                documents = [self._create_single_document(text_content, **kwargs)]
            
        except Exception as e:
            logger.error(f"Error parsing markdown: {e}")
            errors.append({"error": str(e), "source": content})
            
        return ProcessingResult(documents=documents, errors=errors)
    
    def _parse_by_sections(self, content: str, **kwargs) -> List[Document]:
        """Parse markdown by sections (headers)."""
        documents = []
        source = kwargs.get('source', 'unknown')
        
        # Split by headers (# ## ### etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        current_section = []
        current_header = ""
        current_level = 0
        section_count = 0
        
        for line in lines:
            header_match = re.match(header_pattern, line, re.MULTILINE)
            
            if header_match:
                # Save previous section if it exists
                if current_section:
                    section_content = '\n'.join(current_section).strip()
                    if section_content:
                        doc = self._create_document_from_section(
                            section_content, current_header, section_count, source
                        )
                        documents.append(doc)
                        section_count += 1
                
                # Start new section
                current_level = len(header_match.group(1))
                current_header = header_match.group(2).strip()
                current_section = [line]  # Include the header line
            else:
                current_section.append(line)
        
        # Handle last section
        if current_section:
            section_content = '\n'.join(current_section).strip()
            if section_content:
                doc = self._create_document_from_section(
                    section_content, current_header, section_count, source
                )
                documents.append(doc)
        
        # If no sections found, create single document
        if not documents:
            documents = [self._create_single_document(content, **kwargs)]
        
        return documents
    
    def _create_document_from_section(
        self, content: str, header: str, section_num: int, source: str
    ) -> Document:
        """Create document from a markdown section."""
        metadata = {
            "type": "markdown_section",
            "source": source,
            "section": section_num,
            "header": header,
            "content_type": "text/markdown"
        }
        
        # Extract additional metadata
        if self.extract_links:
            links = self._extract_links(content)
            if links:
                metadata["links"] = links
        
        if self.extract_code_blocks:
            code_blocks = self._extract_code_blocks(content)
            if code_blocks:
                metadata["code_blocks"] = len(code_blocks)
                metadata["code_languages"] = list(set(cb.get("language", "unknown") for cb in code_blocks))
        
        # Clean content for better processing
        clean_content = self._clean_markdown_content(content)
        
        doc_id = f"md_section_{section_num}_{hash(content) % 10000}"
        
        return Document(
            id=doc_id,
            content=clean_content,
            metadata=metadata,
            source=source
        )
    
    def _create_single_document(self, content: str, **kwargs) -> Document:
        """Create a single document from all content."""
        source = kwargs.get('source', 'unknown')
        
        metadata = {
            "type": "markdown_document",
            "source": source,
            "content_type": "text/markdown"
        }
        
        # Extract metadata
        if self.extract_headers:
            headers = self._extract_headers(content)
            if headers:
                metadata["headers"] = headers
                metadata["header_count"] = len(headers)
        
        if self.extract_links:
            links = self._extract_links(content)
            if links:
                metadata["links"] = links
                metadata["link_count"] = len(links)
        
        if self.extract_code_blocks:
            code_blocks = self._extract_code_blocks(content)
            if code_blocks:
                metadata["code_blocks"] = len(code_blocks)
                metadata["code_languages"] = list(set(cb.get("language", "unknown") for cb in code_blocks))
        
        # Clean content
        clean_content = self._clean_markdown_content(content)
        
        doc_id = f"md_doc_{hash(content) % 10000}"
        
        return Document(
            id=doc_id,
            content=clean_content,
            metadata=metadata,
            source=source
        )
    
    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract headers from markdown content."""
        headers = []
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        for match in re.finditer(header_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headers.append({
                "level": level,
                "text": text
            })
        
        return headers
    
    def _extract_links(self, content: str) -> List[Dict[str, str]]:
        """Extract links from markdown content."""
        links = []
        
        # Markdown links: [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            links.append({
                "text": match.group(1),
                "url": match.group(2)
            })
        
        # Reference links: [text][ref] and [ref]: url
        ref_pattern = r'^\[([^\]]+)\]:\s*(.+)$'
        ref_links = {}
        for match in re.finditer(ref_pattern, content, re.MULTILINE):
            ref_links[match.group(1)] = match.group(2)
        
        ref_usage_pattern = r'\[([^\]]+)\]\[([^\]]+)\]'
        for match in re.finditer(ref_usage_pattern, content):
            text = match.group(1)
            ref = match.group(2)
            if ref in ref_links:
                links.append({
                    "text": text,
                    "url": ref_links[ref]
                })
        
        return links
    
    def _extract_code_blocks(self, content: str) -> List[Dict[str, str]]:
        """Extract code blocks from markdown content."""
        code_blocks = []
        
        # Fenced code blocks with language
        fenced_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(fenced_pattern, content, re.DOTALL):
            language = match.group(1) or "unknown"
            code = match.group(2)
            code_blocks.append({
                "language": language,
                "code": code
            })
        
        # Indented code blocks
        indented_pattern = r'^    (.+)$'
        indented_code = []
        for match in re.finditer(indented_pattern, content, re.MULTILINE):
            indented_code.append(match.group(1))
        
        if indented_code:
            code_blocks.append({
                "language": "unknown",
                "code": "\n".join(indented_code)
            })
        
        return code_blocks
    
    def _clean_markdown_content(self, content: str) -> str:
        """Clean markdown content for better text processing."""
        # Remove markdown syntax but preserve content
        clean = content
        
        # Remove headers but keep text
        clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)
        
        # Remove bold/italic markers
        clean = re.sub(r'\*\*([^*]+)\*\*', r'\1', clean)  # **bold**
        clean = re.sub(r'\*([^*]+)\*', r'\1', clean)      # *italic*
        clean = re.sub(r'__([^_]+)__', r'\1', clean)      # __bold__
        clean = re.sub(r'_([^_]+)_', r'\1', clean)        # _italic_
        
        # Remove link syntax but keep text
        clean = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', clean)  # [text](url)
        
        # Remove inline code markers
        clean = re.sub(r'`([^`]+)`', r'\1', clean)
        
        # Remove list markers
        clean = re.sub(r'^[\s]*[-*+]\s+', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'^\s*\d+\.\s+', '', clean, flags=re.MULTILINE)
        
        # Remove blockquote markers
        clean = re.sub(r'^>\s*', '', clean, flags=re.MULTILINE)
        
        # Remove horizontal rules
        clean = re.sub(r'^---+$', '', clean, flags=re.MULTILINE)
        clean = re.sub(r'^\*\*\*+$', '', clean, flags=re.MULTILINE)
        
        # Remove code blocks
        clean = re.sub(r'```.*?```', '', clean, flags=re.DOTALL)
        
        # Clean up whitespace
        clean = re.sub(r'\n\s*\n', '\n\n', clean)
        clean = clean.strip()
        
        return clean
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get supported file extensions."""
        return ['.md', '.markdown', '.mdown', '.mkd']
    
    @classmethod
    def get_description(cls) -> str:
        """Get parser description."""
        return "Markdown document parser that extracts text content, headers, links, and code blocks."