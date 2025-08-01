"""Markdown parser for parsing markdown documents with structure preservation."""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from core.base import Parser, Document, ProcessingResult


class MarkdownParser(Parser):
    """Parser for Markdown files with structure preservation and metadata extraction."""

    def __init__(self, name: str = "MarkdownParser", config: Dict[str, Any] = None):
        super().__init__(name, config)
        config = config or {}
        
        # Configuration options
        self.preserve_structure = config.get("preserve_structure", True)
        self.extract_frontmatter = config.get("extract_frontmatter", True)
        self.extract_headers = config.get("extract_headers", True)
        self.extract_links = config.get("extract_links", True)
        self.extract_code_blocks = config.get("extract_code_blocks", True)
        self.chunk_by_sections = config.get("chunk_by_sections", False)
        self.min_section_length = config.get("min_section_length", 100)
        self.include_code_in_content = config.get("include_code_in_content", True)

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.min_section_length < 0:
            raise ValueError("min_section_length must be non-negative")
        return True

    def parse(self, source: str) -> ProcessingResult:
        """Parse Markdown file into documents."""
        documents = []
        errors = []

        try:
            with open(source, "r", encoding="utf-8") as file:
                content = file.read()

            if self.chunk_by_sections:
                documents = self._parse_by_sections(content, source)
            else:
                doc = self._parse_whole_document(content, source)
                documents = [doc]

        except Exception as e:
            errors.append({
                "error": f"Failed to parse Markdown file: {str(e)}", 
                "source": source
            })

        return ProcessingResult(
            documents=documents,
            errors=errors,
            metrics={
                "total_documents": len(documents),
                "parse_errors": len(errors),
                "chunked_by_sections": self.chunk_by_sections,
            },
        )

    def _parse_whole_document(self, content: str, source: str) -> Document:
        """Parse entire document as single Document."""
        # Extract frontmatter
        frontmatter, main_content = self._extract_frontmatter(content)
        
        # Extract metadata
        metadata = self._extract_metadata(main_content, source)
        if frontmatter:
            metadata.update(frontmatter)

        # Process content
        processed_content = self._process_content(main_content)

        doc_id = Path(source).stem
        return Document(
            content=processed_content,
            metadata=metadata,
            id=doc_id,
            source=source
        )

    def _parse_by_sections(self, content: str, source: str) -> List[Document]:
        """Parse document into sections as separate Documents."""
        documents = []
        
        # Extract frontmatter
        frontmatter, main_content = self._extract_frontmatter(content)
        
        # Split by headers
        sections = self._split_by_headers(main_content)
        
        for i, (header, section_content) in enumerate(sections):
            if len(section_content.strip()) < self.min_section_length:
                continue
                
            # Extract section-specific metadata
            metadata = self._extract_metadata(section_content, source)
            metadata.update({
                "source_file": Path(source).name,
                "section_number": i + 1,
                "section_header": header,
                "is_section": True
            })
            
            # Add frontmatter to all sections
            if frontmatter:
                metadata.update(frontmatter)

            # Process content
            processed_content = self._process_content(section_content)
            if header and self.preserve_structure:
                processed_content = f"# {header}\n\n{processed_content}"

            doc_id = f"{Path(source).stem}_section_{i+1}"
            documents.append(Document(
                content=processed_content,
                metadata=metadata,
                id=doc_id,
                source=source
            ))

        return documents

    def _extract_frontmatter(self, content: str) -> tuple[Dict[str, Any], str]:
        """Extract YAML frontmatter from markdown."""
        frontmatter = {}
        main_content = content
        
        if not self.extract_frontmatter:
            return frontmatter, main_content
            
        # Check for YAML frontmatter
        frontmatter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(frontmatter_pattern, content, re.DOTALL)
        
        if match:
            yaml_content = match.group(1)
            main_content = content[match.end():]
            
            # Parse YAML (basic parsing without external dependencies)
            try:
                frontmatter = self._parse_simple_yaml(yaml_content)
            except Exception as e:
                # If YAML parsing fails, continue without frontmatter
                pass
                
        return frontmatter, main_content

    def _parse_simple_yaml(self, yaml_content: str) -> Dict[str, Any]:
        """Simple YAML parser for basic key-value pairs."""
        result = {}
        lines = yaml_content.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line and not line.startswith('#'):
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove quotes if present
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                
                # Try to convert to appropriate type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif re.match(r'^\d+\.\d+$', value):
                    value = float(value)
                
                result[key] = value
                
        return result

    def _split_by_headers(self, content: str) -> List[tuple[str, str]]:
        """Split content by headers."""
        # Find all headers (# ## ### etc.)
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = content.split('\n')
        
        sections = []
        current_header = None
        current_content = []
        
        for line in lines:
            header_match = re.match(header_pattern, line)
            if header_match:
                # Save previous section
                if current_header is not None or current_content:
                    section_text = '\n'.join(current_content).strip()
                    if section_text:
                        sections.append((current_header, section_text))
                
                # Start new section
                current_header = header_match.group(2).strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_header is not None or current_content:
            section_text = '\n'.join(current_content).strip()
            if section_text:
                sections.append((current_header, section_text))
        
        return sections

    def _extract_metadata(self, content: str, source: str) -> Dict[str, Any]:
        """Extract metadata from markdown content."""
        metadata = {
            "source_file": Path(source).name,
            "file_type": "markdown",
            "parser": self.name
        }

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
                metadata["code_blocks"] = code_blocks
                metadata["code_block_count"] = len(code_blocks)

        # Basic statistics
        metadata.update({
            "word_count": len(content.split()),
            "character_count": len(content),
            "line_count": len(content.split('\n'))
        })

        return metadata

    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract all headers from content."""
        headers = []
        header_pattern = r'^(#{1,6})\s+(.+)$'
        
        for line_num, line in enumerate(content.split('\n'), 1):
            match = re.match(header_pattern, line)
            if match:
                level = len(match.group(1))
                text = match.group(2).strip()
                headers.append({
                    "level": level,
                    "text": text,
                    "line": line_num
                })
        
        return headers

    def _extract_links(self, content: str) -> List[Dict[str, Any]]:
        """Extract all links from content."""
        links = []
        
        # Markdown links: [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            links.append({
                "text": match.group(1),
                "url": match.group(2),
                "type": "markdown"
            })
        
        # Plain URLs
        url_pattern = r'https?://[^\s\)]+(?=[\s\)]|$)'
        for match in re.finditer(url_pattern, content):
            url = match.group(0)
            # Avoid duplicates from markdown links
            if not any(link["url"] == url for link in links):
                links.append({
                    "text": url,
                    "url": url,
                    "type": "plain"
                })
        
        return links

    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from content."""
        code_blocks = []
        
        # Fenced code blocks (```language)
        fenced_pattern = r'```(\w+)?\n(.*?)\n```'
        for match in re.finditer(fenced_pattern, content, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2)
            code_blocks.append({
                "language": language,
                "code": code,
                "type": "fenced",
                "line_count": len(code.split('\n'))
            })
        
        # Inline code (`code`)
        inline_pattern = r'`([^`]+)`'
        inline_codes = re.findall(inline_pattern, content)
        for code in inline_codes:
            code_blocks.append({
                "language": "text",
                "code": code,
                "type": "inline",
                "line_count": 1
            })
        
        return code_blocks

    def _process_content(self, content: str) -> str:
        """Process and clean markdown content."""
        processed = content
        
        if not self.include_code_in_content:
            # Remove code blocks
            processed = re.sub(r'```.*?\n(.*?)\n```', '', processed, flags=re.DOTALL)
            processed = re.sub(r'`([^`]+)`', '', processed)
        
        if not self.preserve_structure:
            # Remove markdown formatting for plain text
            processed = re.sub(r'^#{1,6}\s+', '', processed, flags=re.MULTILINE)  # Headers
            processed = re.sub(r'\*\*(.*?)\*\*', r'\1', processed)  # Bold
            processed = re.sub(r'\*(.*?)\*', r'\1', processed)  # Italic
            processed = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', processed)  # Links
        
        # Clean up extra whitespace
        processed = re.sub(r'\n\s*\n', '\n\n', processed)
        processed = processed.strip()
        
        return processed