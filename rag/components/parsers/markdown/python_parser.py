"""Markdown parser using native Python."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import re
from components.parsers.base.base_parser import BaseParser, ParserConfig

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.parsers.markdownthon_parser")


class MarkdownParser_Python(BaseParser):
    """Markdown parser using native Python with basic parsing."""

    def __init__(
        self,
        name: str = "MarkdownParser_Python",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_strategy = self.config.get("chunk_strategy", "sections")
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_code_blocks = self.config.get("extract_code_blocks", True)
        self.extract_links = self.config.get("extract_links", True)

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""

        return ParserConfig(
            name="MarkdownParser_Python",
            display_name="Python Markdown Parser",
            version="1.0.0",
            supported_extensions=[".md", ".markdown", ".mdown", ".mkd"],
            mime_types=["text/markdown", "text/x-markdown"],
            capabilities=[
                "frontmatter_extraction",
                "header_extraction",
                "code_block_extraction",
                "link_extraction",
                "section_based_chunking",
            ],
            dependencies={},
            default_config={
                "chunk_size": 1000,
                "chunk_strategy": "sections",
                "extract_metadata": True,
                "extract_code_blocks": True,
                "extract_links": True,
            }
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file."""
        path = Path(file_path)
        return path.suffix.lower() in {".md", ".markdown", ".mdown", ".mkd"}

    def parse(self, source: str, **kwargs):
        """Parse Markdown using Python."""
        from core.base import Document, ProcessingResult

        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            with open(source, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "Empty Markdown file", "source": source}],
                )

            # Extract metadata
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": self.name,
                "tool": "Python",
                "file_size": path.stat().st_size,
                "line_count": content.count("\n") + 1,
            }

            if self.extract_metadata:
                # Extract frontmatter if present
                frontmatter = self._extract_frontmatter(content)
                if frontmatter:
                    metadata["frontmatter"] = frontmatter
                    # Remove frontmatter from content
                    content = re.sub(r"^---\n.*?\n---\n", "", content, flags=re.DOTALL)

                # Extract headers
                headers = self._extract_headers(content)
                metadata["headers"] = headers
                metadata["header_count"] = len(headers)

                # Extract code blocks
                if self.extract_code_blocks:
                    code_blocks = self._extract_code_blocks(content)
                    metadata["code_blocks"] = len(code_blocks)
                    metadata["code_languages"] = list(
                        set(cb.get("language", "plain") for cb in code_blocks)
                    )

                # Extract links
                if self.extract_links:
                    links = self._extract_links(content)
                    metadata["links"] = links
                    metadata["link_count"] = len(links)

            documents = []

            # Apply chunking based on strategy
            if self.chunk_strategy == "sections" and self.chunk_size > 0:
                sections = self._split_by_sections(content)
                for i, section in enumerate(sections):
                    if not section["content"].strip():
                        continue

                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {
                            "chunk_index": i,
                            "total_chunks": len(sections),
                            "section_title": section["title"],
                            "section_level": section["level"],
                        }
                    )

                    doc = Document(
                        content=section["content"],
                        metadata=chunk_metadata,
                        id=f"{path.stem}_section_{i + 1}",
                        source=str(path),
                    )
                    documents.append(doc)

            elif self.chunk_size and self.chunk_size > 0:
                # Character-based chunking
                chunks = self._chunk_by_size(content)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update(
                        {"chunk_index": i, "total_chunks": len(chunks)}
                    )

                    doc = Document(
                        content=chunk,
                        metadata=chunk_metadata,
                        id=f"{path.stem}_chunk_{i + 1}",
                        source=str(path),
                    )
                    documents.append(doc)
            else:
                # Single document
                doc = Document(
                    content=content, metadata=metadata, id=path.stem, source=str(path)
                )
                documents.append(doc)

            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "parser_type": self.name,
                    "tool": "Python",
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _extract_frontmatter(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract YAML frontmatter from Markdown."""
        match = re.match(r"^---\n(.*?)\n---\n", content, re.DOTALL)
        if match:
            frontmatter_text = match.group(1)
            # Simple key-value parsing (not full YAML)
            frontmatter = {}
            for line in frontmatter_text.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    frontmatter[key.strip()] = value.strip()
            return frontmatter
        return None

    def _extract_headers(self, content: str) -> List[Dict[str, Any]]:
        """Extract all headers from Markdown."""
        headers = []
        pattern = r"^(#{1,6})\s+(.+)$"
        for match in re.finditer(pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headers.append({"level": level, "title": title})
        return headers

    def _extract_code_blocks(self, content: str) -> List[Dict[str, Any]]:
        """Extract code blocks from Markdown."""
        code_blocks = []
        pattern = r"```(\w*)\n(.*?)\n```"
        for match in re.finditer(pattern, content, re.DOTALL):
            language = match.group(1) or "plain"
            code = match.group(2)
            code_blocks.append({"language": language, "code": code})
        return code_blocks

    def _extract_links(self, content: str) -> List[str]:
        """Extract all links from Markdown."""
        links = []
        # Markdown links [text](url)
        pattern = r"\[([^\]]+)\]\(([^)]+)\)"
        for match in re.finditer(pattern, content):
            links.append(match.group(2))
        # Plain URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        for match in re.finditer(url_pattern, content):
            links.append(match.group(0))
        return list(set(links))  # Remove duplicates

    def _split_by_sections(self, content: str) -> List[Dict[str, Any]]:
        """Split Markdown by headers/sections."""
        sections = []
        current_section = {"title": "Introduction", "level": 0, "content": ""}

        lines = content.split("\n")
        for line in lines:
            header_match = re.match(r"^(#{1,6})\s+(.+)$", line)
            if header_match:
                # Save current section if it has content
                if current_section["content"].strip():
                    sections.append(current_section)

                # Start new section
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                current_section = {
                    "title": title,
                    "level": level,
                    "content": f"{line}\n",
                }
            else:
                current_section["content"] += f"{line}\n"

        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)

        # Merge small sections if needed
        if self.chunk_size:
            merged_sections = []
            current_merged = None

            for section in sections:
                if current_merged is None:
                    current_merged = section
                elif (
                    len(current_merged["content"]) + len(section["content"])
                    <= self.chunk_size
                ):
                    current_merged["content"] += "\n" + section["content"]
                    current_merged["title"] += f" / {section['title']}"
                else:
                    merged_sections.append(current_merged)
                    current_merged = section

            if current_merged:
                merged_sections.append(current_merged)

            return merged_sections

        return sections

    def _chunk_by_size(self, content: str) -> List[str]:
        """Simple size-based chunking."""
        chunks = []
        for i in range(0, len(content), self.chunk_size):
            chunk = content[i : i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        return chunks
