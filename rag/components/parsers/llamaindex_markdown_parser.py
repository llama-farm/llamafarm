"""LlamaIndex-based Markdown Parser."""

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_parser import BaseLlamaIndexParser, ParserFactory

logger = logging.getLogger(__name__)

try:
    from llama_index.readers.file import MarkdownReader
    LLAMA_MARKDOWN_AVAILABLE = True
except ImportError:
    LLAMA_MARKDOWN_AVAILABLE = False

try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False


class LlamaIndexMarkdownParser(BaseLlamaIndexParser):
    """LlamaIndex-based parser for Markdown files."""
    
    def __init__(self, name: str = "LlamaIndexMarkdownParser", config: Optional[Dict[str, Any]] = None):
        """
        Initialize LlamaIndex Markdown parser.
        
        Args:
            name: Parser name
            config: Parser configuration
        """
        super().__init__(name=name, config=config or {})
        
        # Markdown-specific configuration
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_headings = self.config.get("extract_headings", True)
        self.extract_links = self.config.get("extract_links", True)
        self.extract_code_blocks = self.config.get("extract_code_blocks", True)
        self.chunk_by_headings = self.config.get("chunk_by_headings", False)
        self.preserve_formatting = self.config.get("preserve_formatting", False)
        self.heading_level_split = self.config.get("heading_level_split", 2)
        
        # Override chunk strategy if chunk_by_headings is enabled and chunk_size is not set
        if self.chunk_by_headings and not self.chunk_size:
            self.chunk_strategy = "headings"
    
    def _get_reader(self):
        """Get the appropriate LlamaIndex reader for Markdown files."""
        if LLAMA_MARKDOWN_AVAILABLE:
            return MarkdownReader()
        else:
            # Will use manual parsing
            return None
    
    def parse(self, file_path: str, **kwargs) -> "ProcessingResult":
        """
        Parse a Markdown file using LlamaIndex.
        
        Args:
            file_path: Path to the Markdown file
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
            if self._reader:
                return self._parse_with_llamaindex(file_path, **kwargs)
            else:
                return self._parse_with_manual_parsing(file_path, **kwargs)
        
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to parse Markdown file: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _parse_with_llamaindex(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse Markdown using LlamaIndex MarkdownReader."""
        try:
            documents = self._reader.load_data(file=str(file_path))
            
            if not documents:
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No content extracted by LlamaIndex", "source": str(file_path)}]
                )
            
            return self._process_llamaindex_documents(documents, file_path)
            
        except Exception as e:
            logger.warning(f"LlamaIndex Markdown reader failed: {e}, falling back to manual parsing")
            return self._parse_with_manual_parsing(file_path, **kwargs)
    
    def _parse_with_manual_parsing(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse Markdown using manual parsing."""
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            if not content.strip():
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No content found in file", "source": str(file_path)}]
                )
            
            # Extract frontmatter and content
            frontmatter, main_content = self._extract_frontmatter(content)
            
            # Extract markdown structure
            structure = self._extract_markdown_structure(main_content)
            
            # Prepare metadata
            metadata = self._create_markdown_metadata(file_path, frontmatter, structure)
            
            # Handle chunking strategies
            if self.chunk_by_headings or self.chunk_strategy == "headings":
                return self._create_heading_based_chunks(main_content, metadata, file_path, structure)
            else:
                return self._create_documents_from_content(main_content, metadata, file_path)
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Manual Markdown parsing failed: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _process_llamaindex_documents(self, documents, file_path: Path) -> "ProcessingResult":
        """Process LlamaIndex documents into our format."""
        from core.base import ProcessingResult, Document
        from utils.hash_utils import generate_document_metadata, generate_chunk_metadata
        
        result_documents = []
        errors = []
        
        # Combine all content if multiple documents
        if len(documents) > 1:
            combined_content = "\n\n".join(doc.text for doc in documents if doc.text)
        else:
            combined_content = documents[0].text if documents else ""
        
        if not combined_content.strip():
            return ProcessingResult(
                documents=[],
                errors=[{"error": "No content extracted", "source": str(file_path)}]
            )
        
        # Extract markdown structure from combined content
        structure = self._extract_markdown_structure(combined_content)
        
        # Generate metadata
        base_metadata = generate_document_metadata(str(file_path), combined_content)
        base_metadata.update({
            "parser_type": self.name,
            "reader_type": "llamaindex_markdown"
        })
        
        # Add structure information
        base_metadata.update(structure)
        
        # Add LlamaIndex metadata from first document
        if documents[0].metadata:
            base_metadata.update(documents[0].metadata)
        
        # Handle chunking
        if self.chunk_by_headings or self.chunk_strategy == "headings":
            return self._create_heading_based_chunks(combined_content, base_metadata, file_path, structure)
        elif self._node_parser and self.chunk_size:
            chunked_docs = self._create_chunked_documents(
                combined_content, base_metadata, documents[0]
            )
            result_documents.extend(chunked_docs)
        else:
            document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
            doc = Document(
                content=combined_content,
                metadata=base_metadata,
                id=document_id,
                source=str(file_path)
            )
            result_documents.append(doc)
        
        return ProcessingResult(
            documents=result_documents,
            errors=errors,
            metrics={
                "total_documents": len(result_documents),
                "total_errors": len(errors),
                "file_processed": str(file_path),
                "parser_type": self.name
            }
        )
    
    def _extract_frontmatter(self, content: str) -> tuple:
        """Extract YAML frontmatter from markdown content."""
        frontmatter = {}
        main_content = content
        
        if content.startswith('---'):
            try:
                # Find the end of frontmatter
                end_match = re.search(r'\n---\s*\n', content)
                if end_match:
                    frontmatter_text = content[3:end_match.start()]
                    main_content = content[end_match.end():]
                    
                    # Parse YAML frontmatter
                    try:
                        import yaml
                        frontmatter = yaml.safe_load(frontmatter_text) or {}
                    except ImportError:
                        # Simple key-value parsing if PyYAML not available
                        for line in frontmatter_text.split('\n'):
                            if ':' in line:
                                key, value = line.split(':', 1)
                                frontmatter[key.strip()] = value.strip()
                    except Exception as e:
                        logger.warning(f"Failed to parse frontmatter: {e}")
            except Exception as e:
                logger.warning(f"Failed to extract frontmatter: {e}")
        
        return frontmatter, main_content
    
    def _extract_markdown_structure(self, content: str) -> Dict[str, Any]:
        """Extract structural information from markdown content."""
        structure = {
            "headings": [],
            "links": [],
            "code_blocks": [],
            "images": [],
            "tables": [],
            "has_toc": False,
            "heading_levels": set()
        }
        
        lines = content.split('\n')
        in_code_block = False
        code_block_lang = None
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Code blocks
            if stripped.startswith('```'):
                if not in_code_block:
                    # Starting a code block
                    in_code_block = True
                    code_block_lang = stripped[3:].strip() if len(stripped) > 3 else "text"
                else:
                    # Ending a code block
                    in_code_block = False
                    if self.extract_code_blocks:
                        structure["code_blocks"].append({
                            "language": code_block_lang,
                            "line": line_num - 1  # Line where block started
                        })
                    code_block_lang = None
                continue
            
            # Skip lines inside code blocks
            if in_code_block:
                continue
            
            # Headings
            if stripped.startswith('#'):
                heading_match = re.match(r'^(#{1,6})\s+(.+)', stripped)
                if heading_match:
                    level = len(heading_match.group(1))
                    text = heading_match.group(2).strip()
                    
                    if self.extract_headings:
                        structure["headings"].append({
                            "level": level,
                            "text": text,
                            "line": line_num,
                            "id": self._generate_heading_id(text)
                        })
                    
                    structure["heading_levels"].add(level)
            
            # Links
            if self.extract_links:
                # Find markdown links [text](url)
                link_matches = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', line)
                for text, url in link_matches:
                    structure["links"].append({
                        "text": text,
                        "url": url,
                        "line": line_num
                    })
                
                # Find reference links [text][ref]
                ref_link_matches = re.findall(r'\[([^\]]+)\]\[([^\]]+)\]', line)
                for text, ref in ref_link_matches:
                    structure["links"].append({
                        "text": text,
                        "reference": ref,
                        "line": line_num
                    })
            
            # Images
            image_matches = re.findall(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            for alt_text, src in image_matches:
                structure["images"].append({
                    "alt": alt_text,
                    "src": src,
                    "line": line_num
                })
            
            # Tables (simple detection)
            if '|' in stripped and stripped.count('|') >= 2:
                structure["tables"].append(line_num)
            
            # Table of Contents detection
            if re.search(r'\b(table of contents|toc)\b', stripped.lower()):
                structure["has_toc"] = True
        
        structure["heading_levels"] = list(structure["heading_levels"])
        return structure
    
    def _generate_heading_id(self, text: str) -> str:
        """Generate an ID for a heading."""
        # Convert to lowercase, replace spaces with hyphens, remove special chars
        heading_id = re.sub(r'[^\w\s-]', '', text.lower())
        heading_id = re.sub(r'[-\s]+', '-', heading_id).strip('-')
        return heading_id
    
    def _create_markdown_metadata(self, file_path: Path, frontmatter: Dict, structure: Dict) -> Dict[str, Any]:
        """Create metadata for markdown document."""
        from utils.hash_utils import generate_document_metadata
        
        # Read content for hash generation
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        
        metadata = generate_document_metadata(str(file_path), content)
        metadata.update({
            "parser_type": self.name,
            "file_type": "markdown",
            "has_frontmatter": bool(frontmatter)
        })
        
        # Add frontmatter
        metadata.update(frontmatter)
        
        # Add structure information
        metadata.update({
            f"md_{key}": value for key, value in structure.items() 
            if key not in ["headings"]  # Don't add the full headings list
        })
        
        # Add summary statistics
        metadata.update({
            "md_heading_count": len(structure["headings"]),
            "md_link_count": len(structure["links"]),
            "md_code_block_count": len(structure["code_blocks"]),
            "md_image_count": len(structure["images"]),
            "md_table_count": len(structure["tables"])
        })
        
        return metadata
    
    def _create_heading_based_chunks(self, content: str, metadata: Dict, file_path: Path, structure: Dict) -> "ProcessingResult":
        """Create document chunks based on heading structure."""
        from core.base import ProcessingResult, Document
        from utils.hash_utils import generate_chunk_metadata
        
        headings = structure["headings"]
        result_documents = []
        
        if not headings:
            # No headings found, create single document
            return self._create_documents_from_content(content, metadata, file_path)
        
        lines = content.split('\n')
        current_chunk = []
        current_heading = None
        chunk_index = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Check if this line is a heading at our split level
            if stripped.startswith('#'):
                heading_match = re.match(r'^(#{1,6})\s+(.+)', stripped)
                if heading_match:
                    level = len(heading_match.group(1))
                    
                    # If this is a heading at our split level or higher, start new chunk
                    if level <= self.heading_level_split:
                        # Save previous chunk if it has content
                        if current_chunk and current_heading:
                            chunk_content = '\n'.join(current_chunk).strip()
                            if chunk_content:
                                doc = self._create_heading_chunk_document(
                                    chunk_content, metadata, current_heading, chunk_index, file_path
                                )
                                result_documents.append(doc)
                                chunk_index += 1
                        
                        # Start new chunk
                        current_chunk = [line]
                        current_heading = {
                            "level": level,
                            "text": heading_match.group(2).strip(),
                            "line": i + 1
                        }
                    else:
                        # Sub-heading, add to current chunk
                        current_chunk.append(line)
                else:
                    current_chunk.append(line)
            else:
                current_chunk.append(line)
        
        # Don't forget the last chunk
        if current_chunk and current_heading:
            chunk_content = '\n'.join(current_chunk).strip()
            if chunk_content:
                doc = self._create_heading_chunk_document(
                    chunk_content, metadata, current_heading, chunk_index, file_path
                )
                result_documents.append(doc)
        
        if not result_documents:
            # Fallback to single document
            return self._create_documents_from_content(content, metadata, file_path)
        
        return ProcessingResult(
            documents=result_documents,
            errors=[],
            metrics={
                "total_documents": len(result_documents),
                "file_processed": str(file_path),
                "parser_type": self.name,
                "chunking_strategy": "headings",
                "heading_split_level": self.heading_level_split
            }
        )
    
    def _create_heading_chunk_document(self, content: str, base_metadata: Dict, heading: Dict, chunk_index: int, file_path: Path) -> "Document":
        """Create a document for a heading-based chunk."""
        from core.base import Document
        from utils.hash_utils import generate_chunk_metadata
        
        chunk_metadata = generate_chunk_metadata(
            base_metadata, content, chunk_index, None  # Total chunks not known yet
        )
        
        chunk_metadata.update({
            "chunk_heading": heading["text"],
            "chunk_heading_level": heading["level"],
            "chunk_heading_line": heading["line"],
            "chunk_strategy": "headings"
        })
        
        return Document(
            content=content,
            metadata=chunk_metadata,
            id=chunk_metadata["chunk_id"],
            source=str(file_path)
        )
    
    def _create_documents_from_content(self, content: str, metadata: Dict, file_path: Path) -> "ProcessingResult":
        """Create documents from content using standard chunking."""
        from core.base import ProcessingResult, Document
        
        result_documents = []
        
        # Handle chunking
        if self._node_parser and self.chunk_size:
            # Create mock LlamaIndex document for chunking
            from llama_index.core.schema import Document as LlamaDocument
            llama_doc = LlamaDocument(text=content, metadata=metadata)
            
            chunked_docs = self._create_chunked_documents(
                content, metadata, llama_doc
            )
            result_documents.extend(chunked_docs)
        else:
            document_id = f"doc_{metadata['document_hash'][:12]}_full"
            doc = Document(
                content=content,
                metadata=metadata,
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
                "parser_type": self.name
            }
        )
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    @staticmethod
    def can_parse_mime_type(mime_type: str) -> bool:
        """Check if this parser can handle the given MIME type."""
        return mime_type in ['text/markdown', 'text/x-markdown']
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions."""
        return ['.md', '.markdown']
    
    @staticmethod
    def get_description() -> str:
        """Get parser description."""
        return "LlamaIndex-based parser for Markdown files"


# Register the parser
ParserFactory.register_parser("LlamaIndexMarkdownParser", LlamaIndexMarkdownParser)