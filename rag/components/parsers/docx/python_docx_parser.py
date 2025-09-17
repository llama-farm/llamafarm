"""DOCX parser using python-docx library."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class DocxParser_PythonDocx:
    """DOCX parser using python-docx library."""
    
    def __init__(self, name: str = "DocxParser_PythonDocx", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_strategy = self.config.get("chunk_strategy", "paragraphs")
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_tables = self.config.get("extract_tables", True)
        self.extract_headers = self.config.get("extract_headers", True)
        self.extract_footers = self.config.get("extract_footers", False)
        self.extract_comments = self.config.get("extract_comments", False)
        
    def validate_config(self) -> bool:
        """Validate configuration."""
        return True
    
    def parse(self, source: str, **kwargs):
        """Parse DOCX using python-docx."""
        from rag.core.base import Document, ProcessingResult
        
        try:
            import docx
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[{"error": "python-docx not installed. Install with: pip install python-docx", "source": source}]
            )
        
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}]
            )
        
        try:
            # Load document
            doc = docx.Document(source)
            
            # Extract text content
            content_parts = []
            
            # Extract paragraphs and tables in order
            for element in self._iter_block_items(doc):
                if isinstance(element, docx.text.paragraph.Paragraph):
                    text = element.text.strip()
                    if text:
                        # Check if it's a heading
                        if element.style and element.style.name and 'Heading' in element.style.name:
                            content_parts.append(f"\n## {text}\n")
                        else:
                            content_parts.append(text)
                            
                elif isinstance(element, docx.table.Table) and self.extract_tables:
                    table_text = self._extract_table(element)
                    if table_text:
                        content_parts.append(f"\n{table_text}\n")
            
            # Extract headers if configured
            if self.extract_headers:
                for section in doc.sections:
                    header = section.header
                    if header:
                        header_text = "\n".join(p.text for p in header.paragraphs if p.text.strip())
                        if header_text:
                            content_parts.insert(0, f"Header: {header_text}")
            
            # Extract footers if configured
            if self.extract_footers:
                for section in doc.sections:
                    footer = section.footer
                    if footer:
                        footer_text = "\n".join(p.text for p in footer.paragraphs if p.text.strip())
                        if footer_text:
                            content_parts.append(f"Footer: {footer_text}")
            
            # Join all content
            full_text = "\n\n".join(content_parts)
            
            if not full_text.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No text extracted from document", "source": source}]
                )
            
            # Extract metadata
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": self.name,
                "tool": "python-docx",
                "paragraphs": len(doc.paragraphs),
                "tables": len(doc.tables) if self.extract_tables else 0
            }
            
            if self.extract_metadata:
                # Extract document properties
                props = doc.core_properties
                if props:
                    metadata.update({
                        "title": props.title,
                        "author": props.author,
                        "subject": props.subject,
                        "keywords": props.keywords,
                        "created": str(props.created) if props.created else None,
                        "modified": str(props.modified) if props.modified else None,
                        "revision": props.revision
                    })
                    # Remove None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}
            
            documents = []
            
            # Apply chunking if needed
            if self.chunk_size and self.chunk_size > 0:
                chunks = self._chunk_text(full_text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "chunk_strategy": self.chunk_strategy
                    })
                    
                    doc = Document(
                        content=chunk,
                        metadata=chunk_metadata,
                        id=f"{path.stem}_chunk_{i+1}",
                        source=str(path)
                    )
                    documents.append(doc)
            else:
                doc = Document(
                    content=full_text,
                    metadata=metadata,
                    id=path.stem,
                    source=str(path)
                )
                documents.append(doc)
            
            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "parser_type": self.name,
                    "tool": "python-docx"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[],
                errors=[{"error": str(e), "source": source}]
            )
    
    def _iter_block_items(self, parent):
        """Iterate through document elements in order."""
        import docx
        
        if isinstance(parent, docx.document.Document):
            parent_elm = parent.element.body
        else:
            raise ValueError("Unsupported parent type")
        
        for child in parent_elm.iterchildren():
            if isinstance(child, docx.oxml.text.paragraph.CT_P):
                yield docx.text.paragraph.Paragraph(child, parent)
            elif isinstance(child, docx.oxml.table.CT_Tbl):
                yield docx.table.Table(child, parent)
    
    def _extract_table(self, table) -> str:
        """Extract table as formatted text."""
        rows = []
        for row in table.rows:
            row_text = []
            for cell in row.cells:
                cell_text = " ".join(p.text for p in cell.paragraphs).strip()
                row_text.append(cell_text)
            rows.append(" | ".join(row_text))
        
        if rows:
            # Add separator after header row
            if len(rows) > 1:
                rows.insert(1, "-" * len(rows[0]))
        
        return "\n".join(rows)
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text based on strategy."""
        if self.chunk_strategy == "paragraphs":
            paragraphs = text.split("\n\n")
            chunks = []
            current_chunk = ""
            
            for para in paragraphs:
                if len(current_chunk) + len(para) + 2 <= self.chunk_size:
                    current_chunk += para + "\n\n"
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = para + "\n\n"
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        else:
            # Character-based chunking
            chunks = []
            for i in range(0, len(text), self.chunk_size):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks