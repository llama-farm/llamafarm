"""PDF parser using PyPDF2 library."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class PDFParser_PyPDF2:
    """PDF parser using PyPDF2 for text extraction."""
    
    def __init__(self, name: str = "PDFParser_PyPDF2", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "sentences")
        self.extract_metadata = self.config.get("extract_metadata", True)
        
    def validate_config(self) -> bool:
        """Validate configuration."""
        return True
    
    def parse(self, source: str, **kwargs):
        """Parse PDF using PyPDF2."""
        from core.base import Document, ProcessingResult
        
        try:
            import PyPDF2
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[{"error": "PyPDF2 not installed. Install with: pip install PyPDF2", "source": source}]
            )
        
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}]
            )
        
        try:
            text = ""
            metadata = {}
            
            with open(source, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                
                # Extract metadata if available
                if self.extract_metadata and pdf_reader.metadata:
                    metadata = {
                        "title": getattr(pdf_reader.metadata, 'title', None),
                        "author": getattr(pdf_reader.metadata, 'author', None),
                        "subject": getattr(pdf_reader.metadata, 'subject', None),
                        "creator": getattr(pdf_reader.metadata, 'creator', None),
                        "pages": len(pdf_reader.pages)
                    }
                    # Remove None values
                    metadata = {k: v for k, v in metadata.items() if v is not None}
                
                # Extract text from all pages
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {page_num + 1} ---\n{page_text}"
            
            if not text.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No text extracted from PDF", "source": source}]
                )
            
            # Add parser info to metadata
            metadata.update({
                "source": str(path),
                "file_name": path.name,
                "parser": self.name,
                "tool": "PyPDF2"
            })
            
            documents = []
            
            # Apply chunking if configured
            if self.chunk_size and self.chunk_size > 0:
                chunks = self._chunk_text(text)
                for i, chunk in enumerate(chunks):
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks)
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
                    content=text,
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
                    "tool": "PyPDF2"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[],
                errors=[{"error": str(e), "source": source}]
            )
    
    def _chunk_text(self, text: str) -> List[str]:
        """Simple text chunking."""
        if self.chunk_strategy == "sentences":
            # Simple sentence splitting
            sentences = text.replace('\n\n', '\n').split('. ')
            chunks = []
            current_chunk = ""
            
            for sentence in sentences:
                if not sentence.strip():
                    continue
                sentence = sentence.strip() + '. '
                
                if len(current_chunk) + len(sentence) <= self.chunk_size:
                    current_chunk += sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
            
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            return chunks
        else:
            # Simple character-based chunking
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size]
                if chunk.strip():
                    chunks.append(chunk)
            return chunks