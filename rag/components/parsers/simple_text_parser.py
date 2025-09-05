"""Simple text parser that doesn't require LlamaIndex."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class SimpleTextParser:
    """Simple text parser for demos without LlamaIndex dependency."""
    
    def __init__(self, name: str = "text", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "sentences")
    
    def parse(self, source: str, **kwargs):
        """Parse a text file."""
        from core.base import Document, ProcessingResult
        
        path = Path(source)
        
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}]
            )
        
        try:
            # Read file
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()
            
            documents = []
            
            # Simple chunking if configured
            if self.chunk_size and self.chunk_size > 0:
                chunks = self._chunk_text(content)
                for i, chunk in enumerate(chunks):
                    doc = Document(
                        content=chunk,
                        metadata={
                            "source": str(path),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            "file_name": path.name,
                            "parser": self.name
                        },
                        id=f"{path.stem}_chunk_{i+1}",
                        source=str(path)
                    )
                    documents.append(doc)
            else:
                # No chunking
                doc = Document(
                    content=content,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "parser": self.name
                    },
                    id=path.stem,
                    source=str(path)
                )
                documents.append(doc)
            
            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "parser_type": self.name
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

# Fallback PDF parser
class SimplePDFParser(SimpleTextParser):
    """Simple PDF parser fallback."""
    
    def __init__(self, name: str = "pdf", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
    
    def parse(self, source: str, **kwargs):
        """Try to parse PDF, fallback to text extraction if possible."""
        try:
            # Try using pypdf2 if available
            import PyPDF2
            
            from core.base import Document, ProcessingResult
            
            text = ""
            with open(source, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            
            # Create a temporary text version and use parent parser
            if text.strip():
                # Save text to process
                self._temp_text = text
                # Mock a text file parse
                path = Path(source)
                
                documents = []
                if self.chunk_size and self.chunk_size > 0:
                    chunks = self._chunk_text(text)
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            content=chunk,
                            metadata={
                                "source": str(path),
                                "chunk_index": i,
                                "total_chunks": len(chunks),
                                "file_name": path.name,
                                "parser": self.name
                            },
                            id=f"{path.stem}_chunk_{i+1}",
                            source=str(path)
                        )
                        documents.append(doc)
                else:
                    doc = Document(
                        content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "parser": self.name
                        },
                        id=path.stem,
                        source=str(path)
                    )
                    documents.append(doc)
                
                return ProcessingResult(
                    documents=documents,
                    errors=[],
                    metrics={"total_documents": len(documents), "parser_type": self.name}
                )
                
        except Exception as e:
            logger.warning(f"PDF parsing failed, trying as text: {e}")
        
        # Fallback to parent text parser
        return super().parse(source, **kwargs)