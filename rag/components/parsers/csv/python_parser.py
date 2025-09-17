"""CSV parser using native Python csv module."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import csv

logger = logging.getLogger(__name__)

class CSVParser_Python:
    """CSV parser using native Python csv module (no external dependencies)."""
    
    def __init__(self, name: str = "CSVParser_Python", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.encoding = self.config.get("encoding", "utf-8")
        self.delimiter = self.config.get("delimiter", ",")
        self.quotechar = self.config.get("quotechar", '"')
        
    def validate_config(self) -> bool:
        """Validate configuration."""
        return True
    
    def parse(self, source: str, **kwargs):
        """Parse CSV using Python's csv module."""
        from rag.core.base import Document, ProcessingResult
        
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}]
            )
        
        try:
            rows = []
            headers = None
            
            with open(source, 'r', encoding=self.encoding) as f:
                reader = csv.reader(f, delimiter=self.delimiter, quotechar=self.quotechar)
                
                for i, row in enumerate(reader):
                    if i == 0:
                        headers = row
                    rows.append(row)
            
            if not rows:
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "Empty CSV file", "source": source}]
                )
            
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": self.name,
                "tool": "Python csv",
                "rows": len(rows) - 1 if headers else len(rows),
                "columns": len(headers) if headers else len(rows[0]) if rows else 0,
                "column_names": headers if headers else []
            }
            
            documents = []
            
            # Create text representation
            if self.chunk_size and self.chunk_size > 0:
                # Chunk by rows
                for chunk_idx in range(0, len(rows), self.chunk_size):
                    chunk_rows = rows[chunk_idx:chunk_idx + self.chunk_size]
                    
                    # Format as text table
                    chunk_text = self._format_rows(chunk_rows, headers)
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": chunk_idx // self.chunk_size,
                        "chunk_rows": len(chunk_rows),
                        "start_row": chunk_idx,
                        "end_row": min(chunk_idx + self.chunk_size, len(rows))
                    })
                    
                    doc = Document(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        id=f"{path.stem}_chunk_{chunk_idx//self.chunk_size+1}",
                        source=str(path)
                    )
                    documents.append(doc)
            else:
                # Single document
                text = self._format_rows(rows, headers)
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
                    "tool": "Python csv"
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[],
                errors=[{"error": str(e), "source": source}]
            )
    
    def _format_rows(self, rows: List[List[str]], headers: Optional[List[str]] = None) -> str:
        """Format rows as a text table."""
        if not rows:
            return ""
        
        # Calculate column widths
        col_widths = []
        num_cols = len(rows[0])
        
        for col_idx in range(num_cols):
            max_width = len(headers[col_idx]) if headers and col_idx < len(headers) else 0
            for row in rows:
                if col_idx < len(row):
                    max_width = max(max_width, len(str(row[col_idx])))
            col_widths.append(min(max_width, 50))  # Cap at 50 chars
        
        # Format table
        lines = []
        
        # Add headers if available
        if headers:
            header_line = " | ".join(
                str(headers[i]).ljust(col_widths[i])[:col_widths[i]]
                for i in range(min(len(headers), num_cols))
            )
            lines.append(header_line)
            lines.append("-" * len(header_line))
        
        # Add data rows
        for row in rows:
            if headers and row == rows[0]:  # Skip header row if we already added it
                continue
            row_line = " | ".join(
                str(row[i] if i < len(row) else "").ljust(col_widths[i])[:col_widths[i]]
                for i in range(num_cols)
            )
            lines.append(row_line)
        
        return "\n".join(lines)