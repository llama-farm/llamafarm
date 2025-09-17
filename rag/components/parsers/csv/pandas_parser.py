"""CSV parser using Pandas library."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class CSVParser_Pandas:
    """CSV parser using Pandas for data extraction."""
    
    def __init__(self, name: str = "CSVParser_Pandas", config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_strategy = self.config.get("chunk_strategy", "rows")
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.encoding = self.config.get("encoding", "utf-8")
        self.delimiter = self.config.get("delimiter", ",")
        self.na_values = self.config.get("na_values", ["", "NA", "N/A", "null", "None"])
        
    def validate_config(self) -> bool:
        """Validate configuration."""
        return True
    
    def parse(self, source: str, **kwargs):
        """Parse CSV using Pandas."""
        from rag.core.base import Document, ProcessingResult
        
        try:
            import pandas as pd
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[{"error": "Pandas not installed. Install with: pip install pandas", "source": source}]
            )
        
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}]
            )
        
        try:
            # Read CSV with error handling
            df = pd.read_csv(
                source,
                encoding=self.encoding,
                delimiter=self.delimiter,
                na_values=self.na_values,
                on_bad_lines='warn'
            )
            
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": self.name,
                "tool": "Pandas",
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": df.columns.tolist()
            }
            
            if self.extract_metadata:
                # Add data statistics
                metadata.update({
                    "memory_usage": df.memory_usage(deep=True).sum(),
                    "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                    "null_counts": df.isnull().sum().to_dict()
                })
            
            documents = []
            
            if self.chunk_strategy == "rows":
                # Chunk by rows
                for i in range(0, len(df), self.chunk_size):
                    chunk_df = df.iloc[i:i+self.chunk_size]
                    chunk_text = chunk_df.to_string()
                    
                    chunk_metadata = metadata.copy()
                    chunk_metadata.update({
                        "chunk_index": i // self.chunk_size,
                        "chunk_rows": len(chunk_df),
                        "start_row": i,
                        "end_row": min(i + self.chunk_size, len(df))
                    })
                    
                    doc = Document(
                        content=chunk_text,
                        metadata=chunk_metadata,
                        id=f"{path.stem}_chunk_{i//self.chunk_size+1}",
                        source=str(path)
                    )
                    documents.append(doc)
                    
            elif self.chunk_strategy == "columns":
                # Create documents per column
                for col in df.columns:
                    col_data = df[col].to_string()
                    
                    col_metadata = metadata.copy()
                    col_metadata.update({
                        "column": col,
                        "dtype": str(df[col].dtype),
                        "unique_values": df[col].nunique(),
                        "null_count": df[col].isnull().sum()
                    })
                    
                    doc = Document(
                        content=f"Column: {col}\n{col_data}",
                        metadata=col_metadata,
                        id=f"{path.stem}_col_{col}",
                        source=str(path)
                    )
                    documents.append(doc)
                    
            else:  # full
                # Single document for entire CSV
                doc = Document(
                    content=df.to_string(),
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
                    "tool": "Pandas",
                    "rows_processed": len(df)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[],
                errors=[{"error": str(e), "source": source}]
            )