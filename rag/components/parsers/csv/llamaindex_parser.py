"""CSV parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


class CSVParser_LlamaIndex:
    """CSV parser using LlamaIndex with Pandas backend."""

    def __init__(
        self,
        name: str = "CSVParser_LlamaIndex",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.config = config or {}

        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 1000)  # rows per chunk
        self.chunk_strategy = self.config.get("chunk_strategy", "rows")

        # CSV specific config
        self.content_fields = self.config.get("content_fields", None)
        self.metadata_fields = self.config.get("metadata_fields", [])
        self.id_field = self.config.get("id_field", None)
        self.combine_content = self.config.get("combine_content", True)
        self.content_separator = self.config.get("content_separator", "\n\n")

        # Pandas options
        self.encoding = self.config.get("encoding", "utf-8")
        self.delimiter = self.config.get("delimiter", ",")
        self.na_values = self.config.get("na_values", ["", "NA", "N/A", "null", "None"])

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse CSV using LlamaIndex."""
        from core.base import Document, ProcessingResult

        try:
            from llama_index.readers.file import PandasCSVReader
            from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
            import pandas as pd
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "LlamaIndex or pandas not installed. Install with: pip install llama-index pandas",
                        "source": source,
                    }
                ],
            )

        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            # Use LlamaIndex PandasCSVReader
            reader = PandasCSVReader(
                concat_rows=False,
                col_joiner=self.content_separator,
                row_joiner="\n",
                pandas_config={
                    "encoding": self.encoding,
                    "delimiter": self.delimiter,
                    "na_values": self.na_values,
                },
            )

            llama_docs = reader.load_data(file=path)
            documents = []

            # If we have specific content/metadata field mapping
            if self.content_fields:
                # Read with pandas for more control
                df = pd.read_csv(
                    source,
                    encoding=self.encoding,
                    delimiter=self.delimiter,
                    na_values=self.na_values,
                )

                # Process by chunks if needed
                if self.chunk_size and self.chunk_strategy == "rows":
                    for chunk_idx in range(0, len(df), self.chunk_size):
                        chunk_df = df.iloc[chunk_idx : chunk_idx + self.chunk_size]

                        # Combine content fields
                        if self.combine_content:
                            content_parts = []
                            for _, row in chunk_df.iterrows():
                                row_content = []
                                for field in self.content_fields:
                                    if field in row and pd.notna(row[field]):
                                        row_content.append(str(row[field]))
                                if row_content:
                                    content_parts.append(
                                        self.content_separator.join(row_content)
                                    )
                            content = "\n".join(content_parts)
                        else:
                            # Create separate documents per row
                            for row_idx, row in chunk_df.iterrows():
                                row_content = []
                                for field in self.content_fields:
                                    if field in row and pd.notna(row[field]):
                                        row_content.append(str(row[field]))

                                metadata = {
                                    "source": str(path),
                                    "file_name": path.name,
                                    "parser": "CSVParser_LlamaIndex",
                                    "tool": "LlamaIndex-Pandas",
                                    "row_index": row_idx,
                                    "chunk_index": chunk_idx // self.chunk_size,
                                }

                                # Add metadata fields
                                for field in self.metadata_fields:
                                    if field in row:
                                        metadata[field] = row[field]

                                # Use ID field if specified
                                doc_id = (
                                    str(row[self.id_field])
                                    if self.id_field and self.id_field in row
                                    else f"{path.stem}_row_{row_idx}"
                                )

                                doc = Document(
                                    content=self.content_separator.join(row_content),
                                    metadata=metadata,
                                    id=doc_id,
                                    source=str(path),
                                )
                                documents.append(doc)
                            continue

                        metadata = {
                            "source": str(path),
                            "file_name": path.name,
                            "parser": "CSVParser_LlamaIndex",
                            "tool": "LlamaIndex-Pandas",
                            "chunk_index": chunk_idx // self.chunk_size,
                            "chunk_rows": len(chunk_df),
                            "start_row": chunk_idx,
                            "end_row": min(chunk_idx + self.chunk_size, len(df)),
                            "total_rows": len(df),
                            "columns": list(df.columns),
                        }

                        doc = Document(
                            content=content,
                            metadata=metadata,
                            id=f"{path.stem}_chunk_{chunk_idx // self.chunk_size + 1}",
                            source=str(path),
                        )
                        documents.append(doc)
                else:
                    # Single document for entire CSV
                    content_parts = []
                    for _, row in df.iterrows():
                        row_content = []
                        for field in self.content_fields or df.columns:
                            if field in row and pd.notna(row[field]):
                                row_content.append(f"{field}: {row[field]}")
                        if row_content:
                            content_parts.append(", ".join(row_content))

                    content = "\n".join(content_parts)

                    metadata = {
                        "source": str(path),
                        "file_name": path.name,
                        "parser": "CSVParser_LlamaIndex",
                        "tool": "LlamaIndex-Pandas",
                        "rows": len(df),
                        "columns": list(df.columns),
                    }

                    doc = Document(
                        content=content,
                        metadata=metadata,
                        id=path.stem,
                        source=str(path),
                    )
                    documents.append(doc)
            else:
                # Use default LlamaIndex document processing
                for i, llama_doc in enumerate(llama_docs):
                    content = (
                        llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                    )

                    metadata = {
                        "source": str(path),
                        "file_name": path.name,
                        "parser": "CSVParser_LlamaIndex",
                        "tool": "LlamaIndex-Pandas",
                        "doc_index": i,
                    }

                    if hasattr(llama_doc, "metadata"):
                        metadata.update(llama_doc.metadata)

                    doc = Document(
                        content=content,
                        metadata=metadata,
                        id=f"{path.stem}_doc_{i + 1}",
                        source=str(path),
                    )
                    documents.append(doc)

            return ProcessingResult(documents=documents, errors=[])

        except Exception as e:
            logger.error(f"Failed to parse CSV file {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )
