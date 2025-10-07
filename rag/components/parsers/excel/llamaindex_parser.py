"""Excel parser using LlamaIndex."""

from pathlib import Path
from typing import Dict, Any, Optional, List

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.parsers.excel.llamaindex_parser")


class ExcelParser_LlamaIndex:
    """Excel parser using LlamaIndex with Pandas backend."""

    def __init__(
        self,
        name: str = "ExcelParser_LlamaIndex",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.config = config or {}

        # Sheet processing
        self.sheets = self.config.get("sheets", None)  # None = all sheets
        self.combine_sheets = self.config.get("combine_sheets", False)

        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 1000)  # rows per chunk
        self.chunk_strategy = self.config.get("chunk_strategy", "rows")

        # Excel specific config
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_formulas = self.config.get("extract_formulas", False)
        self.header_row = self.config.get("header_row", 0)
        self.skiprows = self.config.get("skiprows", None)
        self.na_values = self.config.get("na_values", ["", "NA", "N/A", "null", "None"])

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse Excel using LlamaIndex."""
        from core.base import Document, ProcessingResult

        try:
            from llama_index.readers.file import PandasExcelReader
            import pandas as pd
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "LlamaIndex or pandas not installed. Install with: pip install llama-index pandas openpyxl",
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
            documents = []

            # Read Excel file with pandas for more control
            excel_file = pd.ExcelFile(source)
            sheets_to_process = self.sheets if self.sheets else excel_file.sheet_names

            all_sheet_content = []

            for sheet_name in sheets_to_process:
                if sheet_name not in excel_file.sheet_names:
                    logger.warning(f"Sheet '{sheet_name}' not found in {source}")
                    continue

                # Read sheet
                df = pd.read_excel(
                    source,
                    sheet_name=sheet_name,
                    header=self.header_row,
                    skiprows=self.skiprows,
                    na_values=self.na_values,
                )

                # Extract metadata
                sheet_metadata = {
                    "source": str(path),
                    "file_name": path.name,
                    "parser": "ExcelParser_LlamaIndex",
                    "tool": "LlamaIndex-Pandas",
                    "sheet_name": sheet_name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": list(df.columns),
                }

                if self.extract_metadata:
                    # Add data statistics
                    sheet_metadata["memory_usage"] = df.memory_usage(deep=True).sum()
                    sheet_metadata["dtypes"] = {
                        col: str(dtype) for col, dtype in df.dtypes.items()
                    }
                    sheet_metadata["null_counts"] = df.isnull().sum().to_dict()

                # Convert DataFrame to text
                if self.chunk_strategy == "rows" and self.chunk_size:
                    # Process in chunks
                    for chunk_idx in range(0, len(df), self.chunk_size):
                        chunk_df = df.iloc[chunk_idx : chunk_idx + self.chunk_size]

                        # Convert chunk to text
                        chunk_content = self._dataframe_to_text(chunk_df)

                        if self.combine_sheets:
                            all_sheet_content.append(
                                f"=== Sheet: {sheet_name} (Rows {chunk_idx}-{min(chunk_idx + self.chunk_size, len(df))}) ===\n{chunk_content}"
                            )
                        else:
                            chunk_metadata = sheet_metadata.copy()
                            chunk_metadata.update(
                                {
                                    "chunk_index": chunk_idx // self.chunk_size,
                                    "chunk_rows": len(chunk_df),
                                    "start_row": chunk_idx,
                                    "end_row": min(
                                        chunk_idx + self.chunk_size, len(df)
                                    ),
                                }
                            )

                            doc = Document(
                                content=chunk_content,
                                metadata=chunk_metadata,
                                id=f"{path.stem}_{sheet_name}_chunk_{chunk_idx // self.chunk_size + 1}",
                                source=str(path),
                            )
                            documents.append(doc)
                else:
                    # Process entire sheet as one document
                    sheet_content = self._dataframe_to_text(df)

                    if self.combine_sheets:
                        all_sheet_content.append(
                            f"=== Sheet: {sheet_name} ===\n{sheet_content}"
                        )
                    else:
                        doc = Document(
                            content=sheet_content,
                            metadata=sheet_metadata,
                            id=f"{path.stem}_{sheet_name}",
                            source=str(path),
                        )
                        documents.append(doc)

            # If combining sheets, create a single document
            if self.combine_sheets and all_sheet_content:
                combined_content = "\n\n".join(all_sheet_content)

                metadata = {
                    "source": str(path),
                    "file_name": path.name,
                    "parser": "ExcelParser_LlamaIndex",
                    "tool": "LlamaIndex-Pandas",
                    "total_sheets": len(sheets_to_process),
                    "sheet_names": sheets_to_process,
                    "file_size": path.stat().st_size,
                }

                # Apply chunking if needed
                if self.chunk_size and len(combined_content) > self.chunk_size:
                    from llama_index.core import Document as LlamaDoc
                    from llama_index.core.node_parser import TokenTextSplitter

                    llama_doc = LlamaDoc(text=combined_content, metadata=metadata)
                    splitter = TokenTextSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=100
                    )

                    nodes = splitter.get_nodes_from_documents([llama_doc])

                    for i, node in enumerate(nodes):
                        chunk_metadata = metadata.copy()
                        chunk_metadata.update(
                            {"chunk_index": i, "total_chunks": len(nodes)}
                        )

                        doc = Document(
                            content=node.text if hasattr(node, "text") else str(node),
                            metadata=chunk_metadata,
                            id=f"{path.stem}_combined_chunk_{i + 1}",
                            source=str(path),
                        )
                        documents.append(doc)
                else:
                    doc = Document(
                        content=combined_content,
                        metadata=metadata,
                        id=f"{path.stem}_combined",
                        source=str(path),
                    )
                    documents.append(doc)

            # Alternative: Use LlamaIndex PandasExcelReader for simpler processing
            if not documents:
                reader = PandasExcelReader(
                    concat_rows=False,
                    sheet_name=sheets_to_process[0] if sheets_to_process else None,
                    pandas_config={
                        "header": self.header_row,
                        "skiprows": self.skiprows,
                        "na_values": self.na_values,
                    },
                )

                llama_docs = reader.load_data(file=path)

                for i, llama_doc in enumerate(llama_docs):
                    content = (
                        llama_doc.text if hasattr(llama_doc, "text") else str(llama_doc)
                    )

                    metadata = {
                        "source": str(path),
                        "file_name": path.name,
                        "parser": "ExcelParser_LlamaIndex",
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
            logger.error(f"Failed to parse Excel file {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _dataframe_to_text(self, df) -> str:
        """Convert DataFrame to text representation."""
        # Option 1: Markdown table format
        if len(df) <= 50:  # Small tables can be formatted as markdown
            return (
                df.to_markdown(index=False)
                if hasattr(df, "to_markdown")
                else df.to_string()
            )

        # Option 2: CSV-like format for larger data
        return df.to_csv(index=False)
