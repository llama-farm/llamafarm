"""Excel parser using Pandas library."""

from pathlib import Path
from typing import Dict, Any, List, Optional

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.parsers.excel.pandas_parser")


class ExcelParser_Pandas:
    """Excel parser using Pandas for data analysis."""

    def __init__(
        self, name: str = "ExcelParser_Pandas", config: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.sheets = self.config.get("sheets", None)  # None = all sheets
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.skiprows = self.config.get("skiprows", None)
        self.na_values = self.config.get("na_values", ["", "NA", "N/A", "null", "None"])

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse Excel using Pandas."""
        from core.base import Document, ProcessingResult

        try:
            import pandas as pd
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "Pandas not installed. Install with: pip install pandas openpyxl",
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
            # Read Excel file
            excel_file = pd.ExcelFile(source, engine="openpyxl")

            # Get sheets to process
            if self.sheets:
                sheet_names = [s for s in self.sheets if s in excel_file.sheet_names]
            else:
                sheet_names = excel_file.sheet_names

            documents = []

            for sheet_name in sheet_names:
                # Read sheet
                df = pd.read_excel(
                    excel_file,
                    sheet_name=sheet_name,
                    skiprows=self.skiprows,
                    na_values=self.na_values,
                )

                if df.empty:
                    continue

                metadata = {
                    "source": str(path),
                    "file_name": path.name,
                    "parser": self.name,
                    "tool": "Pandas",
                    "sheet_name": sheet_name,
                    "total_sheets": len(excel_file.sheet_names),
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                }

                if self.extract_metadata:
                    # Add data statistics
                    metadata.update(
                        {
                            "memory_usage": df.memory_usage(deep=True).sum(),
                            "dtypes": {
                                col: str(dtype) for col, dtype in df.dtypes.items()
                            },
                            "null_counts": df.isnull().sum().to_dict(),
                            "numeric_columns": df.select_dtypes(
                                include=["number"]
                            ).columns.tolist(),
                            "text_columns": df.select_dtypes(
                                include=["object"]
                            ).columns.tolist(),
                        }
                    )

                # Apply chunking if needed
                if (
                    self.chunk_size
                    and self.chunk_size > 0
                    and len(df) > self.chunk_size
                ):
                    for chunk_idx in range(0, len(df), self.chunk_size):
                        chunk_df = df.iloc[chunk_idx : chunk_idx + self.chunk_size]
                        chunk_text = f"Sheet: {sheet_name}\n{chunk_df.to_string()}"

                        chunk_metadata = metadata.copy()
                        chunk_metadata.update(
                            {
                                "chunk_index": chunk_idx // self.chunk_size,
                                "chunk_rows": len(chunk_df),
                                "start_row": chunk_idx,
                                "end_row": min(chunk_idx + self.chunk_size, len(df)),
                            }
                        )

                        doc = Document(
                            content=chunk_text,
                            metadata=chunk_metadata,
                            id=f"{path.stem}_{sheet_name}_chunk_{chunk_idx // self.chunk_size + 1}",
                            source=str(path),
                        )
                        documents.append(doc)
                else:
                    # Single document for sheet
                    text = f"Sheet: {sheet_name}\n{df.to_string()}"
                    doc = Document(
                        content=text,
                        metadata=metadata,
                        id=f"{path.stem}_{sheet_name}",
                        source=str(path),
                    )
                    documents.append(doc)

            excel_file.close()

            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "parser_type": self.name,
                    "tool": "Pandas",
                    "sheets_processed": len(sheet_names),
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )
