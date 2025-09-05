"""LlamaIndex-based CSV and Excel Parser."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_parser import BaseLlamaIndexParser, ParserFactory

logger = logging.getLogger(__name__)

try:
    from llama_index.readers.file import PandasCSVReader, PandasExcelReader
    PANDAS_READERS_AVAILABLE = True
except ImportError:
    try:
        from llama_index.readers import PandasCSVReader, PandasExcelReader
        PANDAS_READERS_AVAILABLE = True
    except ImportError:
        PANDAS_READERS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class LlamaIndexCSVExcelParser(BaseLlamaIndexParser):
    """LlamaIndex-based parser for CSV and Excel files."""
    
    def __init__(self, name: str = "LlamaIndexCSVExcelParser", config: Optional[Dict[str, Any]] = None):
        """
        Initialize LlamaIndex CSV/Excel parser.
        
        Args:
            name: Parser name
            config: Parser configuration
        """
        super().__init__(name=name, config=config or {})
        
        # CSV/Excel-specific configuration
        self.content_fields = self.config.get("content_fields", ["subject", "body", "content", "description"])
        self.metadata_fields = self.config.get("metadata_fields", [])
        self.id_field = self.config.get("id_field", None)
        self.combine_content = self.config.get("combine_content", True)
        self.content_separator = self.config.get("content_separator", "\n\n")
        self.priority_mapping = self.config.get("priority_mapping", {})
        self.table_format = self.config.get("table_format", "markdown")  # For Excel
        
        # Excel-specific
        self.sheet_names = self.config.get("sheet_names", None)
        self.combine_sheets = self.config.get("combine_sheets", False)
        self.header_row = self.config.get("header_row", 0)
        
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas is required for CSV/Excel parsing. Install with: pip install pandas")
        
        self._csv_reader = None
        self._excel_reader = None
        self._file_type = None
    
    def _get_reader(self):
        """Get the appropriate LlamaIndex reader - determined per file."""
        # This will be set in parse() method based on file type
        return None
    
    def parse(self, file_path: str, **kwargs) -> "ProcessingResult":
        """
        Parse a CSV or Excel file using LlamaIndex.
        
        Args:
            file_path: Path to the CSV or Excel file
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
        
        # Determine file type and set appropriate reader
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.csv':
                return self._parse_csv(file_path, **kwargs)
            elif extension in ['.xlsx', '.xls']:
                return self._parse_excel(file_path, **kwargs)
            else:
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{
                        "error": f"Unsupported file extension: {extension}",
                        "source": str(file_path)
                    }]
                )
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to parse {extension} file: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _parse_csv(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse CSV file."""
        try:
            if PANDAS_READERS_AVAILABLE:
                return self._parse_csv_with_llamaindex(file_path, **kwargs)
            else:
                return self._parse_csv_with_pandas(file_path, **kwargs)
        except Exception as e:
            logger.error(f"CSV parsing failed: {e}")
            # Fallback to manual pandas parsing
            return self._parse_csv_with_pandas(file_path, **kwargs)
    
    def _parse_excel(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse Excel file."""
        try:
            if PANDAS_READERS_AVAILABLE:
                return self._parse_excel_with_llamaindex(file_path, **kwargs)
            else:
                return self._parse_excel_with_pandas(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Excel parsing failed: {e}")
            # Fallback to manual pandas parsing
            return self._parse_excel_with_pandas(file_path, **kwargs)
    
    def _parse_csv_with_llamaindex(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse CSV using LlamaIndex PandasCSVReader."""
        try:
            reader = PandasCSVReader(concat_rows=False)
            documents = reader.load_data(file=str(file_path))
            
            return self._process_tabular_documents(documents, file_path, "csv")
            
        except Exception as e:
            logger.warning(f"LlamaIndex CSV reader failed: {e}, falling back to pandas")
            return self._parse_csv_with_pandas(file_path, **kwargs)
    
    def _parse_excel_with_llamaindex(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse Excel using LlamaIndex PandasExcelReader."""
        try:
            reader = PandasExcelReader(
                concat_rows=False,
                sheet_name=self.sheet_names[0] if self.sheet_names else None
            )
            
            if self.sheet_names and len(self.sheet_names) > 1:
                # Handle multiple sheets
                all_documents = []
                for sheet_name in self.sheet_names:
                    sheet_reader = PandasExcelReader(concat_rows=False, sheet_name=sheet_name)
                    sheet_docs = sheet_reader.load_data(file=str(file_path))
                    
                    # Add sheet name to metadata
                    for doc in sheet_docs:
                        if not doc.metadata:
                            doc.metadata = {}
                        doc.metadata["sheet_name"] = sheet_name
                    
                    all_documents.extend(sheet_docs)
                
                return self._process_tabular_documents(all_documents, file_path, "excel")
            else:
                documents = reader.load_data(file=str(file_path))
                return self._process_tabular_documents(documents, file_path, "excel")
                
        except Exception as e:
            logger.warning(f"LlamaIndex Excel reader failed: {e}, falling back to pandas")
            return self._parse_excel_with_pandas(file_path, **kwargs)
    
    def _parse_csv_with_pandas(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse CSV using direct pandas."""
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(file_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is None:
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "Failed to read CSV with any encoding", "source": str(file_path)}]
                )
            
            return self._process_dataframe(df, file_path, "csv")
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Pandas CSV parsing failed: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _parse_excel_with_pandas(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse Excel using direct pandas."""
        try:
            if self.sheet_names:
                # Read specific sheets
                sheet_dict = pd.read_excel(file_path, sheet_name=self.sheet_names, header=self.header_row)
                
                if self.combine_sheets:
                    # Combine all sheets into one dataframe
                    combined_df = pd.concat(sheet_dict.values(), ignore_index=True)
                    # Add sheet source information
                    sheet_sources = []
                    for sheet_name, sheet_df in sheet_dict.items():
                        sheet_sources.extend([sheet_name] * len(sheet_df))
                    combined_df['_sheet_source'] = sheet_sources
                    
                    return self._process_dataframe(combined_df, file_path, "excel")
                else:
                    # Process each sheet separately
                    all_documents = []
                    all_errors = []
                    
                    for sheet_name, sheet_df in sheet_dict.items():
                        result = self._process_dataframe(sheet_df, file_path, "excel", sheet_name=sheet_name)
                        all_documents.extend(result.documents)
                        all_errors.extend(result.errors)
                    
                    from core.base import ProcessingResult
                    return ProcessingResult(
                        documents=all_documents,
                        errors=all_errors,
                        metrics={
                            "total_documents": len(all_documents),
                            "total_errors": len(all_errors),
                            "file_processed": str(file_path),
                            "parser_type": self.name,
                            "sheets_processed": len(sheet_dict)
                        }
                    )
            else:
                # Read all sheets
                df = pd.read_excel(file_path, header=self.header_row)
                return self._process_dataframe(df, file_path, "excel")
                
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Pandas Excel parsing failed: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _process_tabular_documents(self, documents, file_path: Path, file_type: str) -> "ProcessingResult":
        """Process LlamaIndex tabular documents."""
        from core.base import ProcessingResult, Document
        from utils.hash_utils import generate_document_metadata
        
        result_documents = []
        errors = []
        
        for i, llama_doc in enumerate(documents):
            try:
                content = llama_doc.text or ""
                
                if not content.strip():
                    continue
                
                # Generate metadata
                base_metadata = generate_document_metadata(str(file_path), content)
                base_metadata.update({
                    "parser_type": self.name,
                    "file_type": file_type,
                    "row_index": i,
                    "llama_doc_id": llama_doc.id_
                })
                
                # Add LlamaIndex metadata
                if hasattr(llama_doc, 'metadata') and llama_doc.metadata:
                    base_metadata.update(llama_doc.metadata)
                
                document_id = f"doc_{base_metadata['document_hash'][:12]}_row_{i}"
                doc = Document(
                    content=content,
                    metadata=base_metadata,
                    id=document_id,
                    source=str(file_path)
                )
                result_documents.append(doc)
                
            except Exception as e:
                errors.append({
                    "error": f"Failed to process row {i}: {str(e)}",
                    "source": str(file_path)
                })
        
        return ProcessingResult(
            documents=result_documents,
            errors=errors,
            metrics={
                "total_documents": len(result_documents),
                "total_errors": len(errors),
                "file_processed": str(file_path),
                "parser_type": self.name,
                "file_type": file_type
            }
        )
    
    def _process_dataframe(self, df: pd.DataFrame, file_path: Path, file_type: str, sheet_name: str = None) -> "ProcessingResult":
        """Process pandas DataFrame into documents."""
        from core.base import ProcessingResult, Document
        from utils.hash_utils import generate_document_metadata
        
        result_documents = []
        errors = []
        
        # Identify content and metadata fields
        available_columns = [col.lower() for col in df.columns]
        content_cols = []
        metadata_cols = []
        id_col = None
        
        # Find content columns
        for field in self.content_fields:
            for col in df.columns:
                if col.lower() == field.lower():
                    content_cols.append(col)
                    break
        
        # Find metadata columns
        for field in self.metadata_fields:
            for col in df.columns:
                if col.lower() == field.lower():
                    metadata_cols.append(col)
                    break
        
        # Find ID column
        if self.id_field:
            for col in df.columns:
                if col.lower() == self.id_field.lower():
                    id_col = col
                    break
        
        # If no content columns specified or found, use all text columns
        if not content_cols:
            for col in df.columns:
                if df[col].dtype == 'object':  # Likely text
                    content_cols.append(col)
        
        # Process each row
        for idx, row in df.iterrows():
            try:
                # Extract content
                content_parts = []
                for col in content_cols:
                    value = row[col]
                    if pd.notna(value) and str(value).strip():
                        if len(content_cols) > 1:
                            content_parts.append(f"{col}: {str(value).strip()}")
                        else:
                            content_parts.append(str(value).strip())
                
                if not content_parts:
                    continue
                
                content = self.content_separator.join(content_parts) if self.combine_content else content_parts[0]
                
                # Extract metadata
                row_metadata = {}
                for col in metadata_cols:
                    value = row[col]
                    if pd.notna(value):
                        # Handle priority mapping
                        if col.lower() in ['priority'] and str(value) in self.priority_mapping:
                            row_metadata[col.lower()] = self.priority_mapping[str(value)]
                        else:
                            row_metadata[col.lower()] = str(value)
                
                # Add all columns as metadata if no specific metadata fields
                if not metadata_cols:
                    for col in df.columns:
                        if col not in content_cols:
                            value = row[col]
                            if pd.notna(value):
                                row_metadata[col.lower()] = str(value)
                
                # Generate document metadata
                base_metadata = generate_document_metadata(str(file_path), content)
                base_metadata.update({
                    "parser_type": self.name,
                    "file_type": file_type,
                    "row_index": int(idx),
                    "total_rows": len(df),
                    "content_fields": content_cols,
                    "metadata_fields": metadata_cols
                })
                
                if sheet_name:
                    base_metadata["sheet_name"] = sheet_name
                
                base_metadata.update(row_metadata)
                
                # Create document ID
                if id_col and pd.notna(row[id_col]):
                    document_id = f"doc_{str(row[id_col])}"
                else:
                    document_id = f"doc_{base_metadata['document_hash'][:12]}_row_{idx}"
                
                doc = Document(
                    content=content,
                    metadata=base_metadata,
                    id=document_id,
                    source=str(file_path)
                )
                result_documents.append(doc)
                
            except Exception as e:
                errors.append({
                    "error": f"Failed to process row {idx}: {str(e)}",
                    "source": str(file_path),
                    "row_index": int(idx)
                })
        
        return ProcessingResult(
            documents=result_documents,
            errors=errors,
            metrics={
                "total_documents": len(result_documents),
                "total_errors": len(errors),
                "total_rows": len(df),
                "file_processed": str(file_path),
                "parser_type": self.name,
                "file_type": file_type,
                "content_fields": content_cols,
                "metadata_fields": metadata_cols
            }
        )
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    @staticmethod
    def can_parse_mime_type(mime_type: str) -> bool:
        """Check if this parser can handle the given MIME type."""
        return mime_type in [
            'text/csv',
            'application/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ]
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions."""
        return ['.csv', '.xlsx', '.xls']
    
    @staticmethod
    def get_description() -> str:
        """Get parser description."""
        return "LlamaIndex-based parser for CSV and Excel files"


# Register the parser
ParserFactory.register_parser("LlamaIndexCSVExcelParser", LlamaIndexCSVExcelParser)