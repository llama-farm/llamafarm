"""CSV and Excel parser using LlamaIndex and pandas."""

from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import yaml
import json

from ..base import LlamaIndexParser, ParserConfig

logger = logging.getLogger(__name__)

# Lazy imports
PANDAS_AVAILABLE = False
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    logger.warning("Pandas not available. Install with: pip install pandas openpyxl")


class CSVExcelParser(LlamaIndexParser):
    """CSV and Excel parser with table-to-text conversion."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize CSV/Excel parser.
        
        Args:
            config: Parser configuration
        """
        if not PANDAS_AVAILABLE:
            raise ImportError("Pandas required. Install with: pip install pandas openpyxl")
        
        super().__init__(config)
        
        # We'll use pandas directly instead of LlamaIndex reader for more control
        self.reader = None
        
        # CSV/Excel-specific options
        self.convert_to = self.config.get("convert_to", "markdown")
        self.include_headers = self.config.get("include_headers", True)
        self.sheet_names = self.config.get("sheet_names", None)
        self.max_rows = self.config.get("max_rows", None)
        self.summarize_large_tables = self.config.get("summarize_large_tables", True)
    
    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata from config.yaml.
        
        Returns:
            ParserConfig object with metadata
        """
        config_path = Path(__file__).parent / "config.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)
                return ParserConfig(**data['parser'])
        
        # Fallback configuration
        return ParserConfig(
            name="csv_excel",
            display_name="CSV/Excel Parser",
            version="2.0.0",
            supported_extensions=[".csv", ".xls", ".xlsx", ".xlsm", ".tsv"],
            mime_types=[
                "text/csv",
                "application/vnd.ms-excel",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "text/tab-separated-values"
            ],
            capabilities=[
                "table_extraction",
                "chunking",
                "metadata_extraction",
                "multi_sheet_support",
                "statistical_summary"
            ],
            dependencies={
                "required": ["pandas"],
                "optional": ["openpyxl", "xlrd", "xlsxwriter"]
            },
            default_config={
                "chunk_size": None,
                "convert_to": "markdown",
                "include_headers": True,
                "sheet_names": None,
                "max_rows": None,
                "summarize_large_tables": True
            }
        )
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file.
        
        Args:
            file_path: Path to file
            
        Returns:
            True if parser can handle the file
        """
        path = Path(file_path)
        return path.suffix.lower() in self.metadata.supported_extensions
    
    def _table_to_markdown(self, df: pd.DataFrame, title: str = None) -> str:
        """Convert DataFrame to markdown table.
        
        Args:
            df: Pandas DataFrame
            title: Optional table title
            
        Returns:
            Markdown formatted table
        """
        if title:
            markdown = f"## {title}\n\n"
        else:
            markdown = ""
        
        # Add summary statistics for large tables
        if self.summarize_large_tables and len(df) > 100:
            markdown += f"*Table contains {len(df)} rows and {len(df.columns)} columns*\n\n"
            markdown += "**Summary Statistics:**\n"
            markdown += df.describe().to_markdown() + "\n\n"
            markdown += "**First 10 rows:**\n"
            df = df.head(10)
        
        markdown += df.to_markdown(index=False)
        return markdown
    
    def _table_to_json(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to JSON string.
        
        Args:
            df: Pandas DataFrame
            
        Returns:
            JSON formatted table
        """
        return df.to_json(orient='records', indent=2)
    
    def _table_to_text(self, df: pd.DataFrame, title: str = None) -> str:
        """Convert DataFrame to plain text.
        
        Args:
            df: Pandas DataFrame
            title: Optional table title
            
        Returns:
            Plain text representation
        """
        text = ""
        if title:
            text += f"{title}\n" + "=" * len(title) + "\n\n"
        
        # Create text representation
        for _, row in df.iterrows():
            row_text = []
            for col, val in row.items():
                if pd.notna(val):
                    row_text.append(f"{col}: {val}")
            text += ", ".join(row_text) + "\n"
        
        return text
    
    def _read_csv(self, file_path: str) -> pd.DataFrame:
        """Read CSV file with encoding detection.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Pandas DataFrame
        """
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # Detect delimiter
                with open(file_path, 'r', encoding=encoding) as f:
                    first_line = f.readline()
                    if '\t' in first_line:
                        delimiter = '\t'
                    elif '|' in first_line:
                        delimiter = '|'
                    else:
                        delimiter = ','
                
                df = pd.read_csv(file_path, encoding=encoding, delimiter=delimiter)
                logger.debug(f"Read CSV with encoding {encoding}")
                return df
            except Exception:
                continue
        
        # If all encodings fail, try with error handling
        return pd.read_csv(file_path, encoding='utf-8', errors='replace')
    
    def parse(self, source: str):
        """Parse CSV or Excel file.
        
        Args:
            source: Path to CSV/Excel file
            
        Returns:
            ProcessingResult with documents
        """
        from core.base import Document, ProcessingResult
        
        path = Path(source)
        documents = []
        errors = []
        
        try:
            # Read file based on extension
            if path.suffix.lower() in ['.csv', '.tsv']:
                df = self._read_csv(source)
                dataframes = {"main": df}
            else:
                # Excel file - may have multiple sheets
                excel_file = pd.ExcelFile(source)
                
                if self.sheet_names:
                    sheets = self.sheet_names
                else:
                    sheets = excel_file.sheet_names
                
                dataframes = {}
                for sheet in sheets:
                    if sheet in excel_file.sheet_names:
                        dataframes[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
            
            # Process each dataframe
            for name, df in dataframes.items():
                # Limit rows if configured
                if self.max_rows:
                    df = df.head(self.max_rows)
                
                # Convert to text based on configuration
                if self.convert_to == "markdown":
                    content = self._table_to_markdown(df, title=name if len(dataframes) > 1 else None)
                elif self.convert_to == "json":
                    content = self._table_to_json(df)
                else:
                    content = self._table_to_text(df, title=name if len(dataframes) > 1 else None)
                
                # Create metadata
                metadata = {
                    "source": source,
                    "sheet_name": name,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "column_names": df.columns.tolist(),
                    "data_types": df.dtypes.astype(str).to_dict(),
                    "parser_type": "CSVExcelParser"
                }
                
                # Add statistical summary for numeric columns
                numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                if len(numeric_cols) > 0:
                    metadata["numeric_summary"] = df[numeric_cols].describe().to_dict()
                
                # Create document
                doc_id = f"{path.stem}_{name}" if len(dataframes) > 1 else path.stem
                doc = Document(
                    content=content,
                    metadata=metadata,
                    id=doc_id,
                    source=source
                )
                documents.append(doc)
                
        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            errors.append({
                'source': source,
                'error': str(e),
                'parser': 'CSVExcelParser'
            })
        
        # Apply chunking if configured
        if documents and self.config.get("chunk_size"):
            documents = self._apply_chunking(documents)
        
        return ProcessingResult(
            documents=documents,
            errors=errors,
            metrics={
                'total_documents': len(documents),
                'total_errors': len(errors),
                'parser_type': 'CSVExcelParser'
            }
        )