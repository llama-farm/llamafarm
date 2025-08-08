"""Directory parser that handles directory inputs and delegates to appropriate parsers."""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import importlib

from core.base import Parser, Document, ProcessingResult

logger = logging.getLogger(__name__)


class DirectoryParser(Parser):
    """Parser that handles directory inputs by iterating through files and delegating to appropriate parsers."""
    
    def __init__(self, name: str = "DirectoryParser", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        config = config or {}
        
        # Configuration options
        self.recursive = config.get("recursive", True)
        self.include_patterns = config.get("include_patterns", ["*"])
        self.exclude_patterns = config.get("exclude_patterns", [])
        self.max_files = config.get("max_files", None)
        self.parser_map = config.get("parser_map", {})  # File extension to parser mapping
        self.fallback_parser = config.get("fallback_parser", "PlainTextParser")
        self.parser_configs = config.get("parser_configs", {})  # Parser-specific configurations
        
        # Cache for loaded parsers
        self._parser_cache = {}
        
        # Initialize default parser mappings
        if not self.parser_map:
            self.parser_map = {
                ".txt": "PlainTextParser",
                ".log": "PlainTextParser",
                ".md": "MarkdownParser",
                ".markdown": "MarkdownParser",
                ".html": "HTMLParser",
                ".htm": "HTMLParser",
                ".pdf": "PDFParser",
                ".docx": "DocxParser",
                ".xlsx": "ExcelParser",
                ".xls": "ExcelParser",
                ".csv": "CSVParser",
                ".json": "JSONParser",
                ".xml": "XMLParser",
            }
    
    def validate_config(self) -> bool:
        """Validate parser configuration."""
        return True
    
    def parse(self, source: str, **kwargs) -> ProcessingResult:
        """Parse all files in a directory."""
        errors = []
        all_documents = []
        
        source_path = Path(source)
        
        # Handle single file case
        if source_path.is_file():
            logger.debug(f"Source is a file, parsing directly: {source_path}")
            return self._parse_file(source_path, **kwargs)
        
        # Handle directory case
        if not source_path.is_dir():
            error_msg = f"Source is neither a file nor a directory: {source}"
            logger.error(error_msg)
            errors.append({"error": error_msg, "source": source})
            return ProcessingResult(documents=[], errors=errors)
        
        logger.info(f"Processing directory: {source_path}")
        
        # Get all files to process
        files_to_process = self._get_files_to_process(source_path)
        
        if not files_to_process:
            logger.warning(f"No files found to process in: {source_path}")
            return ProcessingResult(documents=[], errors=[])
        
        # Apply max_files limit if configured
        if self.max_files and len(files_to_process) > self.max_files:
            logger.info(f"Limiting to {self.max_files} files (found {len(files_to_process)})")
            files_to_process = files_to_process[:self.max_files]
        
        logger.info(f"Processing {len(files_to_process)} files")
        
        # Process each file
        for file_path in files_to_process:
            try:
                result = self._parse_file(file_path, **kwargs)
                all_documents.extend(result.documents)
                if result.errors:
                    errors.extend(result.errors)
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                errors.append({
                    "error": str(e),
                    "source": str(file_path),
                    "parser": "DirectoryParser"
                })
        
        logger.info(f"Processed {len(all_documents)} documents with {len(errors)} errors")
        
        return ProcessingResult(documents=all_documents, errors=errors)
    
    def _get_files_to_process(self, directory: Path) -> List[Path]:
        """Get list of files to process based on configuration."""
        files = []
        
        if self.recursive:
            # Recursively find all files
            for pattern in self.include_patterns:
                files.extend(directory.rglob(pattern))
        else:
            # Only look in the immediate directory
            for pattern in self.include_patterns:
                files.extend(directory.glob(pattern))
        
        # Filter out directories
        files = [f for f in files if f.is_file()]
        
        # Apply exclusion patterns
        if self.exclude_patterns:
            excluded_files = set()
            for pattern in self.exclude_patterns:
                if self.recursive:
                    excluded_files.update(directory.rglob(pattern))
                else:
                    excluded_files.update(directory.glob(pattern))
            
            files = [f for f in files if f not in excluded_files]
        
        # Sort for consistent ordering
        files.sort()
        
        return files
    
    def _parse_file(self, file_path: Path, **kwargs) -> ProcessingResult:
        """Parse a single file using the appropriate parser."""
        # Determine parser based on file extension
        extension = file_path.suffix.lower()
        parser_name = self.parser_map.get(extension, self.fallback_parser)
        
        # Get or create parser instance
        parser = self._get_parser(parser_name)
        
        if not parser:
            error_msg = f"No parser available for {extension} files"
            logger.warning(error_msg)
            return ProcessingResult(
                documents=[],
                errors=[{"error": error_msg, "source": str(file_path)}]
            )
        
        # Parse the file
        logger.debug(f"Parsing {file_path} with {parser_name}")
        
        # Pass source in kwargs for parsers that need it (but not PDFParser which has it as first arg)
        # Most parsers expect the file path as first argument and source in kwargs
        clean_kwargs = {k: v for k, v in kwargs.items() if k != 'source'}
        
        # Only add source to kwargs for parsers that expect it in kwargs
        # PDFParser and CSVParser take file path as first argument
        if parser_name not in ['PDFParser', 'CSVParser', 'CustomerSupportCSVParser']:
            clean_kwargs['source'] = str(file_path)
        
        try:
            result = parser.parse(str(file_path), **clean_kwargs)
            return result
        except Exception as e:
            logger.error(f"Parser {parser_name} failed on {file_path}: {e}")
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": str(e),
                    "source": str(file_path),
                    "parser": parser_name
                }]
            )
    
    def _get_parser(self, parser_name: str) -> Optional[Parser]:
        """Get or create a parser instance."""
        if parser_name in self._parser_cache:
            return self._parser_cache[parser_name]
        
        try:
            # Try to import the parser
            parser_class = None
            
            # Common parser locations
            parser_modules = [
                f"components.parsers.{parser_name.lower().replace('parser', '_parser')}",
                f"components.parsers.{parser_name.lower().replace('parser', '')}",
                f"components.parsers.text_parser",  # For PlainTextParser
                f"components.parsers.html_parser",
                f"components.parsers.markdown_parser",
                f"components.parsers.csv_parser",
                f"components.parsers.pdf_parser",
                f"components.parsers.docx_parser",
                f"components.parsers.excel_parser",
            ]
            
            for module_path in parser_modules:
                try:
                    module = importlib.import_module(module_path)
                    if hasattr(module, parser_name):
                        parser_class = getattr(module, parser_name)
                        break
                except (ImportError, AttributeError):
                    continue
            
            if not parser_class:
                logger.warning(f"Could not find parser class: {parser_name}")
                return None
            
            # Create parser instance with specific config if available
            parser_config = self.parser_configs.get(parser_name, {})
            parser = parser_class(name=parser_name, config=parser_config)
            self._parser_cache[parser_name] = parser
            return parser
            
        except Exception as e:
            logger.error(f"Failed to load parser {parser_name}: {e}")
            return None
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get supported file extensions."""
        # This parser supports directories and delegates to other parsers
        return ["directory"]
    
    @classmethod
    def get_description(cls) -> str:
        """Get parser description."""
        return "Directory parser that iterates through files and delegates to appropriate parsers based on file type."