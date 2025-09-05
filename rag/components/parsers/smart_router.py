"""Smart Router for automatic parser selection based on file content analysis."""

import logging
import magic
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from core.base import Parser, ProcessingResult
from .base_parser import BaseLlamaIndexParser, ParserFactory

logger = logging.getLogger(__name__)


class SmartRouter(Parser):
    """
    Smart router that automatically selects the appropriate parser 
    based on content analysis and file type detection.
    """
    
    def __init__(self, name: str = "SmartRouter", config: Optional[Dict[str, Any]] = None):
        """
        Initialize smart router.
        
        Args:
            name: Router name
            config: Router configuration
        """
        super().__init__(name=name, config=config or {})
        
        # Configuration
        self.fallback_chain = self.config.get("fallback_chain", [
            "PlainTextParser", 
            "MarkdownParser"
        ])
        self.content_detection_enabled = self.config.get("content_detection_enabled", True)
        self.max_file_size_mb = self.config.get("max_file_size_mb", 100)
        self.enable_magic = self.config.get("enable_magic", True)
        
        # Cache for file type detection results
        self._detection_cache = {}
    
    def parse(self, file_path: str, **kwargs) -> ProcessingResult:
        """
        Route file to appropriate parser and parse it.
        
        Args:
            file_path: Path to the file
            **kwargs: Additional parsing options
            
        Returns:
            ProcessingResult from the selected parser
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {file_path}", "source": str(file_path)}]
            )
        
        # Check file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"File too large: {file_size_mb:.1f}MB (max: {self.max_file_size_mb}MB)",
                    "source": str(file_path)
                }]
            )
        
        # Select parser
        parser_info = self._select_parser(file_path)
        
        if not parser_info:
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": "No suitable parser found",
                    "source": str(file_path),
                    "detected_type": self._detection_cache.get(str(file_path), {})
                }]
            )
        
        # Create and use parser
        try:
            parser_class = parser_info["parser_class"]
            parser_config = parser_info.get("config", self.config.copy())
            
            # Remove router-specific config before passing to parser
            for key in ["fallback_chain", "content_detection_enabled", "max_file_size_mb", "enable_magic"]:
                parser_config.pop(key, None)
            
            parser = parser_class(config=parser_config)
            result = parser.parse(str(file_path), **kwargs)
            
            # Add router metadata
            for doc in result.documents:
                doc.metadata.update({
                    "router_used": True,
                    "selected_parser": parser_class.__name__,
                    "selection_method": parser_info.get("selection_method", "unknown"),
                    "detection_confidence": parser_info.get("confidence", "unknown")
                })
            
            # Update metrics
            result.metrics.update({
                "router_used": True,
                "selected_parser": parser_class.__name__,
                "selection_method": parser_info.get("selection_method", "unknown")
            })
            
            return result
            
        except Exception as e:
            # Try fallback chain
            return self._try_fallback_chain(file_path, str(e), **kwargs)
    
    def _select_parser(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Select the most appropriate parser for the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with parser information or None
        """
        file_str = str(file_path)
        
        # Check cache first
        if file_str in self._detection_cache:
            detection_result = self._detection_cache[file_str]
        else:
            # Perform file type detection
            detection_result = self._detect_file_type(file_path)
            self._detection_cache[file_str] = detection_result
        
        # Strategy 1: Content-based detection using MIME type
        if self.content_detection_enabled and detection_result.get("mime_type"):
            parser_class = self._get_parser_by_mime_type(detection_result["mime_type"])
            if parser_class:
                return {
                    "parser_class": parser_class,
                    "selection_method": "mime_type",
                    "confidence": detection_result.get("confidence", "medium"),
                    "config": self.config.copy()
                }
        
        # Strategy 2: Extension-based detection
        extension = file_path.suffix.lower()
        if extension:
            parser_class = self._get_parser_by_extension(extension)
            if parser_class:
                return {
                    "parser_class": parser_class,
                    "selection_method": "extension",
                    "confidence": "medium",
                    "config": self.config.copy()
                }
        
        # Strategy 3: Content analysis for files without extensions
        if not extension and detection_result.get("is_text"):
            # Try to analyze content to determine format
            parser_class = self._analyze_text_content(file_path)
            if parser_class:
                return {
                    "parser_class": parser_class,
                    "selection_method": "content_analysis",
                    "confidence": "low",
                    "config": self.config.copy()
                }
        
        # Strategy 4: Fallback to text parser for text files
        if detection_result.get("is_text"):
            parser_class = ParserFactory._parsers.get("PlainTextParser")
            if parser_class:
                return {
                    "parser_class": parser_class,
                    "selection_method": "text_fallback",
                    "confidence": "low",
                    "config": self.config.copy()
                }
        
        return None
    
    def _detect_file_type(self, file_path: Path) -> Dict[str, Any]:
        """
        Detect file type using multiple methods.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dict with detection results
        """
        detection_result = {
            "file_path": str(file_path),
            "extension": file_path.suffix.lower(),
            "mime_type": None,
            "encoding": None,
            "description": None,
            "is_text": False,
            "confidence": "low"
        }
        
        if not self.enable_magic:
            # Basic extension-based detection
            extension = file_path.suffix.lower()
            text_extensions = {'.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml'}
            detection_result["is_text"] = extension in text_extensions
            detection_result["confidence"] = "extension_only"
            return detection_result
        
        try:
            # Use python-magic for detailed detection
            mime_type = magic.from_file(str(file_path), mime=True)
            description = magic.from_file(str(file_path))
            
            detection_result.update({
                "mime_type": mime_type,
                "description": description,
                "confidence": "high"
            })
            
            # Determine if it's text-based
            if mime_type:
                detection_result["is_text"] = (
                    mime_type.startswith('text/') or
                    mime_type in {
                        'application/json',
                        'application/xml',
                        'application/javascript',
                        'application/x-yaml',
                        'application/csv'
                    }
                )
            
            # Try to detect encoding for text files
            if detection_result["is_text"]:
                try:
                    with open(file_path, 'rb') as f:
                        raw_data = f.read(1024)  # Read first 1KB
                    # Simple heuristic - could be improved with chardet
                    try:
                        raw_data.decode('utf-8')
                        detection_result["encoding"] = "utf-8"
                    except UnicodeDecodeError:
                        detection_result["encoding"] = "unknown"
                except:
                    detection_result["encoding"] = "unknown"
                    
        except Exception as e:
            logger.warning(f"Failed to detect file type for {file_path}: {e}")
            detection_result["error"] = str(e)
            detection_result["confidence"] = "extension_fallback"
            
            # Basic fallback detection
            extension = file_path.suffix.lower()
            text_extensions = {'.txt', '.md', '.csv', '.json', '.xml', '.yaml', '.yml'}
            detection_result["is_text"] = extension in text_extensions
        
        return detection_result
    
    def _get_parser_by_mime_type(self, mime_type: str) -> Optional[Type[BaseLlamaIndexParser]]:
        """Get parser based on MIME type."""
        mime_to_parser = {
            'text/plain': 'PlainTextParser',
            'text/markdown': 'MarkdownParser',
            'text/csv': 'CSVParser',
            'application/csv': 'CSVParser',
            'text/html': 'HTMLParser',
            'application/pdf': 'PDFParser',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'DocxParser',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'ExcelParser',
            'application/msword': 'DocxParser',
            'application/vnd.ms-excel': 'ExcelParser'
        }
        
        parser_name = mime_to_parser.get(mime_type)
        return ParserFactory._parsers.get(parser_name) if parser_name else None
    
    def _get_parser_by_extension(self, extension: str) -> Optional[Type[BaseLlamaIndexParser]]:
        """Get parser based on file extension."""
        extension_to_parser = {
            '.txt': 'PlainTextParser',
            '.md': 'MarkdownParser',
            '.markdown': 'MarkdownParser',
            '.csv': 'CSVParser',
            '.html': 'HTMLParser',
            '.htm': 'HTMLParser',
            '.pdf': 'PDFParser',
            '.docx': 'DocxParser',
            '.xlsx': 'ExcelParser',
            '.xls': 'ExcelParser',
            '.json': 'PlainTextParser',
            '.xml': 'PlainTextParser',
            '.yaml': 'PlainTextParser',
            '.yml': 'PlainTextParser',
            '.log': 'PlainTextParser'
        }
        
        parser_name = extension_to_parser.get(extension)
        return ParserFactory._parsers.get(parser_name) if parser_name else None
    
    def _analyze_text_content(self, file_path: Path) -> Optional[Type[BaseLlamaIndexParser]]:
        """
        Analyze text content to determine the best parser.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Parser class or None
        """
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(2048)  # Read first 2KB
            
            content_lower = content.lower()
            
            # Check for Markdown indicators
            markdown_indicators = ['# ', '## ', '### ', '- [ ]', '- [x]', '```', '**', '__']
            if any(indicator in content for indicator in markdown_indicators):
                return ParserFactory._parsers.get('MarkdownParser')
            
            # Check for HTML indicators
            html_indicators = ['<html', '<head', '<body', '<div', '<p>', '<a href']
            if any(indicator in content_lower for indicator in html_indicators):
                return ParserFactory._parsers.get('HTMLParser')
            
            # Check for CSV indicators (comma-separated values with consistent structure)
            lines = content.split('\n')[:10]  # Check first 10 lines
            if len(lines) > 1:
                first_line_commas = lines[0].count(',')
                if first_line_commas > 0:
                    # Check if other lines have similar comma counts
                    consistent_commas = sum(
                        1 for line in lines[1:] 
                        if line.strip() and abs(line.count(',') - first_line_commas) <= 1
                    )
                    if consistent_commas >= len([l for l in lines[1:] if l.strip()]) * 0.8:
                        return ParserFactory._parsers.get('CSVParser')
            
            # Default to plain text
            return ParserFactory._parsers.get('PlainTextParser')
            
        except Exception as e:
            logger.warning(f"Failed to analyze content of {file_path}: {e}")
            return ParserFactory._parsers.get('PlainTextParser')
    
    def _try_fallback_chain(self, file_path: Path, original_error: str, **kwargs) -> ProcessingResult:
        """
        Try parsers in the fallback chain.
        
        Args:
            file_path: Path to the file
            original_error: Error from the primary parser
            **kwargs: Additional parsing options
            
        Returns:
            ProcessingResult from successful parser or error result
        """
        errors = [{"error": f"Primary parser failed: {original_error}", "source": str(file_path)}]
        
        for parser_name in self.fallback_chain:
            try:
                parser_class = ParserFactory._parsers.get(parser_name)
                if not parser_class:
                    errors.append({
                        "error": f"Fallback parser not found: {parser_name}",
                        "source": str(file_path)
                    })
                    continue
                
                parser_config = self.config.copy()
                # Remove router-specific config
                for key in ["fallback_chain", "content_detection_enabled", "max_file_size_mb", "enable_magic"]:
                    parser_config.pop(key, None)
                
                parser = parser_class(config=parser_config)
                result = parser.parse(str(file_path), **kwargs)
                
                if result.documents:  # Success
                    # Add fallback metadata
                    for doc in result.documents:
                        doc.metadata.update({
                            "router_used": True,
                            "selected_parser": parser_class.__name__,
                            "selection_method": "fallback_chain",
                            "fallback_position": self.fallback_chain.index(parser_name)
                        })
                    
                    result.metrics.update({
                        "router_used": True,
                        "selected_parser": parser_class.__name__,
                        "selection_method": "fallback_chain",
                        "fallback_used": True
                    })
                    
                    # Include previous errors as warnings
                    result.errors.extend(errors)
                    
                    return result
                    
            except Exception as e:
                errors.append({
                    "error": f"Fallback parser {parser_name} failed: {str(e)}",
                    "source": str(file_path)
                })
        
        # All parsers failed
        return ProcessingResult(
            documents=[],
            errors=errors + [{
                "error": "All fallback parsers failed",
                "source": str(file_path)
            }]
        )
    
    def can_parse(self, file_path: str) -> bool:
        """
        Check if router can handle the file (always true - router tries to find a parser).
        
        Args:
            file_path: Path to the file
            
        Returns:
            Always True (router will try to find an appropriate parser)
        """
        return True
    
    def get_supported_extensions(self) -> List[str]:
        """Get all supported extensions from registered parsers."""
        extensions = set()
        for parser_class in ParserFactory._parsers.values():
            if hasattr(parser_class, 'get_supported_extensions'):
                extensions.update(parser_class.get_supported_extensions())
        return list(extensions)
    
    def get_description(self) -> str:
        """Get router description."""
        return "Smart router that automatically selects the best parser based on file content analysis"
    
    def get_detection_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get file detection information without parsing.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detection information
        """
        file_path = Path(file_path)
        detection_result = self._detect_file_type(file_path)
        
        parser_info = self._select_parser(file_path)
        if parser_info:
            detection_result["selected_parser"] = parser_info["parser_class"].__name__
            detection_result["selection_method"] = parser_info["selection_method"]
        else:
            detection_result["selected_parser"] = None
            detection_result["selection_method"] = None
        
        return detection_result