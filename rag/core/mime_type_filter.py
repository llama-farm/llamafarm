"""
MIME Type Filter

Enforces MIME type and file extension restrictions at both strategy and parser levels.
Provides a two-tier filtering system:
1. Strategy-level: Filters files before they reach parsers
2. Parser-level: Each parser can specify what it handles
"""

import mimetypes
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)

# Initialize mimetypes database
mimetypes.init()

# Add custom MIME type mappings for common file types
CUSTOM_MIME_MAPPINGS = {
    '.md': 'text/markdown',
    '.markdown': 'text/markdown',
    '.mdown': 'text/markdown',
    '.mkd': 'text/markdown',
    '.yml': 'text/yaml',
    '.yaml': 'text/yaml',
    '.log': 'text/plain',
    '.txt': 'text/plain',
    '.csv': 'text/csv',
    '.tsv': 'text/tab-separated-values',
    '.pdf': 'application/pdf',
    '.json': 'application/json',
    '.jsonl': 'application/x-jsonlines',
    '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    '.xls': 'application/vnd.ms-excel',
    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    '.doc': 'application/msword',
    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
    '.ppt': 'application/vnd.ms-powerpoint',
}

# Register custom mappings
for ext, mime_type in CUSTOM_MIME_MAPPINGS.items():
    mimetypes.add_type(mime_type, ext)


class MimeTypeFilter:
    """Enforces MIME type and file extension filtering for strategies and parsers."""
    
    def __init__(self):
        """Initialize the MIME type filter."""
        self.mime_cache = {}  # Cache MIME type lookups
    
    def get_mime_type(self, file_path: Path) -> Tuple[Optional[str], str]:
        """
        Get MIME type and extension for a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Tuple of (mime_type, extension)
        """
        # Check cache first
        path_str = str(file_path)
        if path_str in self.mime_cache:
            return self.mime_cache[path_str]
        
        # Get extension
        extension = file_path.suffix.lower()
        
        # Try to detect MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Use custom mapping if standard detection fails
        if not mime_type and extension in CUSTOM_MIME_MAPPINGS:
            mime_type = CUSTOM_MIME_MAPPINGS[extension]
        
        # Cache the result
        result = (mime_type, extension)
        self.mime_cache[path_str] = result
        
        return result
    
    def is_file_allowed_by_strategy(self, file_path: Path, strategy_config: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if a file is allowed by a strategy's supported_files patterns.
        
        Supports three filtering approaches:
        1. supported_files: List of glob patterns (recommended)
        2. allowed_mime_types: List of MIME types (legacy)
        3. allowed_extensions: List of extensions (legacy)
        
        Args:
            file_path: Path to the file
            strategy_config: Strategy configuration with supported_files patterns
            
        Returns:
            Tuple of (is_allowed, reason_if_rejected)
        """
        import fnmatch
        
        # Check new supported_files patterns first (if present)
        supported_files = strategy_config.get('supported_files', [])
        if supported_files:
            file_name = file_path.name
            for pattern in supported_files:
                # Handle different pattern types
                if pattern == "*":  # Accept all files
                    return True, "File allowed by wildcard pattern"
                elif pattern.startswith("*."):  # Extension pattern
                    if fnmatch.fnmatch(file_name.lower(), pattern.lower()):
                        return True, f"File matches pattern '{pattern}'"
                else:  # General glob pattern
                    if fnmatch.fnmatch(file_name, pattern):
                        return True, f"File matches pattern '{pattern}'"
            return False, f"File '{file_name}' doesn't match any supported patterns: {supported_files}"
        
        # Fall back to legacy allowed_mime_types/allowed_extensions if no supported_files
        # Get file info
        mime_type, extension = self.get_mime_type(file_path)
        
        # Check MIME types if specified (legacy)
        allowed_mime_types = strategy_config.get('allowed_mime_types', [])
        if allowed_mime_types:  # Empty list means accept all
            if not mime_type:
                return False, f"Could not determine MIME type for {file_path.name}"
            if mime_type not in allowed_mime_types:
                return False, f"MIME type '{mime_type}' not in allowed types: {allowed_mime_types}"
        
        # Check extensions if specified (legacy)
        allowed_extensions = strategy_config.get('allowed_extensions', [])
        if allowed_extensions:  # Empty list means accept all
            if not extension:
                return False, f"File {file_path.name} has no extension"
            if extension not in allowed_extensions:
                return False, f"Extension '{extension}' not in allowed extensions: {allowed_extensions}"
        
        return True, "File allowed by strategy"
    
    def find_matching_parser(self, file_path: Path, parsers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Find the best matching parser for a file based on MIME type and extension.
        
        Args:
            file_path: Path to the file
            parsers: List of parser configurations
            
        Returns:
            Best matching parser configuration or None
        """
        mime_type, extension = self.get_mime_type(file_path)
        
        # Score each parser
        best_parser = None
        best_score = -1
        
        for parser in parsers:
            score = 0
            
            # Check if parser specifies MIME types
            parser_mime_types = parser.get('mime_types', [])
            if parser_mime_types:
                if mime_type and mime_type in parser_mime_types:
                    score += 10  # High score for MIME type match
                else:
                    continue  # Skip this parser if MIME types specified but don't match
            
            # Check if parser specifies extensions
            parser_extensions = parser.get('file_extensions', [])
            if parser_extensions:
                if extension and extension in parser_extensions:
                    score += 5  # Medium score for extension match
                else:
                    continue  # Skip this parser if extensions specified but don't match
            
            # If parser has no restrictions, give it a base score
            if not parser_mime_types and not parser_extensions:
                score = 1  # Low score for unrestricted parser
            
            # Consider priority if specified
            priority = parser.get('priority', 0)
            score += priority
            
            # Update best parser if this one scores higher
            if score > best_score:
                best_score = score
                best_parser = parser
        
        return best_parser
    
    def validate_strategy_files(self, file_paths: List[Path], strategy_config: Dict[str, Any]) -> Dict[str, List[Path]]:
        """
        Validate multiple files against a strategy's restrictions.
        
        Args:
            file_paths: List of file paths to validate
            strategy_config: Strategy configuration
            
        Returns:
            Dictionary with 'accepted' and 'rejected' file lists
        """
        result = {
            'accepted': [],
            'rejected': [],
            'rejection_reasons': {}
        }
        
        for file_path in file_paths:
            is_allowed, reason = self.is_file_allowed_by_strategy(file_path, strategy_config)
            if is_allowed:
                result['accepted'].append(file_path)
            else:
                result['rejected'].append(file_path)
                result['rejection_reasons'][str(file_path)] = reason
                logger.warning(f"File rejected by strategy: {file_path} - {reason}")
        
        return result
    
    def assign_files_to_parsers(self, file_paths: List[Path], parsers: List[Dict[str, Any]]) -> Dict[str, List[Path]]:
        """
        Assign files to appropriate parsers based on MIME types and extensions.
        
        Args:
            file_paths: List of file paths to assign
            parsers: List of parser configurations
            
        Returns:
            Dictionary mapping parser indices to file lists
        """
        assignments = {}
        unassigned = []
        
        for file_path in file_paths:
            # Find matching parser
            matching_parser = None
            for i, parser in enumerate(parsers):
                if self._parser_accepts_file(file_path, parser):
                    matching_parser = i
                    break
            
            if matching_parser is not None:
                if matching_parser not in assignments:
                    assignments[matching_parser] = []
                assignments[matching_parser].append(file_path)
            else:
                unassigned.append(file_path)
        
        if unassigned:
            # Try to find a generic parser (one with no restrictions)
            for i, parser in enumerate(parsers):
                if not parser.get('mime_types') and not parser.get('file_extensions'):
                    if i not in assignments:
                        assignments[i] = []
                    assignments[i].extend(unassigned)
                    unassigned = []
                    break
        
        if unassigned:
            logger.warning(f"No parser found for files: {unassigned}")
            assignments['unassigned'] = unassigned
        
        return assignments
    
    def _parser_accepts_file(self, file_path: Path, parser_config: Dict[str, Any]) -> bool:
        """
        Check if a parser accepts a specific file.
        
        Args:
            file_path: Path to the file
            parser_config: Parser configuration
            
        Returns:
            True if parser accepts the file
        """
        mime_type, extension = self.get_mime_type(file_path)
        
        # Check MIME types
        parser_mime_types = parser_config.get('mime_types', [])
        if parser_mime_types:
            if not mime_type or mime_type not in parser_mime_types:
                return False
        
        # Check extensions
        parser_extensions = parser_config.get('file_extensions', [])
        if parser_extensions:
            if not extension or extension not in parser_extensions:
                return False
        
        # If no restrictions, parser accepts all files
        if not parser_mime_types and not parser_extensions:
            return True
        
        # If we get here, all checks passed
        return True
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get detailed file information for debugging.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        mime_type, extension = self.get_mime_type(file_path)
        
        return {
            'path': str(file_path),
            'name': file_path.name,
            'extension': extension,
            'mime_type': mime_type,
            'size': file_path.stat().st_size if file_path.exists() else None,
            'exists': file_path.exists()
        }


# Global instance for convenience
mime_filter = MimeTypeFilter()


def filter_files_for_strategy(file_paths: List[Path], strategy_config: Dict[str, Any]) -> List[Path]:
    """
    Convenience function to filter files for a strategy.
    
    Args:
        file_paths: List of file paths
        strategy_config: Strategy configuration
        
    Returns:
        List of accepted file paths
    """
    result = mime_filter.validate_strategy_files(file_paths, strategy_config)
    return result['accepted']


def get_parser_for_file(file_path: Path, parsers: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Convenience function to find the best parser for a file.
    
    Args:
        file_path: Path to the file
        parsers: List of parser configurations
        
    Returns:
        Best matching parser or None
    """
    return mime_filter.find_matching_parser(file_path, parsers)