"""
Link Extractor for finding URLs, email addresses, and references in text.
"""

import logging
import re
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.core.base import Document
from urllib.parse import urlparse

from rag.components.extractors.base import BaseExtractor

logger = logging.getLogger(__name__)


class LinkExtractor(BaseExtractor):
    """Extract URLs, email addresses, and various reference types from text."""
    
    def __init__(self, name: str = "LinkExtractor", config: Optional[Dict[str, Any]] = None):
        """
        Initialize link extractor.
        
        Args:
            name: Extractor name
            config: Extractor configuration
        """
        super().__init__(name=name, config=config or {})
        
        # Configuration options
        self.extract_urls = self.config.get("extract_urls", True)
        self.extract_emails = self.config.get("extract_emails", True)
        self.extract_phone_numbers = self.config.get("extract_phone_numbers", True)
        self.extract_mentions = self.config.get("extract_mentions", True)  # @mentions
        self.extract_hashtags = self.config.get("extract_hashtags", True)  # #hashtags
        self.extract_file_paths = self.config.get("extract_file_paths", True)
        self.validate_urls = self.config.get("validate_urls", True)
        self.categorize_domains = self.config.get("categorize_domains", True)
        
        # URL patterns
        self.url_pattern = re.compile(
            r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[&\w_.=%-])*)?(?:#(?:[\w._%-])*)?)?',
            re.IGNORECASE
        )
        
        # Email patterns
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        )
        
        # Phone number patterns (various formats)
        self.phone_patterns = [
            re.compile(r'\b(?:\+?1[-.]?)?(?:\([2-9]\d{2}\)[-.]?)?[2-9]\d{2}[-.]?\d{4}\b'),  # US format
            re.compile(r'\b\+\d{1,3}[-.]?\d{1,14}\b'),  # International format
            re.compile(r'\b\d{3}-\d{3}-\d{4}\b'),  # XXX-XXX-XXXX
            re.compile(r'\b\(\d{3}\)\s*\d{3}-\d{4}\b'),  # (XXX) XXX-XXXX
        ]
        
        # Social media patterns
        self.mention_pattern = re.compile(r'@[A-Za-z0-9_]+')
        self.hashtag_pattern = re.compile(r'#[A-Za-z0-9_]+')
        
        # File path patterns
        self.file_path_patterns = [
            re.compile(r'[A-Za-z]:\\(?:[^\\/:*?"<>|\r\n]+\\)*[^\\/:*?"<>|\r\n]*'),  # Windows paths
            re.compile(r'/(?:[^/\0]+/)*[^/\0]*'),  # Unix paths
            re.compile(r'\\[\w.-]+\\[\w.$-]+(?:\\[\w.$-]+)*'),  # UNC paths
        ]
    
    def extract(self, documents: List["Document"]) -> List["Document"]:
        """
        Extract links and references from documents.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of enhanced documents with link metadata
        """
        from rag.core.base import Document
        
        for doc in documents:
            try:
                link_data = self._extract_from_text(doc.content, doc.metadata)
                
                # Add to extractors metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}
                
                doc.metadata["extractors"]["links"] = link_data
                
                # Also add a simplified list of links for easy access
                if "urls" in link_data:
                    doc.metadata["links"] = link_data["urls"]
                else:
                    doc.metadata["links"] = []
                
                self.logger.debug(f"Extracted links for document {doc.id}")
                
            except Exception as e:
                self.logger.error(f"Link extraction failed for document {doc.id}: {e}")
        
        return documents
    
    def get_dependencies(self) -> List[str]:
        """Get required dependencies for this extractor."""
        return []  # No external dependencies
    
    def _extract_from_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract links and references from text.
        
        Args:
            text: Input text to analyze
            metadata: Optional metadata context
            
        Returns:
            Dictionary containing extracted link information
        """
        try:
            results = {
                "extractor": "LinkExtractor"
            }
            
            # Extract URLs
            if self.extract_urls:
                urls = self._extract_urls(text)
                results["urls"] = urls
                results["url_count"] = len(urls)
                
                if self.categorize_domains:
                    results["domain_categories"] = self._categorize_domains(urls)
            
            # Extract email addresses
            if self.extract_emails:
                emails = self._extract_emails(text)
                results["emails"] = emails
                results["email_count"] = len(emails)
                results["email_domains"] = list(set(email.split('@')[1] for email in emails if '@' in email))
            
            # Extract phone numbers
            if self.extract_phone_numbers:
                phones = self._extract_phone_numbers(text)
                results["phone_numbers"] = phones
                results["phone_count"] = len(phones)
            
            # Extract social media references
            if self.extract_mentions:
                mentions = self._extract_mentions(text)
                results["mentions"] = mentions
                results["mention_count"] = len(mentions)
            
            if self.extract_hashtags:
                hashtags = self._extract_hashtags(text)
                results["hashtags"] = hashtags
                results["hashtag_count"] = len(hashtags)
            
            # Extract file paths
            if self.extract_file_paths:
                file_paths = self._extract_file_paths(text)
                results["file_paths"] = file_paths
                results["file_path_count"] = len(file_paths)
            
            # Generate summary statistics
            results["total_links"] = sum([
                results.get("url_count", 0),
                results.get("email_count", 0),
                results.get("phone_count", 0),
                results.get("mention_count", 0),
                results.get("hashtag_count", 0),
                results.get("file_path_count", 0)
            ])
            
            return results
            
        except Exception as e:
            logger.error(f"Error in link extraction: {e}")
            return {"error": str(e), "extractor": "LinkExtractor"}
    
    def _extract_urls(self, text: str) -> List[Dict[str, Any]]:
        """Extract and analyze URLs from text."""
        urls = []
        
        for match in self.url_pattern.finditer(text):
            url = match.group(0)
            url_info = {
                "url": url,
                "position": match.span(),
                "context": self._get_context(text, match.span())
            }
            
            # Parse URL components
            if self.validate_urls:
                try:
                    parsed = urlparse(url)
                    url_info.update({
                        "scheme": parsed.scheme,
                        "domain": parsed.netloc,
                        "path": parsed.path,
                        "query": parsed.query,
                        "fragment": parsed.fragment,
                        "is_valid": True
                    })
                except Exception:
                    url_info["is_valid"] = False
            
            urls.append(url_info)
        
        return urls
    
    def _extract_emails(self, text: str) -> List[str]:
        """Extract email addresses from text."""
        emails = []
        
        for match in self.email_pattern.finditer(text):
            email = match.group(0).lower()
            if email not in emails:  # Avoid duplicates
                emails.append(email)
        
        return emails
    
    def _extract_phone_numbers(self, text: str) -> List[Dict[str, Any]]:
        """Extract phone numbers from text."""
        phones = []
        found_numbers = set()  # To avoid duplicates
        
        for pattern in self.phone_patterns:
            for match in pattern.finditer(text):
                number = match.group(0)
                normalized = self._normalize_phone_number(number)
                
                if normalized not in found_numbers:
                    found_numbers.add(normalized)
                    phones.append({
                        "original": number,
                        "normalized": normalized,
                        "position": match.span(),
                        "context": self._get_context(text, match.span())
                    })
        
        return phones
    
    def _extract_mentions(self, text: str) -> List[str]:
        """Extract @mentions from text."""
        mentions = []
        
        for match in self.mention_pattern.finditer(text):
            mention = match.group(0)
            if len(mention) > 1 and mention not in mentions:  # Avoid duplicates and single @
                mentions.append(mention)
        
        return mentions
    
    def _extract_hashtags(self, text: str) -> List[str]:
        """Extract #hashtags from text."""
        hashtags = []
        
        for match in self.hashtag_pattern.finditer(text):
            hashtag = match.group(0)
            if len(hashtag) > 1 and hashtag not in hashtags:  # Avoid duplicates and single #
                hashtags.append(hashtag)
        
        return hashtags
    
    def _extract_file_paths(self, text: str) -> List[Dict[str, Any]]:
        """Extract file paths from text."""
        file_paths = []
        found_paths = set()
        
        for pattern in self.file_path_patterns:
            for match in pattern.finditer(text):
                path = match.group(0)
                
                # Basic validation - should have file extension or be a directory
                if (('.' in path and len(path.split('.')[-1]) <= 5) or 
                    path.endswith('/') or path.endswith('\\')):
                    
                    if path not in found_paths:
                        found_paths.add(path)
                        file_paths.append({
                            "path": path,
                            "type": self._classify_file_path(path),
                            "position": match.span(),
                            "context": self._get_context(text, match.span())
                        })
        
        return file_paths
    
    def _categorize_domains(self, urls: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Categorize URLs by domain type."""
        categories = {
            "social_media": [],
            "documentation": [],
            "repositories": [],
            "academic": [],
            "government": [],
            "commercial": [],
            "other": []
        }
        
        # Define domain patterns for categorization
        domain_patterns = {
            "social_media": [
                "facebook.com", "twitter.com", "instagram.com", "linkedin.com",
                "youtube.com", "tiktok.com", "snapchat.com", "reddit.com"
            ],
            "documentation": [
                "docs.", "documentation.", "wiki.", "confluence.", "notion."
            ],
            "repositories": [
                "github.com", "gitlab.com", "bitbucket.org", "sourceforge.net"
            ],
            "academic": [
                ".edu", "arxiv.org", "scholar.google.", "researchgate.net"
            ],
            "government": [
                ".gov", ".mil"
            ]
        }
        
        for url_info in urls:
            domain = url_info.get("domain", "").lower()
            categorized = False
            
            for category, patterns in domain_patterns.items():
                for pattern in patterns:
                    if pattern in domain:
                        categories[category].append(url_info["url"])
                        categorized = True
                        break
                if categorized:
                    break
            
            if not categorized:
                if any(tld in domain for tld in [".com", ".org", ".net", ".biz"]):
                    categories["commercial"].append(url_info["url"])
                else:
                    categories["other"].append(url_info["url"])
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def _normalize_phone_number(self, phone: str) -> str:
        """Normalize phone number format."""
        # Remove all non-digit characters except +
        normalized = re.sub(r'[^\d+]', '', phone)
        
        # Handle US numbers
        if normalized.startswith('1') and len(normalized) == 11:
            return f"+{normalized}"
        elif len(normalized) == 10:
            return f"+1{normalized}"
        elif normalized.startswith('+'):
            return normalized
        else:
            return f"+{normalized}"
    
    def _classify_file_path(self, path: str) -> str:
        """Classify file path type."""
        if path.startswith('http'):
            return "url_path"
        elif path.startswith('\\\\'):
            return "unc_path"
        elif ':' in path and len(path) > 2 and path[1] == ':':
            return "windows_path"
        elif path.startswith('/'):
            return "unix_path"
        else:
            return "relative_path"
    
    def _get_context(self, text: str, span: tuple, context_size: int = 50) -> str:
        """Get surrounding context for a matched item."""
        start, end = span
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        
        context = text[context_start:context_end]
        # Mark the actual match
        match_start = start - context_start
        match_end = end - context_start
        
        return (
            context[:match_start] + 
            "[" + context[match_start:match_end] + "]" + 
            context[match_end:]
        ).replace('\n', ' ').strip()
    
    @staticmethod
    def get_supported_types() -> List[str]:
        """Get list of supported link types."""
        return ["urls", "emails", "phone_numbers", "mentions", "hashtags", "file_paths"]
    
    @staticmethod
    def get_description() -> str:
        """Get extractor description."""
        return "Extract URLs, email addresses, phone numbers, and other references from text"