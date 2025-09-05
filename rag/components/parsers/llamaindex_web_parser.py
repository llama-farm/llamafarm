"""LlamaIndex-based Web/HTML Parser."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_parser import BaseLlamaIndexParser, ParserFactory

logger = logging.getLogger(__name__)

try:
    from llama_index.readers.web import BeautifulSoupWebReader
    LLAMA_WEB_READER_AVAILABLE = True
except ImportError:
    LLAMA_WEB_READER_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class LlamaIndexWebParser(BaseLlamaIndexParser):
    """LlamaIndex-based parser for web content and HTML files."""
    
    def __init__(self, name: str = "LlamaIndexWebParser", config: Optional[Dict[str, Any]] = None):
        """
        Initialize LlamaIndex web parser.
        
        Args:
            name: Parser name
            config: Parser configuration
        """
        super().__init__(name=name, config=config or {})
        
        # Web/HTML-specific configuration
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_links = self.config.get("extract_links", True)
        self.extract_images = self.config.get("extract_images", True)
        self.preserve_structure = self.config.get("preserve_structure", False)
        self.remove_scripts = self.config.get("remove_scripts", True)
        self.remove_styles = self.config.get("remove_styles", True)
        self.text_only = self.config.get("text_only", False)
        
        # Request settings for URLs
        self.request_timeout = self.config.get("request_timeout", 30)
        self.user_agent = self.config.get("user_agent", 
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36")
        
        if not BS4_AVAILABLE:
            raise ImportError("BeautifulSoup4 is required for HTML parsing. Install with: pip install beautifulsoup4")
    
    def _get_reader(self):
        """Get the appropriate LlamaIndex reader for web content."""
        if LLAMA_WEB_READER_AVAILABLE:
            return BeautifulSoupWebReader()
        else:
            # Will use manual parsing
            return None
    
    def parse(self, source: str, **kwargs) -> "ProcessingResult":
        """
        Parse a web URL or HTML file using LlamaIndex.
        
        Args:
            source: URL or path to HTML file
            **kwargs: Additional parsing options
            
        Returns:
            ProcessingResult containing parsed documents
        """
        try:
            # Determine if source is URL or file path
            if source.startswith(('http://', 'https://')):
                return self._parse_url(source, **kwargs)
            else:
                return self._parse_html_file(Path(source), **kwargs)
        
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to parse web content: {str(e)}",
                    "source": str(source)
                }]
            )
    
    def _parse_url(self, url: str, **kwargs) -> "ProcessingResult":
        """Parse content from a URL."""
        try:
            if self._reader:
                return self._parse_url_with_llamaindex(url, **kwargs)
            else:
                return self._parse_url_with_manual(url, **kwargs)
        
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to parse URL: {str(e)}",
                    "source": url
                }]
            )
    
    def _parse_html_file(self, file_path: Path, **kwargs) -> "ProcessingResult":
        """Parse content from an HTML file."""
        if not file_path.exists():
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {file_path}", "source": str(file_path)}]
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                html_content = f.read()
            
            return self._parse_html_content(html_content, str(file_path), is_file=True)
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to parse HTML file: {str(e)}",
                    "source": str(file_path)
                }]
            )
    
    def _parse_url_with_llamaindex(self, url: str, **kwargs) -> "ProcessingResult":
        """Parse URL using LlamaIndex BeautifulSoupWebReader."""
        try:
            documents = self._reader.load_data([url])
            
            if not documents:
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No content extracted by LlamaIndex", "source": url}]
                )
            
            return self._process_llamaindex_documents(documents, url)
            
        except Exception as e:
            logger.warning(f"LlamaIndex web reader failed: {e}, falling back to manual parsing")
            return self._parse_url_with_manual(url, **kwargs)
    
    def _parse_url_with_manual(self, url: str, **kwargs) -> "ProcessingResult":
        """Parse URL using manual requests and BeautifulSoup."""
        if not REQUESTS_AVAILABLE:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": "requests library required for URL parsing",
                    "source": url
                }]
            )
        
        try:
            headers = {'User-Agent': self.user_agent}
            response = requests.get(url, headers=headers, timeout=self.request_timeout)
            response.raise_for_status()
            
            return self._parse_html_content(response.text, url, is_file=False)
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"Failed to fetch URL: {str(e)}",
                    "source": url
                }]
            )
    
    def _parse_html_content(self, html_content: str, source: str, is_file: bool = False) -> "ProcessingResult":
        """Parse HTML content using BeautifulSoup."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove scripts and styles if requested
            if self.remove_scripts:
                for script in soup(["script", "noscript"]):
                    script.decompose()
            
            if self.remove_styles:
                for style in soup(["style"]):
                    style.decompose()
            
            # Extract metadata
            metadata = self._extract_html_metadata(soup, source, is_file)
            
            # Extract main content
            content = self._extract_main_content(soup)
            
            if not content.strip():
                from core.base import ProcessingResult
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No text content found", "source": source}]
                )
            
            return self._create_documents_from_content(content, metadata, source)
            
        except Exception as e:
            from core.base import ProcessingResult
            return ProcessingResult(
                documents=[],
                errors=[{
                    "error": f"HTML parsing failed: {str(e)}",
                    "source": source
                }]
            )
    
    def _process_llamaindex_documents(self, documents, source: str) -> "ProcessingResult":
        """Process LlamaIndex documents into our format."""
        from core.base import ProcessingResult, Document
        from utils.hash_utils import generate_document_metadata, generate_chunk_metadata
        
        result_documents = []
        errors = []
        
        # Combine all content if multiple documents
        if len(documents) > 1:
            combined_content = "\n\n".join(doc.text for doc in documents if doc.text)
        else:
            combined_content = documents[0].text if documents else ""
        
        if not combined_content.strip():
            return ProcessingResult(
                documents=[],
                errors=[{"error": "No content extracted", "source": source}]
            )
        
        # Generate metadata
        base_metadata = generate_document_metadata(source, combined_content)
        base_metadata.update({
            "parser_type": self.name,
            "reader_type": "llamaindex_web",
            "content_type": "web" if source.startswith('http') else "html_file"
        })
        
        # Add LlamaIndex metadata from first document
        if documents[0].metadata:
            base_metadata.update(documents[0].metadata)
        
        # Handle chunking
        if self._node_parser and self.chunk_size:
            chunked_docs = self._create_chunked_documents(
                combined_content, base_metadata, documents[0]
            )
            result_documents.extend(chunked_docs)
        else:
            document_id = f"doc_{base_metadata['document_hash'][:12]}_full"
            doc = Document(
                content=combined_content,
                metadata=base_metadata,
                id=document_id,
                source=source
            )
            result_documents.append(doc)
        
        return ProcessingResult(
            documents=result_documents,
            errors=errors,
            metrics={
                "total_documents": len(result_documents),
                "total_errors": len(errors),
                "source_processed": source,
                "parser_type": self.name
            }
        )
    
    def _extract_html_metadata(self, soup: BeautifulSoup, source: str, is_file: bool) -> Dict[str, Any]:
        """Extract metadata from HTML document."""
        from utils.hash_utils import generate_document_metadata
        
        # Extract text content for hash generation
        text_content = self._extract_main_content(soup)
        
        metadata = generate_document_metadata(source, text_content)
        metadata.update({
            "parser_type": self.name,
            "content_type": "html_file" if is_file else "web_page"
        })
        
        if is_file:
            file_path = Path(source)
            metadata.update({
                "file_name": file_path.name,
                "file_size": file_path.stat().st_size
            })
        
        # Extract HTML metadata
        try:
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.get_text().strip()
            
            # Meta tags
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property')
                content = meta.get('content')
                
                if name and content:
                    key = f"meta_{name.lower().replace(':', '_')}"
                    metadata[key] = content
            
            # Language
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                metadata["language"] = html_tag.get('lang')
            
            # Extract links if requested
            if self.extract_links:
                links = []
                for link in soup.find_all('a', href=True):
                    links.append({
                        "text": link.get_text().strip()[:100],  # Limit length
                        "url": link['href'],
                        "title": link.get('title', '')
                    })
                metadata["extracted_links"] = links[:50]  # Limit number
                metadata["link_count"] = len(links)
            
            # Extract images if requested
            if self.extract_images:
                images = []
                for img in soup.find_all('img', src=True):
                    images.append({
                        "src": img['src'],
                        "alt": img.get('alt', ''),
                        "title": img.get('title', '')
                    })
                metadata["extracted_images"] = images[:20]  # Limit number
                metadata["image_count"] = len(images)
            
            # Document structure info
            metadata.update({
                "heading_count": len(soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])),
                "paragraph_count": len(soup.find_all('p')),
                "list_count": len(soup.find_all(['ul', 'ol'])),
                "table_count": len(soup.find_all('table'))
            })
            
        except Exception as e:
            logger.warning(f"Failed to extract HTML metadata: {e}")
        
        return metadata
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main text content from HTML."""
        if self.text_only:
            return soup.get_text(separator='\n', strip=True)
        
        # Try to find main content areas
        main_selectors = [
            'main', 'article', '.main', '.content', '.post', '.entry',
            '#main', '#content', '#post', '#entry'
        ]
        
        main_content = None
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                main_content = element
                break
        
        if not main_content:
            # Remove header, footer, nav, sidebar elements
            for tag in soup.find_all(['header', 'footer', 'nav', 'aside', '.sidebar', '.nav']):
                tag.decompose()
            main_content = soup.find('body') or soup
        
        # Extract text with some structure preservation
        if self.preserve_structure:
            content_parts = []
            
            for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'li']):
                text = element.get_text(strip=True)
                if text:
                    if element.name.startswith('h'):
                        content_parts.append(f"\n# {text}\n")
                    elif element.name == 'li':
                        content_parts.append(f"- {text}")
                    else:
                        content_parts.append(text)
            
            return '\n'.join(content_parts)
        else:
            return main_content.get_text(separator='\n', strip=True)
    
    def _create_documents_from_content(self, content: str, metadata: Dict, source: str) -> "ProcessingResult":
        """Create documents from extracted content."""
        from core.base import ProcessingResult, Document
        
        result_documents = []
        
        # Handle chunking
        if self._node_parser and self.chunk_size:
            # Create mock LlamaIndex document for chunking
            from llama_index.core.schema import Document as LlamaDocument
            llama_doc = LlamaDocument(text=content, metadata=metadata)
            
            chunked_docs = self._create_chunked_documents(
                content, metadata, llama_doc
            )
            result_documents.extend(chunked_docs)
        else:
            document_id = f"doc_{metadata['document_hash'][:12]}_full"
            doc = Document(
                content=content,
                metadata=metadata,
                id=document_id,
                source=source
            )
            result_documents.append(doc)
        
        return ProcessingResult(
            documents=result_documents,
            errors=[],
            metrics={
                "total_documents": len(result_documents),
                "source_processed": source,
                "parser_type": self.name
            }
        )
    
    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        # Can parse HTML files or URLs
        if file_path.startswith(('http://', 'https://')):
            return True
        return Path(file_path).suffix.lower() in self.get_supported_extensions()
    
    @staticmethod
    def can_parse_mime_type(mime_type: str) -> bool:
        """Check if this parser can handle the given MIME type."""
        return mime_type in ['text/html', 'application/xhtml+xml']
    
    @staticmethod
    def get_supported_extensions() -> List[str]:
        """Get list of supported file extensions."""
        return ['.html', '.htm']
    
    @staticmethod
    def get_description() -> str:
        """Get parser description."""
        return "LlamaIndex-based parser for web content and HTML files"


# Register the parser
ParserFactory.register_parser("LlamaIndexWebParser", LlamaIndexWebParser)