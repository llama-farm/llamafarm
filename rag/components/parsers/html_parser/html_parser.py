"""HTML document parser implementation."""

import logging
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

from core.base import Parser, Document, ProcessingResult
# Import hash utilities for deduplication
from utils.hash_utils import (
    generate_document_metadata,
    generate_chunk_metadata,
    DeduplicationTracker
)

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None  # Type hint placeholder
    logger.debug("BeautifulSoup4 not available. HTML parsing will use regex fallback.")


class HtmlParser(Parser):
    """Parser for HTML documents."""
    
    def __init__(self, name: str = "HtmlParser", config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        config = config or {}
        self.extract_links = config.get("extract_links", True)
        self.extract_images = config.get("extract_images", True)
        self.extract_meta_tags = config.get("extract_meta_tags", True)
        self.preserve_structure = config.get("preserve_structure", False)
        self.remove_scripts = config.get("remove_scripts", True)
        self.remove_styles = config.get("remove_styles", True)
        self.extract_tables = config.get("extract_tables", False)
    
    def validate_config(self) -> bool:
        """Validate parser configuration."""
        if not HAS_BS4:
            logger.debug("BeautifulSoup4 not available. HTML parsing will be limited.")
        return True
    
    def parse(self, content: str, **kwargs) -> ProcessingResult:
        """Parse HTML content into documents."""
        errors = []
        documents = []
        
        try:
            # Handle file path or content string
            if Path(content).exists():
                with open(content, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
            else:
                text_content = str(content)
            
            if HAS_BS4:
                documents = self._parse_with_bs4(text_content, **kwargs)
            else:
                documents = self._parse_with_regex(text_content, **kwargs)
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            errors.append({"error": str(e), "source": content})
            
        return ProcessingResult(documents=documents, errors=errors)
    
    def _parse_with_bs4(self, content: str, **kwargs) -> List[Document]:
        """Parse HTML using BeautifulSoup."""
        source = kwargs.get('source', 'unknown')
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove unwanted elements
        if self.remove_scripts:
            for script in soup(["script", "style"]):
                script.decompose()
        
        if self.remove_styles:
            for style in soup(["style"]):
                style.decompose()
        
        documents = []
        
        if self.preserve_structure:
            # Split by major sections
            documents = self._parse_by_sections_bs4(soup, source)
        else:
            # Create single document
            documents = [self._create_single_document_bs4(soup, source)]
        
        return documents
    
    def _parse_by_sections_bs4(self, soup: 'BeautifulSoup', source: str) -> List[Document]:
        """Parse HTML by sections using BeautifulSoup."""
        documents = []
        section_count = 0
        
        # Try to find main content areas
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main', 'article'])
        
        if main_content:
            # Split by headers within main content
            headers = main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            
            if headers:
                current_section = []
                current_header = ""
                
                for element in main_content.descendants:
                    if hasattr(element, 'name') and element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        # Save previous section
                        if current_section:
                            section_text = ' '.join(current_section).strip()
                            if section_text:
                                doc = self._create_section_document_bs4(
                                    section_text, current_header, section_count, source, soup
                                )
                                documents.append(doc)
                                section_count += 1
                        
                        # Start new section
                        current_header = element.get_text().strip()
                        current_section = []
                    elif hasattr(element, 'string') and element.string:
                        text = element.string.strip()
                        if text:
                            current_section.append(text)
                
                # Handle last section
                if current_section:
                    section_text = ' '.join(current_section).strip()
                    if section_text:
                        doc = self._create_section_document_bs4(
                            section_text, current_header, section_count, source, soup
                        )
                        documents.append(doc)
        
        if not documents:
            # Fallback to single document
            documents = [self._create_single_document_bs4(soup, source)]
        
        return documents
    
    def _create_single_document_bs4(self, soup: 'BeautifulSoup', source: str) -> Document:
        """Create single document using BeautifulSoup."""
        # Extract text content
        text_content = soup.get_text()
        clean_content = self._clean_text_content(text_content)
        
        # Generate comprehensive metadata with hash utilities
        base_metadata = generate_document_metadata(source, clean_content)
        
        # Add HTML-specific metadata
        metadata = base_metadata.copy()
        metadata.update({
            "type": "html_document",
            "content_type": "text/html",
            "parser_type": "HtmlParser"
        })
        
        # Extract title
        title_tag = soup.find('title')
        if title_tag:
            metadata["title"] = title_tag.get_text().strip()
        
        # Extract meta tags
        if self.extract_meta_tags:
            meta_tags = self._extract_meta_tags_bs4(soup)
            if meta_tags:
                metadata.update(meta_tags)
        
        # Extract links
        if self.extract_links:
            links = self._extract_links_bs4(soup)
            if links:
                metadata["links"] = links
                metadata["link_count"] = len(links)
        
        # Extract images
        if self.extract_images:
            images = self._extract_images_bs4(soup)
            if images:
                metadata["images"] = images
                metadata["image_count"] = len(images)
        
        # Extract tables
        if self.extract_tables:
            tables = self._extract_tables_bs4(soup)
            if tables:
                metadata["table_count"] = len(tables)
        
        # Use hash-based ID from metadata
        doc_id = f"doc_{metadata['document_hash'][:12]}_full"
        
        return Document(
            id=doc_id,
            content=clean_content,
            metadata=metadata,
            source=source
        )
    
    def _create_section_document_bs4(
        self, content: str, header: str, section_num: int, source: str, soup: 'BeautifulSoup'
    ) -> Document:
        """Create section document using BeautifulSoup."""
        # Generate base metadata with hash utilities
        base_metadata = generate_document_metadata(source, content)
        
        # Generate chunk metadata for section
        chunk_metadata = generate_chunk_metadata(
            base_metadata,
            content,
            section_num,
            1  # We'll update total_chunks later if needed
        )
        
        # Add HTML section-specific metadata
        chunk_metadata.update({
            "type": "html_section",
            "header": header,
            "content_type": "text/html",
            "parser_type": "HtmlParser"
        })
        
        # Use chunk ID from metadata
        doc_id = chunk_metadata["chunk_id"]
        
        return Document(
            id=doc_id,
            content=content,
            metadata=chunk_metadata,
            source=source
        )
    
    def _extract_meta_tags_bs4(self, soup: 'BeautifulSoup') -> Dict[str, str]:
        """Extract meta tags using BeautifulSoup."""
        meta_data = {}
        
        # Standard meta tags
        for meta in soup.find_all('meta'):
            name = meta.get('name') or meta.get('property')
            content = meta.get('content')
            
            if name and content:
                meta_data[f"meta_{name}"] = content
        
        return meta_data
    
    def _extract_links_bs4(self, soup: 'BeautifulSoup') -> List[Dict[str, str]]:
        """Extract links using BeautifulSoup."""
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link.get('href')
            text = link.get_text().strip()
            
            if href and text:
                links.append({
                    "text": text,
                    "url": href
                })
        
        return links
    
    def _extract_images_bs4(self, soup: 'BeautifulSoup') -> List[Dict[str, str]]:
        """Extract images using BeautifulSoup."""
        images = []
        
        for img in soup.find_all('img', src=True):
            src = img.get('src')
            alt = img.get('alt', '')
            title = img.get('title', '')
            
            if src:
                image_data = {"url": src}
                if alt:
                    image_data["alt"] = alt
                if title:
                    image_data["title"] = title
                
                images.append(image_data)
        
        return images
    
    def _extract_tables_bs4(self, soup: 'BeautifulSoup') -> List[Dict[str, Any]]:
        """Extract tables using BeautifulSoup."""
        tables = []
        
        for table in soup.find_all('table'):
            rows = []
            for tr in table.find_all('tr'):
                cells = []
                for td in tr.find_all(['td', 'th']):
                    cells.append(td.get_text().strip())
                if cells:
                    rows.append(cells)
            
            if rows:
                tables.append({
                    "rows": len(rows),
                    "columns": len(rows[0]) if rows else 0,
                    "data": rows[:5]  # Store first 5 rows as sample
                })
        
        return tables
    
    def _parse_with_regex(self, content: str, **kwargs) -> List[Document]:
        """Parse HTML using regex fallback."""
        source = kwargs.get('source', 'unknown')
        
        # Basic HTML cleaning with regex
        clean_content = self._clean_html_with_regex(content)
        
        # Generate comprehensive metadata with hash utilities
        base_metadata = generate_document_metadata(source, clean_content)
        
        # Add HTML-specific metadata
        metadata = base_metadata.copy()
        metadata.update({
            "type": "html_document",
            "content_type": "text/html",
            "parser_type": "HtmlParser",
            "parser_method": "regex_fallback"
        })
        
        # Extract basic metadata with regex
        title_match = re.search(r'<title[^>]*>([^<]+)</title>', content, re.IGNORECASE)
        if title_match:
            metadata["title"] = title_match.group(1).strip()
        
        # Extract links with regex
        if self.extract_links:
            links = self._extract_links_regex(content)
            if links:
                metadata["links"] = links
                metadata["link_count"] = len(links)
        
        # Use hash-based ID from metadata
        doc_id = f"doc_{metadata['document_hash'][:12]}_full"
        
        return [Document(
            id=doc_id,
            content=clean_content,
            metadata=metadata,
            source=source
        )]
    
    def _clean_html_with_regex(self, content: str) -> str:
        """Clean HTML content using regex."""
        # Remove scripts and styles
        content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove HTML tags
        content = re.sub(r'<[^>]+>', ' ', content)
        
        # Decode HTML entities
        content = content.replace('&nbsp;', ' ')
        content = content.replace('&amp;', '&')
        content = content.replace('&lt;', '<')
        content = content.replace('&gt;', '>')
        content = content.replace('&quot;', '"')
        content = content.replace('&#39;', "'")
        
        return self._clean_text_content(content)
    
    def _extract_links_regex(self, content: str) -> List[Dict[str, str]]:
        """Extract links using regex."""
        links = []
        
        # Find links with regex
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
        for match in re.finditer(link_pattern, content, re.IGNORECASE):
            href = match.group(1)
            text = match.group(2).strip()
            
            if href and text:
                links.append({
                    "text": text,
                    "url": href
                })
        
        return links
    
    def _clean_text_content(self, content: str) -> str:
        """Clean extracted text content."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        return content.strip()
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get supported file extensions."""
        return ['.html', '.htm', '.xhtml']
    
    @classmethod
    def get_description(cls) -> str:
        """Get parser description."""
        return "HTML document parser that extracts text content, links, images, and metadata."