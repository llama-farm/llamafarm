"""HTML document parser implementation."""

import logging
from typing import List, Dict, Any, Optional
import re
from pathlib import Path

from core.base import Parser, Document

logger = logging.getLogger(__name__)

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    logger.warning("BeautifulSoup4 not available. HTML parsing will use regex fallback.")


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
            logger.warning("BeautifulSoup4 not available. HTML parsing will be limited.")
        return True
    
    def parse(self, content: bytes, **kwargs) -> List[Document]:
        """Parse HTML content into documents."""
        try:
            # Decode bytes to string
            if isinstance(content, bytes):
                text_content = content.decode('utf-8', errors='ignore')
            else:
                text_content = str(content)
            
            if HAS_BS4:
                return self._parse_with_bs4(text_content, **kwargs)
            else:
                return self._parse_with_regex(text_content, **kwargs)
            
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
            return []
    
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
        
        # Extract metadata
        metadata = {
            "type": "html_document",
            "source": source,
            "content_type": "text/html"
        }
        
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
        
        doc_id = f"html_doc_{hash(clean_content) % 10000}"
        
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
        metadata = {
            "type": "html_section",
            "source": source,
            "section": section_num,
            "header": header,
            "content_type": "text/html"
        }
        
        doc_id = f"html_section_{section_num}_{hash(content) % 10000}"
        
        return Document(
            id=doc_id,
            content=content,
            metadata=metadata,
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
        
        metadata = {
            "type": "html_document",
            "source": source,
            "content_type": "text/html",
            "parser_method": "regex_fallback"
        }
        
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
        
        doc_id = f"html_doc_{hash(clean_content) % 10000}"
        
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