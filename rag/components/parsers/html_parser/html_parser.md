# HTML Parser

**Framework:** BeautifulSoup4, lxml

**When to use:** Parse web pages, documentation sites, and HTML exports.

**Schema fields:**
- `extract_links`: Extract hyperlinks
- `extract_images`: Extract image references
- `extract_meta_tags`: Extract HTML metadata
- `preserve_structure`: Keep DOM structure
- `remove_scripts`: Remove JavaScript
- `remove_styles`: Remove CSS

**Best practices:**
- Remove scripts/styles for clean text
- Extract meta tags for metadata
- Preserve structure for navigation
- Handle encoding properly