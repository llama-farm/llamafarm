# Table Extractor

**Framework:** Pure Python (BeautifulSoup for HTML)

**When to use:** Extract tables from HTML, markdown, or structured text documents.

**Schema fields:**
- `format`: Expected table format (html, markdown, csv)
- `extract_headers`: Extract column headers
- `convert_to_dict`: Convert tables to dictionaries
- `min_columns`: Minimum columns to consider as table
- `min_rows`: Minimum rows to consider as table

**Best practices:**
- Specify expected format for accuracy
- Set minimum dimensions to filter noise
- Extract headers for column identification
- Convert to dict for easier processing