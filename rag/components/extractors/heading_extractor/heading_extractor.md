# Heading Extractor

**Framework:** Pure Python

**When to use:** Extract document structure from headings and subheadings in markdown, HTML, or text.

**Schema fields:**
- `extract_hierarchy`: Build heading tree structure
- `include_level`: Include heading level (H1, H2, etc.)
- `min_level`: Minimum heading level to extract
- `max_level`: Maximum heading level to extract

**Best practices:**
- Extract hierarchy for table of contents
- Set level limits based on document depth
- Use for document navigation
- Combine with content chunking by sections