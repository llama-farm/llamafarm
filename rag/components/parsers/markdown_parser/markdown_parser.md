# Markdown Parser

**Framework:** markdown, mistune

**When to use:** Parse markdown documentation, README files, and technical docs.

**Schema fields:**
- `extract_headers`: Extract heading structure
- `extract_links`: Extract hyperlinks
- `extract_code_blocks`: Extract code snippets
- `preserve_structure`: Maintain document hierarchy
- `chunk_by_header`: Split by heading sections

**Best practices:**
- Extract headers for navigation
- Preserve code blocks with language
- Chunk by headers for coherent sections
- Extract links for reference tracking