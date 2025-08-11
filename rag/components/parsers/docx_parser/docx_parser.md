# DOCX Parser

**Framework:** python-docx

**When to use:** Parse Microsoft Word documents including reports and business documents.

**Schema fields:**
- `extract_images`: Extract embedded images
- `extract_tables`: Extract tables as structured data
- `extract_headers`: Extract document headers/footers
- `extract_comments`: Extract review comments
- `preserve_formatting`: Keep text formatting

**Best practices:**
- Extract tables for data analysis
- Include comments for context
- Extract headers for metadata
- Handle large documents in chunks