# PDF Parser

**Framework:** PyPDF2, pdfplumber, or PyMuPDF

**When to use:** Extract text from PDF documents including reports, papers, and scanned documents.

**Schema fields:**
- `extract_images`: Extract embedded images
- `ocr_enabled`: Use OCR for scanned PDFs
- `preserve_layout`: Maintain text layout
- `extract_metadata`: Extract PDF metadata
- `page_separator`: Separator between pages

**Best practices:**
- Enable OCR only when needed (performance)
- Extract metadata for document info
- Preserve layout for tables/forms
- Process by page for large PDFs