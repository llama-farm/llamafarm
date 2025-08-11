# Link Extractor

**Framework:** Pure Python (regex-based)

**When to use:** Extract URLs, email addresses, file paths, and social media references.

**Schema fields:**
- `extract_urls`: Extract web URLs
- `extract_emails`: Extract email addresses
- `extract_phone_numbers`: Extract phone numbers
- `extract_mentions`: Extract @mentions
- `extract_hashtags`: Extract #hashtags
- `extract_file_paths`: Extract file system paths
- `validate_urls`: Validate URL structure
- `categorize_domains`: Group URLs by domain type

**Best practices:**
- Enable only needed extraction types
- Validate URLs for accuracy
- Categorize domains for filtering
- Extract context for link relevance