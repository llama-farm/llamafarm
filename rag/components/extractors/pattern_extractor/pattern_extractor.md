# Pattern Extractor

**Framework:** Pure Python (regex-based)

**When to use:** Extract specific patterns like IPs, credit cards, SSNs, or custom patterns.

**Schema fields:**
- `patterns`: List of patterns to extract (email, phone, url, ip_address, custom)
- `custom_patterns`: Dict of custom regex patterns
- `include_context`: Include surrounding text
- `context_window`: Characters of context to include

**Best practices:**
- Use predefined patterns when possible
- Test custom regex thoroughly
- Include context for validation
- Be mindful of PII extraction