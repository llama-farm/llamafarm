# Entity Extractor

**Framework:** spaCy (with regex fallback)

**When to use:** Extract named entities like people, organizations, locations, emails, and phone numbers.

**Schema fields:**
- `entity_types`: List of entity types to extract (PERSON, ORG, GPE, etc.)
- `use_fallback`: Use regex patterns if spaCy unavailable
- `min_entity_length`: Minimum character length for entities
- `include_context`: Include surrounding text context

**Best practices:**
- Specify only needed entity types for performance
- Use fallback for environments without spaCy
- Set min_length to filter noise
- Enable context for disambiguation