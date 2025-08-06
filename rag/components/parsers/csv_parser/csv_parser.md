# CSV Parser

**Framework:** Pure Python (pandas optional)

**When to use:** Parse CSV files including support tickets, data exports, and tabular data.

**Schema fields:**
- `content_fields`: Fields to use as document content
- `metadata_fields`: Fields to include as metadata
- `id_field`: Field to use as document ID
- `delimiter`: CSV delimiter (default: comma)
- `encoding`: File encoding

**Best practices:**
- Specify content fields explicitly
- Map all relevant fields to metadata
- Use unique ID field when available
- Handle encoding for international data