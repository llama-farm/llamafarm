# DateTime Extractor

**Framework:** Pure Python (dateutil optional)

**When to use:** Extract dates, times, and temporal expressions from text (deadlines, appointments, historical dates).

**Schema fields:**
- `formats`: List of date formats to recognize (default: common formats)
- `include_relative`: Extract relative dates ("next Monday", "in 3 days")
- `timezone`: Default timezone for parsing
- `return_iso`: Return dates in ISO format

**Best practices:**
- Include domain-specific formats
- Set appropriate timezone
- Handle ambiguous dates consistently
- Extract both absolute and relative dates for completeness