# Excel Parser

**Framework:** openpyxl, pandas

**When to use:** Parse Excel spreadsheets with multiple sheets and complex data.

**Schema fields:**
- `sheet_names`: Specific sheets to parse (None for all)
- `header_row`: Row containing headers
- `data_only`: Extract values only (not formulas)
- `merge_sheets`: Combine sheets into one document
- `include_formulas`: Include formula definitions

**Best practices:**
- Specify sheets to avoid noise
- Use header_row for column names
- Extract values for data analysis
- Merge related sheets carefully