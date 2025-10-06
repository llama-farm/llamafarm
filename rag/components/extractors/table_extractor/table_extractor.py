"""
Table Extractor for finding and extracting structured tabular data from documents.
"""

import re
from typing import Any, Dict, List, Optional

from components.extractors.base import BaseExtractor
from core.base import Document


class TableExtractor(BaseExtractor):
    """Extract structured table data from text content."""

    def __init__(
        self, name: str = "TableExtractor", config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize table extractor.

        Args:
            name: Extractor name
            config: Extractor configuration
        """
        super().__init__(name, config)

        # Configuration options
        self.min_columns = self.config.get("min_columns", 2)
        self.min_rows = self.config.get("min_rows", 2)
        self.detect_headers = self.config.get("detect_headers", True)
        self.table_formats = self.config.get("table_formats", ["pipe", "grid", "csv"])
        self.max_cell_length = self.config.get("max_cell_length", 200)

    def extract(self, documents: List[Document]) -> List[Document]:
        """
        Extract table data from documents.

        Args:
            documents: List of documents to process

        Returns:
            List of enhanced documents with table metadata
        """
        for doc in documents:
            try:
                table_data = self._extract_from_text(doc.content, doc.metadata)

                # Add to metadata
                if "extractors" not in doc.metadata:
                    doc.metadata["extractors"] = {}

                doc.metadata["extractors"]["tables"] = table_data

                # Add simplified data for easy access
                doc.metadata["tables"] = table_data.get("tables", [])
                doc.metadata["table_count"] = table_data.get("table_count", 0)

                self.logger.debug(
                    f"Extracted {table_data.get('table_count', 0)} tables from document {doc.id}"
                )

            except Exception as e:
                self.logger.error(f"Table extraction failed for document {doc.id}: {e}")

        return documents

    def get_dependencies(self) -> List[str]:
        """Get required dependencies for this extractor."""
        return []  # No external dependencies

    def _extract_from_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Extract table data from text.

        Args:
            text: Input text to analyze
            metadata: Optional metadata context

        Returns:
            Dictionary containing extracted table information
        """
        try:
            tables = []

            # Look for different table formats
            if "pipe" in self.table_formats:
                pipe_tables = self._extract_pipe_tables(text)
                tables.extend(pipe_tables)

            if "grid" in self.table_formats:
                grid_tables = self._extract_grid_tables(text)
                tables.extend(grid_tables)

            if "csv" in self.table_formats:
                csv_tables = self._extract_csv_like_tables(text)
                tables.extend(csv_tables)

            # Look for Excel/HTML table markers
            html_tables = self._extract_html_table_markers(text)
            tables.extend(html_tables)

            # Filter and validate tables
            valid_tables = []
            for table in tables:
                if self._validate_table(table):
                    valid_tables.append(table)

            # Generate table statistics
            table_stats = self._generate_table_stats(valid_tables)

            return {
                "tables": valid_tables,
                "table_count": len(valid_tables),
                "total_rows": sum(len(table["data"]) for table in valid_tables),
                "total_columns": sum(
                    len(table["data"][0]) if table["data"] else 0
                    for table in valid_tables
                ),
                "statistics": table_stats,
                "extractor": "TableExtractor",
            }

        except Exception as e:
            self.logger.error(f"Error in table extraction: {e}")
            return {"tables": [], "table_count": 0, "error": str(e)}

    def _extract_pipe_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract pipe-delimited tables (Markdown-style)."""
        tables = []
        lines = text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for pipe-delimited lines
            if "|" in line and line.count("|") >= self.min_columns - 1:
                table_lines = [line]
                j = i + 1

                # Collect consecutive pipe-delimited lines
                while j < len(lines):
                    next_line = lines[j].strip()
                    if "|" in next_line or self._is_separator_line(next_line):
                        table_lines.append(next_line)
                        j += 1
                    else:
                        break

                # Parse the table
                table_data = self._parse_pipe_table(table_lines)
                if table_data:
                    tables.append(
                        {
                            "type": "pipe",
                            "data": table_data["rows"],
                            "headers": table_data.get("headers"),
                            "start_line": i + 1,
                            "end_line": j,
                            "raw_lines": table_lines,
                        }
                    )

                i = j
            else:
                i += 1

        return tables

    def _extract_grid_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract grid-style tables with borders."""
        tables = []

        # Look for grid table patterns
        grid_pattern = re.compile(r"^[+|\s-]+$", re.MULTILINE)
        lines = text.split("\n")

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Look for grid table borders
            if grid_pattern.match(line) and "+" in line:
                table_lines = []
                j = i

                # Collect the entire grid table
                while j < len(lines):
                    current_line = lines[j].strip()
                    if (
                        "|" in current_line
                        or "+" in current_line
                        or grid_pattern.match(current_line)
                    ):
                        table_lines.append(current_line)
                        j += 1
                    else:
                        break

                # Parse the grid table
                table_data = self._parse_grid_table(table_lines)
                if table_data:
                    tables.append(
                        {
                            "type": "grid",
                            "data": table_data["rows"],
                            "headers": table_data.get("headers"),
                            "start_line": i + 1,
                            "end_line": j,
                            "raw_lines": table_lines,
                        }
                    )

                i = j
            else:
                i += 1

        return tables

    def _extract_csv_like_tables(self, text: str) -> List[Dict[str, Any]]:
        """Extract CSV-like tables with consistent delimiters."""
        tables = []
        lines = text.split("\n")

        # Common delimiters to check
        delimiters = [",", ";", "\t"]

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check if line looks like CSV
            for delimiter in delimiters:
                if delimiter in line and line.count(delimiter) >= self.min_columns - 1:
                    table_lines = [line]
                    j = i + 1

                    # Collect consecutive lines with same delimiter pattern
                    expected_columns = line.count(delimiter) + 1
                    while j < len(lines):
                        next_line = lines[j].strip()
                        if (
                            delimiter in next_line
                            and abs(next_line.count(delimiter) + 1 - expected_columns)
                            <= 1
                        ):
                            table_lines.append(next_line)
                            j += 1
                        else:
                            break

                    # Only process if we have enough rows
                    if len(table_lines) >= self.min_rows:
                        table_data = self._parse_csv_table(table_lines, delimiter)
                        if table_data:
                            tables.append(
                                {
                                    "type": "csv",
                                    "delimiter": delimiter,
                                    "data": table_data["rows"],
                                    "headers": table_data.get("headers"),
                                    "start_line": i + 1,
                                    "end_line": j,
                                    "raw_lines": table_lines,
                                }
                            )

                    i = j
                    break
            else:
                i += 1

        return tables

    def _extract_html_table_markers(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables marked with [TABLE] tags or similar."""
        tables = []

        # Look for table markers from HTML parser or similar
        table_pattern = re.compile(
            r"\[TABLE\s*(\d+)?\]([\s\S]*?)\[END\s*TABLE\]", re.IGNORECASE
        )

        for match in table_pattern.finditer(text):
            table_num = match.group(1)
            table_content = match.group(2).strip()

            # Parse the table content
            lines = [line.strip() for line in table_content.split("\n") if line.strip()]
            if len(lines) >= self.min_rows:
                # Try to parse as pipe-delimited
                table_data = self._parse_pipe_table(lines)
                if table_data:
                    tables.append(
                        {
                            "type": "html_marker",
                            "table_number": table_num,
                            "data": table_data["rows"],
                            "headers": table_data.get("headers"),
                            "raw_content": table_content,
                        }
                    )

        return tables

    def _parse_pipe_table(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse pipe-delimited table lines."""
        if not lines:
            return None

        rows = []
        headers = None

        for i, line in enumerate(lines):
            # Skip separator lines
            if self._is_separator_line(line):
                continue

            # Split by pipes
            cells = [cell.strip() for cell in line.split("|")]

            # Remove empty cells at start/end (from leading/trailing pipes)
            while cells and not cells[0]:
                cells.pop(0)
            while cells and not cells[-1]:
                cells.pop()

            if len(cells) >= self.min_columns:
                # First non-separator row might be headers
                if (
                    not rows
                    and self.detect_headers
                    and not self._looks_like_data_row(cells)
                ):
                    headers = cells
                else:
                    rows.append(cells)

        if len(rows) >= self.min_rows:
            return {"rows": rows, "headers": headers}

        return None

    def _parse_grid_table(self, lines: List[str]) -> Optional[Dict[str, Any]]:
        """Parse grid-style table."""
        if not lines:
            return None

        # Filter out border lines
        content_lines = []
        for line in lines:
            if "|" in line and not re.match(r"^[+|\s-]+$", line):
                content_lines.append(line)

        # Parse as pipe table
        return self._parse_pipe_table(content_lines)

    def _parse_csv_table(
        self, lines: List[str], delimiter: str
    ) -> Optional[Dict[str, Any]]:
        """Parse CSV-style table."""
        if not lines:
            return None

        rows = []
        headers = None

        for i, line in enumerate(lines):
            cells = [cell.strip() for cell in line.split(delimiter)]

            if len(cells) >= self.min_columns:
                # First row might be headers
                if (
                    not rows
                    and self.detect_headers
                    and not self._looks_like_data_row(cells)
                ):
                    headers = cells
                else:
                    rows.append(cells)

        if len(rows) >= self.min_rows:
            return {"rows": rows, "headers": headers}

        return None

    def _is_separator_line(self, line: str) -> bool:
        """Check if line is a table separator."""
        stripped = line.strip()
        return bool(re.match(r"^[|+:\s-]+$", stripped))

    def _looks_like_data_row(self, cells: List[str]) -> bool:
        """Check if row looks like data rather than headers."""
        # Look for numbers, dates, or other data patterns
        data_indicators = 0
        for cell in cells:
            if re.search(r"\d+", cell):  # Contains numbers
                data_indicators += 1
            elif re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", cell):  # Date pattern
                data_indicators += 1
            elif cell.lower() in ["yes", "no", "true", "false", "n/a", "null"]:
                data_indicators += 1

        # If any cells look like data, treat as data row
        return data_indicators > 0

    def _validate_table(self, table: Dict[str, Any]) -> bool:
        """Validate that table meets minimum requirements."""
        data = table.get("data", [])

        # Check minimum rows and columns
        if len(data) < self.min_rows:
            return False

        if not data or len(data[0]) < self.min_columns:
            return False

        # Check for consistent column count
        first_row_cols = len(data[0])
        for row in data[1:]:
            if abs(len(row) - first_row_cols) > 1:  # Allow some variation
                return False

        # Check cell length limits
        for row in data:
            for cell in row:
                if len(str(cell)) > self.max_cell_length:
                    return False

        return True

    def _generate_table_stats(self, tables: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate statistics about extracted tables."""
        if not tables:
            return {}

        stats = {
            "total_tables": len(tables),
            "table_types": {},
            "avg_rows": 0,
            "avg_columns": 0,
            "largest_table": {"rows": 0, "columns": 0},
            "has_headers": 0,
        }

        total_rows = 0
        total_columns = 0

        for table in tables:
            # Count table types
            table_type = table.get("type", "unknown")
            stats["table_types"][table_type] = (
                stats["table_types"].get(table_type, 0) + 1
            )

            # Calculate size stats
            data = table.get("data", [])
            if data:
                rows = len(data)
                columns = len(data[0]) if data else 0

                total_rows += rows
                total_columns += columns

                if rows > stats["largest_table"]["rows"]:
                    stats["largest_table"]["rows"] = rows
                    stats["largest_table"]["columns"] = columns

                if table.get("headers"):
                    stats["has_headers"] += 1

        if tables:
            stats["avg_rows"] = total_rows / len(tables)
            stats["avg_columns"] = total_columns / len(tables)

        return stats

    @staticmethod
    def get_supported_formats() -> List[str]:
        """Get list of supported table formats."""
        return ["pipe", "grid", "csv", "html_marker"]

    @staticmethod
    def get_description() -> str:
        """Get extractor description."""
        return "Extract structured tabular data from text content"
