"""PaddleOCR preprocessor for high-accuracy OCR with layout analysis and table extraction."""

import tempfile
from pathlib import Path
from typing import Any, Optional

from components.preprocessors.ocr.base_ocr import BaseOCRPreprocessor, PreprocessorResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.preprocessors.ocr.paddleocr")


class PaddleOCRPreprocessor(BaseOCRPreprocessor):
    """OCR using PaddleOCR engine with table extraction and searchable PDF creation."""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize PaddleOCR preprocessor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # PaddleOCR-specific config
        self.use_gpu = self.config.get("use_gpu", False)
        self.use_angle_cls = self.config.get(
            "use_angle_cls", True
        )  # Detect text orientation
        self.lang = self.config.get("language", "en")  # en, ch, fr, german, korean, japan

        # Table extraction config
        self.extract_tables = self.config.get("extract_tables", True)
        self.table_confidence_threshold = self.config.get(
            "table_confidence_threshold", 0.7
        )
        self.merge_tables_inline = self.config.get("merge_tables_inline", False)

        # Output format: markdown (default), text, or searchable_pdf
        self.output_format = self.config.get("output_format", "markdown")

        # Initialize PaddleOCR
        try:
            from paddleocr import PaddleOCR

            # PaddleOCR v3 has a simplified API
            self.ocr = PaddleOCR(
                lang=self.lang,
            )
            logger.info(f"PaddleOCR initialized (lang={self.lang})")

            # Initialize PPStructureV3 for table extraction if enabled
            if self.extract_tables:
                try:
                    from paddleocr import PPStructureV3

                    # PPStructureV3 - converts complex PDFs/images to Markdown with tables
                    self.table_engine = PPStructureV3(
                        lang=self.lang,
                    )
                    logger.info("PPStructureV3 initialized for table extraction")
                except Exception as e:
                    logger.warning(
                        f"Failed to initialize PPStructure for table extraction: {e}"
                    )
                    self.extract_tables = False

        except ImportError as e:
            raise ImportError(
                "PaddleOCR not installed. Install with: uv pip install paddleocr paddlepaddle"
            ) from e

    def preprocess(
        self, file_path: str, metadata: dict[str, Any]
    ) -> PreprocessorResult:
        """Extract text from image or scanned PDF using PaddleOCR.

        Args:
            file_path: Path to input file
            metadata: File metadata

        Returns:
            PreprocessorResult with extracted text, tables, and optionally searchable PDF
        """
        try:
            ext = Path(file_path).suffix.lower()

            # Handle PDFs: convert to images first, then OCR
            if ext == ".pdf":
                return self._process_pdf(file_path, metadata)

            # Handle images directly
            else:
                return self._process_image(file_path, metadata)

        except Exception as e:
            logger.error(f"OCR failed for {file_path}: {e}", exc_info=True)
            return PreprocessorResult(
                content="",
                metadata={"error": str(e)},
                output_format="text",
                success=False,
                errors=[str(e)],
            )

    def _process_image(
        self, image_path: str, metadata: dict[str, Any]
    ) -> PreprocessorResult:
        """OCR a single image with optional table extraction.

        Args:
            image_path: Path to image file
            metadata: File metadata

        Returns:
            PreprocessorResult with extracted content
        """
        # Phase 1: Extract tables (if enabled)
        tables = []
        if self.extract_tables:
            tables = self._extract_tables_from_image(image_path)
            logger.info(f"Extracted {len(tables)} tables from {image_path}")

        # Phase 2: Extract text using PaddleOCR (v3 API - different response structure)
        result = self.ocr.ocr(image_path)

        # PaddleOCR v3 returns: [{'rec_texts': [...], 'dt_polys': [...], 'rec_scores': [...]}]
        lines = []
        layout_info = []
        total_confidence = 0
        num_boxes = 0

        if result and len(result) > 0:
            page_result = result[0]  # First (and usually only) page

            # Extract text strings from rec_texts
            rec_texts = page_result.get('rec_texts', [])
            dt_polys = page_result.get('dt_polys', [])
            rec_scores = page_result.get('rec_scores', [])

            for i, text in enumerate(rec_texts):
                lines.append(text)

                # Store layout info if available
                if i < len(dt_polys):
                    bbox = dt_polys[i].tolist() if hasattr(dt_polys[i], 'tolist') else dt_polys[i]
                    confidence = rec_scores[i] if i < len(rec_scores) else 1.0

                    layout_info.append({
                        "text": text,
                        "bbox": bbox,
                        "confidence": float(confidence),
                    })

                    total_confidence += confidence
                    num_boxes += 1

        # Join text with newlines (preserves reading order)
        text_content = "\n".join(lines)
        avg_confidence = total_confidence / num_boxes if num_boxes > 0 else 0

        # Phase 3: Format as Markdown (structure preservation)
        if self.output_format == "markdown":
            text_content = self._format_as_markdown(text_content, tables, layout_info)
        elif self.extract_tables and tables:
            # Plain text mode - just append tables
            text_content = self._merge_text_and_tables(text_content, tables)

        # Build metadata
        ocr_metadata = {
            "ocr_engine": "PaddleOCR",
            "language": self.lang,
            "num_text_boxes": num_boxes,
            "avg_confidence": avg_confidence,
            "layout": layout_info if self.detect_layout else None,
            "table_count": len(tables) if tables else 0,
            "tables": tables if tables else None,
            **metadata,
        }

        logger.info(
            f"OCR completed for {image_path}",
            text_boxes=num_boxes,
            confidence=avg_confidence,
            tables=len(tables),
        )

        return PreprocessorResult(
            content=text_content,
            metadata=ocr_metadata,
            output_format=self.output_format,
            success=True,
        )

    def _process_pdf(
        self, pdf_path: str, metadata: dict[str, Any]
    ) -> PreprocessorResult:
        """OCR a scanned PDF by converting to images.

        Args:
            pdf_path: Path to PDF file
            metadata: File metadata

        Returns:
            PreprocessorResult with extracted content and optionally searchable PDF
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(pdf_path)
            all_text = []
            all_layout = []
            all_tables = []

            for page_num in range(len(doc)):
                page = doc[page_num]

                # Convert page to image (high DPI for better OCR)
                pix = page.get_pixmap(dpi=300)

                # Save to temp file for PaddleOCR
                with tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ) as tmp:
                    pix.save(tmp.name)

                    # OCR the page image
                    page_result = self._process_image(tmp.name, {})
                    all_text.append(
                        f"=== Page {page_num + 1} ===\n{page_result.content}"
                    )

                    if page_result.metadata.get("layout"):
                        all_layout.extend(
                            [
                                {**item, "page": page_num + 1}
                                for item in page_result.metadata["layout"]
                            ]
                        )

                    if page_result.metadata.get("tables"):
                        for table in page_result.metadata["tables"]:
                            table["page"] = page_num + 1
                            all_tables.append(table)

                    Path(tmp.name).unlink()  # Clean up temp file

            doc.close()

            # Combine all pages
            content = "\n\n".join(all_text)

            # Format as Markdown if requested
            if self.output_format == "markdown" and all_layout:
                # Already formatted by _process_image, just combine
                pass

            result_metadata = {
                "ocr_engine": "PaddleOCR",
                "language": self.lang,
                "num_pages": len(doc),
                "layout": all_layout if self.detect_layout else None,
                "table_count": len(all_tables),
                "tables": all_tables if all_tables else None,
                **metadata,
            }

            return PreprocessorResult(
                content=content,
                metadata=result_metadata,
                output_format=self.output_format,
                success=True,
            )

        except Exception as e:
            logger.error(f"PDF OCR failed for {pdf_path}: {e}", exc_info=True)
            raise

    def _extract_tables_from_image(self, image_path: str) -> list[dict[str, Any]]:
        """Extract tables using PPStructure layout analysis.

        Args:
            image_path: Path to image file

        Returns:
            List of extracted tables with HTML and Markdown formats
        """
        if not self.extract_tables or not hasattr(self, "table_engine"):
            return []

        try:
            result = self.table_engine(image_path)
            tables = []

            for region in result:
                if region["type"] == "table":
                    # PPStructure returns table as HTML
                    table_html = region["res"]["html"]
                    bbox = region["bbox"]  # [x1, y1, x2, y2]
                    confidence = region.get("confidence", 1.0)

                    # Filter by confidence threshold
                    if confidence < self.table_confidence_threshold:
                        logger.debug(
                            f"Skipping table with low confidence: {confidence}"
                        )
                        continue

                    # Convert HTML table to Markdown
                    table_markdown = self._html_table_to_markdown(table_html)

                    tables.append(
                        {
                            "type": "table",
                            "bbox": bbox,
                            "html": table_html,
                            "markdown": table_markdown,
                            "confidence": confidence,
                        }
                    )

            return tables

        except Exception as e:
            logger.warning(f"Table extraction failed for {image_path}: {e}")
            return []

    def _html_table_to_markdown(self, html: str) -> str:
        """Convert HTML table to Markdown format.

        Args:
            html: HTML table string

        Returns:
            Markdown table string
        """
        try:
            # Try using markitdown for conversion
            from markitdown import MarkItDown
            import io

            converter = MarkItDown()
            result = converter.convert_stream(io.StringIO(html))
            return result.text_content

        except ImportError:
            # Fallback: use html2text
            try:
                import html2text

                h = html2text.HTML2Text()
                h.body_width = 0  # Don't wrap lines
                h.ignore_links = False
                return h.handle(html)
            except ImportError:
                logger.warning(
                    "Neither markitdown nor html2text available for table conversion"
                )
                # Last resort: return raw HTML
                return f"```html\n{html}\n```"

    def _format_as_markdown(
        self, text: str, tables: list[dict[str, Any]], layout_info: list[dict[str, Any]]
    ) -> str:
        """Format OCR output as structured Markdown.

        Args:
            text: Extracted text content
            tables: List of extracted tables
            layout_info: Layout information with bboxes

        Returns:
            Markdown-formatted content
        """
        # Start with a document header
        result = "# OCR Extracted Document\n\n"

        # Add the main text content
        if text:
            # Try to detect paragraphs (double newlines or indentation)
            paragraphs = text.split('\n\n')
            if len(paragraphs) == 1:
                # No paragraph breaks found, split on single newlines and group
                lines = text.split('\n')
                result += '\n\n'.join(line for line in lines if line.strip())
            else:
                result += '\n\n'.join(p.strip() for p in paragraphs if p.strip())

        # Add tables if present
        if tables:
            result += "\n\n## Tables\n\n"
            sorted_tables = sorted(tables, key=lambda t: t["bbox"][1])

            for i, table in enumerate(sorted_tables, 1):
                confidence = table.get("confidence", 0)
                result += f"### Table {i}\n"
                result += f"*Confidence: {confidence:.1%}*\n\n"
                result += f"{table['markdown']}\n\n"

        return result

    def _merge_text_and_tables(
        self, text: str, tables: list[dict[str, Any]]
    ) -> str:
        """Merge extracted text with tables (plain text mode).

        Args:
            text: Extracted text content
            tables: List of extracted tables

        Returns:
            Combined content with tables
        """
        if not tables:
            return text

        # Sort tables by vertical position (top to bottom)
        sorted_tables = sorted(tables, key=lambda t: t["bbox"][1])

        # Append tables at the end with markers
        result = f"{text}\n\n## Extracted Tables\n\n"

        for i, table in enumerate(sorted_tables, 1):
            confidence = table.get("confidence", 0)
            result += f"\n### Table {i} (confidence: {confidence:.2%})\n\n"
            result += f"{table['markdown']}\n"

        return result


    def _run_ocr(self, image_path: str) -> dict[str, Any]:
        """Run PaddleOCR engine (implementation of abstract method).

        Args:
            image_path: Path to image file

        Returns:
            Dictionary with OCR results
        """
        return self.ocr.ocr(image_path)
