"""MarkItDown universal document parser."""

from pathlib import Path
from typing import Dict, Any, Optional
import tempfile

from components.parsers.base.base_parser import BaseParser, ParserConfig
from core.base import Document, ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.markitdown.parser")


class MarkItDownParser(BaseParser):
    """Universal document parser using Microsoft MarkItDown."""

    # Centralized list of supported file extensions
    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xls",
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".wav",
        ".mp3",
        ".html",
        ".htm",
        ".csv",
        ".json",
        ".xml",
        ".zip",
        ".epub",
    }

    def __init__(
        self,
        name: str = "MarkItDownParser",
        config: Optional[Dict[str, Any]] = None,
    ):
        # Call parent constructor to initialize base state
        super().__init__(config)

        self.name = name

        # Configuration
        self.preserve_structure = self.config.get("preserve_structure", True)
        self.chain_to_markdown = self.config.get("chain_to_markdown_parser", True)
        self.markdown_parser_name = self.config.get(
            "markdown_parser", "MarkdownParser_Python"
        )

        # Optional features
        self.enable_ocr = self.config.get("enable_ocr", False)
        self.enable_audio = self.config.get("enable_audio_transcription", False)
        self.enable_llm_desc = self.config.get("enable_llm_descriptions", False)

        # Azure Doc Intelligence
        self.use_azure = self.config.get("use_azure_doc_intelligence", False)
        self.azure_endpoint = self.config.get("azure_doc_intelligence_endpoint")
        self.azure_key = self.config.get("azure_doc_intelligence_key")

        # Import MarkItDown
        try:
            from markitdown import MarkItDown

            # Initialize MarkItDown with configured options
            # Note: MarkItDown accepts these parameters at initialization time
            markitdown_kwargs = {}

            # Azure Document Intelligence support
            if self.use_azure and self.azure_endpoint and self.azure_key:
                try:
                    from azure.ai.formrecognizer import DocumentAnalysisClient
                    from azure.core.credentials import AzureKeyCredential

                    markitdown_kwargs["azure_di_client"] = DocumentAnalysisClient(
                        endpoint=self.azure_endpoint,
                        credential=AzureKeyCredential(self.azure_key)
                    )
                    logger.info("Azure Document Intelligence client configured")
                except ImportError:
                    logger.warning(
                        "Azure Document Intelligence requested but azure-ai-formrecognizer not installed"
                    )

            # LLM client for image descriptions (if enabled)
            if self.enable_llm_desc:
                logger.info("LLM descriptions enabled but not yet implemented in this parser")
                # TODO: Add OpenAI/other LLM client configuration when needed
                # markitdown_kwargs["llm_client"] = ...
                # markitdown_kwargs["llm_model"] = ...

            self.markitdown = MarkItDown(**markitdown_kwargs)
            logger.info("MarkItDown initialized successfully")
        except ImportError as e:
            logger.error(
                f"MarkItDown library not installed. Run: uv sync --extra markitdown. Error: {e}"
            )
            raise

    def validate_config(self) -> bool:
        """Validate configuration."""
        # Validate Azure Document Intelligence configuration
        if self.use_azure:
            if not self.azure_endpoint or not self.azure_key:
                logger.error(
                    "Azure Document Intelligence is enabled, but endpoint or key is missing. "
                    "Please provide both 'azure_doc_intelligence_endpoint' and "
                    "'azure_doc_intelligence_key' in the parser configuration."
                )
                return False
            logger.info("Azure Document Intelligence configuration validated")

        # Validate LLM descriptions configuration
        if self.enable_llm_desc:
            logger.warning(
                "LLM descriptions are enabled but not yet fully implemented. "
                "This feature requires additional configuration (OpenAI API key, etc.)"
            )

        return True

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""
        return ParserConfig(
            name="MarkItDownParser",
            display_name="MarkItDown Universal Parser",
            version="1.0.0",
            supported_extensions=sorted(list(self.SUPPORTED_EXTENSIONS)),
            mime_types=[
                "application/pdf",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "image/jpeg",
                "image/png",
                "text/html",
                "application/json",
            ],
            capabilities=[
                "universal_conversion",
                "markdown_output",
                "structure_preservation",
                "metadata_extraction",
            ],
            dependencies={"required": ["markitdown"], "optional": ["pillow", "pytesseract"]},
            default_config=self.config,
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the file."""
        path = Path(file_path)
        return path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def parse(self, source: str, **kwargs) -> ProcessingResult:
        """Convert file to Markdown and optionally chain to Markdown parser."""
        path = Path(source)

        # Validate that source is a file (not a directory)
        if not path.is_file():
            error_msg = f"Not a valid file: {source}"
            if path.is_dir():
                error_msg = f"Source is a directory, not a file: {source}"
            elif not path.exists():
                error_msg = f"File not found: {source}"

            logger.error(error_msg)
            return ProcessingResult(
                documents=[], errors=[{"error": error_msg, "source": source}]
            )

        try:
            # Phase 1: Convert to Markdown using MarkItDown
            logger.info(f"Converting {path.name} to Markdown using MarkItDown")
            result = self.markitdown.convert(str(path))

            markdown_text = result.text_content

            if not markdown_text or not markdown_text.strip():
                logger.warning(f"MarkItDown produced empty output for {source}")
                return ProcessingResult(
                    documents=[],
                    errors=[
                        {"error": "MarkItDown produced empty output", "source": source}
                    ],
                )

            # Extract metadata
            metadata = {
                "source": str(path),
                "file_name": path.name,
                "parser": self.name,
                "tool": "MarkItDown",
                "file_size": path.stat().st_size,
                "original_format": path.suffix.lower(),
                "converted_to": "markdown",
            }

            logger.info(
                f"MarkItDown conversion successful: {len(markdown_text)} characters"
            )

            # Phase 2: Chain to Markdown parser (if enabled)
            if self.chain_to_markdown:
                logger.info(f"Chaining to {self.markdown_parser_name} for chunking")
                return self._chain_to_markdown_parser(markdown_text, metadata, path)

            # Return raw Markdown (standalone mode)
            logger.info("Standalone mode: returning raw Markdown without chunking")
            doc = Document(
                content=markdown_text,
                metadata=metadata,
                id=f"{path.stem}_markitdown",
                source=str(path),
            )

            return ProcessingResult(
                documents=[doc],
                errors=[],
                metrics={
                    "total_documents": 1,
                    "parser_type": self.name,
                    "tool": "MarkItDown",
                    "conversion_successful": True,
                    "mode": "standalone",
                },
            )

        except Exception as e:
            logger.error(f"MarkItDown conversion failed for {source}: {e}", exc_info=True)
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _chain_to_markdown_parser(
        self, markdown_text: str, base_metadata: Dict[str, Any], original_path: Path
    ) -> ProcessingResult:
        """Chain converted Markdown to a Markdown parser for chunking."""
        from components.parsers.parser_factory import ToolAwareParserFactory

        try:
            # Get the configured Markdown parser
            logger.info(f"Loading {self.markdown_parser_name} for chunking")

            # Pass chunking config to markdown parser
            md_parser_config = {
                "chunk_size": self.config.get("chunk_size", 1000),
                "chunk_strategy": self.config.get("chunk_strategy", "sections"),
                "chunk_overlap": self.config.get("chunk_overlap", 100),
            }

            md_parser = ToolAwareParserFactory.create_parser(
                parser_name=self.markdown_parser_name, config=md_parser_config
            )

            # Write markdown to temp file for parser
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".md", delete=False, encoding="utf-8"
            ) as tmp:
                tmp.write(markdown_text)
                tmp_path = tmp.name

            try:
                # Parse the Markdown with chunking
                logger.info(f"Parsing Markdown with {self.markdown_parser_name}")
                result = md_parser.parse(tmp_path)

                logger.info(
                    f"Chunking complete: {len(result.documents)} chunks created"
                )

                # Enhance each chunk with original file metadata
                for i, doc in enumerate(result.documents):
                    doc.metadata.update(base_metadata)
                    doc.metadata["preprocessing"] = "markitdown"
                    doc.metadata["chunking_parser"] = self.markdown_parser_name
                    # Update chunk numbering to be relative to original file
                    doc.metadata["chunk_index"] = i
                    doc.metadata["total_chunks"] = len(result.documents)

                result.metrics = result.metrics or {}
                result.metrics["preprocessing"] = "markitdown"
                result.metrics["original_file"] = str(original_path)
                result.metrics["chunking_parser"] = self.markdown_parser_name
                result.metrics["mode"] = "chained"

                return result

            finally:
                # Clean up temp file
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Failed to chain to Markdown parser: {e}", exc_info=True)
            # Fallback: return raw markdown as single doc
            logger.warning("Falling back to standalone mode due to chaining error")
            doc = Document(
                content=markdown_text,
                metadata={**base_metadata, "chain_error": str(e)},
                id=f"{original_path.stem}_markitdown_fallback",
                source=str(original_path),
            )
            return ProcessingResult(
                documents=[doc],
                errors=[
                    {"error": f"Chaining failed: {e}", "source": str(original_path)}
                ],
                metrics={"fallback": True, "mode": "fallback"},
            )
