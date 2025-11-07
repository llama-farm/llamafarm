"""MarkItDown universal document preprocessor for format conversion."""

from pathlib import Path
from typing import Any, Optional
import yaml

from components.preprocessors.base import BasePreprocessor, PreprocessorResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.preprocessors.markitdown")


class MarkItDownPreprocessor(BasePreprocessor):
    """Universal document preprocessor using Microsoft MarkItDown.

    Converts various document formats (DOCX, PPTX, PDF, HTML, etc.) to Markdown.
    """

    def __init__(self, config: Optional[dict[str, Any]] = None):
        """Initialize MarkItDown preprocessor.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)

        # Load supported extensions from config.yaml (single source of truth)
        config_path = Path(__file__).parent / "config.yaml"
        with open(config_path, "r") as f:
            preprocessor_config = yaml.safe_load(f)
            # Extract supported extensions from first preprocessor definition
            self.supported_extensions = set(
                preprocessor_config["preprocessors"][0]["supported_extensions"]
            )

        # Configuration
        self.preserve_structure = self.config.get("preserve_structure", True)

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
            markitdown_kwargs = {}

            # Azure Document Intelligence support
            if self.use_azure and self.azure_endpoint and self.azure_key:
                try:
                    from azure.ai.formrecognizer import DocumentAnalysisClient
                    from azure.core.credentials import AzureKeyCredential

                    markitdown_kwargs["azure_di_client"] = DocumentAnalysisClient(
                        endpoint=self.azure_endpoint,
                        credential=AzureKeyCredential(self.azure_key),
                    )
                    logger.info("Azure Document Intelligence client configured")
                except ImportError:
                    logger.warning(
                        "Azure Document Intelligence requested but azure-ai-formrecognizer not installed"
                    )

            # LLM client for image descriptions (if enabled)
            if self.enable_llm_desc:
                logger.info(
                    "LLM descriptions enabled but not yet implemented in this preprocessor"
                )
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

    def can_process(self, file_path: str, metadata: dict[str, Any]) -> bool:
        """Check if this preprocessor can handle the file.

        Args:
            file_path: Path to the file
            metadata: File metadata

        Returns:
            True if file extension is supported
        """
        path = Path(file_path)
        ext = path.suffix.lower()
        return ext in self.supported_extensions

    def preprocess(
        self, file_path: str, metadata: dict[str, Any]
    ) -> PreprocessorResult:
        """Convert file to Markdown using MarkItDown.

        Args:
            file_path: Path to input file
            metadata: File metadata

        Returns:
            PreprocessorResult with markdown content
        """
        path = Path(file_path)

        # Validate that source is a file (not a directory)
        if not path.is_file():
            error_msg = f"Not a valid file: {file_path}"
            if path.is_dir():
                error_msg = f"Source is a directory, not a file: {file_path}"
            elif not path.exists():
                error_msg = f"File not found: {file_path}"

            logger.error(error_msg)
            return PreprocessorResult(
                content="",
                metadata=metadata,
                output_format="markdown",
                success=False,
                errors=[error_msg],
            )

        try:
            # Convert to Markdown using MarkItDown
            logger.info(f"Converting {path.name} to Markdown using MarkItDown")
            result = self.markitdown.convert(str(path))

            markdown_text = result.text_content

            if not markdown_text or not markdown_text.strip():
                logger.warning(f"MarkItDown produced empty output for {file_path}")
                return PreprocessorResult(
                    content="",
                    metadata=metadata,
                    output_format="markdown",
                    success=False,
                    errors=["MarkItDown produced empty output"],
                )

            # Build enriched metadata
            enriched_metadata = {
                **metadata,
                "preprocessor": "MarkItDownPreprocessor",
                "tool": "MarkItDown",
                "original_format": path.suffix.lower(),
                "converted_to": "markdown",
                "file_size": path.stat().st_size,
                "content_length": len(markdown_text),
            }

            logger.info(
                f"MarkItDown conversion successful: {len(markdown_text)} characters"
            )

            return PreprocessorResult(
                content=markdown_text,
                metadata=enriched_metadata,
                output_format="markdown",
                success=True,
            )

        except Exception as e:
            logger.error(
                f"MarkItDown conversion failed for {file_path}: {e}", exc_info=True
            )
            return PreprocessorResult(
                content="",
                metadata=metadata,
                output_format="markdown",
                success=False,
                errors=[str(e)],
            )

    def get_supported_formats(self) -> list[str]:
        """Get list of supported file extensions.

        Returns:
            List of supported file extensions
        """
        return sorted(list(self.supported_extensions))
