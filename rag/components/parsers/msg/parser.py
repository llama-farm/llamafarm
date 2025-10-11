"""MSG parser using msg_parser library."""

import mimetypes
from pathlib import Path
from typing import Any, Optional, Union

from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

from components.parsers.base.base_parser import BaseParser, ParserConfig
from components.parsers.msg.msg_utils import (
    MsgChunker,
    MsgDocumentFactory,
    MsgTempFileHandler,
)
from core.base import ProcessingResult
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.msg.parser")


class MsgParser(BaseParser):
    """MSG parser using msg_parser library."""

    def __init__(
        self,
        name: str = "MsgParser",
        config: Optional[dict[str, Any]] = None,
    ):
        super().__init__(config or {})  # Call BaseParser init
        self.name = name

        # Chunking configuration
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "email_sections")

        # Feature flags
        self.extract_metadata = self.config.get("extract_metadata", True)
        self.extract_attachments = self.config.get("extract_attachments", True)
        self.extract_headers = self.config.get("extract_headers", True)
        self.include_attachment_content = self.config.get(
            "include_attachment_content", True
        )
        self.clean_text = self.config.get("clean_text", True)
        self.preserve_formatting = self.config.get("preserve_formatting", False)

    def _load_metadata(self) -> ParserConfig:
        """Load parser metadata."""
        return ParserConfig(
            name="MsgParser_MsgParser",
            display_name="MSG Parser (msg_parser)",
            version="1.0.0",
            supported_extensions=[".msg"],
            mime_types=["application/vnd.ms-outlook", "application/octet-stream"],
            capabilities=[
                "text_extraction",
                "metadata_extraction",
                "attachment_extraction",
                "email_parsing",
                "header_extraction",
                "body_extraction",
            ],
            dependencies={
                "msg-parser": ["msg-parser[rtf]>=1.2.0"],
            },
            default_config={
                "chunk_size": 1000,
                "chunk_overlap": 100,
                "chunk_strategy": "email_sections",
                "extract_metadata": True,
                "extract_attachments": True,
                "extract_headers": True,
            },
        )

    def can_parse(self, file_path: str) -> bool:
        """Check if this parser can handle the given file."""
        path = Path(file_path)

        # First check file extension
        if not path.name.lower().endswith(".msg"):
            return False

        # Check if file exists and is readable
        if not path.exists() or not path.is_file():
            return False

        try:
            # Check file size - MSG files should not be empty and not excessively large
            file_size = path.stat().st_size
            if file_size == 0:
                return False
            if file_size > 100 * 1024 * 1024:  # 100MB limit for safety
                logger.warning(
                    f"MSG file {file_path} is very large ({file_size} bytes), may cause memory issues"
                )

            # Try to read the first few bytes to check for MSG file signature
            with open(path, "rb") as f:
                header = f.read(8)
                # MSG files typically start with specific OLE compound document signatures
                # Check for OLE compound document signature (MSG files are OLE documents)
                if len(header) >= 8:
                    # OLE signature: D0CF11E0A1B11AE1
                    ole_signature = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
                    if header == ole_signature:
                        return True
                    # Sometimes MSG files might have different headers, so we'll be more permissive
                    # but still check for reasonable binary content
                    if header.startswith(b"\xd0\xcf") or b"\x00" in header[:4]:
                        return True

        except (OSError, PermissionError) as e:
            logger.debug(f"Cannot access file {file_path} for validation: {e}")
            return False

        # Fallback to extension-only check if header validation is inconclusive
        logger.debug(
            f"MSG file {file_path} header validation inconclusive, relying on extension"
        )
        return True

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs) -> ProcessingResult:
        """Parse MSG file using msg_parser."""
        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            msg = self._load_msg_file(str(path))
            if isinstance(msg, ProcessingResult):  # Error case
                return msg

            content_parts, metadata = self._extract_msg_content(msg, path)
            documents = self._create_documents_from_content(
                content_parts, metadata, str(path)
            )

            return ProcessingResult(documents=documents, errors=[])

        except Exception as e:
            logger.error(f"Failed to parse MSG file {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _load_msg_file(self, file_path: str):
        """Load MSG file using msg_parser library."""
        try:
            from msg_parser import MsOxMessage
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "msg_parser library not installed. Install with: pip install msg_parser[rtf]",
                        "source": file_path,
                    }
                ],
            )

        return MsOxMessage(file_path)

    def _extract_msg_content(
        self, msg, path: Path
    ) -> tuple[list[tuple[str, str]], dict]:
        """Extract content and metadata from MSG file."""
        content_parts = []
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "parser": "MsgParser_MsgParser",
            "tool": "msg_parser",
            "file_size": path.stat().st_size,
        }

        # Get message properties
        msg_properties = msg.get_properties()
        logger.info(f"Message properties: {msg_properties}")

        # Extract email metadata if enabled
        if self.extract_metadata:
            metadata = self._extract_metadata_from_properties(msg_properties, metadata)

        # Extract headers if enabled
        if self.extract_headers:
            if headers_content := self._extract_headers_from_properties(msg_properties):
                content_parts.append(("headers", headers_content))

        # Extract body content
        if body_content := self._extract_body_content_from_properties(msg_properties):
            content_parts.append(("body", body_content))

        # Extract attachments if enabled
        if self.extract_attachments:
            if attachments_content := self._extract_attachments_from_properties(
                msg_properties
            ):
                content_parts.extend(attachments_content)

        return content_parts, metadata

    def _create_documents_from_content(
        self, content_parts: list[tuple[str, str]], metadata: dict, source: str
    ) -> list:
        """Create documents from extracted content parts."""
        if not content_parts:
            return []

        # Apply chunking based on strategy
        if self.chunk_size:
            if self.chunk_strategy == "email_sections":
                # Chunk by email sections (headers, body, attachments)
                return MsgChunker.chunk_by_email_sections(
                    content_parts, metadata, self.chunk_size, self.chunk_overlap
                )
            else:
                # Combine all content and use standard chunking
                full_content = "\n\n".join([content for _, content in content_parts])
                return self._apply_chunking_strategy(full_content, metadata, source)
        else:
            # Single document with all content
            full_content = "\n\n".join([content for _, content in content_parts])
            return [
                MsgDocumentFactory.create_single_document(
                    full_content, metadata, source
                )
            ]

    def _apply_chunking_strategy(
        self, content: str, metadata: dict, source: str
    ) -> list:
        """Apply the configured chunking strategy to content."""
        if SentenceSplitter is None or TokenTextSplitter is None:
            # Fallback chunking if LlamaIndex not available
            chunks = MsgChunker._split_text_into_chunks(
                content, self.chunk_size, self.chunk_overlap
            )
            return MsgDocumentFactory.create_documents_from_chunks(
                chunks, metadata, source, self.chunk_strategy
            )
        else:
            return self._apply_standard_chunking(content, metadata, source)

    def parse_blob(self, data: bytes, metadata: dict[str, Any] | None = None) -> list:
        """Parse MSG from raw bytes using temporary file."""
        try:
            # msg_parser needs a file on disk, so write temporarily
            with MsgTempFileHandler(data) as tmp_path:
                result = self.parse(tmp_path)

                # Update metadata for blob parsing
                filename = (
                    metadata.get("filename", "message.msg")
                    if metadata
                    else "message.msg"
                )

                for doc in result.documents:
                    doc.metadata["source"] = filename
                    doc.metadata["file_name"] = filename
                    if metadata:
                        doc.metadata.update(metadata)

                return result.documents

        except Exception as e:
            logger.error(f"Failed to parse MSG blob: {e}")
            return []

    def _extract_metadata_from_properties(
        self, msg_properties: dict, metadata: dict
    ) -> dict:
        """Extract metadata from MSG properties dictionary."""
        # Common email properties that msg_parser provides
        email_fields = {
            "subject": ["Subject", "subject"],
            "sender_name": ["SenderName", "sender_name", "from_name"],
            "sender_email": ["SenderEmailAddress", "sender_email", "from_email"],
            "display_to": ["DisplayTo", "display_to", "to"],
            "display_cc": ["DisplayCc", "display_cc", "cc"],
            "display_bcc": ["DisplayBcc", "display_bcc", "bcc"],
            "creation_time": ["CreationTime", "creation_time", "date_created"],
            "last_modification_time": [
                "LastModificationTime",
                "last_modification_time",
                "date_modified",
            ],
            "message_delivery_time": [
                "MessageDeliveryTime",
                "message_delivery_time",
                "date_received",
            ],
            "importance": ["Importance", "importance", "priority"],
            "sensitivity": ["Sensitivity", "sensitivity"],
            "message_class": ["MessageClass", "message_class"],
            "message_size": ["MessageSize", "message_size", "size"],
            "has_attachments": [
                "HasAttachments",
                "has_attachments",
                "attachment_count",
            ],
        }

        for field_name, possible_keys in email_fields.items():
            for key in possible_keys:
                if key in msg_properties and msg_properties[key] is not None:
                    metadata[field_name] = msg_properties[key]
                    break

        return metadata

    def _extract_headers_from_properties(self, msg_properties: dict) -> str:
        """Extract headers information from MSG properties."""
        headers_parts = []

        # Extract key header fields
        header_fields = {
            "Subject": ["Subject", "subject"],
            "From": ["SenderName", "SenderEmailAddress", "sender_name", "from_name"],
            "To": ["DisplayTo", "display_to", "to"],
            "CC": ["DisplayCc", "display_cc", "cc"],
            "BCC": ["DisplayBcc", "display_bcc", "bcc"],
            "Date": [
                "CreationTime",
                "MessageDeliveryTime",
                "creation_time",
                "date_received",
            ],
            "Message-ID": ["InternetMessageId", "message_id"],
            "Importance": ["Importance", "importance", "priority"],
        }

        for header_name, possible_keys in header_fields.items():
            for key in possible_keys:
                if key in msg_properties and msg_properties[key] is not None:
                    if value := msg_properties[key]:
                        headers_parts.append(f"{header_name}: {value}")
                    break

        return "\n".join(headers_parts) if headers_parts else ""

    def _extract_body_content_from_properties(self, msg_properties: dict) -> str:
        """Extract body content from MSG properties."""
        content_parts = []

        # Try to get HTML body first, then plain text
        # msg_parser provides body content in different keys
        body_keys = ["HtmlBody", "html_body", "Body", "body", "RtfBody", "rtf_body"]

        for key in body_keys:
            if key in msg_properties and msg_properties[key]:
                body_content = msg_properties[key]

                if self.clean_text and ("html" in key.lower() or "Html" in key):
                    # Remove HTML tags for cleaner text
                    import re

                    body_content = re.sub(r"<[^>]+>", "", str(body_content))
                    body_content = re.sub(r"\s+", " ", body_content).strip()

                if body_content.strip():
                    content_parts.append(str(body_content).strip())
                    break  # Use the first available body content

        return "\n\n".join(content_parts) if content_parts else ""

    def _extract_attachments_from_properties(
        self, msg_properties: dict
    ) -> list[tuple[str, str]]:
        """Extract attachment information from MSG properties."""
        attachments_content = []

        # msg_parser provides attachments information
        attachments_key = next(
            (
                key
                for key in ["Attachments", "attachments", "attachment_list"]
                if key in msg_properties
            ),
            None,
        )

        if attachments_key and (attachments := msg_properties[attachments_key]):
            # Handle different attachment formats that msg_parser might provide
            if isinstance(attachments, list):
                for i, attachment in enumerate(attachments):
                    attachment_info = self._process_attachment(attachment, i)
                    if attachment_info:
                        attachments_content.append(attachment_info)
            elif isinstance(attachments, dict):
                # Single attachment or dict of attachments
                for i, (_, attachment) in enumerate(attachments.items()):
                    attachment_info = self._process_attachment(attachment, i)
                    if attachment_info:
                        attachments_content.append(attachment_info)

        return attachments_content

    def _get_attachment_property(
        self, attachment: Any, property_names: list[str]
    ) -> Any:
        """Get attachment property with case-insensitive matching."""
        if isinstance(attachment, dict):
            # For dictionary-based attachments, try different case variations
            for prop_name in property_names:
                # Try exact match first
                if prop_name in attachment:
                    return attachment[prop_name]
                # Try case-insensitive match
                for key in attachment.keys():
                    if key.lower() == prop_name.lower():
                        return attachment[key]
        else:
            # For object-based attachments, try different attribute names
            for prop_name in property_names:
                if hasattr(attachment, prop_name):
                    value = getattr(attachment, prop_name)
                    if value is not None:
                        return value
                # Try case variations
                for attr_name in dir(attachment):
                    if attr_name.lower() == prop_name.lower():
                        value = getattr(attachment, attr_name, None)
                        if value is not None:
                            return value
        return None

    def _get_mime_type_from_filename(self, filename: str) -> str:
        """Get MIME type from filename extension."""
        if not filename:
            return "application/octet-stream"

        mime_type, _ = mimetypes.guess_type(filename)
        return mime_type or "application/octet-stream"

    def _is_text_attachment(
        self, filename: str, mime_type: Optional[str] = None
    ) -> bool:
        """Check if attachment is a text-based file that should have content extracted."""
        if not filename:
            return False

        # Check by extension
        text_extensions = [
            ".txt",
            ".log",
            ".csv",
            ".json",
            ".xml",
            ".md",
            ".yaml",
            ".yml",
            ".ini",
            ".cfg",
        ]
        if any(filename.lower().endswith(ext) for ext in text_extensions):
            return True

        # Check by MIME type if provided
        if mime_type:
            text_mime_prefixes = [
                "text/",
                "application/json",
                "application/xml",
                "application/yaml",
            ]
            if any(
                mime_type.lower().startswith(prefix) for prefix in text_mime_prefixes
            ):
                return True

        return False

    def _process_attachment(
        self, attachment: Any, index: int
    ) -> tuple[str, str] | None:
        """Process a single attachment and return attachment info."""
        attachment_info = []

        # Extract basic attachment properties
        filename = self._extract_attachment_filename(attachment)
        if filename:
            attachment_info.append(f"Filename: {filename}")

        # Extract size information
        if size_info := self._extract_attachment_size(attachment):
            attachment_info.append(size_info)

        # Extract content type
        content_type = self._extract_attachment_content_type(attachment, filename)
        if content_type:
            attachment_info.append(f"Type: {content_type}")

        # Extract content if enabled
        if self.include_attachment_content:
            if content_info := self._extract_attachment_content(
                attachment, filename, content_type
            ):
                attachment_info.append(content_info)

        if attachment_info:
            return (f"attachment_{index}", "\n".join(attachment_info))
        return None

    def _extract_attachment_filename(self, attachment: Any) -> str | None:
        """Extract filename from attachment."""
        filename_props = [
            "filename",
            "Filename",
            "name",
            "Name",
            "displayname",
            "DisplayName",
            "longFilename",
            "LongFilename",
        ]
        return self._get_attachment_property(attachment, filename_props)

    def _extract_attachment_size(self, attachment: Any) -> str | None:
        """Extract size information from attachment."""
        size_props = ["size", "Size", "filesize", "FileSize", "length", "Length"]
        size = self._get_attachment_property(attachment, size_props)

        if size is not None:
            try:
                size_val = size if isinstance(size, int) else int(size)
                return f"Size: {size_val:,} bytes"
            except (ValueError, TypeError):
                return f"Size: {size}"
        return None

    def _extract_attachment_content_type(
        self, attachment: Any, filename: str | None
    ) -> str | None:
        """Extract content type from attachment."""
        content_type_props = [
            "content_type",
            "ContentType",
            "mime_type",
            "MimeType",
            "type",
            "Type",
        ]
        content_type = self._get_attachment_property(attachment, content_type_props)

        # If no content type found but we have a filename, compute it from extension
        if not content_type and filename:
            content_type = self._get_mime_type_from_filename(str(filename))

        return content_type

    def _extract_attachment_content(
        self, attachment: Any, filename: str | None, content_type: str | None
    ) -> str | None:
        """Extract content from text-based attachments."""
        content_props = ["content", "Content", "data", "Data", "body", "Body"]
        content = self._get_attachment_property(attachment, content_props)

        if (
            content is not None
            and filename
            and self._is_text_attachment(str(filename), content_type)
        ):
            try:
                # Handle different content formats
                if isinstance(content, bytes):
                    content_str = content.decode("utf-8", errors="ignore")
                elif hasattr(content, "decode"):  # bytes-like object
                    content_str = content.decode("utf-8", errors="ignore")
                else:
                    content_str = str(content)

                if content_str and content_str.strip():
                    # Limit content length to prevent overwhelming output
                    max_content_length = 1000
                    if len(content_str) > max_content_length:
                        content_str = content_str[:max_content_length] + "..."
                    return f"Content:\n{content_str}"
            except Exception as e:
                logger.warning(
                    f"Failed to extract content from attachment {filename}: {e}"
                )

        return None

    def _apply_standard_chunking(
        self, content: str, metadata: dict[str, Any], source: str
    ) -> list:
        """Apply standard chunking strategies."""
        if SentenceSplitter is None or TokenTextSplitter is None:
            # Fallback if LlamaIndex not available
            chunks = MsgChunker._split_text_into_chunks(
                content, self.chunk_size, self.chunk_overlap
            )
            return MsgDocumentFactory.create_documents_from_chunks(
                chunks, metadata, source, self.chunk_strategy
            )

        splitter: Union[SentenceSplitter, TokenTextSplitter]
        if self.chunk_strategy == "sentences":
            splitter = SentenceSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
        elif self.chunk_strategy == "paragraphs":
            # Split by paragraphs and then apply chunking
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            full_text = "\n\n".join(paragraphs)
            chunks = MsgChunker.chunk_by_paragraphs(
                full_text, self.chunk_size, self.chunk_overlap
            )
            return MsgDocumentFactory.create_documents_from_chunks(
                chunks, metadata, source, "paragraphs"
            )
        else:  # characters or tokens
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )

        # Create a temporary document for splitting
        try:
            from llama_index.core.schema import Document as LlamaDocument

            temp_doc = LlamaDocument(text=content)
            nodes = splitter.get_nodes_from_documents([temp_doc])
        except ImportError:
            # Fallback if LlamaIndex not available
            chunks = MsgChunker._split_text_into_chunks(
                content, self.chunk_size, self.chunk_overlap
            )
            return MsgDocumentFactory.create_documents_from_chunks(
                chunks, metadata, source, self.chunk_strategy
            )

        chunks = [node.text if hasattr(node, "text") else str(node) for node in nodes]
        return MsgDocumentFactory.create_documents_from_chunks(
            chunks, metadata, source, self.chunk_strategy
        )
