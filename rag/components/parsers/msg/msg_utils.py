"""Utility functions for MSG parsing."""

import tempfile
import os
from contextlib import contextmanager
from typing import Any

from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.parsers.msg.msg_utils")


class MsgMetadataExtractor:
    """Extract metadata from MSG files."""

    @staticmethod
    def extract_email_properties(msg, base_metadata: dict[str, Any]) -> dict[str, Any]:
        """Extract email properties from MSG file."""
        metadata = base_metadata.copy()

        try:
            # Basic email properties
            if hasattr(msg, "subject") and msg.subject:
                metadata["subject"] = msg.subject

            if hasattr(msg, "sender") and msg.sender:
                metadata["sender"] = msg.sender

            if hasattr(msg, "to") and msg.to:
                metadata["to"] = msg.to

            if hasattr(msg, "cc") and msg.cc:
                metadata["cc"] = msg.cc

            if hasattr(msg, "bcc") and msg.bcc:
                metadata["bcc"] = msg.bcc

            # Date properties
            if hasattr(msg, "date") and msg.date:
                metadata["date"] = str(msg.date)
                metadata["email_date"] = str(msg.date)

            if hasattr(msg, "receivedTime") and msg.receivedTime:
                metadata["received_time"] = str(msg.receivedTime)

            # Message properties
            if hasattr(msg, "messageId") and msg.messageId:
                metadata["message_id"] = msg.messageId

            if hasattr(msg, "importance") and msg.importance is not None:
                metadata["importance"] = msg.importance

            if hasattr(msg, "priority") and msg.priority is not None:
                metadata["priority"] = msg.priority

            # Attachment count
            if hasattr(msg, "attachments") and msg.attachments:
                metadata["attachment_count"] = len(msg.attachments)
                attachment_names = []
                for attachment in msg.attachments:
                    name = getattr(attachment, "longFilename", "") or getattr(
                        attachment, "shortFilename", ""
                    )
                    if name:
                        attachment_names.append(name)
                if attachment_names:
                    metadata["attachment_names"] = attachment_names
            else:
                metadata["attachment_count"] = 0

            # Message size
            if hasattr(msg, "size") and msg.size:
                metadata["message_size"] = msg.size

            # Content type information
            has_html = bool(getattr(msg, "htmlBody", None))
            has_text = bool(getattr(msg, "body", None))
            metadata["has_html_body"] = has_html
            metadata["has_text_body"] = has_text

            if has_html and has_text:
                metadata["body_format"] = "both"
            elif has_html:
                metadata["body_format"] = "html"
            elif has_text:
                metadata["body_format"] = "text"
            else:
                metadata["body_format"] = "none"

        except Exception as e:
            logger.warning(f"Failed to extract some email properties: {e}")

        return metadata

    @staticmethod
    def extract_headers_as_text(msg) -> str:
        """Extract email headers as formatted text."""
        headers = []

        try:
            if hasattr(msg, "subject") and msg.subject:
                headers.append(f"Subject: {msg.subject}")

            if hasattr(msg, "sender") and msg.sender:
                headers.append(f"From: {msg.sender}")

            if hasattr(msg, "to") and msg.to:
                headers.append(f"To: {msg.to}")

            if hasattr(msg, "cc") and msg.cc:
                headers.append(f"CC: {msg.cc}")

            if hasattr(msg, "bcc") and msg.bcc:
                headers.append(f"BCC: {msg.bcc}")

            if hasattr(msg, "date") and msg.date:
                headers.append(f"Date: {msg.date}")

            if hasattr(msg, "messageId") and msg.messageId:
                headers.append(f"Message-ID: {msg.messageId}")

        except Exception as e:
            logger.warning(f"Failed to extract headers: {e}")

        return "\n".join(headers) if headers else ""


class MsgChunker:
    """Chunking utilities for MSG content."""

    @staticmethod
    def chunk_by_email_sections(
        content_parts: list[tuple[str, str]],
        base_metadata: dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        """Chunk content by email sections (headers, body, attachments)."""
        documents = []

        for section_type, content in content_parts:
            if not content.strip():
                continue

            # Create metadata for this section
            section_metadata = base_metadata.copy()
            section_metadata["section_type"] = section_type

            # If content is small enough, create single document
            if len(content) <= chunk_size:
                documents.append(
                    MsgDocumentFactory.create_single_document(
                        content, section_metadata, base_metadata.get("source", "")
                    )
                )
            else:
                # Split large sections into chunks
                chunks = MsgChunker._split_text_into_chunks(
                    content, chunk_size, chunk_overlap
                )
                section_docs = MsgDocumentFactory.create_documents_from_chunks(
                    chunks,
                    section_metadata,
                    base_metadata.get("source", ""),
                    section_type,
                )
                documents.extend(section_docs)

        return documents

    @staticmethod
    def chunk_by_paragraphs(
        text: str, chunk_size: int, chunk_overlap: int
    ) -> list[str]:
        """Chunk text by paragraphs, respecting chunk size limits."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        if not paragraphs:
            return []

        chunks = []
        current_chunk: list[str] = []
        current_size = 0

        for paragraph in paragraphs:
            para_size = len(paragraph)

            # If adding this paragraph would exceed chunk size
            if current_size + para_size > chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)

                # Start new chunk with overlap
                if chunk_overlap > 0:
                    # Calculate how much overlap to include
                    overlap_text = (
                        chunk_text[-chunk_overlap:]
                        if len(chunk_text) > chunk_overlap
                        else chunk_text
                    )
                    current_chunk = [overlap_text, paragraph]
                    current_size = len(overlap_text) + para_size + 2  # +2 for \n\n
                else:
                    current_chunk = [paragraph]
                    current_size = para_size
            else:
                # Add paragraph to current chunk
                current_chunk.append(paragraph)
                current_size += para_size + (
                    2 if current_chunk else 0
                )  # +2 for \n\n separator

        # Add final chunk if it exists
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))

        return chunks

    @staticmethod
    def _split_text_into_chunks(
        text: str, chunk_size: int, chunk_overlap: int
    ) -> list[str]:
        """Split text into chunks with overlap."""
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for a space or newline near the end
                for i in range(
                    min(50, chunk_size // 4)
                ):  # Look back up to 50 chars or 1/4 of chunk
                    if end - i > start and text[end - i] in [" ", "\n", "\t"]:
                        end = end - i
                        break

            if chunk := text[start:end].strip():
                chunks.append(chunk)

            # Move start position with overlap
            start = end - chunk_overlap if chunk_overlap > 0 else end

            # Prevent infinite loop
            if start >= len(text):
                break

        return chunks


class MsgDocumentFactory:
    """Factory for creating Document objects from MSG content."""

    @staticmethod
    def create_single_document(
        content: str, metadata: dict[str, Any], source: str
    ) -> Document:
        """Create a single document from content."""
        doc_metadata = metadata.copy()
        doc_metadata["chunk_index"] = 0
        doc_metadata["total_chunks"] = 1
        doc_metadata["chunk_type"] = "full_document"

        return Document(content=content, metadata=doc_metadata, source=source)

    @staticmethod
    def create_documents_from_chunks(
        chunks: list[str],
        base_metadata: dict[str, Any],
        source: str,
        chunk_strategy: str,
    ) -> list[Document]:
        """Create documents from text chunks."""
        documents = []

        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue

            chunk_metadata = base_metadata.copy() | {
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_strategy": chunk_strategy,
                "chunk_type": "chunk",
            }

            documents.append(
                Document(content=chunk, metadata=chunk_metadata, source=source)
            )

        return documents


@contextmanager
def MsgTempFileHandler(data: bytes):
    """Context manager for handling temporary MSG files."""
    tmp_fd = None
    tmp_path = None

    try:
        # Create temporary file with .msg extension
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".msg")

        # Write data to temporary file
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(data)
            tmp_fd = None  # File is now closed

        yield tmp_path

    finally:
        # Clean up
        if tmp_fd is not None:
            try:
                os.close(tmp_fd)
            except Exception:
                pass

        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
