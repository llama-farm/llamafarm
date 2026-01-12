"""Simple text chunker with multiple strategies."""

import re
from typing import Any

from .base import BaseChunker, ChunkingStrategy, ChunkResult


class SimpleChunker(BaseChunker):
    """Simple chunker supporting multiple splitting strategies.

    Strategies:
    - characters: Fixed character count splitting
    - sentences: Split on sentence boundaries
    - paragraphs: Split on paragraph boundaries (double newlines)
    - sections: Split on markdown headers
    - pages: Split on page markers from metadata
    - tokens: Token-based splitting (requires tokenizer)
    """

    # Sentence boundary patterns
    SENTENCE_ENDINGS = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')

    # Paragraph boundary pattern (two or more newlines)
    PARAGRAPH_BOUNDARY = re.compile(r'\n\s*\n')

    # Markdown header pattern
    MARKDOWN_HEADER = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)

    # Page marker pattern (commonly used in parsed documents)
    PAGE_MARKER = re.compile(r'\[Page\s*(\d+)\]|\n---\s*Page\s*(\d+)\s*---\n|<!--\s*page:\s*(\d+)\s*-->', re.IGNORECASE)

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 0,
        strategy: ChunkingStrategy | str = ChunkingStrategy.CHARACTERS,
        config: dict | None = None,
    ):
        """Initialize SimpleChunker.

        Args:
            chunk_size: Target size for chunks
            overlap: Number of characters/sentences/etc. to overlap
            strategy: Chunking strategy to use
            config: Additional configuration
        """
        # Convert string to enum if needed
        if isinstance(strategy, str):
            strategy = ChunkingStrategy(strategy.lower())

        super().__init__(
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
            config=config,
        )

        # Strategy-specific settings from config
        self.min_chunk_size = config.get("min_chunk_size", 50) if config else 50
        self.max_chunk_size = config.get("max_chunk_size", chunk_size * 2) if config else chunk_size * 2
        self.preserve_whitespace = config.get("preserve_whitespace", False) if config else False

    def chunk(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> list[ChunkResult]:
        """Split text into chunks using the configured strategy.

        Args:
            text: Text content to chunk
            metadata: Optional metadata to include in chunks

        Returns:
            List of ChunkResult objects
        """
        if not text or not text.strip():
            return []

        # Select chunking method based on strategy
        if self.strategy == ChunkingStrategy.CHARACTERS:
            chunks = self._chunk_by_characters(text)
        elif self.strategy == ChunkingStrategy.SENTENCES:
            chunks = self._chunk_by_sentences(text)
        elif self.strategy == ChunkingStrategy.PARAGRAPHS:
            chunks = self._chunk_by_paragraphs(text)
        elif self.strategy == ChunkingStrategy.SECTIONS:
            chunks = self._chunk_by_sections(text)
        elif self.strategy == ChunkingStrategy.PAGES:
            chunks = self._chunk_by_pages(text, metadata)
        elif self.strategy == ChunkingStrategy.TOKENS:
            chunks = self._chunk_by_tokens(text)
        else:
            # Default to character-based
            chunks = self._chunk_by_characters(text)

        # Build results with metadata
        total_chunks = len(chunks)
        results = []

        for i, (content, start, end, overlap_before, overlap_after) in enumerate(chunks):
            chunk_metadata = self._build_chunk_metadata(metadata, i, total_chunks)

            result = ChunkResult(
                content=content,
                metadata=chunk_metadata,
                chunk_index=i,
                total_chunks=total_chunks,
                start_offset=start,
                end_offset=end,
                strategy=self.strategy.value,
                overlap_before=overlap_before,
                overlap_after=overlap_after,
            )
            results.append(result)

        return results

    def _chunk_by_characters(self, text: str) -> list[tuple[str, int, int, int, int]]:
        """Split text by character count with overlap.

        Returns: List of (content, start, end, overlap_before, overlap_after)
        """
        chunks = []
        text_len = len(text)

        if text_len <= self.chunk_size:
            return [(text, 0, text_len, 0, 0)]

        start = 0
        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # Calculate overlap content
            overlap_before = 0
            if start > 0 and self.overlap > 0:
                overlap_start = max(0, start - self.overlap)
                overlap_before = start - overlap_start

            overlap_after = 0
            if end < text_len and self.overlap > 0:
                overlap_after = min(self.overlap, text_len - end)

            # Extract chunk with overlap
            actual_start = start - overlap_before
            actual_end = end + overlap_after
            content = text[actual_start:actual_end]

            chunks.append((content, start, end, overlap_before, overlap_after))

            # Move to next chunk (non-overlapping portion)
            start = end

        return chunks

    def _chunk_by_sentences(self, text: str) -> list[tuple[str, int, int, int, int]]:
        """Split text by sentence boundaries.

        Groups sentences to approximately match chunk_size.
        """
        # Split into sentences
        sentences = self.SENTENCE_ENDINGS.split(text)

        if len(sentences) <= 1:
            return [(text, 0, len(text), 0, 0)]

        chunks = []
        current_chunk = []
        current_length = 0
        current_start = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_len = len(sentence)

            # If adding this sentence exceeds chunk_size, save current chunk
            if current_length + sentence_len > self.chunk_size and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunk_end = current_start + len(chunk_text)
                chunks.append((chunk_text, current_start, chunk_end, 0, 0))

                # Handle overlap (keep some sentences from end)
                if self.overlap > 0:
                    # Keep last few sentences for overlap
                    overlap_sentences = []
                    overlap_len = 0
                    for s in reversed(current_chunk):
                        if overlap_len + len(s) <= self.overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break
                    current_chunk = overlap_sentences
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0

                current_start = chunk_end

            current_chunk.append(sentence)
            current_length += sentence_len + 1  # +1 for space

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunks.append((chunk_text, current_start, current_start + len(chunk_text), 0, 0))

        return chunks

    def _chunk_by_paragraphs(self, text: str) -> list[tuple[str, int, int, int, int]]:
        """Split text by paragraph boundaries.

        Groups paragraphs to approximately match chunk_size.
        """
        # Split into paragraphs
        paragraphs = self.PARAGRAPH_BOUNDARY.split(text)

        if len(paragraphs) <= 1:
            return [(text, 0, len(text), 0, 0)]

        chunks = []
        current_chunk = []
        current_length = 0
        current_start = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_len = len(para)

            # If adding this paragraph exceeds chunk_size, save current chunk
            if current_length + para_len > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunk_end = current_start + len(chunk_text)
                chunks.append((chunk_text, current_start, chunk_end, 0, 0))

                # Handle overlap
                if self.overlap > 0:
                    overlap_paras = []
                    overlap_len = 0
                    for p in reversed(current_chunk):
                        if overlap_len + len(p) <= self.overlap:
                            overlap_paras.insert(0, p)
                            overlap_len += len(p) + 2
                        else:
                            break
                    current_chunk = overlap_paras
                    current_length = overlap_len
                else:
                    current_chunk = []
                    current_length = 0

                current_start = chunk_end

            current_chunk.append(para)
            current_length += para_len + 2  # +2 for paragraph separator

        # Add final chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append((chunk_text, current_start, current_start + len(chunk_text), 0, 0))

        return chunks

    def _chunk_by_sections(self, text: str) -> list[tuple[str, int, int, int, int]]:
        """Split text by markdown headers.

        Creates a chunk for each section (header + content until next header).
        """
        # Find all headers
        headers = list(self.MARKDOWN_HEADER.finditer(text))

        if not headers:
            # No headers, fall back to paragraphs
            return self._chunk_by_paragraphs(text)

        chunks = []

        # Add content before first header (if any)
        if headers[0].start() > 0:
            pre_content = text[:headers[0].start()].strip()
            if pre_content:
                chunks.append((pre_content, 0, headers[0].start(), 0, 0))

        # Process each section
        for i, header in enumerate(headers):
            start = header.start()

            # Find end of section (start of next header or end of text)
            if i + 1 < len(headers):
                end = headers[i + 1].start()
            else:
                end = len(text)

            section_text = text[start:end].strip()

            # If section is too large, split it further by paragraphs
            if len(section_text) > self.max_chunk_size:
                # Keep header and split content
                header_text = header.group(0)
                content = section_text[len(header_text):].strip()

                # First chunk is header + some content
                para_chunks = self._chunk_by_paragraphs(content)
                for j, (para_content, p_start, p_end, _, _) in enumerate(para_chunks):
                    if j == 0:
                        chunk_text = f"{header_text}\n\n{para_content}"
                    else:
                        chunk_text = para_content
                    chunks.append((chunk_text, start + p_start, start + p_end, 0, 0))
            else:
                chunks.append((section_text, start, end, 0, 0))

        return chunks

    def _chunk_by_pages(self, text: str, metadata: dict | None = None) -> list[tuple[str, int, int, int, int]]:
        """Split text by page markers or page metadata.

        Uses page markers in text or page_numbers from metadata.
        """
        # Try to find page markers in text
        page_markers = list(self.PAGE_MARKER.finditer(text))

        if page_markers:
            chunks = []

            # Add content before first page marker
            if page_markers[0].start() > 0:
                pre_content = text[:page_markers[0].start()].strip()
                if pre_content:
                    chunks.append((pre_content, 0, page_markers[0].start(), 0, 0))

            # Process each page
            for i, marker in enumerate(page_markers):
                start = marker.start()

                if i + 1 < len(page_markers):
                    end = page_markers[i + 1].start()
                else:
                    end = len(text)

                page_content = text[start:end].strip()
                if page_content:
                    chunks.append((page_content, start, end, 0, 0))

            return chunks if chunks else [(text, 0, len(text), 0, 0)]

        # Check metadata for page_numbers
        if metadata and "page_numbers" in metadata:
            page_numbers = metadata.get("page_numbers", [])
            if len(page_numbers) > 1:
                # This is already chunked by page from parser
                # Return single chunk with page info
                return [(text, 0, len(text), 0, 0)]

        # No page info, fall back to paragraphs
        return self._chunk_by_paragraphs(text)

    def _chunk_by_tokens(self, text: str) -> list[tuple[str, int, int, int, int]]:
        """Split text by token count.

        Uses simple word-based tokenization by default.
        For more accurate tokenization, use Docling's HybridChunker.
        """
        # Simple word tokenization (approximately 1 token per word)
        words = text.split()

        if len(words) <= self.chunk_size:
            return [(text, 0, len(text), 0, 0)]

        chunks = []
        current_words = []
        current_start = 0
        char_pos = 0

        for word in words:
            word_pos = text.find(word, char_pos)
            char_pos = word_pos + len(word)

            if len(current_words) >= self.chunk_size:
                chunk_text = " ".join(current_words)
                chunk_end = current_start + len(chunk_text)
                chunks.append((chunk_text, current_start, chunk_end, 0, 0))

                # Handle overlap
                if self.overlap > 0:
                    current_words = current_words[-self.overlap:]
                else:
                    current_words = []

                current_start = chunk_end

            current_words.append(word)

        # Add final chunk
        if current_words:
            chunk_text = " ".join(current_words)
            chunks.append((chunk_text, current_start, current_start + len(chunk_text), 0, 0))

        return chunks

    def get_chunk_count_estimate(self, text: str) -> int:
        """Estimate number of chunks without actually chunking.

        Useful for planning and progress reporting.
        """
        text_len = len(text)

        if self.strategy == ChunkingStrategy.CHARACTERS:
            if text_len <= self.chunk_size:
                return 1
            return (text_len + self.chunk_size - 1) // self.chunk_size

        elif self.strategy == ChunkingStrategy.SENTENCES:
            sentences = self.SENTENCE_ENDINGS.split(text)
            return max(1, len(sentences) // max(1, self.chunk_size // 100))

        elif self.strategy == ChunkingStrategy.PARAGRAPHS:
            paragraphs = self.PARAGRAPH_BOUNDARY.split(text)
            return max(1, len([p for p in paragraphs if p.strip()]))

        elif self.strategy == ChunkingStrategy.SECTIONS:
            headers = self.MARKDOWN_HEADER.findall(text)
            return max(1, len(headers))

        elif self.strategy == ChunkingStrategy.TOKENS:
            words = len(text.split())
            return max(1, (words + self.chunk_size - 1) // self.chunk_size)

        else:
            return (text_len + self.chunk_size - 1) // self.chunk_size
