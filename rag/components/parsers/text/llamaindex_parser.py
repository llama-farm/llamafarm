"""LlamaIndex text parser with advanced chunking capabilities."""

from pathlib import Path
from typing import Dict, Any, List, Optional

from core.logging import RAGStructLogger
logger = RAGStructLogger("rag.components.parsers.text.llamaindex_parser")


class TextParser_LlamaIndex:
    """Advanced text parser using LlamaIndex node parsers for semantic and code-aware chunking."""

    def __init__(
        self,
        name: str = "TextParser_LlamaIndex",
        config: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.config = config or {}
        self.chunk_size = self.config.get("chunk_size", 1000)
        self.chunk_overlap = self.config.get("chunk_overlap", 100)
        self.chunk_strategy = self.config.get("chunk_strategy", "semantic")
        self.encoding = self.config.get("encoding", "utf-8")
        self.clean_text = self.config.get("clean_text", True)
        self.extract_metadata = self.config.get("extract_metadata", True)

        # LlamaIndex-specific config
        self.semantic_buffer_size = self.config.get("semantic_buffer_size", 1)
        self.semantic_breakpoint_percentile_threshold = self.config.get(
            "semantic_breakpoint_percentile_threshold", 95
        )
        self.token_model = self.config.get("token_model", "gpt-3.5-turbo")
        self.preserve_code_structure = self.config.get("preserve_code_structure", True)
        self.detect_language = self.config.get("detect_language", True)
        self.include_prev_next_rel = self.config.get("include_prev_next_rel", True)

    def validate_config(self) -> bool:
        """Validate configuration."""
        return True

    def parse(self, source: str, **kwargs):
        """Parse text using LlamaIndex node parsers."""
        from core.base import Document, ProcessingResult

        try:
            # Try importing LlamaIndex components
            from llama_index.core.node_parser import (
                SentenceSplitter,
                TokenTextSplitter,
                SemanticSplitterNodeParser,
                CodeSplitter,
            )
            from llama_index.core import Document as LlamaDocument
        except ImportError:
            return ProcessingResult(
                documents=[],
                errors=[
                    {
                        "error": "LlamaIndex not installed. Install with: pip install llama-index llama-index-core",
                        "source": source,
                    }
                ],
            )

        path = Path(source)
        if not path.exists():
            return ProcessingResult(
                documents=[],
                errors=[{"error": f"File not found: {source}", "source": source}],
            )

        try:
            # Read the file with encoding detection
            text = self._read_file(path)
            if not text.strip():
                return ProcessingResult(
                    documents=[],
                    errors=[{"error": "No text content found", "source": source}],
                )

            # Detect file type for code parsing
            file_extension = path.suffix.lower()
            is_code_file = self._is_code_file(file_extension)

            # Create LlamaIndex document
            llama_doc = LlamaDocument(
                text=text, metadata={"source": str(path), "file_name": path.name}
            )

            # Choose appropriate parser based on strategy and file type
            if self.chunk_strategy == "semantic":
                try:
                    # Import embedding model for semantic chunking
                    from llama_index.embeddings.openai import OpenAIEmbedding

                    embed_model = OpenAIEmbedding()

                    parser = SemanticSplitterNodeParser(
                        buffer_size=self.semantic_buffer_size,
                        breakpoint_percentile_threshold=self.semantic_breakpoint_percentile_threshold,
                        embed_model=embed_model,
                    )
                except ImportError:
                    # Fallback to sentence splitter if OpenAI embeddings not available
                    logger.warning(
                        "OpenAI embeddings not available, falling back to sentence splitting"
                    )
                    parser = SentenceSplitter(
                        chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
                    )
            elif self.chunk_strategy == "code" and is_code_file:
                # Use code splitter for programming languages
                language = self._detect_programming_language(file_extension)
                parser = CodeSplitter(
                    language=language,
                    chunk_lines=40,  # Reasonable default for code
                    chunk_lines_overlap=15,
                    max_chars=self.chunk_size,
                )
            elif self.chunk_strategy == "tokens":
                parser = TokenTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separator=" ",
                    backup_separators=["\n", ".", "!", "?"],
                    tokenizer=self._get_tokenizer_fn(),
                )
            else:
                # Default to sentence splitter
                parser = SentenceSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                    separator=" ",
                    paragraph_separator="\n\n",
                    secondary_chunking_regex="[^.!?]+[.!?]",
                )

            # Parse the document
            nodes = parser.get_nodes_from_documents([llama_doc])

            # Convert to our Document format
            documents = []
            base_metadata = self._extract_metadata(path, text, is_code_file)

            for i, node in enumerate(nodes):
                node_metadata = base_metadata.copy()
                node_metadata.update(
                    {
                        "chunk_index": i,
                        "total_chunks": len(nodes),
                        "chunk_strategy": self.chunk_strategy,
                        "node_id": node.node_id,
                        "chunk_size_actual": len(node.text),
                    }
                )

                # Add relationships if available
                if hasattr(node, "relationships") and node.relationships:
                    relationships = {}
                    for rel_type, rel_node in node.relationships.items():
                        relationships[str(rel_type)] = (
                            rel_node.node_id
                            if hasattr(rel_node, "node_id")
                            else str(rel_node)
                        )
                    node_metadata["relationships"] = relationships

                doc = Document(
                    content=node.text,
                    metadata=node_metadata,
                    id=f"{path.stem}_chunk_{i + 1}",
                    source=str(path),
                )
                documents.append(doc)

            return ProcessingResult(
                documents=documents,
                errors=[],
                metrics={
                    "total_documents": len(documents),
                    "parser_type": self.name,
                    "tool": "LlamaIndex",
                    "chunk_strategy": self.chunk_strategy,
                    "is_code_file": is_code_file,
                },
            )

        except Exception as e:
            logger.error(f"Failed to parse {source}: {e}")
            return ProcessingResult(
                documents=[], errors=[{"error": str(e), "source": source}]
            )

    def _read_file(self, path: Path) -> str:
        """Read file with encoding detection."""
        encodings = [self.encoding, "utf-8", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(path, "r", encoding=encoding) as f:
                    text = f.read()
                if self.clean_text:
                    text = self._clean_text(text)
                return text
            except UnicodeDecodeError:
                continue

        # Last resort with error replacement
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    def _clean_text(self, text: str) -> str:
        """Clean text while preserving structure."""
        if not self.clean_text:
            return text

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove excessive whitespace but preserve indentation for code
            if line.strip():
                # Preserve leading whitespace for code structure
                leading_spaces = len(line) - len(line.lstrip())
                cleaned_content = " ".join(line.split())
                cleaned_lines.append(" " * leading_spaces + cleaned_content.lstrip())
            else:
                cleaned_lines.append("")

        return "\n".join(cleaned_lines)

    def _is_code_file(self, extension: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".cs",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".swift",
            ".kt",
            ".scala",
            ".r",
            ".sql",
            ".sh",
            ".bash",
            ".zsh",
            ".ps1",
            ".bat",
            ".jsx",
            ".tsx",
            ".vue",
            ".html",
            ".css",
            ".scss",
            ".less",
            ".yaml",
            ".yml",
            ".json",
            ".xml",
            ".toml",
            ".ini",
            ".cfg",
            ".conf",
        }
        return extension in code_extensions

    def _detect_programming_language(self, extension: str) -> str:
        """Map file extension to LlamaIndex CodeSplitter language."""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".cs": "c_sharp",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".swift": "swift",
            ".kt": "kotlin",
            ".scala": "scala",
            ".r": "r",
            ".sql": "sql",
            ".sh": "bash",
            ".bash": "bash",
            ".html": "html",
            ".css": "css",
        }
        return language_map.get(extension, "text")

    def _get_tokenizer_fn(self):
        """Get tokenizer function for token-based chunking."""
        try:
            import tiktoken

            encoding = tiktoken.encoding_for_model(self.token_model)
            return encoding.encode
        except ImportError:
            logger.warning(
                "tiktoken not available, falling back to simple word tokenizer"
            )
            return lambda text: text.split()

    def _extract_metadata(
        self, path: Path, text: str, is_code_file: bool
    ) -> Dict[str, Any]:
        """Extract comprehensive metadata."""
        metadata = {
            "source": str(path),
            "file_name": path.name,
            "parser": self.name,
            "parser_type": self.name,
            "tool": "LlamaIndex",
            "file_size": path.stat().st_size,
            "encoding": self.encoding,
            "is_code_file": is_code_file,
        }

        if self.extract_metadata:
            # Basic text statistics
            lines = text.split("\n")
            metadata.update(
                {
                    "line_count": len(lines),
                    "word_count": len(text.split()),
                    "char_count": len(text),
                    "non_empty_lines": len([line for line in lines if line.strip()]),
                }
            )

            # Code-specific metadata
            if is_code_file:
                metadata.update(self._extract_code_metadata(text, path.suffix))

        return metadata

    def _extract_code_metadata(self, text: str, extension: str) -> Dict[str, Any]:
        """Extract code-specific metadata."""
        lines = text.split("\n")
        code_metadata = {
            "programming_language": self._detect_programming_language(extension),
            "blank_lines": len([line for line in lines if not line.strip()]),
            "comment_lines": 0,
            "estimated_functions": 0,
            "estimated_classes": 0,
        }

        # Simple heuristics for code analysis
        comment_patterns = {
            ".py": ["#"],
            ".js": ["//", "/*"],
            ".jsx": ["//", "/*"],
            ".ts": ["//", "/*"],
            ".tsx": ["//", "/*"],
            ".java": ["//", "/*"],
            ".cpp": ["//", "/*"],
            ".c": ["//", "/*"],
            ".cs": ["//", "/*"],
            ".php": ["//", "#", "/*"],
            ".rb": ["#"],
            ".go": ["//"],
            ".rs": ["//"],
            ".sql": ["--", "/*"],
        }

        function_patterns = {
            ".py": ["def "],
            ".js": ["function ", "const ", "let ", "=>"],
            ".jsx": ["function ", "const ", "let ", "=>"],
            ".ts": ["function ", "const ", "let ", "=>"],
            ".tsx": ["function ", "const ", "let ", "=>"],
            ".java": ["public ", "private ", "protected "],
            ".cpp": ["int ", "void ", "bool "],
            ".c": ["int ", "void ", "bool "],
            ".cs": ["public ", "private ", "protected "],
            ".php": ["function "],
            ".rb": ["def "],
            ".go": ["func "],
            ".rs": ["fn "],
        }

        class_patterns = {
            ".py": ["class "],
            ".java": ["class ", "interface ", "enum "],
            ".cpp": ["class ", "struct "],
            ".c": ["struct "],
            ".cs": ["class ", "interface ", "struct ", "enum "],
            ".rb": ["class ", "module "],
            ".go": ["type "],
            ".rs": ["struct ", "impl ", "trait ", "enum "],
        }

        # Count comments, functions, and classes
        comment_chars = comment_patterns.get(extension, [])
        function_keywords = function_patterns.get(extension, [])
        class_keywords = class_patterns.get(extension, [])
        for line in lines:
            stripped = line.strip()
            if any(stripped.startswith(comment) for comment in comment_chars):
                code_metadata["comment_lines"] += 1
            if any(keyword in stripped for keyword in function_keywords):
                code_metadata["estimated_functions"] += 1
            if any(keyword in stripped for keyword in class_keywords):
                code_metadata["estimated_classes"] += 1

        return code_metadata
