"""
Service for loading and matching documentation files for contextual injection.

Provides a lightweight alternative to full RAG by matching user queries to relevant
documentation using keyword-based heuristics.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from core.logging import FastAPIStructLogger

logger = FastAPIStructLogger(__name__)

# Keyword to documentation file mappings
# Keys are lowercase keywords/phrases, values are lists of doc paths relative to docs/website/docs/
KEYWORD_TO_DOCS: Dict[str, List[str]] = {
    "rag": ["rag/index.md"],
    "retrieval": ["rag/index.md"],
    "vector": ["rag/index.md"],
    "embedding": ["rag/index.md"],
    "dataset": ["cli/lf-datasets.md", "quickstart/index.md"],
    "datasets": ["cli/lf-datasets.md", "quickstart/index.md"],
    "ingest": ["cli/lf-datasets.md", "rag/index.md"],
    "cli": ["cli/index.md"],
    "command": ["cli/index.md"],
    "commands": ["cli/index.md"],
    "config": ["configuration/index.md"],
    "configuration": ["configuration/index.md"],
    "yaml": ["configuration/index.md"],
    "llamafarm.yaml": ["configuration/index.md"],
    "quickstart": ["quickstart/index.md"],
    "getting started": ["quickstart/index.md"],
    "get started": ["quickstart/index.md"],
    "start": ["cli/lf-start.md", "quickstart/index.md"],
    "init": ["cli/lf-init.md", "quickstart/index.md"],
    "model": ["models/index.md"],
    "models": ["models/index.md"],
    "runtime": ["models/index.md"],
    "ollama": ["models/index.md"],
    "provider": ["models/index.md"],
    "extend": ["extending/index.md"],
    "extending": ["extending/index.md"],
    "plugin": ["extending/index.md"],
    "custom": ["extending/index.md"],
    "troubleshoot": ["troubleshooting/index.md"],
    "troubleshooting": ["troubleshooting/index.md"],
    "error": ["troubleshooting/index.md"],
    "problem": ["troubleshooting/index.md"],
    "issue": ["troubleshooting/index.md"],
    "example": ["examples/index.md"],
    "examples": ["examples/index.md"],
    "prompt": ["prompts/index.md"],
    "prompts": ["prompts/index.md"],
    "chat": ["cli/lf-chat.md"],
    "query": ["cli/lf-rag.md"],
}


class DocsContextService:
    """Service for loading and matching documentation files."""

    def __init__(self, docs_root: Optional[Path] = None):
        """
        Initialize the docs context service.

        Args:
            docs_root: Path to the docs/website/docs directory. If None, auto-detects from server location.
        """
        if docs_root is None:
            # Auto-detect: server/ is sibling to docs/
            server_dir = Path(__file__).parent.parent
            docs_root = server_dir.parent / "docs" / "website" / "docs"

        self.docs_root = docs_root
        self._doc_cache: Dict[str, str] = {}
        self._load_doc_cache()

    def _load_doc_cache(self) -> None:
        """Load all documentation files into memory cache."""
        if not self.docs_root.exists():
            logger.warning(
                f"Docs root not found at {self.docs_root}, doc injection disabled"
            )
            return

        # Collect all unique doc paths from mappings
        unique_docs = set()
        for doc_list in KEYWORD_TO_DOCS.values():
            unique_docs.update(doc_list)

        loaded_count = 0
        for doc_rel_path in unique_docs:
            doc_path = self.docs_root / doc_rel_path
            if doc_path.exists() and doc_path.is_file():
                try:
                    content = doc_path.read_text(encoding="utf-8")
                    self._doc_cache[doc_rel_path] = content
                    loaded_count += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to load doc {doc_rel_path}: {e}", exc_info=True
                    )
            else:
                logger.warning(f"Doc file not found: {doc_path}")

        logger.info(
            f"Loaded {loaded_count} documentation files into cache",
            docs_root=str(self.docs_root),
            unique_docs=len(unique_docs),
        )

    def match_docs_for_query(
        self, query: str, max_docs: int = 2, max_lines_per_doc: int = 150
    ) -> List[tuple[str, str]]:
        """
        Match documentation files based on keywords in the query.

        Args:
            query: User query string
            max_docs: Maximum number of docs to return
            max_lines_per_doc: Maximum lines per doc (truncates long docs)

        Returns:
            List of (doc_path, doc_content) tuples
        """
        if not self._doc_cache:
            return []

        query_lower = query.lower()

        # Find matching docs using keyword mapping
        matched_docs: Dict[str, int] = {}  # doc_path -> priority score
        for keyword, doc_paths in KEYWORD_TO_DOCS.items():
            if keyword in query_lower:
                for doc_path in doc_paths:
                    # Higher score for exact keyword matches, bonus for shorter keywords (more specific)
                    score = matched_docs.get(doc_path, 0) + 10 - len(keyword.split())
                    matched_docs[doc_path] = score

        if not matched_docs:
            logger.debug(f"No docs matched for query: {query[:100]}")
            return []

        # Sort by score (highest first), then by doc path length (shorter = more specific)
        sorted_docs = sorted(
            matched_docs.items(), key=lambda x: (-x[1], len(x[0]))
        )[:max_docs]

        results = []
        for doc_path, score in sorted_docs:
            content = self._doc_cache.get(doc_path, "")
            if content:
                # Truncate long docs
                lines = content.splitlines()
                if len(lines) > max_lines_per_doc:
                    content = "\n".join(lines[:max_lines_per_doc])
                    content += f"\n\n... (truncated, {len(lines) - max_lines_per_doc} more lines)"

                results.append((doc_path, content))
                logger.debug(
                    f"Matched doc for query",
                    doc_path=doc_path,
                    score=score,
                    query_preview=query[:50],
                )

        return results

    def format_doc_for_context(self, doc_path: str, doc_content: str) -> str:
        """
        Format a documentation file for context injection.

        Args:
            doc_path: Relative path to the doc file
            doc_content: Content of the doc file

        Returns:
            Formatted markdown string
        """
        return f"""## Documentation: {doc_path}

{doc_content}

---
"""

    def format_multiple_docs(self, docs: List[tuple[str, str]]) -> str:
        """
        Format multiple docs for context injection.

        Args:
            docs: List of (doc_path, doc_content) tuples

        Returns:
            Combined formatted markdown string
        """
        if not docs:
            return ""

        formatted = []
        for doc_path, doc_content in docs:
            formatted.append(self.format_doc_for_context(doc_path, doc_content))

        return "\n".join(formatted)


# Global service instance
_service: Optional[DocsContextService] = None


def get_docs_service() -> DocsContextService:
    """Get or create the global DocsContextService instance."""
    global _service
    if _service is None:
        _service = DocsContextService()
    return _service
