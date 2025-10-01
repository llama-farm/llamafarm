"""
Context provider for injecting documentation into agent context.
"""

from atomic_agents.context import BaseDynamicContextProvider
from typing import List


class DocsContextProvider(BaseDynamicContextProvider):
    """
    Context provider that injects relevant documentation based on user queries.

    This provides a lightweight alternative to full RAG by matching keywords
    to pre-loaded documentation files.
    """

    def __init__(self, title: str = "Relevant Documentation"):
        """
        Initialize the docs context provider.

        Args:
            title: Title for the context section
        """
        super().__init__(title=title)
        self.docs: List[tuple[str, str]] = []  # List of (doc_path, content) tuples

    def set_docs(self, docs: List[tuple[str, str]]) -> None:
        """
        Set the documentation to be injected.

        Args:
            docs: List of (doc_path, content) tuples
        """
        self.docs = docs

    def clear_docs(self) -> None:
        """Clear all documentation."""
        self.docs = []

    def get_info(self) -> str:
        """
        Get formatted documentation for context injection.

        Returns:
            Formatted markdown string with all docs
        """
        if not self.docs:
            return ""

        formatted_docs = []
        for doc_path, content in self.docs:
            formatted_docs.append(f"## Documentation: {doc_path}\n\n{content}\n\n{'-' * 80}")

        return "\n\n".join(formatted_docs)
