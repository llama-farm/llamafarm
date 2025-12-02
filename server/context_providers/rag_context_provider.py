from atomic_agents.context import BaseDynamicContextProvider
from pydantic import BaseModel


class ChunkItem(BaseModel):
    content: str
    metadata: dict


class RAGContextProvider(BaseDynamicContextProvider):
    def __init__(self, title: str):
        super().__init__(title=title)
        self.chunks: list[ChunkItem] = []

    def get_info(self) -> str:
        if not self.chunks:
            return ""

        formatted_chunks = []
        source_counter = 1

        # Process all chunks and assign global source numbers
        for item in self.chunks:
            metadata = item.metadata
            source_file = metadata.get("source", metadata.get("filename", "unknown"))
            score = round(metadata.get("score", 0.0), 3)

            # Build citation info parts
            citation_parts = [f'"{source_file}"']

            # Add page number if available
            if "page" in metadata:
                citation_parts.append(f"Page {metadata['page']}")
            elif "page_number" in metadata:
                citation_parts.append(f"Page {metadata['page_number']}")

            # Add relevance score
            citation_parts.append(f"Relevance: {score}")

            # Format as "Source N: filename, page X, relevance: 0.XX"
            citation_str = ", ".join(citation_parts)

            chunk_text = (
                f"<chunk>\n"
                f"  <chunk_name>Source {source_counter}: {citation_str}</chunk_name>\n"
                f"  <chunk_content>{item.content}</chunk_content>\n"
                f"</chunk>"
            )
            formatted_chunks.append(chunk_text)
            source_counter += 1

        # Add instruction header to prevent markup from appearing in responses
        instruction = (
            "The following information has been retrieved from your knowledge base to help answer the user's question. "
            "Use this information to provide accurate responses, but NEVER include the <chunk>, <chunk_name>, or <chunk_content> "
            "tags in your response. Only provide the actual information and insights based on the content.\n\n"
            "<chunks>\n"
        )

        return instruction + "\n\n".join(formatted_chunks) + "\n</chunks>"
