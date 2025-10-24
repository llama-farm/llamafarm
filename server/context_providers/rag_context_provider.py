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
        return "\n\n".join(
            [
                f"Chunk {idx}:\nMetadata: {item.metadata}\nContent:\n{item.content}\n{'-' * 80}"
                for idx, item in enumerate(self.chunks, 1)
            ]
        )
