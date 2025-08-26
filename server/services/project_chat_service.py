import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from agents.project_chat_orchestrator import (
    ProjectChatOrchestratorAgent,
    ProjectChatOrchestratorAgentInputSchema,
)
from context_providers.project_chat_context_provider import (
    ChunkItem,
    ProjectChatContextProvider,
)
from core.logging import FastAPIStructLogger
from services.rag_subprocess import search_with_rag

logger = FastAPIStructLogger()
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig  # noqa: E402


class ProjectChatService:
    def __init__(self):
        self.rag_api_cache: dict[str, Any] = {}

    def _create_rag_config_from_strategy(self, strategy) -> dict[str, Any]:
        """Convert LlamaFarm strategy config to RAG API compatible config."""
        components = strategy.components

        # Ensure JSON-serializable content: use enum .value and json-mode dumps
        def enum_value(value: Any) -> Any:
            return getattr(value, "value", value)

        embedder_type = enum_value(components.embedder.type)
        embedder_config = components.embedder.config.model_dump(mode="json")

        vector_store_type = enum_value(components.vector_store.type)
        vector_store_config = components.vector_store.config.model_dump(mode="json")

        retrieval_type_raw = components.retrieval_strategy.type
        retrieval_type = enum_value(retrieval_type_raw)
        # Some Literal types are already plain strings; keep as-is
        if not isinstance(retrieval_type, str | int | float | bool | type(None)):
            retrieval_type = str(retrieval_type)
        retrieval_config = components.retrieval_strategy.config.model_dump(mode="json")

        return {
            "version": "2.0",
            "embedders": {
                "default": {
                    "type": embedder_type,
                    "config": embedder_config,
                }
            },
            "vector_stores": {
                "default": {
                    "type": vector_store_type,
                    "config": vector_store_config,
                }
            },
            "retrieval_strategies": {
                "default": {
                    "type": retrieval_type,
                    "config": retrieval_config,
                }
            },
        }

    def _perform_rag_search(
        self, project_config: LlamaFarmConfig, message: str, top_k: int = 5
    ) -> list[Any]:
        """Perform RAG search using the project's RAG configuration.

        This implementation shells out to the rag subsystem in its own
        Python environment to avoid cross-venv import issues.
        """
        logger.info(f"Performing RAG search for message: {message}")

        # For now, use the first available strategy
        if not project_config.rag.strategies:
            logger.error("No RAG strategies found in project config")
            return []

        strategy = project_config.rag.strategies[0]
        # Use shared helper to run RAG search
        results = search_with_rag(strategy, message, top_k=top_k)
        if results is None:
            results = []

        normalized = [
            type(
                "RagResult",
                (),
                {
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                    "score": item.get("score", 0.0),
                },
            )()
            for item in results
        ]
        logger.info(f"RAG search returned {len(normalized)} results")
        return normalized

    def chat(
        self,
        project_config: LlamaFarmConfig,
        chat_agent: ProjectChatOrchestratorAgent,
        message: str,
    ) -> ChatCompletion:
        context_provider = ProjectChatContextProvider(title="Project Chat Context")

        chat_agent.register_context_provider("project_chat_context", context_provider)

        # Use the RAG subsystem to perform RAG based on the project config
        rag_results = self._perform_rag_search(project_config, message)

        # Store the result from the RAG subsystem in the agent's context provider
        for idx, result in enumerate(rag_results):
            chunk_item = ChunkItem(
                content=result.content,
                metadata={
                    "source": result.metadata.get("source", "unknown"),
                    "score": getattr(result, "score", 0.0),
                    "chunk_index": idx,
                    "retrieval_method": "rag_search",
                    **result.metadata,
                },
            )
            context_provider.chunks.append(chunk_item)

        input_schema = ProjectChatOrchestratorAgentInputSchema(chat_message=message)
        agent_response = chat_agent.run(input_schema)

        response_message = agent_response.chat_message

        completion = ChatCompletion(
            id=f"chat-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=project_config.runtime.model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content=response_message,
                    ),
                    finish_reason="stop",
                )
            ],
        )
        return completion


project_chat_service = ProjectChatService()
