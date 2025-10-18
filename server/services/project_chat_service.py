import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from config.datamodel import LlamaFarmConfig  # noqa: E402
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

from agents.chat_orchestrator import ChatOrchestratorAgent
from agents.llamagent.agent import LlamAgent
from agents.llamagent.history import LlamAgentChatMessage
from context_providers.rag_context_provider import (
    ChunkItem,
    RAGContextProvider,
)
from core.logging import FastAPIStructLogger
from services.rag_service import search_with_rag

logger = FastAPIStructLogger()


FALLBACK_ECHO_RESPONSE = (
    "I notice my previous response wasn't very helpful. Let me try to provide a better answer. "
    "Could you provide more specific details about what you're looking for? "
    "For example, if you're asking about a particular feature or need help with something specific, "
    "please let me know and I'll do my best to assist you properly."
)

# Echo detection constants for better code clarity
MIN_LENGTH_RATIO = 0.3  # Minimum length ratio (candidate/input) to avoid echo detection
LENGTH_EXTENSION_FACTOR = 1.2  # Factor by which candidate must exceed input length
SIMILARITY_THRESHOLD = 0.8  # Minimum word similarity ratio to trigger echo detection
SIMILARITY_LENGTH_FACTOR = (
    1.5  # Maximum length multiplier for similarity-based echo detection
)


class ProjectChatService:
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
        self,
        project_dir: str,
        project_config: LlamaFarmConfig,
        message: str,
        top_k: int = 5,
        database: str | None = None,
    ) -> list[Any]:
        """Perform RAG search using the project's RAG configuration.

        This implementation searches the database directly, not through datasets.
        """

        # First, make sure rag is enabled
        if not project_config.rag:
            logger.warning("RAG is not enabled in project config. Skipping.")
            return []

        logger.info(f"Performing RAG search for message: {message}")

        # Find the database configuration
        if not database:
            # Use the first database as default
            if project_config.rag.databases:
                database = str(project_config.rag.databases[0].name)
                logger.info(f"Using default database: {database}")
            else:
                logger.error("No databases found in project config")
                return []

        # Use shared helper to run RAG search on database
        results = search_with_rag(project_dir, database, message, top_k=top_k)
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

    def _clear_rag_context_provider(self, chat_agent: LlamAgent) -> None:
        try:
            chat_agent.remove_context_provider("project_chat_context")
        except Exception:
            logger.warning("Failed to clear RAG context provider", exc_info=True)

    async def chat(
        self,
        *,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: ChatOrchestratorAgent,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> ChatCompletion:
        response_message = ""
        async for chunk in self.stream_chat(
            project_dir=project_dir,
            project_config=project_config,
            chat_agent=chat_agent,
            message=message,
            rag_enabled=rag_enabled,
            database=database,
            rag_top_k=rag_top_k,
            rag_score_threshold=rag_score_threshold,
        ):
            response_message += chunk

        completion = ChatCompletion(
            id=f"chat-{uuid.uuid4()}",
            object="chat.completion",
            created=int(time.time()),
            model=chat_agent.model_name,
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

    async def stream_chat(
        self,
        *,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: LlamAgent,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield assistant content chunks, using agent-native streaming if available."""
        self._clear_rag_context_provider(chat_agent)
        context_provider = RAGContextProvider(title="Project Chat Context")
        chat_agent.register_context_provider("project_chat_context", context_provider)

        # Use config defaults if not explicitly provided (same logic as chat method)
        if rag_enabled is None:
            rag_enabled = bool(project_config.rag and project_config.rag.databases)
            if rag_enabled:
                logger.info("RAG enabled by default based on project configuration")

        if rag_enabled and project_config.rag:
            if database is None and project_config.rag.databases:
                database = project_config.rag.databases[0].name
                logger.info(f"Using default database from config: {database}")

            if rag_top_k is None:
                if project_config.rag.databases:
                    for db in project_config.rag.databases:
                        if db.name == database:
                            for strategy in db.retrieval_strategies or []:
                                if strategy.default:
                                    rag_top_k = (
                                        strategy.config.top_k
                                        if (
                                            strategy.config
                                            and hasattr(strategy.config, "top_k")
                                        )
                                        else 5
                                    )
                                    break
                            break
                if rag_top_k is None:
                    rag_top_k = 5

        rag_results = []
        if rag_enabled:
            rag_results = self._perform_rag_search(
                project_dir,
                project_config,
                message,
                top_k=rag_top_k or 5,
                database=database,
            )
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

        user_input = LlamAgentChatMessage(role="user", content=message)
        try:
            logger.info("Running async stream")
            previous_response = ""
            emitted = False
            async for chunk in chat_agent.run_async_stream(user_input=user_input):
                if chunk:
                    emitted = True
                    yield chunk

            if not emitted:
                yield previous_response
        except Exception:
            logger.error(
                "Model call failed",
                exc_info=True,
            )


project_chat_service = ProjectChatService()
