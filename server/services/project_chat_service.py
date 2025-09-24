import sys
import time
import uuid
from collections.abc import AsyncGenerator
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

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from config.datamodel import LlamaFarmConfig  # noqa: E402

logger = FastAPIStructLogger()


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

    def _clear_rag_context_provider(
        self, chat_agent: ProjectChatOrchestratorAgent
    ) -> None:
        try:
            if (
                hasattr(chat_agent, "context_providers")
                and chat_agent.context_providers
            ):
                chat_agent.context_providers.pop("project_chat_context", None)
        except Exception:
            logger.warning("Failed to clear RAG context provider", exc_info=True)

    async def chat(
        self,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: ProjectChatOrchestratorAgent,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> ChatCompletion:
        self._clear_rag_context_provider(chat_agent)
        context_provider = ProjectChatContextProvider(title="Project Chat Context")
        chat_agent.register_context_provider("project_chat_context", context_provider)

        # Use config defaults if not explicitly provided
        # If rag_enabled is None, check if RAG is configured
        if rag_enabled is None:
            rag_enabled = bool(project_config.rag and project_config.rag.databases)
            if rag_enabled:
                logger.info("RAG enabled by default based on project configuration")

        # Use config defaults for other parameters if not provided
        if rag_enabled and project_config.rag:
            # If no database specified, use the first database
            if database is None and project_config.rag.databases:
                database = project_config.rag.databases[0].name
                logger.info(f"Using default database from config: {database}")

            # If no top_k specified, check if there's a default in retrieval strategies
            if rag_top_k is None:
                # Look for default retrieval strategy's top_k
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
                    rag_top_k = 5  # Fallback default

        # Use the RAG subsystem to perform RAG based on the project config
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

        input_schema = ProjectChatOrchestratorAgentInputSchema(chat_message=message)
        logger.info(f"Input schema: {input_schema}")
        agent_response = await chat_agent.run_async(input_schema)
        logger.info(f"Agent response: {agent_response}")

        response_message = agent_response.chat_message
        
        # Check if response is echoing input
        if response_message == message:
            logger.warning(f"Response is echoing input! Input: {message}, Response: {response_message}")
            
            # Clear the corrupted history to prevent learning from bad responses
            if hasattr(chat_agent, 'history'):
                logger.warning("Clearing agent history due to echo response")
                # Remove the last exchange if it was an echo
                if len(chat_agent.history) >= 2:
                    # Remove the last assistant response (the echo)
                    chat_agent.history = chat_agent.history[:-1]
            
            # Generate a fallback response
            response_message = (
                "I apologize, but I'm having trouble processing your question properly. "
                "Could you please try rephrasing it or asking something else? "
                "If this continues, you may want to start a new session."
            )
            logger.info(f"Using fallback response instead of echo")

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

    async def stream_chat(
        self,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: ProjectChatOrchestratorAgent,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> AsyncGenerator[str, None]:
        """Yield assistant content chunks, using agent-native streaming if available."""
        self._clear_rag_context_provider(chat_agent)
        context_provider = ProjectChatContextProvider(title="Project Chat Context")
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

        input_schema = ProjectChatOrchestratorAgentInputSchema(chat_message=message)
        try:
            logger.info("Running async stream")
            previous_response = ""
            async for chunk in chat_agent.run_async_stream(input_schema):
                if hasattr(chunk, "chat_message") and chunk.chat_message:
                    logger.info("Processing partial response", message=chunk)
                    current_response = chunk.chat_message

                    # Skip duplicates
                    if current_response == previous_response:
                        continue
                    
                    # Skip if response is just echoing the input
                    if current_response == message:
                        logger.warning(f"Skipping echoed input in stream: {current_response}")
                        continue

                    # If this is the first chunk, yield it entirely
                    if not previous_response:
                        yield current_response
                    # Otherwise, yield only the incremental part
                    elif len(current_response) > len(previous_response):
                        incremental = current_response[len(previous_response) :]
                        yield incremental

                    previous_response = current_response
            return
        except Exception:
            logger.error(
                "Model call failed",
                exc_info=True,
            )


project_chat_service = ProjectChatService()
