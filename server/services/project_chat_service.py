from collections.abc import AsyncGenerator
from typing import Any
import uuid

from config.datamodel import LlamaFarmConfig  # noqa: E402

from agents.base.agent import LFAgent
from agents.base.clients.client import LFChatCompletion, LFChatCompletionChunk
from agents.base.history import (
    LFChatCompletionUserMessageParam,
)
from agents.chat_orchestrator import ChatOrchestratorAgent
from context_providers.rag_context_provider import (
    ChunkItem,
    RAGContextProvider,
)
from core.logging import FastAPIStructLogger
from observability.event_logger import EventLogger
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


class RAGParameters:
    """Container for resolved RAG parameters."""

    def __init__(
        self,
        rag_enabled: bool = False,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ):
        self.rag_enabled = rag_enabled
        self.database = database
        self.retrieval_strategy = retrieval_strategy
        self.rag_top_k = rag_top_k
        self.rag_score_threshold = rag_score_threshold


class ProjectChatService:
    def _create_event_logger(
        self,
        project_config: LlamaFarmConfig,
        event_type: str = "inference",
    ) -> EventLogger | None:
        """Create event logger, returning None if it fails (e.g., in tests with mocks)."""
        try:
            request_id = f"req_{uuid.uuid4().hex[:12]}"
            return EventLogger(
                event_type=event_type,
                request_id=request_id,
                namespace=project_config.namespace,
                project=project_config.name,
                config=project_config,
            )
        except Exception:
            # Event logging failed (likely tests with mocks), skip it
            return None

    def _log_event(self, event_logger: EventLogger | None, event_name: str, data: dict):
        """Log event if logger exists."""
        if event_logger:
            try:
                event_logger.log_event(event_name, data)
            except Exception:
                # Ignore logging errors
                pass

    def _complete_event(self, event_logger: EventLogger | None):
        """Complete event if logger exists."""
        if event_logger:
            try:
                event_logger.complete_event()
            except Exception:
                # Ignore logging errors
                pass

    def _fail_event(self, event_logger: EventLogger | None, error: str):
        """Fail event if logger exists."""
        if event_logger:
            try:
                event_logger.fail_event(error)
            except Exception:
                # Ignore logging errors
                pass

    async def _perform_rag_with_logging(
        self,
        event_logger: EventLogger | None,
        chat_agent: LFAgent,
        project_dir: str,
        project_config: LlamaFarmConfig,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> None:
        """
        Perform RAG search with event logging.

        Wraps _perform_rag_search_and_add_to_context to capture RAG metrics
        for observability without duplicating RAG logic.
        """
        # Resolve RAG parameters to check if RAG is enabled
        rag_params = self._resolve_rag_parameters(
            project_config,
            rag_enabled=rag_enabled,
            database=database,
            retrieval_strategy=retrieval_strategy,
            rag_top_k=rag_top_k,
            rag_score_threshold=rag_score_threshold,
        )

        # Log RAG query start if enabled
        if rag_params.rag_enabled:
            self._log_event(event_logger, "rag_query_start", {
                "database": rag_params.database,
                "query": message,
                "top_k": rag_params.rag_top_k,
                "retrieval_strategy": rag_params.retrieval_strategy,
            })

        # Perform RAG search using existing helper
        await self._perform_rag_search_and_add_to_context(
            chat_agent,
            project_dir,
            project_config,
            message,
            rag_enabled=rag_enabled,
            database=database,
            retrieval_strategy=retrieval_strategy,
            rag_top_k=rag_top_k,
            rag_score_threshold=rag_score_threshold,
        )

        # Extract results from context provider to log completion metrics
        if rag_params.rag_enabled:
            # Always log rag_retrieval_complete when RAG is enabled
            chunks = []
            avg_score = 0.0

            try:
                # Access the RAG context provider to get results
                context_provider = chat_agent.context_providers.get("rag_context")
                if context_provider and hasattr(context_provider, "chunks"):
                    chunks = context_provider.chunks

                    # Calculate average score
                    if chunks:
                        scores = [chunk.metadata.get("score", 0.0) for chunk in chunks]
                        avg_score = sum(scores) / len(scores) if scores else 0.0
            except Exception:
                # If we can't extract metrics, log with empty results
                pass

            # Always log the completion event (even if 0 chunks found)
            self._log_event(event_logger, "rag_retrieval_complete", {
                "chunks_retrieved": len(chunks),
                "avg_score": round(avg_score, 3),
                "top_chunks": [
                    {
                        "rank": idx + 1,
                        "content_preview": chunk.content[:100] if len(chunk.content) > 100 else chunk.content,
                        "source": chunk.metadata.get("source", "unknown"),
                        "score": round(chunk.metadata.get("score", 0.0), 3),
                    }
                    for idx, chunk in enumerate(chunks[:2])  # Top 2 chunks
                ] if chunks else []
            })

    async def chat(
        self,
        *,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: ChatOrchestratorAgent,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> LFChatCompletion:
        # Create event logger (gracefully handles test mocks)
        event_logger = self._create_event_logger(project_config)

        # Log request
        self._log_event(event_logger, "request_received", {
            "message_length": len(message),
            "model": chat_agent.model_name,
            "rag_enabled": rag_enabled,
        })

        try:
            # Perform RAG search with event logging
            await self._perform_rag_with_logging(
                event_logger,
                chat_agent,
                project_dir,
                project_config,
                message,
                rag_enabled=rag_enabled,
                database=database,
                retrieval_strategy=retrieval_strategy,
                rag_top_k=rag_top_k,
                rag_score_threshold=rag_score_threshold,
            )

            user_input = LFChatCompletionUserMessageParam(role="user", content=message)

            result = await chat_agent.run_async(user_input=user_input)

            # Log response (handle both dict and object responses)
            if hasattr(result, "choices"):
                response_content = result.choices[0].message.content if result.choices else ""
                finish_reason = result.choices[0].finish_reason if result.choices else "unknown"
            else:
                # Handle dict response (from tests)
                choices = result.get("choices", [])
                response_content = choices[0].get("message", {}).get("content", "") if choices else ""
                finish_reason = choices[0].get("finish_reason", "unknown") if choices else "unknown"

            self._log_event(event_logger, "response_generated", {
                "response_length": len(response_content),
                "finish_reason": finish_reason,
            })

            self._complete_event(event_logger)
            return result

        except Exception as e:
            self._fail_event(event_logger, str(e))
            raise

    async def stream_chat(
        self,
        *,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: LFAgent,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Yield assistant content chunks, using agent-native streaming if available."""

        # Create event logger (gracefully handles test mocks)
        event_logger = self._create_event_logger(project_config)

        # Log request
        self._log_event(event_logger, "request_received", {
            "message_length": len(message),
            "model": chat_agent.model_name,
            "rag_enabled": rag_enabled,
        })

        try:
            # Perform RAG search with event logging
            await self._perform_rag_with_logging(
                event_logger,
                chat_agent,
                project_dir,
                project_config,
                message,
                rag_enabled=rag_enabled,
                database=database,
                retrieval_strategy=retrieval_strategy,
                rag_top_k=rag_top_k,
                rag_score_threshold=rag_score_threshold,
            )

            user_input = LFChatCompletionUserMessageParam(role="user", content=message)

            logger.info("Running async stream")
            async for chunk in chat_agent.run_async_stream(user_input=user_input):
                yield chunk

            # Log stream complete
            self._log_event(event_logger, "stream_complete", {
                "finish_reason": "stop",
            })

            self._complete_event(event_logger)

        except Exception as e:
            self._fail_event(event_logger, str(e))
            logger.error(
                "Model call failed",
                exc_info=True,
            )
            raise

    def _resolve_rag_parameters(
        self,
        project_config: LlamaFarmConfig,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> RAGParameters:
        """
        Resolve RAG parameters with intelligent cascading defaults.

        Priority order for each parameter:
        1. Explicit request parameters (highest priority)
        2. Database defaults from config
        3. Strategy defaults from config
        4. Strategy config values (top_k, score_threshold)
        5. Hardcoded fallbacks (lowest priority)

        Returns:
            RAGParameters object with all resolved values
        """
        # Step 0: Determine if RAG is enabled
        if rag_enabled is None:
            rag_enabled = bool(project_config.rag and project_config.rag.databases)
            if rag_enabled:
                logger.info("RAG enabled by default based on project configuration")

        if not rag_enabled or not project_config.rag:
            return RAGParameters(rag_enabled=False)

        # Step 1: Resolve database (request > default_database > first database)
        if database is None:
            if project_config.rag.default_database:
                database = str(project_config.rag.default_database)
                logger.info(f"Using configured default_database: {database}")
            elif project_config.rag.databases:
                database = str(project_config.rag.databases[0].name)
                logger.info(f"Using first database as default: {database}")

        # Step 2: Find the database configuration
        db_config = None
        if database and project_config.rag.databases:
            for db in project_config.rag.databases:
                if db.name == database:
                    db_config = db
                    break

        if not db_config:
            logger.warning(f"Database '{database}' not found in configuration")
            return RAGParameters(rag_enabled=False)

        # Step 3: Resolve retrieval strategy (request > default_retrieval_strategy > first strategy)
        if retrieval_strategy is None:
            if db_config.default_retrieval_strategy:
                retrieval_strategy = str(db_config.default_retrieval_strategy)
                logger.info(f"Using default_retrieval_strategy: {retrieval_strategy}")
            elif db_config.retrieval_strategies:
                # Check for strategy marked as default
                for strategy in db_config.retrieval_strategies:
                    if hasattr(strategy, "default") and strategy.default:
                        retrieval_strategy = str(strategy.name)
                        logger.info(
                            f"Using default-marked strategy: {retrieval_strategy}"
                        )
                        break
                # If no default marked, use first strategy
                if retrieval_strategy is None:
                    retrieval_strategy = str(db_config.retrieval_strategies[0].name)
                    logger.info(
                        f"Using first strategy as default: {retrieval_strategy}"
                    )
            else:
                logger.warning(
                    f"No retrieval strategies defined for database '{database}'"
                )
                return RAGParameters(rag_enabled=False)

        # Step 4: Resolve parameters from strategy config (only if not explicitly provided)
        if retrieval_strategy and db_config.retrieval_strategies:
            for strategy in db_config.retrieval_strategies:
                if strategy.name == retrieval_strategy:
                    # Get top_k from strategy config if not provided
                    if rag_top_k is None and strategy.config:
                        rag_top_k = self._extract_config_value(
                            strategy.config, "top_k", retrieval_strategy
                        )

                    # Get score_threshold from strategy config if not provided
                    if rag_score_threshold is None and strategy.config:
                        rag_score_threshold = self._extract_config_value(
                            strategy.config, "score_threshold", retrieval_strategy
                        )
                    break

        # Step 5: Final fallback defaults if still None
        if rag_top_k is None:
            rag_top_k = 5
            logger.info("Using fallback default top_k: 5")
        if rag_score_threshold is None:
            rag_score_threshold = 0.0
            logger.info("Using fallback default score_threshold: 0.0")

        return RAGParameters(
            rag_enabled=True,
            database=database,
            retrieval_strategy=retrieval_strategy,
            rag_top_k=rag_top_k,
            rag_score_threshold=rag_score_threshold,
        )

    def _extract_config_value(
        self, config: dict | Any, key: str, strategy_name: str
    ) -> Any:
        """Extract a value from strategy config, handling both dict and object types."""
        value = None
        if isinstance(config, dict):
            if key in config:
                value = config[key]
                logger.info(f"Using {key} from strategy '{strategy_name}': {value}")
        elif hasattr(config, key):
            value = getattr(config, key)
            logger.info(f"Using {key} from strategy '{strategy_name}': {value}")
        return value

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
        retrieval_strategy: str | None = None,
        score_threshold: float | None = None,
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
            # Check for explicit default_database first, then fall back to first database
            if project_config.rag.default_database:
                database = str(project_config.rag.default_database)
                logger.info(f"Using configured default database: {database}")
            elif project_config.rag.databases:
                database = str(project_config.rag.databases[0].name)
                logger.info(f"Using first database as default: {database}")
            else:
                logger.error("No databases found in project config")
                return []

        # Use shared helper to run RAG search on database
        results = search_with_rag(
            project_dir,
            database,
            message,
            top_k=top_k,
            retrieval_strategy=retrieval_strategy,
            score_threshold=score_threshold,
        )
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

    def _clear_rag_context_provider(self, chat_agent: LFAgent) -> None:
        try:
            chat_agent.remove_context_provider("rag_context")
        except Exception:
            logger.warning("Failed to clear RAG context provider", exc_info=True)

    async def _perform_rag_search_and_add_to_context(
        self,
        chat_agent: LFAgent,
        project_dir: str,
        project_config: LlamaFarmConfig,
        message: str,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
    ) -> None:
        self._clear_rag_context_provider(chat_agent)
        context_provider = RAGContextProvider(title="Project Chat Context")
        chat_agent.register_context_provider("rag_context", context_provider)

        # Resolve RAG parameters using shared helper
        rag_params = self._resolve_rag_parameters(
            project_config,
            rag_enabled=rag_enabled,
            database=database,
            retrieval_strategy=retrieval_strategy,
            rag_top_k=rag_top_k,
            rag_score_threshold=rag_score_threshold,
        )

        rag_results = []
        if rag_params.rag_enabled:
            rag_results = self._perform_rag_search(
                project_dir,
                project_config,
                message,
                top_k=rag_params.rag_top_k or 5,
                database=rag_params.database,
                retrieval_strategy=rag_params.retrieval_strategy,
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

project_chat_service = ProjectChatService()
