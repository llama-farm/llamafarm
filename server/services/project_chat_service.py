import asyncio
import contextlib
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from config.datamodel import LlamaFarmConfig  # noqa: E402
from observability.event_logger import EventLogger

from agents.base.agent import LFAgent
from agents.base.clients.client import LFChatCompletion, LFChatCompletionChunk
from agents.base.history import (
    LFChatCompletionMessageParam,
)
from agents.base.types import ToolDefinition
from agents.chat_orchestrator import ChatOrchestratorAgent
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


class RAGParameters:
    """Container for resolved RAG parameters."""

    def __init__(
        self,
        rag_enabled: bool = False,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
        rag_queries: list[str] | None = None,
    ):
        self.rag_enabled = rag_enabled
        self.database = database
        self.retrieval_strategy = retrieval_strategy
        self.rag_top_k = rag_top_k
        self.rag_score_threshold = rag_score_threshold
        self.rag_queries = rag_queries


class ProjectChatService:
    def _extract_latest_user_message(
        self, messages: list[LFChatCompletionMessageParam]
    ) -> str:
        """Extract the content from the latest user message in the messages list."""
        latest_user_message = next(
            (
                msg
                for msg in reversed(messages)
                if msg.get("role", None) == "user" and msg.get("content", None)
            ),
            None,
        )
        if latest_user_message:
            return latest_user_message.get("content", "")
        return ""

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
            with contextlib.suppress(Exception):
                event_logger.log_event(event_name, data)

    def _complete_event(self, event_logger: EventLogger | None):
        """Complete event if logger exists."""
        if event_logger:
            with contextlib.suppress(Exception):
                event_logger.complete_event()

    def _fail_event(self, event_logger: EventLogger | None, error: str):
        """Fail event if logger exists."""
        if event_logger:
            with contextlib.suppress(Exception):
                event_logger.fail_event(error)

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
        rag_queries: list[str] | None = None,
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
            rag_queries=rag_queries,
        )

        if not rag_params.rag_enabled:
            return

        # Determine the query for logging
        log_query = message
        if rag_params.rag_queries:
            if len(rag_params.rag_queries) == 1:
                log_query = rag_params.rag_queries[0]
            else:
                log_query = f"[{len(rag_params.rag_queries)} custom queries]"

        # Log RAG query start if enabled
        self._log_event(
            event_logger,
            "rag_query_start",
            {
                "database": rag_params.database,
                "query": log_query,
                "top_k": rag_params.rag_top_k,
                "retrieval_strategy": rag_params.retrieval_strategy,
                "rag_queries_count": len(rag_params.rag_queries)
                if rag_params.rag_queries
                else 0,
            },
        )

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
            rag_queries=rag_queries,
        )

        # Extract results from context provider to log completion metrics
        if rag_params.rag_enabled:
            # Always log rag_retrieval_complete when RAG is enabled
            chunks = []
            avg_score = 0.0

            try:
                # Access the RAG context provider to get results
                # Check if context_providers exists and is accessible
                if hasattr(chat_agent, "context_providers"):
                    context_provider = chat_agent.context_providers.get("rag_context")
                    if context_provider and hasattr(context_provider, "chunks"):
                        chunks = context_provider.chunks

                        # Calculate average score
                        if chunks:
                            scores = [
                                chunk.metadata.get("score", 0.0) for chunk in chunks
                            ]
                            avg_score = sum(scores) / len(scores) if scores else 0.0
            except Exception as e:
                # If we can't extract metrics, log with empty results
                logger.warning(f"Could not extract RAG metrics: {e}", exc_info=False)

            # Always log the completion event (even if 0 chunks found)
            self._log_event(
                event_logger,
                "rag_retrieval_complete",
                {
                    "chunks_retrieved": len(chunks),
                    "avg_score": round(avg_score, 3),
                    "top_chunks": [
                        {
                            "rank": idx + 1,
                            "content_preview": chunk.content[:100]
                            if len(chunk.content) > 100
                            else chunk.content,
                            "source": chunk.metadata.get("source", "unknown"),
                            "score": round(chunk.metadata.get("score", 0.0), 3),
                        }
                        for idx, chunk in enumerate(chunks[:2])  # Top 2 chunks
                    ]
                    if chunks
                    else [],
                },
            )

    async def chat(
        self,
        *,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: ChatOrchestratorAgent,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
        n_ctx: int | None = None,
        rag_queries: list[str] | None = None,
        think: bool | None = None,
        thinking_budget: int | None = None,
        max_tokens: int | None = None,
    ) -> LFChatCompletion:
        # Create event logger (gracefully handles test mocks)
        event_logger = self._create_event_logger(project_config)

        # Log request
        self._log_event(
            event_logger,
            "request_received",
            {
                "message_length": len(messages),
                "model": chat_agent.model_name,
                "rag_enabled": rag_enabled,
            },
        )

        latest_user_message = next(
            (
                msg
                for msg in reversed(messages)
                if msg.get("role", None) == "user" and msg.get("content", None)
            ),
            None,
        )

        if latest_user_message:
            try:
                # Perform RAG search with event logging
                await self._perform_rag_with_logging(
                    event_logger,
                    chat_agent,
                    project_dir,
                    project_config,
                    latest_user_message.get("content", ""),
                    rag_enabled=rag_enabled,
                    database=database,
                    retrieval_strategy=retrieval_strategy,
                    rag_top_k=rag_top_k,
                    rag_score_threshold=rag_score_threshold,
                    rag_queries=rag_queries,
                )
            except Exception as e:
                self._fail_event(event_logger, str(e))
                raise

        try:
            # Build extra_body dict for runtime-specific parameters
            extra_body = {}
            if n_ctx is not None:
                extra_body["n_ctx"] = n_ctx
            # Thinking/reasoning model parameters (universal runtime)
            if think is not None:
                extra_body["think"] = think
            if thinking_budget is not None:
                extra_body["thinking_budget"] = thinking_budget
            if max_tokens is not None:
                extra_body["max_tokens"] = max_tokens

            response = await chat_agent.run_async(
                messages=messages,
                tools=tools,
                extra_body=extra_body if extra_body else None,
            )

            # Log response (handle both dict and object responses)
            if hasattr(response, "model_dump"):
                response_data = response.model_dump(mode="json")
            elif hasattr(response, "choices"):
                # Pydantic object but use dict serialization
                response_content = (
                    response.choices[0].message.content if response.choices else ""
                )
                response_data = {
                    "response_length": len(response_content),
                    "finish_reason": response.choices[0].finish_reason
                    if response.choices
                    else "unknown",
                }
            else:
                # Handle dict response (from tests)
                choices = response.get("choices", [])
                response_content = (
                    choices[0].get("message", {}).get("content", "") if choices else ""
                )
                response_data = {
                    "response_length": len(response_content),
                    "finish_reason": choices[0].get("finish_reason", "unknown")
                    if choices
                    else "unknown",
                }

            self._log_event(
                event_logger,
                "response_generated",
                response_data,
            )

            self._complete_event(event_logger)
            return response
        except Exception as e:
            self._fail_event(event_logger, str(e))
            raise

    async def stream_chat(
        self,
        *,
        project_dir: str,
        project_config: LlamaFarmConfig,
        chat_agent: LFAgent,
        messages: list[LFChatCompletionMessageParam],
        tools: list[ToolDefinition] | None = None,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
        n_ctx: int | None = None,
        rag_queries: list[str] | None = None,
        think: bool | None = None,
        thinking_budget: int | None = None,
        max_tokens: int | None = None,
    ) -> AsyncGenerator[LFChatCompletionChunk]:
        """Yield assistant content chunks, using agent-native streaming if available."""
        # Create event logger (gracefully handles test mocks)
        event_logger = self._create_event_logger(project_config)

        # Log request
        self._log_event(
            event_logger,
            "request_received",
            {
                "message_length": len(messages),
                "model": chat_agent.model_name,
                "rag_enabled": rag_enabled,
            },
        )

        latest_user_message = next(
            (
                msg
                for msg in reversed(messages)
                if msg.get("role", None) == "user" and msg.get("content", None)
            ),
            None,
        )

        event_failed = False
        first_token_logged = False

        try:
            if latest_user_message:
                await self._perform_rag_with_logging(
                    event_logger,
                    chat_agent,
                    project_dir,
                    project_config,
                    latest_user_message.get("content", ""),
                    rag_enabled=rag_enabled,
                    database=database,
                    retrieval_strategy=retrieval_strategy,
                    rag_top_k=rag_top_k,
                    rag_score_threshold=rag_score_threshold,
                    rag_queries=rag_queries,
                )

            logger.debug("Running async stream")

            # Log LLM inference start
            self._log_event(
                event_logger,
                "llm_inference_start",
                {
                    "model": chat_agent.model_name,
                },
            )

            # Build extra_body dict for runtime-specific parameters
            extra_body = {}
            if n_ctx is not None:
                extra_body["n_ctx"] = n_ctx
            # Thinking/reasoning model parameters (universal runtime)
            if think is not None:
                extra_body["think"] = think
            if thinking_budget is not None:
                extra_body["thinking_budget"] = thinking_budget
            if max_tokens is not None:
                extra_body["max_tokens"] = max_tokens

            async for chunk in chat_agent.run_async_stream(
                messages=messages,
                tools=tools,
                extra_body=extra_body if extra_body else None,
            ):
                # Log time to first token (only once)
                if not first_token_logged and event_logger:
                    self._log_event(event_logger, "llm_first_token", {})
                    first_token_logged = True

                yield chunk

            # Log LLM inference complete after streaming finishes
            self._log_event(
                event_logger,
                "llm_inference_complete",
                {
                    "finish_reason": "stop",
                },
            )

        except Exception:
            event_failed = True
            raise
        finally:
            if event_failed:
                self._fail_event(event_logger, "stream_failed")
            else:
                self._log_event(event_logger, "stream_complete", {})
                self._complete_event(event_logger)

    def _resolve_rag_parameters(
        self,
        project_config: LlamaFarmConfig,
        rag_enabled: bool | None = None,
        database: str | None = None,
        retrieval_strategy: str | None = None,
        rag_top_k: int | None = None,
        rag_score_threshold: float | None = None,
        rag_queries: list[str] | None = None,
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
            rag_queries=rag_queries,
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

    def _execute_single_rag_query(
        self,
        project_dir: str,
        query: str,
        database: str,
        top_k: int = 5,
        retrieval_strategy: str | None = None,
        score_threshold: float | None = None,
    ) -> list[Any]:
        """Execute a single RAG query and return normalized results.

        This is the low-level search method that calls the RAG service directly.
        For most use cases, use _perform_rag_search which handles custom queries
        and concurrent execution.
        """
        logger.info(f"Executing RAG query: {query[:50]}...")

        # Use shared helper to run RAG search on database
        results = search_with_rag(
            project_dir,
            database,
            query,
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
        logger.info(f"RAG query returned {len(normalized)} results")
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
        rag_queries: list[str] | None = None,
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
            rag_queries=rag_queries,
        )

        rag_results = []
        if rag_params.rag_enabled:
            rag_results = await self._perform_rag_search(
                project_dir=project_dir,
                message=message,
                rag_params=rag_params,
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

    async def _perform_rag_search(
        self,
        project_dir: str,
        message: str,
        rag_params: RAGParameters,
    ) -> list[Any]:
        """
        Perform RAG search using the project's RAG configuration.

        Handles two cases:
        1. rag_queries (list): Execute queries concurrently and merge/deduplicate results
        2. Default: Use the user message as the query

        Args:
            project_dir: Path to the project directory
            message: The user message (used as default query if no custom queries)
            rag_params: Resolved RAG parameters including database, strategy, and queries
        """
        if not rag_params.rag_enabled or not rag_params.database:
            return []

        top_k = rag_params.rag_top_k or 5

        # Determine queries to execute
        queries = [message]  # Default to user message
        if rag_params.rag_queries:
            valid_queries = [q for q in rag_params.rag_queries if q and q.strip()]
            if valid_queries:
                queries = valid_queries

        # Single query - execute directly without concurrent overhead
        if len(queries) == 1:
            logger.info(f"Performing RAG search with query: {queries[0][:50]}...")
            return self._execute_single_rag_query(
                project_dir=project_dir,
                query=queries[0],
                database=rag_params.database,
                top_k=top_k,
                retrieval_strategy=rag_params.retrieval_strategy,
                score_threshold=rag_params.rag_score_threshold,
            )

        # Multiple queries - execute concurrently
        logger.info(f"Performing RAG search with {len(queries)} queries concurrently")

        # Create concurrent search tasks using asyncio.to_thread for sync function
        search_tasks = [
            asyncio.to_thread(
                self._execute_single_rag_query,
                project_dir=project_dir,
                query=query,
                database=rag_params.database,
                top_k=top_k,
                retrieval_strategy=rag_params.retrieval_strategy,
                score_threshold=rag_params.rag_score_threshold,
            )
            for query in queries
        ]

        # Execute all searches concurrently
        list_of_results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Flatten, deduplicate, and sort the aggregated results
        all_results = []
        seen_content_hashes: set[str] = set()

        for results in list_of_results:
            # Skip failed searches
            if isinstance(results, Exception):
                logger.warning(f"RAG search failed for one query: {results}")
                continue

            for result in results:
                # Create a hash from the first 200 chars of content for deduplication
                content_hash = hash(result.content[:200] if result.content else "")
                if content_hash not in seen_content_hashes:
                    seen_content_hashes.add(content_hash)
                    all_results.append(result)

        # Sort by score (descending) and limit to top_k
        all_results.sort(key=lambda x: getattr(x, "score", 0.0), reverse=True)
        return all_results[:top_k]


project_chat_service = ProjectChatService()
