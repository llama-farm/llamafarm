"""Multi-turn RAG strategy with query decomposition and parallel retrieval."""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from components.retrievers.base import RetrievalResult, RetrievalStrategy
from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.retrievers.multi_turn")


class MultiTurnRAGStrategy(RetrievalStrategy):
    """
    Multi-turn RAG strategy for complex queries with extensive context.

    This strategy detects query complexity and decomposes complex queries into
    focused sub-queries. Each sub-query is processed independently with optional
    reranking, then results are merged and deduplicated.

    Best for:
    - Context-heavy queries with multiple aspects
    - Questions requiring synthesis from multiple sources
    - User-provided documents that need cross-referencing
    - Queries that benefit from breaking down into simpler components

    Performance: Medium-Slow (1-3s depending on decomposition)
    Complexity: High
    Accuracy: Very High for complex queries
    """

    def __init__(
        self,
        name: str = "MultiTurnRAGStrategy",
        config: dict[str, Any] | None = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}

        # Configuration
        self.model_name = config.get("model_name")  # LLM for query decomposition
        self.model_base_url = config.get("model_base_url")  # Resolved by RAGManager
        self.model_id = config.get("model_id")  # Resolved by RAGManager
        self.api_key = config.get("api_key")  # Optional, for endpoints requiring auth

        # Query decomposition settings
        self.max_sub_queries = config.get("max_sub_queries", 3)
        self.complexity_threshold = config.get("complexity_threshold", 50)  # chars
        self.min_query_length = config.get("min_query_length", 20)

        # Retrieval settings
        self.base_strategy_name = config.get("base_strategy", "BasicSimilarityStrategy")
        self.base_strategy_config = config.get("base_strategy_config", {})
        self.sub_query_top_k = config.get("sub_query_top_k", 10)
        self.final_top_k = config.get("final_top_k", 10)
        self.initial_k = config.get("initial_k", 30)  # Initial candidates for reranking

        # Reranking settings (optional)
        self.enable_reranking = config.get("enable_reranking", False)
        self.reranker_strategy_name = config.get("reranker_strategy", "CrossEncoderRerankedStrategy")
        self.reranker_config = config.get("reranker_config", {})

        # Deduplication settings
        self.dedup_similarity_threshold = config.get("dedup_similarity_threshold", 0.95)

        # Parallel execution settings
        self.max_workers = config.get("max_workers", 3)

        # State
        self._base_strategy: RetrievalStrategy | None = None
        self._reranker_strategy: RetrievalStrategy | None = None
        self._llm_client = None

    def _initialize_base_strategy(self):
        """Lazy initialization of base retrieval strategy."""
        if self._base_strategy is not None:
            return

        from components.retrievers.basic_similarity.basic_similarity import (
            BasicSimilarityStrategy,
        )
        from components.retrievers.metadata_filtered.metadata_filtered import (
            MetadataFilteredStrategy,
        )

        strategy_map = {
            "BasicSimilarityStrategy": BasicSimilarityStrategy,
            "MetadataFilteredStrategy": MetadataFilteredStrategy,
        }

        strategy_class = strategy_map.get(self.base_strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown base strategy: {self.base_strategy_name}")

        self._base_strategy = strategy_class(
            name=f"{self.name}_base",
            config=self.base_strategy_config,
            project_dir=self.project_dir,
        )

        logger.info(f"Initialized base strategy: {self.base_strategy_name}")

    def _initialize_reranker(self):
        """Lazy initialization of reranking strategy (optional)."""
        if not self.enable_reranking or self._reranker_strategy is not None:
            return

        from components.retrievers.cross_encoder_reranked.cross_encoder_reranked import (
            CrossEncoderRerankedStrategy,
        )

        strategy_map = {
            "CrossEncoderRerankedStrategy": CrossEncoderRerankedStrategy,
        }

        strategy_class = strategy_map.get(self.reranker_strategy_name)
        if not strategy_class:
            raise ValueError(f"Unknown reranker strategy: {self.reranker_strategy_name}")

        self._reranker_strategy = strategy_class(
            name=f"{self.name}_reranker",
            config=self.reranker_config,
            project_dir=self.project_dir,
        )

        logger.info(f"Initialized reranker strategy: {self.reranker_strategy_name}")

    def _rerank_results(
        self,
        result: RetrievalResult,
        query_text: str,
        top_k: int,
    ) -> RetrievalResult:
        """
        Rerank retrieval results using the configured reranker strategy.

        Args:
            result: Initial retrieval result to rerank
            query_text: Original query text for reranking
            top_k: Number of final results to return

        Returns:
            RetrievalResult with reranked documents
        """
        if not self._reranker_strategy or not result.documents:
            return result

        try:
            # Get the reranker's internal rerank method
            from components.retrievers.cross_encoder_reranked.cross_encoder_reranked import (
                CrossEncoderRerankedStrategy,
            )

            if isinstance(self._reranker_strategy, CrossEncoderRerankedStrategy):
                # Use the reranker's internal method directly
                reranked_docs = self._reranker_strategy._rerank_with_universal_runtime(
                    query_text=query_text,
                    documents=result.documents,
                )

                # Filter and select top_k
                final_docs = reranked_docs[:top_k]

                # Add reranker metadata to documents before building result
                for i, (doc, score) in enumerate(final_docs):
                    doc.metadata["reranker_score"] = score
                    doc.metadata["rerank_position"] = i + 1

                # Build result lists from annotated documents
                documents = [doc for doc, _ in final_docs]
                scores = [score for _, score in final_docs]

                return RetrievalResult(
                    documents=documents,
                    scores=scores,
                    strategy_metadata={
                        **result.strategy_metadata,
                        "reranked": True,
                        "reranker_model": self._reranker_strategy.model_id,
                        "initial_count": len(result.documents),
                        "final_count": len(documents),
                    }
                )
            else:
                logger.warning(f"Unknown reranker type: {type(self._reranker_strategy)}")
                return result

        except Exception as e:
            logger.error(f"Reranking failed: {e}", exc_info=True)
            # Return original results on failure
            result.strategy_metadata["reranking_error"] = str(e)
            return result

    def _initialize_llm_client(self):
        """Initialize LLM client for query decomposition."""
        if self._llm_client is not None:
            return

        if not self.model_base_url or not self.model_id:
            raise ValueError(
                f"Model configuration not resolved for '{self.model_name}'. "
                "Ensure the model exists in runtime.models."
            )

        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "openai package is required for query decomposition. "
                "Install with: pip install openai"
            ) from e

        # Use configured API key, environment variable, or placeholder
        # Ollama and many local endpoints don't require authentication
        import os
        api_key = self.api_key or os.environ.get("OPENAI_API_KEY", "not-needed")

        self._llm_client = OpenAI(
            base_url=self.model_base_url,
            api_key=api_key,
        )

        logger.info(
            "Initialized LLM client for query decomposition",
            model_name=self.model_name,
            base_url=self.model_base_url,
        )

    def _detect_query_complexity(self, query_text: str) -> bool:
        """
        Detect if a query is complex enough to benefit from decomposition.

        Simple heuristics:
        - Length (longer queries are more complex)
        - Multiple questions (contains "and", "also", etc.)
        - Conditional logic ("if", "when", etc.)
        """
        # Length check
        if len(query_text) < self.complexity_threshold:
            return False

        # Multiple question markers
        multi_question_patterns = [
            r'\band\b',
            r'\balso\b',
            r'\badditionally\b',
            r'\bfurthermore\b',
            r'\bmoreover\b',
            r'\?.*\?',  # Multiple question marks
        ]

        for pattern in multi_question_patterns:
            if re.search(pattern, query_text, re.IGNORECASE):
                logger.info("Query complexity detected", reason=f"pattern: {pattern}")
                return True

        return False

    def _decompose_query(self, query_text: str) -> list[str]:
        """
        Decompose a complex query into focused sub-queries using LLM.

        Returns:
            List of sub-queries (max: self.max_sub_queries)
        """
        self._initialize_llm_client()

        system_prompt = """Break complex questions into 2-3 simple questions.

Example:
Input: What are llama and alpaca fibers, and how do they compare?
Output:
<question>What is llama fiber?</question>
<question>What is alpaca fiber?</question>
<question>How do llama and alpaca fibers compare?</question>

Always use <question> tags. Be direct."""

        user_prompt = f"Input: {query_text}\nOutput:"

        try:
            logger.info(
                "Calling LLM for query decomposition",
                model_id=self.model_id,
                query_preview=query_text[:100] + "..." if len(query_text) > 100 else query_text,
            )

            response = self._llm_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=200,
                stop=["Input:", "\n\n\n"],  # Stop if it starts rambling
            )

            # Debug: log full response structure
            raw_content = response.choices[0].message.content
            logger.info(
                "LLM decomposition response received",
                raw_content_type=type(raw_content).__name__,
                raw_content_length=len(raw_content) if raw_content else 0,
                raw_content_preview=repr(raw_content[:500]) if raw_content else "None/Empty",
                finish_reason=response.choices[0].finish_reason,
            )

            content = raw_content.strip() if raw_content else ""

            # Strip out <think>...</think> blocks (some models like Qwen3 use thinking mode)
            # We only want the actual output, not the internal reasoning
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL | re.IGNORECASE)
            content = content.strip()

            # Extract questions using regex
            question_pattern = r'<question>(.*?)</question>'
            matches = re.findall(question_pattern, content, re.DOTALL | re.IGNORECASE)

            logger.info(
                "Question extraction results",
                matches_found=len(matches),
                matches_preview=matches[:3] if matches else [],
            )

            if matches:
                # Clean up and filter questions
                sub_queries = []
                for match in matches[:self.max_sub_queries]:
                    question = match.strip()
                    # Remove extra whitespace and newlines
                    question = re.sub(r'\s+', ' ', question)
                    if len(question) >= self.min_query_length:
                        sub_queries.append(question)

                if sub_queries:
                    logger.info(
                        f"Decomposed query into {len(sub_queries)} sub-queries",
                        sub_queries=sub_queries
                    )
                    return sub_queries
            else:
                logger.warning(
                    "No <question> tags found in LLM response. "
                    "The model may not be following the prompt format, or may have a 'thinking mode' "
                    "that produces empty visible output. Consider using a different model for query_decomposer.",
                    content_preview=repr(content[:300]) if content else "EMPTY",
                    content_length=len(content),
                    model_id=self.model_id,
                )

        except Exception as e:
            logger.error(f"Query decomposition failed: {e}", exc_info=True)

        # Fallback: return original query
        return [query_text]

    def _retrieve_for_subquery(
        self,
        sub_query: str,
        embedder,
        vector_store,
        **kwargs
    ) -> tuple[str, RetrievalResult]:
        """
        Retrieve documents for a single sub-query.

        Args:
            sub_query: The decomposed sub-query text
            embedder: Embedder instance to embed the sub-query
            vector_store: Vector store to search
            **kwargs: Additional arguments passed to strategies

        Returns:
            Tuple of (sub_query, RetrievalResult)
        """
        try:
            # Embed the sub-query (critical fix: each sub-query needs its own embedding)
            sub_query_embedding = embedder.embed_text(sub_query)

            # Use base strategy for retrieval with sub-query embedding
            result = self._base_strategy.retrieve(
                query_embedding=sub_query_embedding,
                vector_store=vector_store,
                top_k=self.sub_query_top_k,
                **kwargs
            )

            # Optionally rerank results
            if self.enable_reranking and self._reranker_strategy:
                result = self._reranker_strategy.retrieve(
                    query_embedding=sub_query_embedding,
                    vector_store=vector_store,
                    top_k=self.sub_query_top_k,
                    query_text=sub_query,
                    embedder=embedder,  # Pass embedder along
                    **kwargs
                )

            return (sub_query, result)

        except Exception:
            logger.error(f"Retrieval failed for sub-query: {sub_query}", exc_info=True)
            return (sub_query, RetrievalResult(documents=[], scores=[]))

    def _content_similarity(self, text1: str, text2: str) -> float:
        """
        Compute content similarity between two texts using simple n-gram overlap.

        This is a lightweight similarity metric that doesn't require embeddings.
        For more accurate similarity, consider using embeddings, but that would
        be much slower.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score between 0 and 1
        """
        # Tokenize into words (simple whitespace split)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        # Handle empty texts
        if not words1 or not words2:
            return 0.0

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return 0.0

        return intersection / union

    def _merge_and_deduplicate(
        self,
        results: list[tuple[str, RetrievalResult]],
        top_k: int
    ) -> RetrievalResult:
        """
        Merge results from multiple sub-queries and remove duplicates.

        Deduplication strategy:
        - Use document ID for exact matches
        - Use content similarity for near-duplicates (configurable threshold)

        Args:
            results: List of (sub_query, RetrievalResult) tuples
            top_k: Number of final results to return (respects caller's request)

        Returns:
            RetrievalResult with deduplicated documents
        """
        all_docs: list[Document] = []
        all_scores: list[float] = []
        seen_ids: set[str] = set()

        # Collect all documents with their scores (ID-based deduplication)
        for _sub_query, result in results:
            for doc, score in zip(result.documents, result.scores, strict=False):
                if doc.id not in seen_ids:
                    seen_ids.add(doc.id)
                    all_docs.append(doc)
                    all_scores.append(score)

        # Content-based deduplication (near-duplicate detection)
        if len(all_docs) > 1 and self.dedup_similarity_threshold < 1.0:
            deduplicated_docs = []
            deduplicated_scores = []

            for doc, score in zip(all_docs, all_scores, strict=False):
                is_duplicate = False

                # Check against already selected documents
                for existing_doc in deduplicated_docs:
                    similarity = self._content_similarity(doc.content, existing_doc.content)

                    if similarity >= self.dedup_similarity_threshold:
                        is_duplicate = True
                        logger.debug(
                            f"Document {doc.id} is near-duplicate (similarity: {similarity:.3f})"
                        )
                        break

                if not is_duplicate:
                    deduplicated_docs.append(doc)
                    deduplicated_scores.append(score)

            all_docs = deduplicated_docs
            all_scores = deduplicated_scores

        # Sort by score (descending) and take top_k (respects caller's request)
        if all_docs:
            # Use min of caller's top_k and configured final_top_k
            effective_top_k = min(top_k, self.final_top_k)

            sorted_pairs = sorted(
                zip(all_docs, all_scores, strict=False),
                key=lambda x: x[1],
                reverse=True
            )[:effective_top_k]

            final_docs = [doc for doc, _ in sorted_pairs]
            final_scores = [score for _, score in sorted_pairs]
        else:
            final_docs = []
            final_scores = []

        return RetrievalResult(
            documents=final_docs,
            scores=final_scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "sub_queries_count": len(results),
                "total_retrieved": len(all_docs),
                "final_count": len(final_docs),
                "dedup_threshold": self.dedup_similarity_threshold,
            }
        )

    def retrieve(
        self,
        query_embedding: list[float],
        vector_store,
        top_k: int = 5,
        query_text: str = "",
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve documents using multi-turn strategy with query decomposition.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of final results to return
            query_text: Original query text (required for decomposition)
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with merged and deduplicated documents
        """
        if not query_text:
            raise ValueError("query_text is required for multi-turn RAG")

        # Initialize components
        self._initialize_base_strategy()
        if self.enable_reranking:
            self._initialize_reranker()

        # Step 1: Detect query complexity
        is_complex = self._detect_query_complexity(query_text)

        if not is_complex:
            # Simple query: use base strategy, then optionally rerank
            logger.info("Query is simple, using base strategy directly")
            result = self._base_strategy.retrieve(
                query_embedding=query_embedding,
                vector_store=vector_store,
                top_k=max(top_k, self.initial_k) if self.enable_reranking else top_k,
                **kwargs
            )

            # Apply reranking if enabled
            if self.enable_reranking and self._reranker_strategy:
                logger.info("Applying reranking to simple query results")
                result = self._rerank_results(result, query_text, top_k)

            result.strategy_metadata["strategy"] = self.name
            result.strategy_metadata["decomposed"] = False
            return result

        # Step 2: Decompose query
        logger.info("Query is complex, decomposing into sub-queries")
        sub_queries = self._decompose_query(query_text)

        if len(sub_queries) == 1:
            # Decomposition returned single query, use base strategy then optionally rerank
            logger.info("Decomposition returned single query")
            result = self._base_strategy.retrieve(
                query_embedding=query_embedding,
                vector_store=vector_store,
                top_k=max(top_k, self.initial_k) if self.enable_reranking else top_k,
                **kwargs
            )

            # Apply reranking if enabled
            if self.enable_reranking and self._reranker_strategy:
                logger.info("Applying reranking to fallback results")
                result = self._rerank_results(result, query_text, top_k)

            result.strategy_metadata["strategy"] = self.name
            result.strategy_metadata["decomposed"] = False
            return result

        # Step 3: Parallel retrieval for sub-queries
        logger.info(f"Retrieving for {len(sub_queries)} sub-queries in parallel")

        # Extract embedder from kwargs (required for sub-query embedding)
        embedder = kwargs.pop("embedder", None)  # Remove from kwargs to avoid duplicate argument
        if not embedder:
            logger.warning("No embedder provided, falling back to base strategy with original embedding")
            # Fallback: use base strategy with original embedding
            result = self._base_strategy.retrieve(
                query_embedding=query_embedding,
                vector_store=vector_store,
                top_k=top_k,
                **kwargs
            )
            result.strategy_metadata["strategy"] = self.name
            result.strategy_metadata["decomposed"] = False
            result.strategy_metadata["fallback_reason"] = "no_embedder"
            return result

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._retrieve_for_subquery,
                    sub_query,
                    embedder,
                    vector_store,
                    **kwargs
                )
                for sub_query in sub_queries
            ]

            results = [future.result() for future in futures]

        # Step 4: Merge and deduplicate
        logger.info("Merging and deduplicating results")
        final_result = self._merge_and_deduplicate(results, top_k)
        final_result.strategy_metadata["decomposed"] = True
        final_result.strategy_metadata["sub_queries"] = sub_queries

        return final_result

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Universal support - works with any vector store."""
        return True

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.max_sub_queries < 1 or self.max_sub_queries > 5:
            return False
        if self.sub_query_top_k < 1:
            return False
        return self.final_top_k >= 1

    def get_config_schema(self) -> dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of model from runtime.models to use for query decomposition",
                },
                "max_sub_queries": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
                "complexity_threshold": {"type": "integer", "minimum": 20, "default": 50},
                "min_query_length": {"type": "integer", "minimum": 10, "default": 20},
                "base_strategy": {
                    "type": "string",
                    "enum": ["BasicSimilarityStrategy", "MetadataFilteredStrategy"],
                    "default": "BasicSimilarityStrategy",
                },
                "sub_query_top_k": {"type": "integer", "minimum": 5, "default": 10},
                "final_top_k": {"type": "integer", "minimum": 1, "default": 10},
                "enable_reranking": {"type": "boolean", "default": False},
                "reranker_strategy": {"type": "string", "default": "CrossEncoderRerankedStrategy"},
                "max_workers": {"type": "integer", "minimum": 1, "maximum": 10, "default": 3},
            },
        }

    def get_performance_info(self) -> dict[str, Any]:
        """Get performance characteristics."""
        return {
            "speed": "medium-slow",
            "memory_usage": "medium",
            "complexity": "high",
            "accuracy": "very_high",
            "best_for": [
                "complex_queries",
                "context_heavy_questions",
                "multi_aspect_queries",
                "synthesis_tasks",
            ],
            "notes": f"Multi-turn RAG with query decomposition using model: {self.model_name}",
        }
