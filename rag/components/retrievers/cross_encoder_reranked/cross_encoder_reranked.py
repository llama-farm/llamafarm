"""Cross-encoder reranking strategy using models from runtime config."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from components.retrievers.base import RetrievalStrategy, RetrievalResult
from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.retrievers.cross_encoder_reranked")


class CrossEncoderRerankedStrategy(RetrievalStrategy):
    """
    Cross-encoder reranking strategy using models from runtime.models config.

    This strategy performs initial retrieval using a base strategy, then uses
    a cross-encoder model to compute precise relevance scores by jointly
    encoding the query and each document.

    Cross-encoders are 10-100x faster than LLM-based reranking and often
    more accurate for relevance scoring.

    Recommended models:
    - bge-reranker-v2-m3 (Best for production, multilingual)
    - bce-reranker-base (Good quantized option)

    Use Cases:
    - Simple, focused questions requiring accurate ranking
    - Production systems requiring fast, accurate reranking
    - High-throughput retrieval pipelines
    - Cost-sensitive deployments (local compute only)

    Performance: Fast (50-400 docs/sec depending on model)
    Complexity: Medium
    Accuracy: Very High
    """

    def __init__(
        self,
        name: str = "CrossEncoderRerankedStrategy",
        config: Optional[Dict[str, Any]] = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}

        # Configuration
        self.model_name = config.get("model_name", "reranker")  # Name from runtime.models
        self.model_base_url = config.get("model_base_url")  # Resolved by RAGManager
        self.model_id = config.get("model_id")  # Resolved by RAGManager
        self.initial_k = config.get("initial_k", 30)
        self.final_k = config.get("final_k", 10)
        self.base_strategy_name = config.get("base_strategy", "BasicSimilarityStrategy")
        self.base_strategy_config = config.get("base_strategy_config", {})
        self.relevance_threshold = config.get("relevance_threshold", 0.0)
        self.batch_size = config.get("batch_size", 32)
        self.normalize_scores = config.get("normalize_scores", True)
        self.max_chars_per_doc = config.get("max_chars_per_doc", 1000)

        # Model state
        self._base_strategy: Optional[RetrievalStrategy] = None
        self._reranker_client = None

    def _initialize_base_strategy(self):
        """Lazy initialization of base strategy."""
        if self._base_strategy is not None:
            return

        # Import dynamically to avoid circular dependencies
        from components.retrievers.basic_similarity.basic_similarity import BasicSimilarityStrategy
        from components.retrievers.metadata_filtered.metadata_filtered import MetadataFilteredStrategy

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
        """Initialize the cross-encoder reranking model."""
        if self._reranker_client is not None:
            return

        if not self.model_base_url or not self.model_id:
            raise ValueError(
                f"Model configuration not resolved for '{self.model_name}'. "
                "Ensure the model exists in runtime.models."
            )

        # Create OpenAI client with the resolved model config
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for cross-encoder reranking. "
                "Install with: pip install openai"
            )

        self._reranker_client = OpenAI(
            base_url=self.model_base_url,
            api_key="dummy",  # OpenAI-compatible endpoints may not require real key
        )

        logger.info(
            f"Initialized cross-encoder reranker",
            model_name=self.model_name,
            base_url=self.model_base_url,
        )

    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        query_text: str = "",
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve and rerank documents using cross-encoder.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of final results to return
            query_text: Original query text (required for reranking)
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with reranked documents
        """
        if not query_text:
            raise ValueError("query_text is required for cross-encoder reranking")

        # Initialize components
        self._initialize_base_strategy()
        self._initialize_reranker()

        # Step 1: Initial retrieval
        logger.info(f"Performing initial retrieval with {self.base_strategy_name}")
        initial_result = self._base_strategy.retrieve(
            query_embedding=query_embedding,
            vector_store=vector_store,
            top_k=self.initial_k,
            **kwargs,
        )

        if not initial_result.documents:
            logger.warning("No documents retrieved by base strategy")
            return RetrievalResult(
                documents=[],
                scores=[],
                strategy_metadata={
                    "strategy": self.name,
                    "version": "1.0.0",
                    "model_name": self.model_name,
                    "initial_retrieved": 0,
                },
            )

        # Step 2: Cross-encoder reranking
        logger.info(
            f"Reranking {len(initial_result.documents)} documents with {self.model_name}"
        )
        reranked_docs = self._rerank_with_cross_encoder(
            query_text=query_text,
            documents=initial_result.documents,
        )

        # Step 3: Filter and select top_k
        filtered_docs = [
            (doc, score) for doc, score in reranked_docs if score >= self.relevance_threshold
        ]

        final_docs = filtered_docs[: min(top_k, self.final_k)]

        # Add metadata
        documents = [doc for doc, _ in final_docs]
        scores = [score for _, score in final_docs]

        for i, (doc, score) in enumerate(final_docs):
            doc.metadata["reranker_score"] = score
            doc.metadata["rerank_position"] = i + 1
            doc.metadata["reranker_model"] = self.model_name

        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "model_name": self.model_name,
                "base_strategy": self.base_strategy_name,
                "initial_retrieved": len(initial_result.documents),
                "candidates_reranked": len(reranked_docs),
                "threshold_filtered": len(filtered_docs),
                "final_count": len(documents),
            },
        )

    def _rerank_with_cross_encoder(
        self,
        query_text: str,
        documents: List[Document],
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents using cross-encoder model.

        This uses the embedding API to compute similarity scores
        between the query and each document. The embeddings API with
        reranker models provides relevance scores.

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        results = []

        # Process in batches
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_results = self._rerank_batch(query_text, batch)
            results.extend(batch_results)

        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _rerank_batch(
        self,
        query_text: str,
        documents: List[Document],
    ) -> List[tuple[Document, float]]:
        """Rerank a batch of documents using embeddings API."""

        results = []

        try:
            # Get query embedding
            query_response = self._reranker_client.embeddings.create(
                model=self.model_id,
                input=query_text,
            )
            query_embedding = query_response.data[0].embedding

            # Get document embeddings and compute scores
            for doc in documents:
                # Truncate content to avoid token limits
                content = doc.content[: self.max_chars_per_doc]

                # Get document embedding
                doc_response = self._reranker_client.embeddings.create(
                    model=self.model_id,
                    input=content,
                )
                doc_embedding = doc_response.data[0].embedding

                # Compute cosine similarity
                score = self._cosine_similarity(query_embedding, doc_embedding)
                results.append((doc, score))

        except Exception as e:
            logger.error(f"Error during reranking batch: {e}", exc_info=True)
            # Fallback: return documents with neutral scores
            results = [(doc, 0.5) for doc in documents]

        # Normalize scores if requested
        if self.normalize_scores and results:
            results = self._normalize_results(results)

        return results

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _normalize_results(
        self, results: List[tuple[Document, float]]
    ) -> List[tuple[Document, float]]:
        """Normalize scores to 0-1 range using min-max normalization."""
        if not results:
            return results

        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)

        if max_score - min_score < 1e-6:
            # All scores are identical
            return [(doc, 0.5) for doc, _ in results]

        normalized = []
        for doc, score in results:
            norm_score = (score - min_score) / (max_score - min_score)
            normalized.append((doc, norm_score))

        return normalized

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Universal support - works with any vector store."""
        return True

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.initial_k < 1 or self.final_k < 1:
            return False
        if self.batch_size < 1:
            return False
        if self.relevance_threshold < 0 or self.relevance_threshold > 1:
            return False
        return True

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of model from runtime.models to use",
                },
                "initial_k": {"type": "integer", "minimum": 10, "default": 30},
                "final_k": {"type": "integer", "minimum": 1, "default": 10},
                "base_strategy": {
                    "type": "string",
                    "enum": ["BasicSimilarityStrategy", "MetadataFilteredStrategy"],
                    "default": "BasicSimilarityStrategy",
                },
                "relevance_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "batch_size": {"type": "integer", "minimum": 1, "default": 32},
                "normalize_scores": {"type": "boolean", "default": True},
                "max_chars_per_doc": {"type": "integer", "minimum": 100, "default": 1000},
            },
        }

    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance characteristics."""
        return {
            "speed": "fast",
            "memory_usage": "low-medium",
            "complexity": "medium",
            "accuracy": "very_high",
            "best_for": [
                "simple_questions",
                "production_systems",
                "high_throughput",
                "cost_effective_reranking",
            ],
            "notes": f"Cross-encoder reranking using model: {self.model_name}",
        }
