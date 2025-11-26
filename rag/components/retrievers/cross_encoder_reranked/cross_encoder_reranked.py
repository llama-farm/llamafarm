"""Cross-encoder reranked retrieval strategy."""

import requests
from pathlib import Path
from typing import Any, Dict, List, Optional

from components.retrievers.base import RetrievalStrategy, RetrievalResult
from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.retrievers.cross_encoder_reranked")


class CrossEncoderRerankedStrategy(RetrievalStrategy):
    """
    Cross-encoder reranking strategy for accurate document ranking.

    This strategy performs proper cross-encoder reranking where query and
    document are jointly encoded (not independently like bi-encoders).

    Uses the Universal Runtime's /v1/rerank endpoint with HuggingFace
    cross-encoder models for fast, accurate reranking.

    Best for:
    - Production deployments requiring fast reranking
    - When accuracy is critical (cross-encoders outperform bi-encoders)
    - Local/on-prem deployments (no external API calls)

    Performance: Fast (50-400 docs/sec depending on model and hardware)
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
        self.model_name = config.get("model_name")  # Name from runtime.models
        self.model_base_url = config.get("model_base_url")  # Resolved by RAGManager
        self.model_id = config.get("model_id")  # HuggingFace model ID
        self.initial_k = config.get("initial_k", 30)
        self.final_k = config.get("final_k", 10)
        self.base_strategy_name = config.get("base_strategy", "BasicSimilarityStrategy")
        self.base_strategy_config = config.get("base_strategy_config", {})
        self.relevance_threshold = config.get("relevance_threshold", 0.0)
        self.timeout = config.get("timeout", 60)  # Request timeout in seconds

        # Model state
        self._base_strategy: Optional[RetrievalStrategy] = None

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

    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        query_text: str = "",
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve and rerank documents using Universal Runtime cross-encoder.

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
            raise ValueError("query_text is required for reranking")

        if not self.model_base_url or not self.model_id:
            raise ValueError(
                f"Model configuration not resolved for '{self.model_name}'. "
                "Ensure the model exists in runtime.models."
            )

        # Initialize base strategy
        self._initialize_base_strategy()

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
                    "model_id": self.model_id,
                    "initial_retrieved": 0,
                },
            )

        # Step 2: Rerank using Universal Runtime
        logger.info(
            f"Reranking {len(initial_result.documents)} documents with {self.model_id}"
        )
        reranked_docs = self._rerank_with_universal_runtime(
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
            doc.metadata["reranker_model"] = self.model_id

        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "model_name": self.model_name,
                "model_id": self.model_id,
                "base_url": self.model_base_url,
                "initial_retrieved": len(initial_result.documents),
                "after_reranking": len(filtered_docs),
                "final_count": len(final_docs),
            },
        )

    def _rerank_with_universal_runtime(
        self,
        query_text: str,
        documents: List[Document],
    ) -> List[tuple[Document, float]]:
        """
        Rerank documents using Universal Runtime's /v1/rerank endpoint.

        Normalizes cross-encoder logits to 0-1 range for intuitive thresholding.

        Returns:
            List of (document, score) tuples sorted by relevance
        """
        try:
            # Prepare request
            doc_texts = [doc.content for doc in documents]

            # Construct URL - add /v1 if not already present
            base_url = self.model_base_url.rstrip('/')
            if base_url.endswith('/v1'):
                url = f"{base_url}/rerank"
            else:
                url = f"{base_url}/v1/rerank"

            payload = {
                "model": self.model_id,
                "query": query_text,
                "documents": doc_texts,
                "return_documents": False,  # We already have the documents
            }

            # Call Universal Runtime
            response = requests.post(
                url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            result = response.json()

            # Map results back to documents and collect raw scores
            raw_scores = []
            doc_map = []
            for item in result["data"]:
                doc_idx = item["index"]
                raw_score = item["relevance_score"]
                raw_scores.append(raw_score)
                doc_map.append((documents[doc_idx], raw_score))

            # Normalize scores to 0-1 range using min-max normalization
            # This makes threshold filtering intuitive (0.0 = keep relevant docs)
            if raw_scores:
                min_score = min(raw_scores)
                max_score = max(raw_scores)
                score_range = max_score - min_score

                if score_range > 0:
                    # Normalize to 0-1 range
                    reranked = [
                        (doc, (score - min_score) / score_range)
                        for doc, score in doc_map
                    ]
                else:
                    # All scores are identical, give them 0.5
                    reranked = [(doc, 0.5) for doc, _ in doc_map]
            else:
                reranked = []

            logger.info(
                f"Successfully reranked {len(reranked)} documents",
                raw_score_range=f"{min(raw_scores):.2f} to {max(raw_scores):.2f}" if raw_scores else "N/A",
                normalized_top_score=reranked[0][1] if reranked else None
            )

            return reranked

        except requests.exceptions.Timeout:
            logger.error(f"Timeout calling Universal Runtime rerank endpoint")
            # Fallback: return documents with neutral scores
            return [(doc, 0.5) for doc in documents]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Universal Runtime: {e}", exc_info=True)
            # Fallback: return documents with neutral scores
            return [(doc, 0.5) for doc in documents]

        except Exception as e:
            logger.error(f"Unexpected error during reranking: {e}", exc_info=True)
            # Fallback: return documents with neutral scores
            return [(doc, 0.5) for doc in documents]

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Universal support - works with any vector store."""
        return True

    def validate_config(self) -> bool:
        """Validate configuration."""
        if self.initial_k < 1:
            return False
        if self.final_k < 1:
            return False
        if not self.model_id:
            return False
        if not self.model_base_url:
            return False
        return True

    def get_config_schema(self) -> Dict[str, Any]:
        """Get configuration schema."""
        return {
            "type": "object",
            "properties": {
                "model_name": {
                    "type": "string",
                    "description": "Name of model from runtime.models to use for reranking",
                },
                "initial_k": {"type": "integer", "minimum": 10, "maximum": 100, "default": 30},
                "final_k": {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
                "base_strategy": {
                    "type": "string",
                    "enum": ["BasicSimilarityStrategy", "MetadataFilteredStrategy"],
                    "default": "BasicSimilarityStrategy",
                },
                "relevance_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.0,
                },
                "timeout": {"type": "integer", "minimum": 10, "maximum": 300, "default": 60},
            },
        }

    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance characteristics."""
        return {
            "speed": "fast",
            "memory_usage": "medium",
            "complexity": "medium",
            "accuracy": "very_high",
            "best_for": [
                "production_deployments",
                "accurate_reranking",
                "local_inference",
                "high_throughput",
            ],
            "notes": f"Universal Runtime cross-encoder reranking with model: {self.model_id}",
        }
