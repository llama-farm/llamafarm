# Multi-Turn RAG with Cross-Encoder Reranking - Implementation Plan

## Executive Summary

This document outlines the implementation of two new retrieval strategies for LlamaFarm:
1. **CrossEncoderRerankedStrategy** - Uses specialized cross-encoder models for fast, accurate reranking
2. **MultiTurnRAGStrategy** - Decomposes complex queries into multiple sub-queries, performs retrieval for each, and reranks results

These strategies address the core limitation that RAG works well for simple questions but struggles when users provide extensive context or ask complex multi-part questions.

**Key Design Principle:** All models (including rerankers) are configured in `runtime.models` and referenced by name, keeping configuration centralized and easy to track.

## Problem Statement

### Current Limitations
- **Simple queries work well**: "What are the FDA requirements for clinical trials?" → RAG retrieves relevant chunks effectively
- **Context-heavy queries fail**: When users paste paragraphs of context or ask multi-faceted questions, single-pass RAG retrieves irrelevant content
- **No intelligent query decomposition**: Complex questions need to be broken down into simpler sub-queries
- **Score-based reranking is limited**: The existing `RerankedStrategy` uses metadata and recency factors but doesn't understand semantic relevance deeply

### Core Use Cases

#### Use Case 1: Simple Chat Question (CrossEncoderRerankedStrategy)
**Example:** "Describe differences in llama vs alpaca fibers"

**Flow:**
1. User asks focused question
2. Single vector similarity search (initial_k=30)
3. Cross-encoder reranks to top-k=5
4. Return best results

**Why this works:**
- Query is clear and focused
- No decomposition needed
- Cross-encoder provides semantic understanding
- Fast: ~150-200ms total

#### Use Case 2: Context-Heavy Query (MultiTurnRAGStrategy)
**Example:** User uploads PDF of test results and asks "What do these results mean?"

**Flow:**
1. Detect high complexity (extensive context provided)
2. Use fast LLM to generate focused queries:
   - "What are the key metrics in the test results?"
   - "What do abnormal values indicate?"
   - "What are normal ranges for each measurement?"
3. Retrieve documents for EACH sub-query (parallel)
4. Cross-encoder reranks results from each sub-query
5. Merge and deduplicate
6. Return top results

**Why this works:**
- Breaks down complex context into manageable queries
- Each sub-query gets accurate retrieval
- Cross-encoder ensures relevance at each step
- Fast model for decomposition (~500ms)
- Cross-encoder for reranking (~150ms per sub-query)

### Additional Use Cases
1. **Complex research questions**: "Compare the FDA guidance on data integrity from 2020 vs 2024, focusing on electronic records and audit trails"
2. **Multi-document synthesis**: "Summarize the common themes across all warning letters regarding manufacturing deviations"
3. **Ambiguous queries**: Questions that could be interpreted multiple ways

---

## Architecture Overview

### Strategy 1: CrossEncoderRerankedStrategy

**Purpose**: Use specialized cross-encoder models to rerank search results based on semantic relevance.

**Key Features**:
- Takes initial retrieval results (from any base strategy)
- Uses a cross-encoder model via Ollama (configured in `runtime.models`) to assess relevance
- 10-100x faster than LLM-based reranking
- Returns reranked results with precise relevance scores
- Model is referenced by name (e.g., "reranker") from runtime config
- Uses GGUF quantized models for efficient local inference

**When to Use**:
- **Use Case 1**: Simple, focused questions
- Production systems requiring speed + accuracy
- High-throughput retrieval pipelines
- Any reranking scenario (this should be the default)

**Performance**:
- Speed: 50-400 docs/sec (depends on model and quantization)
- Latency: 50-200ms for 30 candidates (GGUF on CPU/GPU)
- Cost: Near-zero (local compute via Ollama)

### Strategy 2: MultiTurnRAGStrategy

**Purpose**: Decompose complex queries into multiple focused sub-queries, retrieve for each, and intelligently combine results.

**Key Features**:
- Analyzes incoming query complexity using a fast LLM (from `runtime.models`)
- Determines if decomposition is needed
- Generates optimal sub-queries using the configured LLM
- Performs parallel retrieval for each sub-query
- Uses cross-encoder reranking for EACH sub-query (from `runtime.models`)
- Merges and deduplicates results
- Falls back to simple retrieval for straightforward queries

**When to Use**:
- **Use Case 2**: Context-heavy queries (PDFs, long text, etc.)
- Complex, multi-part questions
- Comparative or analytical queries
- When you want to maximize recall while maintaining precision

**Performance**:
- Simple query (fallback): ~100ms
- Complex query: ~1-2s (depends on number of sub-queries)
  - Complexity analysis: ~200ms (fast model)
  - Query generation: ~300-500ms (fast model)
  - Parallel retrieval: ~500ms
  - Cross-encoder reranking: ~150ms per sub-query (parallel)

---

## Detailed Design

### 1. CrossEncoderRerankedStrategy

#### Location
- `rag/components/retrievers/cross_encoder_reranked/`
- `rag/components/retrievers/cross_encoder_reranked/__init__.py`
- `rag/components/retrievers/cross_encoder_reranked/cross_encoder_reranked.py`

#### Configuration Schema
```yaml
# In rag/schema.yaml, add to retrievalStrategies section
crossEncoderRerankedConfig:
  type: object
  additionalProperties: false
  properties:
    model_name:
      type: string
      description: "Name of the model from runtime.models to use for reranking (e.g., 'reranker', 'fast-reranker')"
      example: "reranker"
    initial_k:
      type: integer
      default: 30
      minimum: 10
      maximum: 100
      description: "Number of initial candidates to retrieve before reranking"
    final_k:
      type: integer
      default: 10
      minimum: 1
      maximum: 50
      description: "Number of results to return after reranking"
    base_strategy:
      type: string
      enum: [BasicSimilarityStrategy, MetadataFilteredStrategy]
      default: BasicSimilarityStrategy
      description: "Base retrieval strategy to use for initial retrieval"
    base_strategy_config:
      type: object
      description: "Configuration for the base strategy"
      additionalProperties: true
    relevance_threshold:
      type: number
      default: 0.5
      minimum: 0.0
      maximum: 1.0
      description: "Minimum relevance score (0-1) from LLM to include a result"
    include_explanations:
      type: boolean
      default: false
      description: "Whether to include LLM explanations for ranking decisions"
    batch_size:
      type: integer
      default: 10
      minimum: 1
      maximum: 50
      description: "Number of documents to rerank per LLM call"
    temperature:
      type: number
      default: 0.1
      minimum: 0.0
      maximum: 1.0
      description: "LLM temperature for reranking (lower = more deterministic)"
```

#### Implementation

```python
# rag/components/retrievers/llm_reranked/llm_reranked.py

"""LLM-based reranking strategy."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import asyncio

from components.retrievers.base import RetrievalStrategy, RetrievalResult
from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.retrievers.llm_reranked")


class LLMRerankedStrategy(RetrievalStrategy):
    """
    LLM-based reranking strategy using OpenAI-compatible models.

    This strategy performs initial retrieval using a base strategy, then uses
    an LLM to assess semantic relevance and rerank results. The LLM evaluates
    each document's relevance to the query on a 0-1 scale.

    Use Cases:
    - Production systems requiring semantic understanding
    - Queries where keyword/vector similarity alone is insufficient
    - When you need explainable ranking decisions
    - As a refinement step after initial retrieval

    Performance: Slower (due to LLM calls)
    Complexity: High
    Accuracy: Very High
    """

    def __init__(
        self,
        name: str = "LLMRerankedStrategy",
        config: Optional[Dict[str, Any]] = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}

        # Configuration
        self.model_name = config.get("model_name", "fast")
        self.initial_k = config.get("initial_k", 30)
        self.final_k = config.get("final_k", 10)
        self.base_strategy_name = config.get("base_strategy", "BasicSimilarityStrategy")
        self.base_strategy_config = config.get("base_strategy_config", {})
        self.relevance_threshold = config.get("relevance_threshold", 0.5)
        self.include_explanations = config.get("include_explanations", False)
        self.batch_size = config.get("batch_size", 10)
        self.temperature = config.get("temperature", 0.1)

        # Will be initialized when needed
        self._base_strategy: Optional[RetrievalStrategy] = None
        self._llm_client = None
        self._project_config = None

    def _initialize_base_strategy(self, vector_store):
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

    def _initialize_llm_client(self, project_dir: Path):
        """Initialize LLM client from project configuration."""
        if self._llm_client is not None:
            return

        # Load project config
        from config.datamodel import LlamaFarmConfig
        from services.model_service import ModelService
        from services.runtime_service import RuntimeService
        import yaml

        config_path = project_dir / "llamafarm.yaml"
        with open(config_path) as f:
            raw_config = yaml.safe_load(f)

        self._project_config = LlamaFarmConfig(**raw_config)

        # Get model configuration
        model_config = ModelService.get_model(self._project_config, self.model_name)

        # Get runtime provider
        provider = RuntimeService.get_provider(model_config)
        self._llm_client = provider.get_client()
        self._model_id = model_config.model

        logger.info(
            f"Initialized LLM client for reranking",
            model_name=self.model_name,
            model_id=self._model_id,
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
        Retrieve and rerank documents using LLM.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of final results to return
            query_text: Original query text (required for LLM reranking)
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with LLM-reranked documents
        """
        if not query_text:
            raise ValueError("query_text is required for LLM reranking")

        # Initialize components
        self._initialize_base_strategy(vector_store)
        self._initialize_llm_client(self.project_dir)

        # Step 1: Initial retrieval using base strategy
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
                    "base_strategy": self.base_strategy_name,
                    "initial_retrieved": 0,
                    "final_count": 0,
                },
            )

        # Step 2: LLM-based reranking
        logger.info(f"Reranking {len(initial_result.documents)} documents with LLM")
        reranked_docs = self._rerank_with_llm(
            query_text=query_text,
            documents=initial_result.documents,
            initial_scores=initial_result.scores,
        )

        # Step 3: Filter by threshold and take top_k
        filtered_docs = [
            (doc, score, explanation)
            for doc, score, explanation in reranked_docs
            if score >= self.relevance_threshold
        ]

        final_docs = filtered_docs[:min(top_k, self.final_k)]

        # Prepare results
        documents = [doc for doc, _, _ in final_docs]
        scores = [score for _, score, _ in final_docs]

        # Add metadata
        for i, (doc, score, explanation) in enumerate(final_docs):
            doc.metadata["llm_relevance_score"] = score
            doc.metadata["rerank_position"] = i + 1
            if self.include_explanations and explanation:
                doc.metadata["relevance_explanation"] = explanation

        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "base_strategy": self.base_strategy_name,
                "model_name": self.model_name,
                "model_id": self._model_id,
                "initial_retrieved": len(initial_result.documents),
                "candidates_reranked": len(reranked_docs),
                "threshold_filtered": len(filtered_docs),
                "final_count": len(documents),
                "relevance_threshold": self.relevance_threshold,
            },
        )

    def _rerank_with_llm(
        self,
        query_text: str,
        documents: List[Document],
        initial_scores: List[float],
    ) -> List[tuple[Document, float, str]]:
        """
        Rerank documents using LLM relevance assessment.

        Returns:
            List of (document, llm_score, explanation) tuples, sorted by score
        """
        results = []

        # Process in batches to avoid token limits
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i : i + self.batch_size]
            batch_scores = initial_scores[i : i + self.batch_size]

            batch_results = self._rerank_batch(query_text, batch, batch_scores)
            results.extend(batch_results)

        # Sort by LLM relevance score (descending)
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def _rerank_batch(
        self,
        query_text: str,
        documents: List[Document],
        initial_scores: List[float],
    ) -> List[tuple[Document, float, str]]:
        """Rerank a batch of documents using LLM."""

        # Prepare prompt
        system_prompt = """You are a relevance assessment system. Your task is to evaluate how relevant each document is to the user's query.

For each document, provide:
1. A relevance score from 0.0 to 1.0 (where 1.0 is perfectly relevant, 0.0 is completely irrelevant)
2. A brief explanation of your reasoning (only if explanations are requested)

Respond with valid JSON only, no additional text."""

        docs_text = []
        for idx, doc in enumerate(documents):
            # Truncate content to avoid token limits (keep first 500 chars)
            content = doc.content[:500] + "..." if len(doc.content) > 500 else doc.content
            docs_text.append(f"Document {idx + 1}:\n{content}")

        user_prompt = f"""Query: {query_text}

Documents to assess:
{chr(10).join(docs_text)}

Assess the relevance of each document to the query. Return a JSON array with this structure:
[
  {{"document_index": 1, "relevance_score": 0.85, "explanation": "Brief explanation"}},
  ...
]

{"Include brief explanations for each score." if self.include_explanations else "Omit the explanation field."}"""

        try:
            # Call LLM
            response = self._llm_client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=2000,
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Try to extract JSON (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            assessments = json.loads(content)

            # Map back to documents
            results = []
            for assessment in assessments:
                idx = assessment.get("document_index", 0) - 1
                if 0 <= idx < len(documents):
                    score = float(assessment.get("relevance_score", 0.5))
                    explanation = assessment.get("explanation", "")
                    results.append((documents[idx], score, explanation))

            # Fill in any missing documents with low scores
            assessed_indices = {r[0] for r in results}
            for idx, doc in enumerate(documents):
                if doc not in assessed_indices:
                    results.append((doc, 0.3, "No assessment provided"))

            return results

        except Exception as e:
            logger.error(f"Error during LLM reranking: {e}", exc_info=True)
            # Fallback: use initial scores
            return [
                (doc, initial_scores[i] if i < len(initial_scores) else 0.5, "")
                for i, doc in enumerate(documents)
            ]

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """This is universal - works with any vector store."""
        return True

    def validate_config(self) -> bool:
        """Validate strategy configuration."""
        if self.initial_k < 1 or self.final_k < 1:
            return False
        if self.relevance_threshold < 0 or self.relevance_threshold > 1:
            return False
        if self.batch_size < 1:
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
                "initial_k": {
                    "type": "integer",
                    "minimum": 10,
                    "maximum": 100,
                    "default": 30,
                },
                "final_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
                "base_strategy": {
                    "type": "string",
                    "enum": ["BasicSimilarityStrategy", "MetadataFilteredStrategy"],
                    "default": "BasicSimilarityStrategy",
                },
                "relevance_threshold": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.5,
                },
                "include_explanations": {
                    "type": "boolean",
                    "default": False,
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                },
                "temperature": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.1,
                },
            },
        }

    def get_performance_info(self) -> Dict[str, Any]:
        """Get performance characteristics."""
        return {
            "speed": "slow",
            "memory_usage": "medium",
            "complexity": "high",
            "accuracy": "very_high",
            "best_for": [
                "semantic_relevance",
                "production_systems",
                "explainable_ranking",
            ],
            "notes": f"Uses LLM ({self.model_name}) for semantic reranking - slower but most accurate",
        }
```

### 2. MultiTurnRAGStrategy

#### Location
- `rag/components/retrievers/multi_turn_rag/`
- `rag/components/retrievers/multi_turn_rag/__init__.py`
- `rag/components/retrievers/multi_turn_rag/multi_turn_rag.py`

#### Configuration Schema
```yaml
# In rag/schema.yaml, add to retrievalStrategies section
multiTurnRAGConfig:
  type: object
  additionalProperties: false
  properties:
    model_name:
      type: string
      description: "Name of the model from runtime.models to use for query decomposition and reranking"
      example: "fast"
    complexity_threshold:
      type: number
      default: 0.6
      minimum: 0.0
      maximum: 1.0
      description: "Threshold for determining if query needs decomposition (0-1 scale)"
    max_subqueries:
      type: integer
      default: 4
      minimum: 1
      maximum: 10
      description: "Maximum number of sub-queries to generate"
    min_subqueries:
      type: integer
      default: 2
      minimum: 1
      maximum: 5
      description: "Minimum number of sub-queries to generate for complex queries"
    results_per_subquery:
      type: integer
      default: 10
      minimum: 1
      maximum: 50
      description: "Number of results to retrieve per sub-query"
    final_k:
      type: integer
      default: 10
      minimum: 1
      maximum: 50
      description: "Final number of results to return after combining and reranking"
    base_strategy:
      type: string
      enum: [BasicSimilarityStrategy, MetadataFilteredStrategy]
      default: BasicSimilarityStrategy
      description: "Base retrieval strategy to use for each sub-query"
    base_strategy_config:
      type: object
      description: "Configuration for the base strategy"
      additionalProperties: true
    enable_llm_reranking:
      type: boolean
      default: true
      description: "Use LLM for final reranking (vs simple score aggregation)"
    temperature:
      type: number
      default: 0.3
      minimum: 0.0
      maximum: 1.0
      description: "LLM temperature for query decomposition"
    deduplication_threshold:
      type: number
      default: 0.9
      minimum: 0.0
      maximum: 1.0
      description: "Similarity threshold for deduplicating results (cosine similarity)"
```

#### Implementation Outline

```python
# rag/components/retrievers/multi_turn_rag/multi_turn_rag.py

"""Multi-turn RAG strategy with query decomposition."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np

from components.retrievers.base import RetrievalStrategy, RetrievalResult
from core.base import Document
from core.logging import RAGStructLogger

logger = RAGStructLogger("rag.components.retrievers.multi_turn_rag")


class MultiTurnRAGStrategy(RetrievalStrategy):
    """
    Multi-turn RAG strategy with intelligent query decomposition.

    This strategy analyzes query complexity and decomposes complex queries
    into multiple focused sub-queries, retrieves for each, and combines
    results using LLM-based reranking.

    Workflow:
    1. Analyze query complexity using LLM
    2. If complex: decompose into sub-queries
    3. Perform retrieval for each sub-query in parallel
    4. Deduplicate and merge results
    5. Use LLM to rerank combined results
    6. Return top-k most relevant documents

    Use Cases:
    - Complex, multi-part questions
    - Queries with extensive context
    - Comparative or analytical queries
    - Maximizing recall while maintaining precision

    Performance: Slow (multiple retrievals + LLM calls)
    Complexity: Very High
    Accuracy: Highest
    """

    def __init__(
        self,
        name: str = "MultiTurnRAGStrategy",
        config: Optional[Dict[str, Any]] = None,
        project_dir: Path | None = None,
    ):
        super().__init__(name, config, project_dir)
        config = config or {}

        # Configuration
        self.model_name = config.get("model_name", "fast")
        self.complexity_threshold = config.get("complexity_threshold", 0.6)
        self.max_subqueries = config.get("max_subqueries", 4)
        self.min_subqueries = config.get("min_subqueries", 2)
        self.results_per_subquery = config.get("results_per_subquery", 10)
        self.final_k = config.get("final_k", 10)
        self.base_strategy_name = config.get("base_strategy", "BasicSimilarityStrategy")
        self.base_strategy_config = config.get("base_strategy_config", {})
        self.enable_llm_reranking = config.get("enable_llm_reranking", True)
        self.temperature = config.get("temperature", 0.3)
        self.deduplication_threshold = config.get("deduplication_threshold", 0.9)

        # Will be initialized when needed
        self._base_strategy: Optional[RetrievalStrategy] = None
        self._llm_client = None
        self._project_config = None
        self._embedder = None

    def retrieve(
        self,
        query_embedding: List[float],
        vector_store,
        top_k: int = 5,
        query_text: str = "",
        **kwargs,
    ) -> RetrievalResult:
        """
        Retrieve documents using multi-turn RAG.

        Args:
            query_embedding: The embedded query vector
            vector_store: The vector store to search
            top_k: Number of final results to return
            query_text: Original query text (required)
            **kwargs: Additional arguments

        Returns:
            RetrievalResult with documents from multi-turn retrieval
        """
        if not query_text:
            raise ValueError("query_text is required for multi-turn RAG")

        # Initialize components
        self._initialize_components(vector_store)

        # Step 1: Analyze query complexity
        logger.info("Analyzing query complexity")
        complexity_score, needs_decomposition = self._analyze_complexity(query_text)

        if not needs_decomposition:
            # Simple query - use base strategy directly
            logger.info("Query is simple, using base strategy directly")
            return self._simple_retrieval(query_embedding, vector_store, top_k, **kwargs)

        # Step 2: Decompose query into sub-queries
        logger.info("Query is complex, decomposing into sub-queries")
        sub_queries = self._decompose_query(query_text)

        # Step 3: Retrieve for each sub-query
        logger.info(f"Retrieving for {len(sub_queries)} sub-queries")
        all_results = self._parallel_retrieve(sub_queries, vector_store)

        # Step 4: Deduplicate and merge
        logger.info("Deduplicating and merging results")
        merged_docs = self._deduplicate_results(all_results)

        # Step 5: Final reranking
        if self.enable_llm_reranking and len(merged_docs) > 0:
            logger.info(f"LLM reranking {len(merged_docs)} merged results")
            final_docs = self._llm_rerank(query_text, merged_docs, top_k)
        else:
            # Simple score-based ranking
            final_docs = self._score_based_rerank(merged_docs, top_k)

        documents = [doc for doc, _ in final_docs]
        scores = [score for _, score in final_docs]

        return RetrievalResult(
            documents=documents,
            scores=scores,
            strategy_metadata={
                "strategy": self.name,
                "version": "1.0.0",
                "model_name": self.model_name,
                "complexity_score": complexity_score,
                "decomposed": needs_decomposition,
                "num_subqueries": len(sub_queries) if needs_decomposition else 0,
                "total_candidates": len(merged_docs),
                "final_count": len(documents),
                "llm_reranking": self.enable_llm_reranking,
            },
        )

    def _analyze_complexity(self, query_text: str) -> Tuple[float, bool]:
        """
        Analyze query complexity to determine if decomposition is needed.

        Returns:
            (complexity_score, needs_decomposition)
        """
        system_prompt = """You are a query complexity analyzer. Determine if a query is complex enough to benefit from decomposition into multiple sub-queries.

Complex queries include:
- Multi-part questions ("Compare X and Y...", "What are A, B, and C?")
- Queries with extensive context (multiple paragraphs)
- Analytical questions requiring synthesis across documents
- Questions with multiple conditions or constraints

Simple queries include:
- Single, focused questions
- Direct factual queries
- Questions with minimal context

Respond with JSON only:
{"complexity_score": 0.0-1.0, "needs_decomposition": true/false, "reasoning": "brief explanation"}"""

        user_prompt = f"""Analyze this query:

{query_text}

Determine its complexity and whether it needs decomposition."""

        try:
            response = self._llm_client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)

            complexity_score = float(result.get("complexity_score", 0.5))
            needs_decomposition = result.get("needs_decomposition", False)

            # Override with threshold if score provided
            if complexity_score >= self.complexity_threshold:
                needs_decomposition = True

            logger.info(
                f"Query complexity: {complexity_score:.2f}, decompose: {needs_decomposition}",
                reasoning=result.get("reasoning", ""),
            )

            return complexity_score, needs_decomposition

        except Exception as e:
            logger.warning(f"Error analyzing complexity: {e}, defaulting to simple retrieval")
            return 0.5, False

    def _decompose_query(self, query_text: str) -> List[str]:
        """Decompose query into focused sub-queries using LLM."""

        system_prompt = f"""You are a query decomposition expert. Break down complex queries into {self.min_subqueries}-{self.max_subqueries} focused sub-queries that together address all aspects of the original query.

Guidelines:
- Each sub-query should be self-contained and specific
- Cover all aspects of the original query
- Avoid redundancy between sub-queries
- Keep sub-queries concise and focused
- Generate {self.min_subqueries}-{self.max_subqueries} sub-queries

Respond with JSON only:
{{"sub_queries": ["query 1", "query 2", ...]}}"""

        user_prompt = f"""Decompose this query:

{query_text}

Generate {self.min_subqueries}-{self.max_subqueries} focused sub-queries."""

        try:
            response = self._llm_client.chat.completions.create(
                model=self._model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=1000,
            )

            content = response.choices[0].message.content.strip()

            # Extract JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            result = json.loads(content)
            sub_queries = result.get("sub_queries", [])

            # Limit to max_subqueries
            sub_queries = sub_queries[: self.max_subqueries]

            logger.info(f"Generated {len(sub_queries)} sub-queries", sub_queries=sub_queries)

            return sub_queries

        except Exception as e:
            logger.error(f"Error decomposing query: {e}", exc_info=True)
            # Fallback: return original query
            return [query_text]

    def _parallel_retrieve(
        self,
        sub_queries: List[str],
        vector_store,
    ) -> List[List[Tuple[Document, float]]]:
        """Retrieve documents for each sub-query."""

        all_results = []

        for sub_query in sub_queries:
            # Get embedding for sub-query
            sub_query_embedding = self._embedder.embed_text(sub_query)

            # Retrieve using base strategy
            result = self._base_strategy.retrieve(
                query_embedding=sub_query_embedding,
                vector_store=vector_store,
                top_k=self.results_per_subquery,
            )

            # Combine docs and scores
            sub_results = list(zip(result.documents, result.scores))
            all_results.append(sub_results)

            logger.debug(f"Sub-query '{sub_query[:50]}...' retrieved {len(sub_results)} results")

        return all_results

    def _deduplicate_results(
        self,
        all_results: List[List[Tuple[Document, float]]],
    ) -> List[Tuple[Document, float, int]]:
        """
        Deduplicate results from multiple sub-queries.

        Returns:
            List of (document, best_score, frequency) tuples
        """
        # Track documents by content hash
        doc_map: Dict[str, Tuple[Document, float, int]] = {}

        for sub_results in all_results:
            for doc, score in sub_results:
                # Create content hash for deduplication
                content_hash = hash(doc.content[:1000])  # Use first 1000 chars

                if content_hash in doc_map:
                    # Document seen before - update score and frequency
                    existing_doc, existing_score, frequency = doc_map[content_hash]
                    # Keep best score, increment frequency
                    doc_map[content_hash] = (
                        doc,
                        max(existing_score, score),
                        frequency + 1,
                    )
                else:
                    # New document
                    doc_map[content_hash] = (doc, score, 1)

        # Convert to list and sort by score * frequency
        merged = []
        for doc, score, freq in doc_map.values():
            # Add frequency information to metadata
            doc.metadata["subquery_frequency"] = freq
            doc.metadata["base_score"] = score
            # Boost score by frequency
            combined_score = score * (1.0 + 0.1 * (freq - 1))  # +10% per additional occurrence
            merged.append((doc, combined_score, freq))

        # Sort by combined score
        merged.sort(key=lambda x: x[1], reverse=True)

        logger.info(
            f"Deduplicated {sum(len(r) for r in all_results)} results to {len(merged)} unique documents"
        )

        return merged

    def _llm_rerank(
        self,
        query_text: str,
        merged_docs: List[Tuple[Document, float, int]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Use LLM to rerank merged results."""

        # Similar to LLMRerankedStrategy but adapted for merged results
        # Implementation details omitted for brevity
        # Would call LLM to assess relevance of each document to original query

        pass

    def _score_based_rerank(
        self,
        merged_docs: List[Tuple[Document, float, int]],
        top_k: int,
    ) -> List[Tuple[Document, float]]:
        """Simple score-based ranking fallback."""

        sorted_docs = sorted(merged_docs, key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score, _ in sorted_docs[:top_k]]

    # Additional helper methods omitted for brevity...
    # _initialize_components, _simple_retrieval, etc.

    def supports_vector_store(self, vector_store_type: str) -> bool:
        """Universal support."""
        return True
```

---

## Schema Updates

### Update `rag/schema.yaml`

Add new retrieval strategy configurations:

```yaml
# In retrievalStrategyConfig oneOf section, add:
- required: [type, config]
  properties:
    type:
      type: string
      const: LLMRerankedStrategy
      description: LLM-based reranking strategy
    config:
      $ref: "#/definitions/retrievalStrategies/llmRerankedConfig"

- required: [type, config]
  properties:
    type:
      type: string
      const: MultiTurnRAGStrategy
      description: Multi-turn RAG with query decomposition
    config:
      $ref: "#/definitions/retrievalStrategies/multiTurnRAGConfig"
```

Add to enum in `databaseDefinition.retrieval_strategies[].type`:
```yaml
enum:
  [
    # ... existing strategies ...
    LLMRerankedStrategy,
    MultiTurnRAGStrategy,
  ]
```

---

## Configuration Example

```yaml
# llamafarm.yaml example

version: v1
name: research-assistant
namespace: default

runtime:
  default_model: balanced

  models:
    - name: fast
      description: "Fast model for query decomposition and reranking"
      provider: ollama
      model: gemma3:1b
      base_url: http://localhost:11434/v1

    - name: balanced
      description: "Balanced model for main chat"
      provider: ollama
      model: qwen3:4b
      base_url: http://localhost:11434/v1

rag:
  default_database: main_db

  databases:
    - name: main_db
      type: ChromaStore
      config:
        persist_directory: ./data/chroma_db
        distance_function: cosine
        collection_name: documents

      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: multi_turn_search

      embedding_strategies:
        - name: default_embeddings
          type: UniversalEmbedder
          config:
            model: nomic-ai/nomic-embed-text-v1.5
            dimension: 768

      retrieval_strategies:
        # Simple similarity search
        - name: simple_search
          type: BasicSimilarityStrategy
          config:
            top_k: 10
            distance_metric: cosine

        # LLM-based reranking
        - name: llm_reranked_search
          type: LLMRerankedStrategy
          config:
            model_name: fast  # Uses "fast" model from runtime.models
            initial_k: 30
            final_k: 10
            base_strategy: BasicSimilarityStrategy
            relevance_threshold: 0.5
            include_explanations: false
            batch_size: 10

        # Multi-turn RAG with decomposition
        - name: multi_turn_search
          type: MultiTurnRAGStrategy
          default: true
          config:
            model_name: fast  # Uses "fast" model for decomposition
            complexity_threshold: 0.6
            max_subqueries: 4
            min_subqueries: 2
            results_per_subquery: 10
            final_k: 10
            base_strategy: BasicSimilarityStrategy
            enable_llm_reranking: true
            temperature: 0.3
            deduplication_threshold: 0.9

  data_processing_strategies:
    - name: universal_processor
      # ... existing config ...
```

---

## Complete Configuration Example

### Full `llamafarm.yaml` with Reranker Models

```yaml
version: v1
name: fda-assistant
namespace: default

runtime:
  default_model: balanced

  models:
    # Fast model for query decomposition and complexity analysis
    fast:
      description: "Fast model for query processing"
      provider: ollama
      model: gemma3:1b
      base_url: "http://localhost:11434/v1"

    # Balanced model for main chat
    balanced:
      description: "Balanced model for general use"
      provider: ollama
      model: qwen3:4b
      base_url: "http://localhost:11434/v1"

    # CROSS-ENCODER RERANKING MODELS (configured just like other models)
    # These run via Ollama using GGUF quantized versions
    reranker:
      description: "Production cross-encoder reranker via Ollama"
      provider: ollama
      model: bge-reranker-v2-m3  # GGUF version via Ollama
      base_url: "http://localhost:11434/v1"
      # Note: Pull with `ollama pull bge-reranker-v2-m3` first

    fast-reranker:
      description: "Fast cross-encoder for high throughput"
      provider: ollama
      model: nomic-embed-text  # Can also be used for reranking
      base_url: "http://localhost:11434/v1"

prompts:
  - name: default
    messages:
      - role: system
        content: >-
          You are an FDA specialist. Answer concisely and cite sources.

rag:
  default_database: main_db

  databases:
    - name: main_db
      type: ChromaStore
      config:
        persist_directory: ./data/chroma_db
        distance_function: cosine

      default_embedding_strategy: default_embeddings
      default_retrieval_strategy: smart_search

      embedding_strategies:
        - name: default_embeddings
          type: UniversalEmbedder
          config:
            model: nomic-ai/nomic-embed-text-v1.5
            dimension: 768

      retrieval_strategies:
        # USE CASE 1: Simple questions
        - name: simple_search
          type: CrossEncoderRerankedStrategy
          config:
            model_name: reranker  # References "reranker" from runtime.models
            initial_k: 30
            final_k: 5
            base_strategy: BasicSimilarityStrategy
            batch_size: 32

        # USE CASE 2: Context-heavy queries (DEFAULT)
        - name: smart_search
          type: MultiTurnRAGStrategy
          default: true
          config:
            # Query decomposition model
            model_name: fast  # References "fast" from runtime.models

            # Complexity settings
            complexity_threshold: 0.6
            max_subqueries: 4
            results_per_subquery: 10
            final_k: 10

            # Cross-encoder reranking (NOT expensive LLM)
            enable_cross_encoder_reranking: true
            cross_encoder_model_name: reranker  # References "reranker"
            enable_llm_reranking: false

  data_processing_strategies:
    - name: pdf_ingest
      description: "Process PDFs with semantic chunking"
      parsers:
        - type: PDFParser_LlamaIndex
          file_include_patterns: ["*.pdf"]
          config:
            chunk_size: 1500
            chunk_overlap: 200
            chunk_strategy: semantic
      extractors:
        - type: HeadingExtractor
        - type: ContentStatisticsExtractor

datasets:
  - name: research-notes
    data_processing_strategy: pdf_ingest
    database: main_db
    files: []
```

### Key Configuration Points

1. **Reranker models are just models**: They live in `runtime.models` like chat models
2. **Use Ollama provider**: All models use Ollama (GGUF format)
3. **Reference by name**: Strategies reference models like `model_name: reranker`
4. **Easy to swap**: Change model in one place (e.g., `bge-reranker-v2-m3` to `nomic-embed-text`)
5. **Pull models first**: Run `ollama pull bge-reranker-v2-m3` before using
6. **Two use cases covered**:
   - `simple_search` for Use Case 1 (simple questions)
   - `smart_search` for Use Case 2 (context-heavy, auto-detects complexity)

---

## Usage Examples

### Use Case 1: Simple Chat Question (CLI)

```bash
# Simple, focused question - uses CrossEncoderRerankedStrategy
lf rag query --database main_db \
  --retrieval-strategy simple_search \
  "Describe differences in llama vs alpaca fibers"

# Alternative: let the system auto-detect (uses smart_search = MultiTurnRAG)
# But complexity analysis will detect it's simple and fall back to base strategy
lf rag query --database main_db \
  "What are the key requirements for clinical trials?"

# Output:
# → Complexity Score: 0.3 (simple)
# → Using BasicSimilarity + CrossEncoder reranking
# → Retrieved 30 candidates, reranked to 5
# → Total time: 180ms
```

### Use Case 2: Context-Heavy Query (CLI)

```bash
# User provides extensive context (e.g., paste from PDF or long document)
lf rag query --database main_db \
  "$(cat my_test_results.pdf)

  These are my latest test results showing cholesterol at 240,
  blood pressure at 145/95, and glucose at 110. I also have a family
  history of heart disease and I'm 45 years old. What do these results
  mean for my health?"

# Output:
# → Complexity Score: 0.85 (complex - extensive context)
# → Decomposing into sub-queries...
# → Generated 4 sub-queries:
#    1. "What is the normal range for cholesterol levels?"
#    2. "What does blood pressure of 145/95 indicate?"
#    3. "What is a normal glucose level?"
#    4. "How does family history affect cardiovascular risk?"
# → Retrieving for each sub-query (parallel)...
# → Sub-query 1: Retrieved 10, reranked to 5 (150ms)
# → Sub-query 2: Retrieved 10, reranked to 5 (150ms)
# → Sub-query 3: Retrieved 10, reranked to 5 (150ms)
# → Sub-query 4: Retrieved 10, reranked to 5 (150ms)
# → Merging and deduplicating: 40 candidates → 15 unique
# → Final reranking to top 10
# → Total time: 1.8s
```

### Additional CLI Examples

```bash
# Use multi-turn RAG explicitly
lf rag query --database main_db \
  --retrieval-strategy smart_search \
  "Compare the FDA guidance on data integrity from 2020 vs 2024"

# Use simple search for straightforward queries
lf rag query --database main_db \
  --retrieval-strategy simple_search \
  "FDA phone number"
```

### API Usage

#### Use Case 1: Simple Question
```bash
# Simple focused question - fast cross-encoder reranking
curl -X POST http://localhost:8000/v1/projects/default/fda-assistant/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Describe differences in llama vs alpaca fibers",
    "database": "main_db",
    "retrieval_strategy": "simple_search",
    "top_k": 5
  }'

# Response:
# {
#   "query": "Describe differences in llama vs alpaca fibers",
#   "results": [...],
#   "total_results": 5,
#   "processing_time_ms": 180,
#   "retrieval_strategy_used": "simple_search",
#   "strategy_metadata": {
#     "strategy": "CrossEncoderRerankedStrategy",
#     "model_name": "reranker",
#     "initial_retrieved": 30,
#     "final_count": 5
#   }
# }
```

#### Use Case 2: Context-Heavy Query
```bash
# Complex query with extensive context
curl -X POST http://localhost:8000/v1/projects/default/fda-assistant/rag/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Here are my test results: [extensive medical data...]. What do these results mean for my health?",
    "database": "main_db",
    "retrieval_strategy": "smart_search",
    "top_k": 10
  }'

# Response:
# {
#   "query": "Here are my test results...",
#   "results": [...],
#   "total_results": 10,
#   "processing_time_ms": 1850,
#   "retrieval_strategy_used": "smart_search",
#   "strategy_metadata": {
#     "strategy": "MultiTurnRAGStrategy",
#     "model_name": "fast",
#     "complexity_score": 0.85,
#     "decomposed": true,
#     "num_subqueries": 4,
#     "total_candidates": 40,
#     "unique_after_dedup": 15,
#     "final_count": 10
#   }
# }
```

### Python SDK Usage

```python
from llamafarm import LlamaFarmClient

client = LlamaFarmClient()

# USE CASE 1: Simple question
response = client.rag.query(
    namespace="default",
    project="fda-assistant",
    query="Describe differences in llama vs alpaca fibers",
    database="main_db",
    retrieval_strategy="simple_search",
    top_k=5
)

print(f"Processing time: {response.processing_time_ms}ms")
print(f"Strategy: {response.strategy_metadata['strategy']}")
print(f"Model: {response.strategy_metadata['model_name']}")

for result in response.results:
    print(f"Score: {result.score:.2f}")
    print(f"Content: {result.content[:200]}...")
    print("---")

# USE CASE 2: Context-heavy query
with open("my_test_results.pdf") as f:
    context = f.read()

response = client.rag.query(
    namespace="default",
    project="fda-assistant",
    query=f"{context}\n\nWhat do these results mean for my health?",
    database="main_db",
    retrieval_strategy="smart_search",
    top_k=10
)

print(f"Processing time: {response.processing_time_ms}ms")
print(f"Complexity score: {response.strategy_metadata['complexity_score']}")
print(f"Decomposed: {response.strategy_metadata['decomposed']}")
if response.strategy_metadata['decomposed']:
    print(f"Sub-queries: {response.strategy_metadata['num_subqueries']}")

for result in response.results:
    print(f"Score: {result.score:.2f}")
    print(f"Content: {result.content[:200]}...")
    print(f"Matched queries: {result.metadata.get('subquery_frequency', 1)}")
    print("---")
```

---

## Implementation Plan

### Phase 1: CrossEncoderRerankedStrategy (Week 1-2)
1. **Day 1-2**: Implement base structure and model loading from runtime.models
2. **Day 3-4**: Implement cross-encoder reranking logic and batch processing
3. **Day 5-6**: Add schema definitions and update `rag/schema.yaml`
4. **Day 7-8**: Write unit tests and integration tests
5. **Day 9-10**: Documentation and example configurations for Use Case 1

### Phase 2: MultiTurnRAGStrategy (Week 3-4)
1. **Day 1-3**: Implement complexity analysis and query decomposition (using fast model from runtime.models)
2. **Day 4-6**: Implement parallel retrieval with cross-encoder reranking per sub-query
3. **Day 7-9**: Implement deduplication and result merging
4. **Day 10-12**: Write comprehensive tests for Use Case 2
5. **Day 13-14**: Documentation, examples, and performance tuning

### Phase 3: Integration & Testing (Week 5)
1. **Day 1-2**: Add example configurations showing opt-in usage
2. **Day 3-4**: End-to-end testing with both use cases (simple + context-heavy)
3. **Day 5**: Performance benchmarking (compare simple vs multi-turn)
4. **Day 6-7**: Documentation updates emphasizing opt-in nature (README, RAG guide, API docs)

### Phase 4: Advanced Features (Optional - Week 6)
1. Caching for query decomposition results
2. Adaptive complexity threshold tuning
3. Query expansion integration
4. Hybrid strategies combining multiple approaches

---

## Testing Strategy

### Unit Tests

```python
# tests/test_llm_reranked_strategy.py
def test_llm_reranked_basic():
    """Test basic LLM reranking with mock LLM."""
    # Test with mock documents and mock LLM responses
    pass

def test_llm_reranked_batch_processing():
    """Test batch processing of documents."""
    pass

def test_llm_reranked_threshold_filtering():
    """Test relevance threshold filtering."""
    pass

# tests/test_multi_turn_rag_strategy.py
def test_complexity_analysis():
    """Test query complexity detection."""
    pass

def test_query_decomposition():
    """Test query decomposition into sub-queries."""
    pass

def test_result_deduplication():
    """Test deduplication of results from sub-queries."""
    pass

def test_simple_query_fallback():
    """Test that simple queries use base strategy directly."""
    pass
```

### Integration Tests

```python
# tests/integration/test_rag_strategies.py
def test_llm_reranked_end_to_end():
    """Test LLM reranking with real vector store and model."""
    # Use test project with configured model
    pass

def test_multi_turn_rag_complex_query():
    """Test multi-turn RAG with complex real-world query."""
    pass

def test_multi_turn_rag_simple_query():
    """Test that simple queries don't trigger decomposition."""
    pass
```

### Performance Tests

```python
# tests/performance/test_retrieval_performance.py
def benchmark_llm_reranked_vs_basic():
    """Compare performance and accuracy."""
    pass

def benchmark_multi_turn_overhead():
    """Measure overhead of complexity analysis and decomposition."""
    pass
```

---

## Documentation Updates

### 1. Update `docs/website/docs/rag/index.md`

Add new sections:
- **LLM-Based Reranking**: Explanation and use cases
- **Multi-Turn RAG**: When and why to use it
- **Strategy Comparison Table**: Performance, accuracy, use cases

### 2. Update `docs/website/docs/configuration/example-configs.md`

Add examples of:
- Basic LLM reranking configuration
- Multi-turn RAG configuration
- Hybrid strategies combining multiple approaches

### 3. Update `README.md`

Add to RAG features:
- Intelligent query decomposition
- LLM-powered reranking
- Multi-turn retrieval for complex queries

### 4. Create Tutorial

New file: `docs/website/docs/examples/multi-turn-rag-tutorial.md`
- Step-by-step guide
- Real-world examples
- Performance comparison

---

## Performance Considerations

### CrossEncoderRerankedStrategy
- **Latency**: +50-200ms depending on model and batch size
- **Cost**: Near-zero (local compute only)
- **Optimization**: Batch processing, model caching, GPU acceleration

### MultiTurnRAGStrategy
- **Latency**:
  - Simple queries (fallback): +100-200ms (no decomposition)
  - Complex queries: +1000-2000ms (includes decomposition + parallel retrieval + reranking)
    - Complexity analysis: ~200ms (fast model)
    - Query decomposition: ~300-500ms (fast model)
    - Parallel retrieval + reranking: ~500-800ms (4 sub-queries × ~150ms each in parallel)
- **Cost**: Low (fast model for decomposition, cross-encoder for reranking)
- **Optimization**:
  - Use fast model (gemma3:1b) for decomposition
  - Parallel retrieval for all sub-queries
  - Cross-encoder (not LLM) for reranking
  - Complexity threshold tuning
  - Query result caching

### Recommendations
1. **Use Case 1 (Simple Questions)**: Use `CrossEncoderRerankedStrategy`
   - Fast: ~150-200ms total
   - Accurate: Cross-encoder semantic understanding
   - Cost-effective: Local compute

2. **Use Case 2 (Context-Heavy)**: Use `MultiTurnRAGStrategy`
   - Auto-detects complexity
   - Falls back to simple for easy queries
   - Uses fast model for decomposition
   - Uses cross-encoder for reranking each sub-query

3. **Default Strategy**: Set `MultiTurnRAGStrategy` as default
   - Handles both use cases automatically
   - Falls back to simple retrieval when appropriate
   - Only adds overhead for complex queries

4. **Model Selection (Ollama + GGUF Only)**:
   - **Decomposition**: Fast GGUF model (`gemma3:1b` via Ollama, ~500ms)
   - **Reranking**: Cross-encoder GGUF (`bge-reranker-v2-m3` via Ollama, ~150ms/batch)
   - **All models**: Served via Ollama provider for consistency
   - **All configuration**: Centralized in `runtime.models` for easy tracking
   - **No external dependencies**: No HuggingFace, universal provider, or API calls needed

---

## Adoption Notes

### Completely Opt-In

These strategies are **100% additive** - existing projects continue working exactly as before:

✅ **No breaking changes** - All existing strategies remain unchanged
✅ **No required updates** - Projects can keep using BasicSimilarityStrategy, MetadataFilteredStrategy, etc.
✅ **Opt-in adoption** - Add new strategies when you're ready
✅ **Mix and match** - Use new strategies alongside existing ones

### Adding to Existing Projects

Simply add the new strategies to your existing `retrieval_strategies` list:

```yaml
# Existing project configuration stays the same
retrieval_strategies:
  - name: default_search
    type: BasicSimilarityStrategy
    default: true  # Keep this as is

  # ADD NEW: Optional cross-encoder reranking
  - name: reranked_search
    type: CrossEncoderRerankedStrategy
    config:
      model_name: reranker
      initial_k: 30
      final_k: 10

  # ADD NEW: Optional multi-turn for complex queries
  - name: multi_turn_search
    type: MultiTurnRAGStrategy
    config:
      model_name: fast
      cross_encoder_model_name: reranker
```

**That's it!** Use the new strategies explicitly when you want them:
```bash
# Keep using existing strategy
lf rag query --database main_db "simple query"

# Try new reranking strategy when ready
lf rag query --database main_db --retrieval-strategy reranked_search "query"
```

---

## Success Metrics

### Quantitative
- **Accuracy**: Measure precision@k and recall@k on benchmark queries
- **Latency**: P50, P95, P99 latency for different query types
- **Relevance**: MRR (Mean Reciprocal Rank) improvement over baseline

### Qualitative
- **User Feedback**: Survey on result quality
- **Query Decomposition Quality**: Manual review of sub-query generation
- **Reranking Decisions**: Spot-check LLM relevance assessments

### Targets
- 15-25% improvement in precision@5 for complex queries
- <5s P95 latency for multi-turn RAG
- 90%+ accuracy in complexity detection

---

## Open Questions & Future Work

### Questions to Resolve
1. **Model selection**: Should decomposition and reranking use same model or different models?
2. **Caching strategy**: How to cache query decomposition results?
3. **Adaptive thresholds**: Should complexity threshold adapt based on user feedback?

### Future Enhancements
1. **Query expansion integration**: Combine with existing query expansion
2. **Cross-encoder reranking**: Support for dedicated reranker models (e.g., bge-reranker)
3. **Streaming results**: Stream sub-query results as they arrive
4. **Confidence scores**: Provide confidence metrics for decomposition decisions
5. **A/B testing framework**: Built-in support for strategy comparison

---

## Conclusion

This implementation plan provides:

1. **Two complementary strategies** that address the two core use cases:
   - **Use Case 1**: Simple questions → CrossEncoderRerankedStrategy
   - **Use Case 2**: Context-heavy queries → MultiTurnRAGStrategy with query decomposition

2. **Centralized model configuration**: All models (chat, decomposition, reranking) live in `runtime.models`

3. **Cross-encoder reranking**: 10-100x faster than LLM reranking with similar accuracy

4. **Smart query handling**: Automatic complexity detection and routing

5. **Backward compatibility** with existing configurations

6. **Comprehensive testing strategy** covering both use cases

7. **Clear performance characteristics**:
   - Simple queries: ~150-200ms total
   - Complex queries: ~1-2s (still fast for multi-document synthesis)

The strategies leverage LlamaFarm's existing multi-model architecture, making it easy to reference models by name and keep all configuration in one place for easy tracking.

**Key Design Decisions**:
- ✅ All models configured in `runtime.models` (including rerankers)
- ✅ **Ollama + GGUF only** - All models use Ollama provider with GGUF format
- ✅ Cross-encoder reranking (fast + accurate via GGUF)
- ✅ Fast model for query decomposition (gemma3:1b GGUF)
- ✅ Automatic fallback for simple queries
- ✅ Parallel retrieval for sub-queries
- ✅ Cross-encoder reranking for each sub-query (not expensive LLM)
- ✅ **100% opt-in** - no mandatory changes to existing projects

**Next Steps**:
1. Review and refine this plan
2. Set up development branch
3. Begin Phase 1: CrossEncoderRerankedStrategy
4. Test Use Case 1 thoroughly
5. Begin Phase 2: MultiTurnRAGStrategy
6. Test Use Case 2 with real context-heavy queries
7. Performance benchmarking and optimization
