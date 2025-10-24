/**
 * AUTO-GENERATED FILE - DO NOT EDIT
 * 
 * Generated from rag/schema.yaml by designer/generate-types.ts
 * Run: cd designer && ./generate-types.sh
 */

// ============================================================================
// Vector Store / Database Types
// ============================================================================

export const VECTOR_STORE_TYPES = [
  "ChromaStore",
  "FAISSStore",
  "PineconeStore",
  "QdrantStore",
] as const

export type VectorStoreType = typeof VECTOR_STORE_TYPES[number]

// ============================================================================
// Embedder Types
// ============================================================================

export const EMBEDDER_TYPES = [
  "HuggingFaceEmbedder",
  "OllamaEmbedder",
  "OpenAIEmbedder",
  "SentenceTransformerEmbedder",
] as const

export type EmbedderType = typeof EMBEDDER_TYPES[number]

// ============================================================================
// Retrieval Strategy Types
// ============================================================================

export const RETRIEVAL_STRATEGY_TYPES = [
  "BM25Retriever",
  "BasicSimilarityStrategy",
  "ElasticRetriever",
  "GraphRetriever",
  "HybridRetriever",
  "HybridUniversalStrategy",
  "MetadataFilteredStrategy",
  "MultiQueryStrategy",
  "RerankedRetriever",
  "RerankedStrategy",
  "VectorRetriever",
] as const

export type RetrievalStrategyType = typeof RETRIEVAL_STRATEGY_TYPES[number]

// ============================================================================
// Default Configurations - Vector Stores
// ============================================================================

export function getDefaultVectorStoreConfig(storeType: VectorStoreType): Record<string, any> {
  const configs: Record<VectorStoreType, Record<string, any>> = {
    "ChromaStore":       {
        "collection_name": "documents",
        "host": null,
        "port": 8000,
        "distance_function": "cosine",
        "distance_metric": "cosine",
        "embedding_dimension": 768,
        "enable_deduplication": true,
        "embedding_function": null
      },
    "FAISSStore":       {},
    "PineconeStore":       {
        "environment": "us-east-1-aws",
        "metric": "cosine",
        "namespace": "",
        "replicas": 1
      },
    "QdrantStore":       {
        "host": "localhost",
        "port": 6333,
        "grpc_port": 6334,
        "api_key": null,
        "collection_name": "documents",
        "distance": "Cosine",
        "on_disk": false
      },
  }
  return configs[storeType] || {}
}

// ============================================================================
// Default Configurations - Embedders
// ============================================================================

export function getDefaultEmbedderConfig(embedderType: EmbedderType): Record<string, any> {
  const configs: Record<EmbedderType, Record<string, any>> = {
    "HuggingFaceEmbedder":       {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "auto",
        "batch_size": 32,
        "normalize_embeddings": true,
        "show_progress_bar": false,
        "cache_folder": null
      },
    "OllamaEmbedder":       {
        "model": "nomic-embed-text",
        "base_url": "http://localhost:11434",
        "dimension": 768,
        "batch_size": 16,
        "timeout": 60,
        "auto_pull": true
      },
    "OpenAIEmbedder":       {},
    "SentenceTransformerEmbedder":       {
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "device": "cpu"
      },
  }
  return configs[embedderType] || {}
}

// ============================================================================
// Default Configurations - Retrieval Strategies
// ============================================================================

export function getDefaultRetrievalStrategyConfig(strategyType: RetrievalStrategyType): Record<string, any> {
  const configs: Record<RetrievalStrategyType, Record<string, any>> = {
    "BM25Retriever":       {},
    "BasicSimilarityStrategy":       {
        "top_k": 10,
        "distance_metric": "cosine",
        "score_threshold": null
      },
    "ElasticRetriever":       {},
    "GraphRetriever":       {},
    "HybridRetriever":       {},
    "HybridUniversalStrategy":       {
        "combination_method": "weighted_average",
        "final_k": 10
      },
    "MetadataFilteredStrategy":       {
        "top_k": 10,
        "filters": {},
        "filter_mode": "pre",
        "fallback_multiplier": 3
      },
    "MultiQueryStrategy":       {
        "num_queries": 3,
        "top_k": 10,
        "aggregation_method": "weighted",
        "query_weights": null
      },
    "RerankedRetriever":       {},
    "RerankedStrategy":       {
        "initial_k": 30,
        "final_k": 10,
        "normalize_scores": true
      },
    "VectorRetriever":       {},
  }
  return configs[strategyType] || {}
}

// ============================================================================
// Schema Metadata
// ============================================================================

export interface VectorStoreSchema {
  type: VectorStoreType
  title: string
  description: string
  category: 'local' | 'cloud' | 'memory'
}

export interface EmbedderSchema {
  type: EmbedderType
  title: string
  description: string
  category: 'local' | 'cloud' | 'huggingface'
}

export interface RetrievalStrategySchema {
  type: RetrievalStrategyType
  title: string
  description: string
  complexity: 'basic' | 'intermediate' | 'advanced'
}

export const VECTOR_STORE_SCHEMAS: Record<VectorStoreType, VectorStoreSchema> = {
  "ChromaStore": {
    type: "ChromaStore",
    title: "Chroma Store Configuration",
    description: "",
    category: "local",
  },
  "FAISSStore": {
    type: "FAISSStore",
    title: "FAISSStore",
    description: "",
    category: "memory",
  },
  "PineconeStore": {
    type: "PineconeStore",
    title: "Pinecone Store Configuration",
    description: "",
    category: "cloud",
  },
  "QdrantStore": {
    type: "QdrantStore",
    title: "Qdrant Store Configuration",
    description: "",
    category: "local",
  },
}

export const EMBEDDER_SCHEMAS: Record<EmbedderType, EmbedderSchema> = {
  "HuggingFaceEmbedder": {
    type: "HuggingFaceEmbedder",
    title: "HuggingFace Embedder Configuration",
    description: "",
    category: "huggingface",
  },
  "OllamaEmbedder": {
    type: "OllamaEmbedder",
    title: "Ollama Embedder Configuration",
    description: "",
    category: "local",
  },
  "OpenAIEmbedder": {
    type: "OpenAIEmbedder",
    title: "OpenAIEmbedder",
    description: "",
    category: "cloud",
  },
  "SentenceTransformerEmbedder": {
    type: "SentenceTransformerEmbedder",
    title: "Sentence Transformer Configuration",
    description: "",
    category: "huggingface",
  },
}

export const RETRIEVAL_STRATEGY_SCHEMAS: Record<RetrievalStrategyType, RetrievalStrategySchema> = {
  "BM25Retriever": {
    type: "BM25Retriever",
    title: "BM25Retriever",
    description: "",
    complexity: "basic",
  },
  "BasicSimilarityStrategy": {
    type: "BasicSimilarityStrategy",
    title: "Basic Similarity Configuration",
    description: "",
    complexity: "basic",
  },
  "ElasticRetriever": {
    type: "ElasticRetriever",
    title: "ElasticRetriever",
    description: "",
    complexity: "basic",
  },
  "GraphRetriever": {
    type: "GraphRetriever",
    title: "GraphRetriever",
    description: "",
    complexity: "basic",
  },
  "HybridRetriever": {
    type: "HybridRetriever",
    title: "HybridRetriever",
    description: "",
    complexity: "advanced",
  },
  "HybridUniversalStrategy": {
    type: "HybridUniversalStrategy",
    title: "Hybrid Universal Configuration",
    description: "",
    complexity: "advanced",
  },
  "MetadataFilteredStrategy": {
    type: "MetadataFilteredStrategy",
    title: "Metadata Filtered Configuration",
    description: "",
    complexity: "intermediate",
  },
  "MultiQueryStrategy": {
    type: "MultiQueryStrategy",
    title: "Multi Query Configuration",
    description: "",
    complexity: "advanced",
  },
  "RerankedRetriever": {
    type: "RerankedRetriever",
    title: "RerankedRetriever",
    description: "",
    complexity: "intermediate",
  },
  "RerankedStrategy": {
    type: "RerankedStrategy",
    title: "Reranked Configuration",
    description: "",
    complexity: "intermediate",
  },
  "VectorRetriever": {
    type: "VectorRetriever",
    title: "VectorRetriever",
    description: "",
    complexity: "basic",
  },
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get all vector stores by category
 */
export function getVectorStoresByCategory(category: 'local' | 'cloud' | 'memory'): VectorStoreType[] {
  return VECTOR_STORE_TYPES.filter(type => VECTOR_STORE_SCHEMAS[type].category === category)
}

/**
 * Get all embedders by category
 */
export function getEmbeddersByCategory(category: 'local' | 'cloud' | 'huggingface'): EmbedderType[] {
  return EMBEDDER_TYPES.filter(type => EMBEDDER_SCHEMAS[type].category === category)
}

/**
 * Get all retrieval strategies by complexity
 */
export function getRetrievalStrategiesByComplexity(complexity: 'basic' | 'intermediate' | 'advanced'): RetrievalStrategyType[] {
  return RETRIEVAL_STRATEGY_TYPES.filter(type => RETRIEVAL_STRATEGY_SCHEMAS[type].complexity === complexity)
}
