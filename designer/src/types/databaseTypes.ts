/**
 * AUTO-GENERATED FILE - DO NOT EDIT
 *
 * Generated from rag/schema.yaml by designer/generate-types.ts
 * Run: cd designer && ./generate-types.sh
 */

// ============================================================================
// Vector Store Types
// ============================================================================

export const VECTOR_STORE_TYPES = ["ChromaStore","FAISSStore","PineconeStore","QdrantStore"] as const

export type VectorStoreType = typeof VECTOR_STORE_TYPES[number]

// ============================================================================
// Embedder Types
// ============================================================================

export const EMBEDDER_TYPES = ["OllamaEmbedder","HuggingFaceEmbedder","OpenAIEmbedder","SentenceTransformerEmbedder"] as const

export type EmbedderType = typeof EMBEDDER_TYPES[number]

// ============================================================================
// Retrieval Strategy Types
// ============================================================================

export const RETRIEVAL_STRATEGY_TYPES = ["VectorRetriever","HybridRetriever","BM25Retriever","RerankedRetriever","GraphRetriever","ElasticRetriever","BasicSimilarityStrategy","MetadataFilteredStrategy","MultiQueryStrategy","RerankedStrategy","HybridUniversalStrategy"] as const

export type RetrievalStrategyType = typeof RETRIEVAL_STRATEGY_TYPES[number]

// ============================================================================
// Default Configurations
// ============================================================================

const VECTOR_STORE_DEFAULTS = {
  "ChromaStore": {
    "collection_name": "documents",
    "host": null,
    "port": 8000,
    "distance_function": "cosine",
    "distance_metric": "cosine",
    "embedding_dimension": 768,
    "enable_deduplication": true,
    "embedding_function": null
  },
  "PineconeStore": {
    "environment": "us-east-1-aws",
    "metric": "cosine",
    "namespace": "",
    "replicas": 1
  },
  "QdrantStore": {
    "host": "localhost",
    "port": 6333,
    "grpc_port": 6334,
    "api_key": null,
    "collection_name": "documents",
    "distance": "Cosine",
    "on_disk": false
  }
} as const

export function getDefaultVectorStoreConfig(storeType: VectorStoreType): Record<string, any> {
  return (VECTOR_STORE_DEFAULTS as any)[storeType] || {}
}

const EMBEDDER_DEFAULTS = {
  "OllamaEmbedder": {
    "model": "nomic-embed-text",
    "base_url": "http://localhost:11434",
    "dimension": 768,
    "batch_size": 16,
    "timeout": 60,
    "auto_pull": true
  },
  "HuggingFaceEmbedder": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "auto",
    "batch_size": 32,
    "normalize_embeddings": true,
    "show_progress_bar": false,
    "cache_folder": null
  },
  "SentenceTransformerEmbedder": {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "device": "cpu"
  }
} as const

export function getDefaultEmbedderConfig(embedderType: EmbedderType): Record<string, any> {
  return (EMBEDDER_DEFAULTS as any)[embedderType] || {}
}

const RETRIEVAL_STRATEGY_DEFAULTS = {
  "BasicSimilarityStrategy": {
    "top_k": 10,
    "distance_metric": "cosine",
    "score_threshold": null
  },
  "MetadataFilteredStrategy": {
    "top_k": 10,
    "filters": {},
    "filter_mode": "pre",
    "fallback_multiplier": 3
  },
  "MultiQueryStrategy": {
    "num_queries": 3,
    "top_k": 10,
    "aggregation_method": "weighted",
    "query_weights": null
  },
  "RerankedStrategy": {
    "initial_k": 30,
    "final_k": 10,
    "normalize_scores": true
  },
  "HybridUniversalStrategy": {
    "combination_method": "weighted_average",
    "final_k": 10
  }
} as const

export function getDefaultRetrievalStrategyConfig(
  strategyType: RetrievalStrategyType
): Record<string, any> {
  return (RETRIEVAL_STRATEGY_DEFAULTS as any)[strategyType] || {}
}

// ============================================================================
// Schema Metadata
// ============================================================================

export interface VectorStoreSchema {
  type: VectorStoreType
  title: string
  description: string
  category: 'local' | 'cloud' | 'memory'
  properties: Record<string, any>
  required?: string[]
}

export interface EmbedderSchema {
  type: EmbedderType
  title: string
  description: string
  category: 'local' | 'cloud' | 'huggingface'
  properties: Record<string, any>
  required?: string[]
}

export interface RetrievalStrategySchema {
  type: RetrievalStrategyType
  title: string
  description: string
  complexity: 'basic' | 'intermediate' | 'advanced'
  properties: Record<string, any>
  required?: string[]
}

export const VECTOR_STORE_SCHEMAS: Partial<Record<VectorStoreType, VectorStoreSchema>> = {
  "ChromaStore": {
    "type": "ChromaStore",
    "title": "Chroma Store Configuration",
    "description": "",
    "properties": {
      "collection_name": {
        "type": "string",
        "default": "documents",
        "pattern": "^[a-zA-Z0-9_-]+$",
        "description": "Collection name"
      },
      "host": {
        "type": [
          "string",
          "null"
        ],
        "default": null,
        "description": "Server host"
      },
      "port": {
        "type": "integer",
        "default": 8000,
        "minimum": 1,
        "maximum": 65535,
        "description": "Server port"
      },
      "distance_function": {
        "type": "string",
        "enum": [
          "cosine",
          "l2",
          "ip"
        ],
        "default": "cosine",
        "description": "Distance metric"
      },
      "distance_metric": {
        "type": "string",
        "enum": [
          "cosine",
          "l2",
          "ip"
        ],
        "default": "cosine",
        "description": "Alternative distance metric name"
      },
      "embedding_dimension": {
        "type": "integer",
        "default": 768,
        "minimum": 1,
        "maximum": 4096,
        "description": "Embedding dimension"
      },
      "enable_deduplication": {
        "type": "boolean",
        "default": true,
        "description": "Enable document deduplication"
      },
      "embedding_function": {
        "type": [
          "string",
          "null"
        ],
        "default": null,
        "description": "Built-in embedding function"
      }
    },
    "required": [],
    "category": "local"
  },
  "PineconeStore": {
    "type": "PineconeStore",
    "title": "Pinecone Store Configuration",
    "description": "",
    "properties": {
      "api_key": {
        "type": "string",
        "description": "Pinecone API key"
      },
      "environment": {
        "type": "string",
        "default": "us-east-1-aws",
        "description": "Pinecone environment"
      },
      "index_name": {
        "type": "string",
        "pattern": "^[a-z0-9-]+$",
        "description": "Index name"
      },
      "dimension": {
        "type": "integer",
        "minimum": 1,
        "maximum": 20000,
        "description": "Vector dimension"
      },
      "metric": {
        "type": "string",
        "enum": [
          "euclidean",
          "cosine",
          "dotproduct"
        ],
        "default": "cosine",
        "description": "Distance metric"
      },
      "namespace": {
        "type": "string",
        "default": "",
        "description": "Namespace for isolation"
      },
      "replicas": {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "maximum": 20,
        "description": "Number of replicas"
      }
    },
    "required": [
      "api_key",
      "index_name",
      "dimension"
    ],
    "category": "cloud"
  },
  "QdrantStore": {
    "type": "QdrantStore",
    "title": "Qdrant Store Configuration",
    "description": "",
    "properties": {
      "host": {
        "type": "string",
        "default": "localhost",
        "description": "Server host"
      },
      "port": {
        "type": "integer",
        "default": 6333,
        "minimum": 1,
        "maximum": 65535,
        "description": "Server port"
      },
      "grpc_port": {
        "type": "integer",
        "default": 6334,
        "minimum": 1,
        "maximum": 65535,
        "description": "gRPC port"
      },
      "api_key": {
        "type": [
          "string",
          "null"
        ],
        "default": null,
        "description": "API key"
      },
      "collection_name": {
        "type": "string",
        "default": "documents",
        "pattern": "^[a-zA-Z0-9_-]+$",
        "description": "Collection name"
      },
      "vector_size": {
        "type": "integer",
        "minimum": 1,
        "maximum": 65536,
        "description": "Vector dimension"
      },
      "distance": {
        "type": "string",
        "enum": [
          "Cosine",
          "Euclid",
          "Dot"
        ],
        "default": "Cosine",
        "description": "Distance metric"
      },
      "on_disk": {
        "type": "boolean",
        "default": false,
        "description": "Store vectors on disk"
      }
    },
    "required": [
      "vector_size"
    ],
    "category": "local"
  }
}

export const EMBEDDER_SCHEMAS: Partial<Record<EmbedderType, EmbedderSchema>> = {
  "OllamaEmbedder": {
    "type": "OllamaEmbedder",
    "title": "Ollama Embedder Configuration",
    "description": "",
    "properties": {
      "model": {
        "type": "string",
        "default": "nomic-embed-text",
        "description": "Ollama model name"
      },
      "base_url": {
        "type": "string",
        "format": "uri",
        "default": "http://localhost:11434",
        "description": "Ollama API endpoint (preferred)"
      },
      "dimension": {
        "type": "integer",
        "default": 768,
        "minimum": 128,
        "maximum": 4096,
        "description": "Embedding dimension"
      },
      "batch_size": {
        "type": "integer",
        "default": 16,
        "minimum": 1,
        "maximum": 128,
        "description": "Batch processing size"
      },
      "timeout": {
        "type": "integer",
        "default": 60,
        "minimum": 10,
        "description": "Request timeout (seconds)"
      },
      "auto_pull": {
        "type": "boolean",
        "default": true,
        "description": "Auto-pull missing models"
      }
    },
    "required": [],
    "category": "local"
  },
  "HuggingFaceEmbedder": {
    "type": "HuggingFaceEmbedder",
    "title": "HuggingFace Embedder Configuration",
    "description": "",
    "properties": {
      "model_name": {
        "type": "string",
        "default": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "HuggingFace model ID"
      },
      "device": {
        "type": "string",
        "enum": [
          "cpu",
          "cuda",
          "mps",
          "auto"
        ],
        "default": "auto",
        "description": "Computation device"
      },
      "batch_size": {
        "type": "integer",
        "default": 32,
        "minimum": 1,
        "maximum": 256,
        "description": "Batch size"
      },
      "normalize_embeddings": {
        "type": "boolean",
        "default": true,
        "description": "L2 normalize embeddings"
      },
      "show_progress_bar": {
        "type": "boolean",
        "default": false,
        "description": "Show progress bar"
      },
      "cache_folder": {
        "type": [
          "string",
          "null"
        ],
        "default": null,
        "description": "Model cache directory"
      }
    },
    "required": [],
    "category": "huggingface"
  },
  "SentenceTransformerEmbedder": {
    "type": "SentenceTransformerEmbedder",
    "title": "Sentence Transformer Configuration",
    "description": "",
    "properties": {
      "model_name": {
        "type": "string",
        "default": "sentence-transformers/all-MiniLM-L6-v2",
        "description": "Model name"
      },
      "device": {
        "type": "string",
        "default": "cpu",
        "enum": [
          "cpu",
          "cuda",
          "mps"
        ],
        "description": "Computation device"
      }
    },
    "required": [],
    "category": "huggingface"
  }
}

export const RETRIEVAL_STRATEGY_SCHEMAS: Partial<Record<
  RetrievalStrategyType,
  RetrievalStrategySchema
>> = {
  "BasicSimilarityStrategy": {
    "type": "BasicSimilarityStrategy",
    "title": "Basic Similarity Configuration",
    "description": "",
    "properties": {
      "top_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 1000,
        "description": "Number of results"
      },
      "distance_metric": {
        "type": "string",
        "default": "cosine",
        "enum": [
          "cosine",
          "euclidean",
          "manhattan",
          "dot"
        ],
        "description": "Distance metric"
      },
      "score_threshold": {
        "type": [
          "number",
          "null"
        ],
        "default": null,
        "minimum": 0,
        "maximum": 1,
        "description": "Minimum similarity score"
      }
    },
    "required": [],
    "complexity": "basic"
  },
  "MetadataFilteredStrategy": {
    "type": "MetadataFilteredStrategy",
    "title": "Metadata Filtered Configuration",
    "description": "",
    "properties": {
      "top_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 1000,
        "description": "Number of results"
      },
      "filters": {
        "type": "object",
        "default": {},
        "description": "Metadata filters",
        "additionalProperties": {
          "oneOf": [
            {
              "type": "string"
            },
            {
              "type": "number"
            },
            {
              "type": "boolean"
            },
            {
              "type": "array",
              "items": {
                "oneOf": [
                  {
                    "type": "string"
                  },
                  {
                    "type": "number"
                  },
                  {
                    "type": "boolean"
                  }
                ]
              }
            }
          ]
        }
      },
      "filter_mode": {
        "type": "string",
        "enum": [
          "pre",
          "post"
        ],
        "default": "pre",
        "description": "When to apply filters"
      },
      "fallback_multiplier": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 10,
        "description": "Multiplier for post-filtering"
      }
    },
    "required": [],
    "complexity": "intermediate"
  },
  "MultiQueryStrategy": {
    "type": "MultiQueryStrategy",
    "title": "Multi Query Configuration",
    "description": "",
    "properties": {
      "num_queries": {
        "type": "integer",
        "default": 3,
        "minimum": 1,
        "maximum": 10,
        "description": "Number of query variations"
      },
      "top_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 1000,
        "description": "Results per query"
      },
      "aggregation_method": {
        "type": "string",
        "default": "weighted",
        "enum": [
          "max",
          "mean",
          "weighted",
          "reciprocal_rank"
        ],
        "description": "Result aggregation method"
      },
      "query_weights": {
        "type": [
          "array",
          "null"
        ],
        "items": {
          "type": "number",
          "minimum": 0,
          "maximum": 1
        },
        "default": null,
        "description": "Weights for each query"
      }
    },
    "required": [],
    "complexity": "advanced"
  },
  "RerankedStrategy": {
    "type": "RerankedStrategy",
    "title": "Reranked Configuration",
    "description": "",
    "properties": {
      "initial_k": {
        "type": "integer",
        "default": 30,
        "minimum": 10,
        "maximum": 1000,
        "description": "Initial candidates"
      },
      "final_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 100,
        "description": "Final results"
      },
      "rerank_factors": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "similarity_weight": {
            "type": "number",
            "default": 0.7,
            "minimum": 0,
            "maximum": 1
          },
          "recency_weight": {
            "type": "number",
            "default": 0.1,
            "minimum": 0,
            "maximum": 1
          },
          "length_weight": {
            "type": "number",
            "default": 0.1,
            "minimum": 0,
            "maximum": 1
          },
          "metadata_weight": {
            "type": "number",
            "default": 0.1,
            "minimum": 0,
            "maximum": 1
          }
        },
        "description": "Reranking factor weights"
      },
      "normalize_scores": {
        "type": "boolean",
        "default": true,
        "description": "Normalize scores before combining"
      }
    },
    "required": [],
    "complexity": "intermediate"
  },
  "HybridUniversalStrategy": {
    "type": "HybridUniversalStrategy",
    "title": "Hybrid Universal Configuration",
    "description": "",
    "properties": {
      "strategies": {
        "type": "array",
        "minItems": 2,
        "maxItems": 5,
        "items": {
          "type": "object",
          "required": [
            "type"
          ],
          "additionalProperties": false,
          "properties": {
            "type": {
              "type": "string",
              "enum": [
                "BasicSimilarityStrategy",
                "MetadataFilteredStrategy",
                "MultiQueryStrategy",
                "RerankedStrategy"
              ]
            },
            "weight": {
              "type": "number",
              "default": 1,
              "minimum": 0,
              "maximum": 1
            },
            "config": {
              "type": "object"
            }
          }
        },
        "description": "Sub-strategies to combine"
      },
      "combination_method": {
        "type": "string",
        "default": "weighted_average",
        "enum": [
          "weighted_average",
          "rank_fusion",
          "score_fusion"
        ],
        "description": "Combination method"
      },
      "final_k": {
        "type": "integer",
        "default": 10,
        "minimum": 1,
        "maximum": 1000,
        "description": "Final number of results"
      }
    },
    "required": [],
    "complexity": "advanced"
  }
}

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Get all vector stores by category
 */
export function getVectorStoresByCategory(
  category: 'local' | 'cloud' | 'memory'
): VectorStoreType[] {
  return VECTOR_STORE_TYPES.filter((type) => VECTOR_STORE_SCHEMAS[type]?.category === category)
}

/**
 * Get all embedders by category
 */
export function getEmbeddersByCategory(
  category: 'local' | 'cloud' | 'huggingface'
): EmbedderType[] {
  return EMBEDDER_TYPES.filter((type) => EMBEDDER_SCHEMAS[type]?.category === category)
}

/**
 * Get all retrieval strategies by complexity
 */
export function getRetrievalStrategiesByComplexity(
  complexity: 'basic' | 'intermediate' | 'advanced'
): RetrievalStrategyType[] {
  return RETRIEVAL_STRATEGY_TYPES.filter(
    (type) => RETRIEVAL_STRATEGY_SCHEMAS[type]?.complexity === complexity
  )
}
