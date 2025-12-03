/**
 * Factory function to create mock Embedding Strategy and Database objects
 * Use this to generate test data for RAG-related tests
 * 
 * @example
 * ```tsx
 * const strategies = createMockEmbeddingStrategies()
 * const databases = createMockDatabases()
 * ```
 */

/**
 * Mock retrieval strategy shape
 */
export interface MockRetrievalStrategy {
  id: string
  type: string
  config: {
    top_k: number
    similarity_threshold: number
  }
  database: string
}

/**
 * Create a list of mock embedding strategies
 */
export function createMockEmbeddingStrategies(): string[] {
  return [
    'pdf_ingest',
    'markdown_ingest',
    'csv_ingest',
    'json_ingest',
    'text_ingest',
  ]
}

/**
 * Create a list of mock databases
 */
export function createMockDatabases(): string[] {
  return ['main_db', 'secondary_db', 'archive_db']
}

/**
 * Create a mock embedding strategy configuration
 */
export function createMockEmbeddingStrategyConfig(
  strategyName: string = 'pdf_ingest'
) {
  return {
    name: strategyName,
    type: 'data_processing',
    parsers: [
      {
        type: 'pdf',
        config: {
          chunk_size: 1000,
          chunk_overlap: 200,
        },
      },
    ],
    extractors: [
      {
        type: 'text',
        config: {
          min_length: 10,
        },
      },
    ],
  }
}

/**
 * Create a mock database configuration
 */
export function createMockDatabaseConfig(databaseName: string = 'main_db') {
  return {
    name: databaseName,
    type: 'chroma',
    embedding_model: 'all-MiniLM-L6-v2',
    collection_name: `${databaseName}_collection`,
    metadata: {
      created_at: new Date().toISOString(),
      description: `Mock database: ${databaseName}`,
    },
  }
}

/**
 * Create a mock retrieval strategy
 */
export function createMockRetrievalStrategy(
  strategyId: string = 'strategy-1',
  type: string = 'semantic_search'
): MockRetrievalStrategy {
  return {
    id: strategyId,
    type,
    config: {
      top_k: 5,
      similarity_threshold: 0.7,
    },
    database: 'main_db',
  }
}

/**
 * Create a list of mock retrieval strategies
 */
export function createMockRetrievalStrategiesList(count: number = 3): MockRetrievalStrategy[] {
  const strategies: MockRetrievalStrategy[] = []
  const types = ['semantic_search', 'hybrid_search', 'keyword_search']

  for (let i = 0; i < count; i++) {
    strategies.push(
      createMockRetrievalStrategy(`strategy-${i + 1}`, types[i % types.length])
    )
  }

  return strategies
}

