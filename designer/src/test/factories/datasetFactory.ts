import {
  ListDatasetsResponse,
  Dataset,
} from '../../types/datasets'

/**
 * Factory function to create mock Dataset objects
 * Use this to generate test data for dataset-related tests
 * 
 * @example
 * ```tsx
 * const dataset = createMockDataset({ name: 'test-dataset' })
 * const datasets = createMockDatasetsList(3) // Creates 3 datasets
 * ```
 */

interface MockDatasetOptions {
  name?: string
  data_processing_strategy?: string
  database?: string
  files?: string[]
}

/**
 * Create a single mock dataset
 */
export function createMockDataset(
  options: MockDatasetOptions = {}
): Dataset {
  const {
    name = 'test-dataset',
    data_processing_strategy = 'pdf_ingest',
    database = 'main_db',
    files = [],
  } = options

  return {
    name,
    data_processing_strategy,
    database,
    files,
  }
}

/**
 * Create a list of mock datasets
 */
export function createMockDatasetsList(
  count: number = 2
): ListDatasetsResponse {
  const datasets: Dataset[] = []

  for (let i = 0; i < count; i++) {
    datasets.push({
      name: `dataset-${i + 1}`,
      data_processing_strategy: 'pdf_ingest',
      database: 'main_db',
      files: [],
    })
  }

  return {
    total: datasets.length,
    datasets,
  }
}

/**
 * Create a mock dataset with files
 */
export function createMockDatasetWithFiles(
  name: string,
  fileCount: number = 3
): Dataset {
  const files = Array.from({ length: fileCount }, (_, i) => `hash-${i + 1}-${Date.now()}`)

  return createMockDataset({
    name,
    files,
  })
}

/**
 * Create a mock dataset in processing state
 */
export function createMockProcessingDataset(name: string): Dataset {
  return createMockDataset({
    name,
    files: ['hash-1', 'hash-2', 'hash-3'],
  })
}

/**
 * Create a mock task status response
 */
export function createMockTaskStatus(
  taskId: string,
  status: 'PENDING' | 'PROCESSING' | 'SUCCESS' | 'FAILURE' = 'SUCCESS'
) {
  return {
    task_id: taskId,
    status,
    result:
      status === 'SUCCESS'
        ? { processed: 10, failed: 0 }
        : status === 'FAILURE'
        ? { error: 'Processing failed' }
        : null,
  }
}

