/**
 * Hugging Face API service
 * Direct calls to HF APIs for search, backend call for import
 */

import apiClient from './client'
import type {
  HFDatasetSearchResult,
  HFDatasetImportRequest,
  HFDatasetImportResponse,
} from '../types/huggingface'

const HF_API_BASE = 'https://huggingface.co/api'
const HF_DATASETS_SERVER = 'https://datasets-server.huggingface.co'

/** Config info from HF datasets-server */
export interface HFDatasetConfigInfo {
  config: string
  splits: string[]
}

/** Allowed formats for text-compatible datasets */
const ALLOWED_FORMATS = ['csv', 'json', 'jsonl', 'text', 'parquet']

/** Task categories that indicate text-based datasets */
const TEXT_TASK_CATEGORIES = [
  'text-classification',
  'question-answering',
  'summarization',
  'text-generation',
  'text2text-generation',
  'translation',
  'fill-mask',
  'token-classification',
  'table-question-answering',
  'conversational',
  'sentence-similarity',
  'feature-extraction',
]

/** Task categories to exclude (image, audio, video) */
const EXCLUDED_TASK_CATEGORIES = [
  'image-classification',
  'image-segmentation',
  'object-detection',
  'image-to-text',
  'text-to-image',
  'image-to-image',
  'video-classification',
  'audio-classification',
  'automatic-speech-recognition',
  'text-to-speech',
  'text-to-audio',
  'voice-activity-detection',
  'depth-estimation',
  'image-feature-extraction',
]

/** Size categories that work well with HF datasets-server (avoid large Parquet files) */
const SAFE_SIZE_CATEGORIES = [
  'n<1K',
  '1K<n<10K',
  '10K<n<100K',
  '100K<n<1M',
]

/**
 * Search HF datasets
 * Filters to text-compatible, public datasets that are likely to work with the API
 */
export async function searchDatasets(
  query: string,
  limit = 10
): Promise<HFDatasetSearchResult[]> {
  const params = new URLSearchParams({
    search: query,
    limit: String(limit * 5), // Fetch extra to account for filtering
    full: 'true',
  })

  const response = await fetch(`${HF_API_BASE}/datasets?${params}`)
  if (!response.ok) {
    throw new Error('Failed to search datasets')
  }

  const results: HFDatasetSearchResult[] = await response.json()

  // Filter and sort for best compatibility
  const filtered = results
    .filter(ds => {
      // Skip gated/private datasets
      if (ds.tags?.includes('gated')) return false

      // Check for excluded task categories (images, audio, video)
      const hasExcludedTask = ds.tags?.some(tag => {
        const tagLower = tag.toLowerCase()
        return EXCLUDED_TASK_CATEGORIES.some(excluded =>
          tagLower === `task_categories:${excluded}` || tagLower === excluded
        )
      })
      if (hasExcludedTask) return false

      // Check for text-compatible task categories
      const hasTextTask = ds.tags?.some(tag => {
        const tagLower = tag.toLowerCase()
        return TEXT_TASK_CATEGORIES.some(textTask =>
          tagLower === `task_categories:${textTask}` || tagLower === textTask
        )
      })

      // Check for text-compatible format tags
      const hasTextFormat = ds.tags?.some(tag =>
        ALLOWED_FORMATS.some(fmt => tag.toLowerCase().includes(fmt))
      )

      // Include if has text task OR text format, exclude if neither
      return hasTextTask || hasTextFormat
    })
    // Sort by size category (smaller first) and downloads
    .sort((a, b) => {
      const sizeA = a.cardData?.size_categories?.[0]
      const sizeB = b.cardData?.size_categories?.[0]

      const safeIndexA = sizeA ? SAFE_SIZE_CATEGORIES.indexOf(sizeA) : -1
      const safeIndexB = sizeB ? SAFE_SIZE_CATEGORIES.indexOf(sizeB) : -1

      // Prefer datasets with known safe sizes
      if (safeIndexA !== -1 && safeIndexB === -1) return -1
      if (safeIndexA === -1 && safeIndexB !== -1) return 1

      // If both have safe sizes, prefer smaller
      if (safeIndexA !== -1 && safeIndexB !== -1) {
        if (safeIndexA !== safeIndexB) return safeIndexA - safeIndexB
      }

      // Fall back to download count (more downloads = more tested/reliable)
      return b.downloads - a.downloads
    })
    .slice(0, limit)

  return filtered
}

/**
 * Validate that a dataset is accessible via the HF datasets-server API
 * Tests by fetching just 1 row - this will fail if Parquet files are too large
 */
export async function validateDatasetAccess(
  datasetId: string,
  config: string,
  split: string
): Promise<{ valid: boolean; error?: string }> {
  try {
    const url = `${HF_DATASETS_SERVER}/rows?dataset=${encodeURIComponent(datasetId)}&config=${encodeURIComponent(config)}&split=${encodeURIComponent(split)}&offset=0&length=1`
    const response = await fetch(url)

    if (response.ok) {
      return { valid: true }
    }

    // Parse error response
    try {
      const errorData = await response.json()
      const errorMsg = errorData?.error || ''

      if (errorMsg.toLowerCase().includes('size limit exceeded')) {
        return {
          valid: false,
          error: 'This dataset has files too large for the API. Please try a smaller dataset.',
        }
      }

      if (response.status === 401) {
        return {
          valid: false,
          error: 'This dataset requires authentication.',
        }
      }

      return {
        valid: false,
        error: `Dataset not accessible: ${errorMsg.slice(0, 100)}`,
      }
    } catch {
      return {
        valid: false,
        error: `Dataset not accessible (HTTP ${response.status})`,
      }
    }
  } catch (err) {
    return {
      valid: false,
      error: 'Failed to validate dataset access',
    }
  }
}

/**
 * Get available configs and splits for a dataset
 * Uses the HF datasets-server API to get valid config/split combinations
 */
export async function getDatasetConfigs(
  datasetId: string
): Promise<HFDatasetConfigInfo[]> {
  const response = await fetch(
    `${HF_DATASETS_SERVER}/splits?dataset=${encodeURIComponent(datasetId)}`
  )

  if (!response.ok) {
    // If we can't get configs, return a default that might work
    console.warn(`Failed to get configs for ${datasetId}, using fallback`)
    return [{ config: 'default', splits: ['train'] }]
  }

  const data = await response.json()

  // The API returns { splits: [{ dataset, config, split }, ...] }
  const splits = data.splits || []

  // Group by config
  const configMap = new Map<string, string[]>()
  for (const s of splits) {
    const existing = configMap.get(s.config) || []
    existing.push(s.split)
    configMap.set(s.config, existing)
  }

  // Convert to array
  const configs: HFDatasetConfigInfo[] = []
  for (const [config, splitList] of configMap) {
    configs.push({ config, splits: splitList })
  }

  // If empty, return default fallback
  if (configs.length === 0) {
    return [{ config: 'default', splits: ['train'] }]
  }

  return configs
}

/**
 * Import HF dataset via backend
 */
export async function importDataset(
  request: HFDatasetImportRequest
): Promise<HFDatasetImportResponse> {
  console.log('[huggingface.ts] Sending import request:', request)
  const { data } = await apiClient.post<HFDatasetImportResponse>(
    '/huggingface/datasets/import',
    request
  )
  console.log('[huggingface.ts] Import response:', data)
  return data
}

export default {
  searchDatasets,
  getDatasetConfigs,
  importDataset,
}
