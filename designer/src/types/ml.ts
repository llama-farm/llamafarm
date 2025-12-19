/**
 * Types for ML API endpoints (classifier and anomaly detection)
 * Mirrors backend schemas from server/api/routers/ml/types.py
 */

// =============================================================================
// Classifier Types
// =============================================================================

export interface ClassifierTrainingData {
  text: string
  label: string
}

export interface ClassifierFitRequest {
  model: string
  base_model?: string // default: "sentence-transformers/all-MiniLM-L6-v2"
  training_data: ClassifierTrainingData[]
  num_iterations?: number // default: 20
  batch_size?: number // default: 16
  overwrite?: boolean // default: false - creates versioned model
  description?: string // optional model description
}

export interface ClassifierFitResponse {
  model: string
  status: string
  labels: string[]
  training_samples: number
  base_name: string
  versioned_name: string
  overwrite: boolean
}

export interface ClassifierPredictRequest {
  model: string
  texts: string[]
}

export interface ClassifierPrediction {
  text: string
  label: string
  score: number // API uses 'score', not 'confidence'
  all_scores?: Record<string, number>
}

export interface ClassifierPredictResponse {
  object: string
  model: string
  data: ClassifierPrediction[] // API uses 'data', not 'predictions'
  total_count: number
}

export interface ClassifierSaveRequest {
  model: string
}

export interface ClassifierSaveResponse {
  model: string
  path: string
  status: string
}

export interface ClassifierLoadRequest {
  model: string
}

export interface ClassifierLoadResponse {
  model: string
  status: string
  labels: string[]
}

export interface ClassifierModelInfo {
  name: string
  path: string
  labels?: string[]
  base_name?: string
  created?: string
  is_versioned?: boolean
  description?: string
}

export interface ClassifierListModelsResponse {
  models: ClassifierModelInfo[]
}

// =============================================================================
// Anomaly Detection Types
// =============================================================================

export type AnomalyBackend =
  | 'isolation_forest'
  | 'one_class_svm'
  | 'local_outlier_factor'
  | 'autoencoder'

// Map display names to API identifiers
export const ANOMALY_BACKEND_MAP: Record<string, AnomalyBackend> = {
  'auto': 'isolation_forest', // default
  'isolation-forest': 'isolation_forest',
  'one-class-svm': 'one_class_svm',
  'autoencoder': 'autoencoder',
  'local-outlier-factor': 'local_outlier_factor',
}

export const ANOMALY_BACKEND_DISPLAY: Record<AnomalyBackend, string> = {
  'isolation_forest': 'Isolation Forest',
  'one_class_svm': 'One-Class SVM',
  'local_outlier_factor': 'Local Outlier Factor',
  'autoencoder': 'Autoencoder',
}

// Schema encoding types for mixed data
export type FeatureEncodingType =
  | 'numeric' // pass through as-is (int/float)
  | 'hash' // MD5 hash to integer (high-cardinality strings)
  | 'label' // category → integer mapping (low-cardinality)
  | 'onehot' // one-hot encoding (< 20 categories)
  | 'binary' // boolean-like (yes/no, true/false → 0/1)
  | 'frequency' // encode as occurrence frequency

export const ENCODING_TYPE_OPTIONS: {
  value: FeatureEncodingType
  label: string
  description: string
}[] = [
  { value: 'numeric', label: 'Numeric', description: 'Numbers (int/float)' },
  { value: 'label', label: 'Text', description: 'Text values (auto-encoded)' },
]

// Full encoding options (for advanced use or future expansion)
export const ENCODING_TYPE_OPTIONS_FULL: {
  value: FeatureEncodingType
  label: string
  description: string
}[] = [
  { value: 'numeric', label: 'Numeric', description: 'Numbers (int/float)' },
  { value: 'hash', label: 'Hash', description: 'High-cardinality text (IDs, user agents)' },
  { value: 'label', label: 'Label', description: 'Categories (< 20 unique values)' },
  { value: 'onehot', label: 'One-Hot', description: 'Low-cardinality categories' },
  { value: 'binary', label: 'Binary', description: 'Yes/No, True/False' },
  { value: 'frequency', label: 'Frequency', description: 'Encode by occurrence count' },
]

// Feature schema definition for table view
export interface FeatureColumn {
  name: string
  type: FeatureEncodingType
}

export interface AnomalyFitRequest {
  model: string
  backend?: AnomalyBackend // default: "isolation_forest"
  data: number[][] | Record<string, unknown>[] // numeric arrays OR dict-based with schema
  schema?: Record<string, FeatureEncodingType> // required for dict-based data
  contamination?: number // 0-0.5, default: 0.1
  epochs?: number // for autoencoder, default: 100
  batch_size?: number // for autoencoder, default: 32
  overwrite?: boolean // default: false - creates versioned model
  description?: string // optional model description
}

export interface AnomalyFitResponse {
  model: string
  status: string
  backend: AnomalyBackend
  training_samples: number
  base_name: string
  versioned_name: string
  overwrite: boolean
}

export interface AnomalyScoreRequest {
  model: string
  backend?: AnomalyBackend
  data: number[][] | Record<string, unknown>[]
  schema?: Record<string, FeatureEncodingType>
  threshold?: number
}

export interface AnomalyScoreResult {
  index: number
  data: number[]
  score: number
  is_anomaly: boolean
  raw_score?: number
}

export interface AnomalyScoreResponse {
  model: string
  backend: AnomalyBackend
  results: AnomalyScoreResult[]
  threshold: number
  anomaly_count: number
  total_count: number
}

export interface AnomalySaveRequest {
  model: string
  backend?: AnomalyBackend
}

export interface AnomalySaveResponse {
  model: string
  backend: AnomalyBackend
  path: string
  status: string
}

export interface AnomalyLoadRequest {
  model: string
  backend?: AnomalyBackend
}

export interface AnomalyLoadResponse {
  model: string
  backend: AnomalyBackend
  status: string
}

export interface AnomalyModelInfo {
  name: string
  filename: string
  base_name: string
  backend: AnomalyBackend
  path: string
  size_bytes: number
  created: string
  is_versioned: boolean
  description?: string
}

export interface AnomalyListModelsResponse {
  models: AnomalyModelInfo[]
}

// =============================================================================
// Shared Types
// =============================================================================

export interface MLHealthResponse {
  status: string
  universal_runtime?: string
}

export interface MLDeleteResponse {
  model: string
  status: string
  deleted: boolean
}

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * Parse a versioned model name to extract base name and timestamp
 * e.g., "fraud-detector_20251218_143022" -> { baseName: "fraud-detector", timestamp: "20251218_143022" }
 */
export function parseVersionedModelName(name: string): {
  baseName: string
  timestamp: string | null
  isVersioned: boolean
} {
  const match = name.match(/^(.+)_(\d{8}_\d{6})$/)
  if (match) {
    return {
      baseName: match[1],
      timestamp: match[2],
      isVersioned: true,
    }
  }
  return {
    baseName: name,
    timestamp: null,
    isVersioned: false,
  }
}

/**
 * Format a timestamp string (YYYYMMDD_HHMMSS) to a readable date
 */
export function formatModelTimestamp(timestamp: string): string {
  if (!timestamp || timestamp.length !== 15) return timestamp

  const year = timestamp.slice(0, 4)
  const month = timestamp.slice(4, 6)
  const day = timestamp.slice(6, 8)
  const hour = timestamp.slice(9, 11)
  const minute = timestamp.slice(11, 13)

  const date = new Date(`${year}-${month}-${day}T${hour}:${minute}:00`)
  return date.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

/**
 * Parse numeric training data from text input
 * Supports comma or newline separated values, with each row being a feature vector
 * Returns null if parsing fails
 */
export function parseNumericTrainingData(input: string): number[][] | null {
  try {
    const lines = input
      .split(/[\n]/)
      .map(line => line.trim())
      .filter(Boolean)

    const data: number[][] = []

    for (const line of lines) {
      const values = line
        .split(/[,\s]+/)
        .map(v => v.trim())
        .filter(Boolean)
        .map(v => parseFloat(v))

      // Check if all values are valid numbers
      if (values.some(isNaN)) {
        return null
      }

      if (values.length > 0) {
        data.push(values)
      }
    }

    return data.length > 0 ? data : null
  } catch {
    return null
  }
}

/**
 * Validate that all rows have the same number of features
 */
export function validateFeatureConsistency(data: number[][]): {
  valid: boolean
  featureCount: number | null
  error: string | null
} {
  if (data.length === 0) {
    return { valid: false, featureCount: null, error: 'No data provided' }
  }

  const featureCount = data[0].length
  const inconsistentRow = data.findIndex(row => row.length !== featureCount)

  if (inconsistentRow !== -1) {
    return {
      valid: false,
      featureCount: null,
      error: `Row ${inconsistentRow + 1} has ${data[inconsistentRow].length} features, expected ${featureCount}`,
    }
  }

  return { valid: true, featureCount, error: null }
}

/**
 * Generate a unique model name by appending a number suffix if the name already exists.
 * e.g., if "new-anomaly-model" exists, returns "new-anomaly-model-2"
 * if "new-anomaly-model-2" also exists, returns "new-anomaly-model-3", etc.
 */
export function generateUniqueModelName(
  baseName: string,
  existingBaseNames: Set<string>
): string {
  if (!existingBaseNames.has(baseName)) {
    return baseName
  }

  // Check if baseName already ends with a number suffix (e.g., "model-2")
  const suffixMatch = baseName.match(/^(.+)-(\d+)$/)
  let nameRoot: string
  let startNum: number

  if (suffixMatch) {
    // Name already has a suffix, increment from there
    nameRoot = suffixMatch[1]
    startNum = parseInt(suffixMatch[2], 10) + 1
  } else {
    // No suffix, start at 2
    nameRoot = baseName
    startNum = 2
  }

  // Find the next available number
  let num = startNum
  while (existingBaseNames.has(`${nameRoot}-${num}`)) {
    num++
  }

  return `${nameRoot}-${num}`
}
