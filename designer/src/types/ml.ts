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
  description?: string
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
  object: string
  data: ClassifierModelInfo[]  // API returns 'data', not 'models'
  models_dir: string
  total: number
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

// =============================================================================
// Score Normalization Types
// =============================================================================

/**
 * Score normalization methods for anomaly detection.
 *
 * - "standardization" (default): Sigmoid transformation to 0-1 range using median/IQR.
 *   Scores approach 0.5 for normal data, approach 1.0 for anomalies.
 *   Default threshold: 0.5
 *
 * - "zscore": Z-score normalization using mean and standard deviation.
 *   Scores represent standard deviations from the training mean.
 *   2.0 = unusual, 3.0 = rare, 4.0+ = extreme anomaly.
 *   Default threshold: 2.0
 *
 * - "raw": No normalization, returns backend-native scores.
 *   Ranges vary by backend. Best for debugging or advanced users.
 *   Default threshold: 0.0 (user should set their own)
 */
export type NormalizationMethod = 'standardization' | 'zscore' | 'raw'

export interface NormalizationOption {
  value: NormalizationMethod
  label: string
  description: string
  defaultThreshold: number
}

export const NORMALIZATION_OPTIONS: NormalizationOption[] = [
  {
    value: 'standardization',
    label: 'Standardized (0-1)',
    description: 'Scores 0-1, threshold ~0.6. Best for general use.',
    defaultThreshold: 0.6,
  },
  {
    value: 'zscore',
    label: 'Z-Score',
    description: 'Standard deviations from mean. 2+ is unusual, 3+ is rare.',
    defaultThreshold: 2.0,
  },
  {
    value: 'raw',
    label: 'Raw',
    description: 'Backend-native scores. For debugging/advanced use.',
    defaultThreshold: 0.0,
  },
]

/**
 * Get the default threshold for a normalization method.
 */
export function getDefaultThreshold(normalization: NormalizationMethod): number {
  const option = NORMALIZATION_OPTIONS.find(o => o.value === normalization)
  return option?.defaultThreshold ?? 0.5
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
  normalization?: NormalizationMethod // default: "standardization"
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
  normalization?: NormalizationMethod // default: "standardization"
}

export interface AnomalyScoreResult {
  index: number
  data: number[]
  score: number
  is_anomaly: boolean
  raw_score?: number
}

export interface AnomalyScoreSummary {
  total_points: number
  anomaly_count: number
  anomaly_rate: number
  threshold: number
}

export interface AnomalyScoreResponse {
  object: string
  model: string
  backend: AnomalyBackend
  data: AnomalyScoreResult[]  // API returns 'data', not 'results'
  total_count: number
  summary: AnomalyScoreSummary
}

export interface AnomalySaveRequest {
  model: string
  backend?: AnomalyBackend
  description?: string
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
  object: string
  data: AnomalyModelInfo[]  // API returns 'data', not 'models'
  models_dir: string
  total: number
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

// =============================================================================
// Document Scanning Types
// =============================================================================

export type DocumentScanningBackend = 'surya' | 'easyocr' | 'tesseract'

export const DOCUMENT_SCANNING_BACKEND_DISPLAY: Record<
  DocumentScanningBackend,
  { label: string; description: string }
> = {
  surya: {
    label: 'Surya',
    description: 'Best accuracy, transformer-based, layout-aware (recommended)',
  },
  easyocr: {
    label: 'EasyOCR',
    description: 'Good multilingual support (80+ languages)',
  },
  tesseract: {
    label: 'Tesseract',
    description: 'Classic OCR engine, CPU-only, widely deployed',
  },
}

export const DOCUMENT_SCANNING_LANGUAGES: Array<{ code: string; label: string }> = [
  { code: 'en', label: 'English' },
  { code: 'de', label: 'German' },
  { code: 'fr', label: 'French' },
  { code: 'es', label: 'Spanish' },
  { code: 'it', label: 'Italian' },
  { code: 'pt', label: 'Portuguese' },
  { code: 'zh', label: 'Chinese' },
  { code: 'ja', label: 'Japanese' },
  { code: 'ko', label: 'Korean' },
  { code: 'ar', label: 'Arabic' },
  { code: 'ru', label: 'Russian' },
]

export interface DocumentScanningResultItem {
  index: number
  text: string
  confidence: number
}

export interface DocumentScanningResponse {
  object: string // "list"
  data: DocumentScanningResultItem[]
  model: string
  usage: {
    images_processed: number
    pages_combined?: number // Present when pages were combined into single result
  }
}

export interface DocumentScanningHistoryEntry {
  id: string
  timestamp: Date
  fileName: string
  pageCount: number
  pagesCombined?: number // Original page count when pages were combined into one
  avgConfidence: number
  previewText: string
  backend: string
  results: DocumentScanningResultItem[]
  error?: string
}

// =============================================================================
// Encoder Types (Embeddings & Reranking)
// =============================================================================

export type EncoderSubMode = 'embedding' | 'reranking'

// Embedding types
export interface EmbeddingRequest {
  model: string
  input: string | string[]
  encoding_format?: 'float' | 'base64'
}

export interface EmbeddingData {
  object: string // "embedding"
  index: number
  embedding: number[]
}

export interface EmbeddingResponse {
  object: string // "list"
  data: EmbeddingData[]
  model: string
  usage: {
    prompt_tokens: number
    total_tokens: number
  }
}

// Reranking types
export interface RerankRequest {
  model: string
  query: string
  documents: string[]
  top_k?: number
  return_documents?: boolean
}

export interface RerankResult {
  index: number
  relevance_score: number
  document?: string
}

export interface RerankResponse {
  object: string // "list"
  data: RerankResult[]
  model: string
  usage: {
    prompt_tokens: number
    total_tokens: number
  }
}

// History entry for encoder testing
export interface EncoderHistoryEntry {
  id: string
  timestamp: Date
  mode: EncoderSubMode
  modelName: string
  // Embedding mode
  texts?: string[]
  similarity?: number // single cosine similarity score for two-text comparison
  // Reranking mode
  query?: string
  documents?: string[]
  results?: RerankResult[]
  error?: string
}

// Common embedding models (HuggingFace)
export const COMMON_EMBEDDING_MODELS = [
  {
    value: 'sentence-transformers/all-MiniLM-L6-v2',
    label: 'all-MiniLM-L6-v2',
    description: 'Fast, balanced (384 dim)',
  },
  {
    value: 'sentence-transformers/all-mpnet-base-v2',
    label: 'all-mpnet-base-v2',
    description: 'Higher quality (768 dim)',
  },
  {
    value: 'BAAI/bge-small-en-v1.5',
    label: 'BGE Small',
    description: 'MTEB top performer (384 dim)',
  },
  {
    value: 'BAAI/bge-base-en-v1.5',
    label: 'BGE Base',
    description: 'MTEB top performer (768 dim)',
  },
  {
    value: 'nomic-ai/nomic-embed-text-v1.5',
    label: 'Nomic Embed',
    description: 'Long context (768 dim)',
  },
]

// Common reranking models
export const COMMON_RERANKING_MODELS = [
  {
    value: 'cross-encoder/ms-marco-MiniLM-L-6-v2',
    label: 'MS-MARCO MiniLM L6',
    description: 'Fast, good quality',
  },
  {
    value: 'cross-encoder/ms-marco-TinyBERT-L-2-v2',
    label: 'MS-MARCO TinyBERT',
    description: 'Fastest, smaller',
  },
  {
    value: 'BAAI/bge-reranker-base',
    label: 'BGE Reranker Base',
    description: 'High quality',
  },
  {
    value: 'BAAI/bge-reranker-large',
    label: 'BGE Reranker Large',
    description: 'Highest quality',
  },
]

// Sample data for embedding similarity testing
export const EMBEDDING_SAMPLES = [
  {
    textA: 'The cat sat on the mat',
    textB: 'A feline rested on the rug',
  },
  {
    textA: 'How do I reset my password?',
    textB: 'I forgot my login credentials and need help',
  },
  {
    textA: 'The bank approved my loan application',
    textB: 'I sat by the river bank watching the ducks',
  },
  {
    textA: "I love this product, it's amazing!",
    textB: "I hate this product, it's terrible!",
  },
  {
    textA: 'The dog chased the cat across the yard',
    textB: 'The cat chased the dog across the yard',
  },
  {
    textA: 'She sold seashells by the seashore',
    textB: 'Seashells were sold near the ocean by a woman',
  },
  {
    textA: 'The server crashed and we lost all our data',
    textB: 'Our waiter tripped and dropped all the plates',
  },
  {
    textA: 'Apply a thin layer of glaze before firing',
    textB: 'Brush on glaze coat prior to kiln firing',
  },
  {
    textA: 'The Python script failed with a syntax error',
    textB: 'The snake escaped from its enclosure at the zoo',
  },
  {
    textA: "It's raining cats and dogs outside",
    textB: 'The weather is extremely rainy right now',
  },
]

// Sample data for document reranking testing
export const RERANKING_SAMPLES = [
  {
    query: 'How do I make sourdough bread?',
    documents: [
      "To make sourdough, you'll need a starter, flour, water, and salt. Mix, fold, proof overnight, and bake at 450°F in a dutch oven.",
      'Bread has been a staple food for thousands of years across many cultures.',
      'Sourdough starter is made by fermenting flour and water over 5-7 days until bubbly and active.',
    ],
  },
  {
    query: 'why is my code returning undefined',
    documents: [
      'A function returns undefined when it has no return statement or the return statement has no value.',
      'JavaScript was created by Brendan Eich in 1995.',
      "Make sure you're not trying to access an object property before the object is initialized.",
      'The console.log() method outputs a message to the web console.',
      "Check if your async function is being awaited—forgetting await will give you a Promise instead of the value.",
    ],
  },
  {
    query: 'best plants for low light apartments',
    documents: [
      'Pothos, snake plants, and ZZ plants thrive in low light and are nearly impossible to kill.',
      'Most vegetables need 6-8 hours of direct sunlight.',
      'Fiddle leaf figs are a popular houseplant but require bright indirect light.',
    ],
  },
  {
    query: 'what to do if a llama spits at you',
    documents: [
      "Llamas spit when threatened or annoyed. If hit, calmly back away and clean up with water—their spit is mostly grass and saliva.",
      'Llamas are domesticated South American camelids used as pack animals.',
      'Alpacas are smaller than llamas and known for their soft fleece.',
      "If a llama pins its ears back, it's a warning sign—give it space.",
    ],
  },
  {
    query: 'how to center a div',
    documents: [
      'Use flexbox: set the parent to display: flex; justify-content: center; align-items: center;',
      'CSS Grid was introduced in 2017 and is now supported by all major browsers.',
      'The <div> element is a block-level container in HTML.',
      "For horizontal centering only, margin: 0 auto; works on block elements with a set width.",
      "Tailwind CSS classes like 'flex items-center justify-center' make centering trivial.",
      'The float property was commonly used for layouts before flexbox but caused many issues.',
    ],
  },
  {
    query: 'is mercury in retrograde right now',
    documents: [
      'Mercury retrograde is an optical illusion where the planet appears to move backward in the sky. Check an ephemeris for current dates.',
      'Mercury is the smallest planet in our solar system and closest to the Sun.',
      'Astrology is a belief system that interprets celestial positions as influencing human affairs.',
    ],
  },
  {
    query: 'ceramic glaze food safe',
    documents: [
      'For food-safe ceramics, use glazes labeled dinnerware-safe and fire to the proper cone. Avoid glazes with lead or cadmium.',
      'Pottery has been used for food storage since ancient times.',
      'Cone 6 glazes mature at approximately 2232°F (1222°C).',
      'Always test glazes on a small piece before applying to finished work.',
    ],
  },
  {
    query: 'my docker container keeps restarting',
    documents: [
      "Check your container logs with docker logs <container_id> to see why it's crashing.",
      'Docker is a platform for developing, shipping, and running applications in containers.',
      "The restart policy 'unless-stopped' will restart containers unless explicitly stopped.",
      "Make sure your container's entrypoint script isn't exiting immediately—it needs a foreground process.",
      'Kubernetes is an orchestration platform that manages Docker containers at scale.',
    ],
  },
  {
    query: 'good running shoes for flat feet',
    documents: [
      'Runners with flat feet typically benefit from stability or motion control shoes with arch support, like Brooks Adrenaline or ASICS Gel-Kayano.',
      'Running is an excellent cardiovascular exercise with many health benefits.',
      'Minimalist running shoes have less cushioning and encourage natural foot movement.',
    ],
  },
  {
    query: 'how to tell if avocado is ripe',
    documents: [
      "A ripe avocado yields slightly to gentle pressure and has dark green to black skin. Check under the stem—if it's green underneath, it's ready.",
      'Avocados are native to Mexico and Central America.',
      'Store unripe avocados at room temperature; refrigerate once ripe to slow further ripening.',
      'Guacamole is made from mashed avocados with lime juice, salt, and optional additions like cilantro and onion.',
    ],
  },
]

// =============================================================================
// Speech Types (STT, TTS, Voice Cloning)
// =============================================================================

export type SpeechModelStatus = 'ready' | 'downloading' | 'not_downloaded' | 'error'

// Speech-to-Text (STT) Models - faster-whisper models
export interface STTModel {
  id: string
  name: string
  size: string
  description?: string
}

// Available Whisper models from the Universal Runtime (faster-whisper)
// Valid model IDs from backend MODEL_SIZES dict
export const STT_MODELS: STTModel[] = [
  { id: 'distil-large-v3-turbo', name: 'Whisper Distil Large V3 Turbo', size: '~800M', description: 'Default - fast & accurate' },
  { id: 'large-v3-turbo', name: 'Whisper Large V3 Turbo', size: '~800M', description: 'Recommended - fast & accurate' },
  { id: 'distil-large-v3', name: 'Whisper Distil Large V3', size: '~800M', description: 'Distilled, high quality' },
  { id: 'large-v3', name: 'Whisper Large V3', size: '1.5B', description: 'Best accuracy, slower' },
  { id: 'large-v2', name: 'Whisper Large V2', size: '1.5B', description: 'Previous best' },
  { id: 'large-v1', name: 'Whisper Large V1', size: '1.5B', description: 'Original large' },
  { id: 'medium', name: 'Whisper Medium', size: '769M', description: 'High accuracy' },
  { id: 'medium.en', name: 'Whisper Medium (English)', size: '769M', description: 'English-only' },
  { id: 'small', name: 'Whisper Small', size: '244M', description: 'Good accuracy' },
  { id: 'small.en', name: 'Whisper Small (English)', size: '244M', description: 'English-only' },
  { id: 'base', name: 'Whisper Base', size: '74M', description: 'Basic accuracy' },
  { id: 'base.en', name: 'Whisper Base (English)', size: '74M', description: 'English-only' },
  { id: 'tiny', name: 'Whisper Tiny', size: '39M', description: 'Fastest, lowest accuracy' },
  { id: 'tiny.en', name: 'Whisper Tiny (English)', size: '39M', description: 'English-only, fast' },
  { id: 'distil-medium.en', name: 'Whisper Distil Medium (English)', size: '~400M', description: 'Smaller distilled' },
  { id: 'distil-small.en', name: 'Whisper Distil Small (English)', size: '~200M', description: 'Smallest distilled' },
]

// Text-to-Speech (TTS) Models
export interface TTSModel {
  id: string
  name: string
  size: string
  description?: string
  supportsVoiceCloning?: boolean
  supportsSpeed?: boolean
}

// Available TTS models from the Universal Runtime
export const TTS_MODELS: TTSModel[] = [
  { id: 'kokoro', name: 'Kokoro', size: '82M', description: 'Fast, GPU-optimized (~100ms)', supportsVoiceCloning: false, supportsSpeed: true },
  // Chatterbox Turbo hidden - requires HuggingFace auth (gated model)
  // { id: 'chatterbox-turbo', name: 'Chatterbox Turbo', size: '350M', description: 'Voice cloning, sub-200ms', supportsVoiceCloning: true, supportsSpeed: false },
  { id: 'pocket-tts', name: 'Pocket TTS', size: '100M', description: 'CPU-only, ~6x realtime', supportsVoiceCloning: false, supportsSpeed: false },
]

// Voice info from backend
export interface VoiceInfo {
  id: string
  name: string
  language: string
  model: string
  preview_url?: string | null
}

// Voice presets - matches backend voices
export interface VoicePreset {
  id: string
  name: string
  gender: 'male' | 'female' | 'neutral'
  language: string
  model: string
  isCustom?: boolean
  duration?: number // in seconds, for custom voices
  createdAt?: string
}

// Kokoro built-in voices (American and British English)
export const KOKORO_VOICES: VoicePreset[] = [
  // American English
  { id: 'af_heart', name: 'Heart (American Female)', gender: 'female', language: 'en-US', model: 'kokoro' },
  { id: 'af_bella', name: 'Bella (American Female)', gender: 'female', language: 'en-US', model: 'kokoro' },
  { id: 'af_nicole', name: 'Nicole (American Female)', gender: 'female', language: 'en-US', model: 'kokoro' },
  { id: 'af_sarah', name: 'Sarah (American Female)', gender: 'female', language: 'en-US', model: 'kokoro' },
  { id: 'af_sky', name: 'Sky (American Female)', gender: 'female', language: 'en-US', model: 'kokoro' },
  { id: 'am_adam', name: 'Adam (American Male)', gender: 'male', language: 'en-US', model: 'kokoro' },
  { id: 'am_michael', name: 'Michael (American Male)', gender: 'male', language: 'en-US', model: 'kokoro' },
  // British English
  { id: 'bf_emma', name: 'Emma (British Female)', gender: 'female', language: 'en-GB', model: 'kokoro' },
  { id: 'bf_isabella', name: 'Isabella (British Female)', gender: 'female', language: 'en-GB', model: 'kokoro' },
  { id: 'bm_george', name: 'George (British Male)', gender: 'male', language: 'en-GB', model: 'kokoro' },
  { id: 'bm_lewis', name: 'Lewis (British Male)', gender: 'male', language: 'en-GB', model: 'kokoro' },
]

// Chatterbox Turbo built-in voices
export const CHATTERBOX_VOICES: VoicePreset[] = [
  { id: 'cb_male_calm', name: 'Male Calm', gender: 'male', language: 'en', model: 'chatterbox-turbo' },
  { id: 'cb_female_warm', name: 'Female Warm', gender: 'female', language: 'en', model: 'chatterbox-turbo' },
  { id: 'cb_male_energetic', name: 'Male Energetic', gender: 'male', language: 'en', model: 'chatterbox-turbo' },
  { id: 'cb_female_professional', name: 'Female Professional', gender: 'female', language: 'en', model: 'chatterbox-turbo' },
]

// Pocket TTS built-in voices
export const POCKET_TTS_VOICES: VoicePreset[] = [
  { id: 'alba', name: 'Alba', gender: 'female', language: 'en', model: 'pocket-tts' },
  { id: 'marius', name: 'Marius', gender: 'male', language: 'en', model: 'pocket-tts' },
  { id: 'javert', name: 'Javert', gender: 'male', language: 'en', model: 'pocket-tts' },
  { id: 'jean', name: 'Jean', gender: 'male', language: 'en', model: 'pocket-tts' },
  { id: 'fantine', name: 'Fantine', gender: 'female', language: 'en', model: 'pocket-tts' },
  { id: 'cosette', name: 'Cosette', gender: 'female', language: 'en', model: 'pocket-tts' },
  { id: 'eponine', name: 'Eponine', gender: 'female', language: 'en', model: 'pocket-tts' },
  { id: 'azelma', name: 'Azelma', gender: 'female', language: 'en', model: 'pocket-tts' },
]

// All preset voices combined
export const PRESET_VOICES: VoicePreset[] = [
  ...KOKORO_VOICES,
  ...CHATTERBOX_VOICES,
  ...POCKET_TTS_VOICES,
]

// Get voices for a specific TTS model
export function getVoicesForModel(modelId: string): VoicePreset[] {
  switch (modelId) {
    case 'kokoro':
      return KOKORO_VOICES
    case 'chatterbox-turbo':
      return CHATTERBOX_VOICES
    case 'pocket-tts':
      return POCKET_TTS_VOICES
    default:
      return KOKORO_VOICES // Default to kokoro
  }
}

// Languages for STT
export const STT_LANGUAGES: Array<{ code: string; name: string }> = [
  { code: 'auto', name: 'Auto-detect' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'it', name: 'Italian' },
  { code: 'ru', name: 'Russian' },
  { code: 'ar', name: 'Arabic' },
]

// Microphone permission states
export type MicPermissionState = 'prompt' | 'granted' | 'denied' | 'error'

// Recording states
export type RecordingState = 'idle' | 'recording' | 'processing' | 'error'

// Custom voice clone
export interface VoiceClone {
  id: string
  name: string
  duration: number // in seconds
  createdAt: string
  audioBlob?: Blob
}

// Transcription result
export interface TranscriptionResult {
  text: string
  language?: string
  confidence?: number
  duration?: number
  segments?: TranscriptionSegment[]
}

export interface TranscriptionSegment {
  id: number
  start: number
  end: number
  text: string
  confidence?: number
}

// Speech message for conversation view
export interface SpeechMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  audioUrl?: string
  timestamp: Date
  transcription?: TranscriptionResult
}

// Speech test history entry
export interface SpeechHistoryEntry {
  id: string
  timestamp: Date
  mode: 'stt' | 'tts' | 'conversation'
  // STT fields
  transcription?: TranscriptionResult
  inputAudioUrl?: string
  // TTS fields
  inputText?: string
  outputAudioUrl?: string
  voiceId?: string
  // Shared
  error?: string
}
