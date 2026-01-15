/**
 * TypeScript types for Hugging Face dataset integration
 */

/** Search result from HF datasets API */
export interface HFDatasetSearchResult {
  id: string
  author?: string
  downloads: number
  likes: number
  tags: string[]
  description?: string
  cardData?: {
    pretty_name?: string
    size_categories?: string[]
    task_categories?: string[]
    language?: string[]
  }
}

/** Request to import HF dataset via backend */
export interface HFDatasetImportRequest {
  namespace: string
  project: string
  dataset: string
  hf_dataset_id: string
  config?: string
  split?: string
  max_rows?: number
  format?: 'jsonl' | 'csv'
  data_processing_strategy: string
  database: string
}

/** Response from HF dataset import */
export interface HFDatasetImportResponse {
  project: string
  namespace: string
  dataset: string
  file_count: number
  row_count: number
  task_id?: string
  // Auto-detected schema info
  detected_text_field?: string
  detected_label_field?: string
  available_fields?: string[]
}

/** Selected HF dataset for onboarding state */
export interface SelectedHFDataset {
  id: string
  name: string
  rowCount: number
  config: string
  split: string
}
