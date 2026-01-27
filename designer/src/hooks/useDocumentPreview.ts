/**
 * Hook for document preview API calls.
 */

import { useMutation } from '@tanstack/react-query'
import { apiClient } from '../api/client'

export interface PreviewChunk {
  chunk_index: number
  content: string
  start_position: number
  end_position: number
  char_count: number
  word_count: number
  metadata?: Record<string, unknown>
}

export interface DocumentPreviewResponse {
  original_text: string
  chunks: PreviewChunk[]
  filename: string
  size_bytes: number
  content_type: string | null
  parser_used: string
  chunk_strategy: string
  chunk_size: number
  chunk_overlap: number
  total_chunks: number
  avg_chunk_size: number
  total_size_with_overlaps: number
  avg_overlap_size: number
  warnings: string[]
}

export interface DocumentPreviewParams {
  database: string
  dataset_id?: string
  file_hash?: string
  file_content?: string
  filename?: string
  data_processing_strategy?: string
  chunk_size?: number
  chunk_overlap?: number
  chunk_strategy?: 'characters' | 'sentences' | 'paragraphs'
}

export function useDocumentPreview(namespace: string, project: string) {
  return useMutation({
    mutationFn: async (params: DocumentPreviewParams) => {
      const response = await apiClient.post<DocumentPreviewResponse>(
        `/projects/${namespace}/${project}/rag/databases/${params.database}/preview`,
        {
          dataset_id: params.dataset_id,
          file_hash: params.file_hash,
          file_content: params.file_content,
          filename: params.filename,
          data_processing_strategy: params.data_processing_strategy,
          chunk_size: params.chunk_size,
          chunk_overlap: params.chunk_overlap,
          chunk_strategy: params.chunk_strategy,
        }
      )
      return response.data
    },
  })
}
