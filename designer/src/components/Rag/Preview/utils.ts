/**
 * Shared utilities for document preview components.
 */

import type { PreviewChunk } from '../../../hooks/useDocumentPreview'

// Chunk background colors (alternating, dark mode aware)
export const CHUNK_COLORS = [
  'bg-purple-200/60 dark:bg-purple-500/40',
  'bg-pink-200/60 dark:bg-pink-500/40',
  'bg-indigo-200/60 dark:bg-indigo-500/40',
]

export interface Segment {
  type: 'chunk' | 'overlap'
  index?: number
  fromChunk?: number
  toChunk?: number
  text: string
  color?: string
}

/**
 * Build segments from chunks for rendering with consistent line wrapping.
 * Used by both PreviewPanel (with colors) and OriginalDocumentPanel (without).
 */
export function buildSegmentsFromChunks(
  originalText: string,
  chunks: PreviewChunk[],
  chunkOverlap: number,
  includeColors: boolean = false
): Segment[] {
  if (!chunks.length) return []

  const result: Segment[] = []

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i]
    const nextChunk = chunks[i + 1]

    // Skip chunks not found in text
    if (chunk.start_position < 0) {
      result.push({
        type: 'chunk',
        index: i,
        text: chunk.content,
        color: includeColors ? CHUNK_COLORS[i % CHUNK_COLORS.length] : undefined,
      })
      continue
    }

    // Calculate non-overlap portion
    const nonOverlapEnd =
      nextChunk && chunkOverlap > 0
        ? Math.max(chunk.start_position, chunk.end_position - chunkOverlap)
        : chunk.end_position

    // Add non-overlap portion of chunk
    if (nonOverlapEnd > chunk.start_position) {
      result.push({
        type: 'chunk',
        index: i,
        text: originalText.slice(chunk.start_position, nonOverlapEnd),
        color: includeColors ? CHUNK_COLORS[i % CHUNK_COLORS.length] : undefined,
      })
    }

    // Add overlap region (if exists)
    if (nextChunk && chunkOverlap > 0 && nonOverlapEnd < chunk.end_position) {
      result.push({
        type: 'overlap',
        fromChunk: i,
        toChunk: i + 1,
        text: originalText.slice(nonOverlapEnd, chunk.end_position),
      })
    }
  }

  return result
}
