/**
 * Panel displaying document text with highlighted chunks.
 */

import { useRef, useEffect, useMemo } from 'react'
import { cn } from '../../../lib/utils'
import type { PreviewChunk } from '../../../hooks/useDocumentPreview'

// Chunk background colors (alternating, dark mode aware)
const CHUNK_COLORS = [
  'bg-purple-200/60 dark:bg-purple-500/40',
  'bg-pink-200/60 dark:bg-pink-500/40',
  'bg-indigo-200/60 dark:bg-indigo-500/40',
]

interface Segment {
  type: 'chunk' | 'overlap' | 'gap'
  index?: number
  fromChunk?: number
  toChunk?: number
  text: string
  color?: string
}

interface PreviewPanelProps {
  originalText: string
  chunks: PreviewChunk[]
  chunkOverlap: number
  selectedChunkIndex?: number | null
  onChunkRef?: (ref: HTMLElement | null) => void
  onChunkClick?: (index: number) => void
}

export function PreviewPanel({
  originalText,
  chunks,
  chunkOverlap,
  selectedChunkIndex,
  onChunkRef,
  onChunkClick,
}: PreviewPanelProps) {
  const chunkRefs = useRef<Map<number, HTMLElement>>(new Map())

  // Scroll to selected chunk
  useEffect(() => {
    if (selectedChunkIndex !== null && selectedChunkIndex !== undefined) {
      const ref = chunkRefs.current.get(selectedChunkIndex)
      if (ref) {
        ref.scrollIntoView({ behavior: 'smooth', block: 'center' })
        onChunkRef?.(ref)
      }
    }
  }, [selectedChunkIndex, onChunkRef])

  // Build segments for rendering
  const segments = useMemo(() => {
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
          color: CHUNK_COLORS[i % CHUNK_COLORS.length],
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
          color: CHUNK_COLORS[i % CHUNK_COLORS.length],
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
  }, [originalText, chunks, chunkOverlap])

  return (
    <div
      data-testid="preview-panel"
      className="font-mono text-sm leading-6 whitespace-pre-wrap p-4 pl-8 bg-muted rounded-lg border border-border relative"
    >
      {segments.map((seg, idx) => {
        if (seg.type === 'chunk') {
          const isSelected = selectedChunkIndex === seg.index
          return (
            <span
              key={idx}
              ref={el => {
                if (el && seg.index !== undefined) {
                  chunkRefs.current.set(seg.index, el)
                }
              }}
              data-testid={`chunk-${seg.index}`}
              onClick={() =>
                seg.index !== undefined && onChunkClick?.(seg.index)
              }
              className={cn(
                seg.color,
                'relative [box-decoration-break:clone]',
                isSelected && 'ring-2 ring-primary',
                seg.index !== undefined &&
                  chunks[seg.index]?.start_position < 0 &&
                  'opacity-50',
                onChunkClick &&
                  'cursor-pointer hover:ring-1 hover:ring-primary/50 transition-shadow'
              )}
            >
              <span className="absolute -left-7 text-xs text-muted-foreground font-bold select-none">
                {(seg.index ?? 0) + 1}
              </span>
              {seg.text}
            </span>
          )
        }

        if (seg.type === 'overlap') {
          return (
            <span
              key={idx}
              data-testid={`overlap-${seg.fromChunk}-${seg.toChunk}`}
              className="bg-orange-200/60 dark:bg-orange-400/50 [box-decoration-break:clone]"
              title={`Overlap between chunk ${(seg.fromChunk ?? 0) + 1} and ${(seg.toChunk ?? 0) + 1}`}
            >
              {seg.text}
            </span>
          )
        }

        return <span key={idx}>{seg.text}</span>
      })}

      {!segments.length && (
        <span className="text-muted-foreground italic">
          No chunks to display
        </span>
      )}
    </div>
  )
}
