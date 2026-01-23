/**
 * Panel displaying the original document text with optional chunk highlighting.
 * Renders with the same segment structure as PreviewPanel for identical line wrapping.
 */

import React, { useEffect, useMemo } from 'react'
import { FileText, AlertCircle } from 'lucide-react'
import type { PreviewChunk } from '../../../hooks/useDocumentPreview'
import { buildSegmentsFromChunks } from './utils'
import { VALID_FILE_TYPES } from '../../../utils/fileValidation'

interface OriginalDocumentPanelProps {
  originalText: string
  contentType: string | null
  filename: string
  chunks: PreviewChunk[]
  chunkOverlap: number
  selectedRange?: { start: number; end: number } | null
}

// Binary file extensions derived from VALID_FILE_TYPES
const BINARY_EXTENSIONS = Object.keys(VALID_FILE_TYPES).filter(ext =>
  ['.pdf', '.docx', '.xlsx', '.xls', '.parquet'].includes(ext)
)

function isBinaryFile(filename: string, contentType: string | null): boolean {
  const lowerFilename = filename.toLowerCase()
  if (BINARY_EXTENSIONS.some(ext => lowerFilename.endsWith(ext))) {
    return true
  }
  if (contentType) {
    return (
      contentType.includes('pdf') ||
      contentType.includes('msword') ||
      contentType.includes('officedocument') ||
      contentType.includes('spreadsheet')
    )
  }
  return false
}

export function OriginalDocumentPanel({
  originalText,
  contentType,
  filename,
  chunks,
  chunkOverlap,
  selectedRange,
}: OriginalDocumentPanelProps) {
  const highlightRef = React.useRef<HTMLSpanElement>(null)

  const isBinary = isBinaryFile(filename, contentType)

  // Scroll to selected range when it changes
  useEffect(() => {
    if (selectedRange && highlightRef.current) {
      highlightRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [selectedRange])

  // Build segments matching PreviewPanel structure for identical line wrapping
  const segments = useMemo(() => {
    if (!chunks.length) {
      return [{ type: 'chunk' as const, text: originalText }]
    }
    return buildSegmentsFromChunks(originalText, chunks, chunkOverlap, false)
  }, [originalText, chunks, chunkOverlap])

  // Check if a position is within the selected range
  const isInSelectedRange = (segmentStart: number, segmentEnd: number): boolean => {
    if (!selectedRange) return false
    return segmentStart < selectedRange.end && segmentEnd > selectedRange.start
  }

  // Track position while rendering
  let currentPosition = 0

  return (
    <div className="flex flex-col">
      <div
        data-testid="original-document-panel"
        className="font-mono text-sm leading-6 whitespace-pre-wrap p-4 pl-8 bg-muted rounded-lg border border-border relative"
      >
        {originalText ? (
          segments.map((seg, idx) => {
            const segStart = currentPosition
            const segEnd = currentPosition + seg.text.length
            currentPosition = segEnd

            const isHighlighted = isInSelectedRange(segStart, segEnd)

            // Render with same structure as PreviewPanel (span with relative positioning)
            // but without visible colors
            return (
              <span
                key={idx}
                ref={isHighlighted ? highlightRef : undefined}
                data-testid={isHighlighted ? 'selected-range-highlight' : undefined}
                className={`relative [box-decoration-break:clone]${isHighlighted ? ' bg-primary/20' : ''}`}
              >
                {seg.text}
              </span>
            )
          })
        ) : (
          <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
            <FileText className="w-12 h-12 mb-2 opacity-50" />
            <span className="italic">No text content available</span>
          </div>
        )}
      </div>

      {isBinary && (
        <div
          data-testid="binary-file-banner"
          className="flex items-center gap-2 p-3 mt-2 bg-blue-50 dark:bg-blue-950/30 text-blue-700 dark:text-blue-300 rounded-lg border border-blue-200 dark:border-blue-800"
        >
          <AlertCircle className="w-4 h-4 flex-shrink-0" />
          <span className="text-sm">
            This is a binary file ({filename.split('.').pop()?.toUpperCase()}). The text shown above was extracted from the document.
          </span>
        </div>
      )}
    </div>
  )
}
