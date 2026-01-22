/**
 * Modal for document preview with all controls and panels.
 */

import { useState, useCallback, useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '../../ui/dialog'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '../../ui/select'
import { useDocumentPreview } from '../../../hooks/useDocumentPreview'
import { useAvailableStrategies } from '../../../hooks/useDatasets'
import { PreviewPanel } from './PreviewPanel'
import { OriginalDocumentPanel } from './OriginalDocumentPanel'
import { Loader2, RefreshCw } from 'lucide-react'
import { Button } from '../../ui/button'

interface DocumentPreviewModalProps {
  isOpen: boolean
  onClose: () => void
  namespace: string
  project: string
  database: string
  fileHash?: string
  datasetId?: string
  filename?: string
}

export function DocumentPreviewModal({
  isOpen,
  onClose,
  namespace,
  project,
  database,
  fileHash,
  datasetId,
  filename: _filename,
}: DocumentPreviewModalProps) {
  // Strategy selection - undefined means use dataset's default
  const [selectedStrategy, setSelectedStrategy] = useState<string | undefined>(
    undefined
  )

  // Override settings - undefined means use strategy's default (not sent to API)
  // Note: setters prefixed with _ as chunk override UI is not yet implemented
  const [chunkSize, _setChunkSize] = useState<number | undefined>(undefined)
  const [chunkOverlap, _setChunkOverlap] = useState<number | undefined>(undefined)
  const [chunkStrategy, _setChunkStrategy] = useState<
    'characters' | 'sentences' | 'paragraphs' | undefined
  >(undefined)

  const [selectedChunkIndex, setSelectedChunkIndex] = useState<number | null>(
    null
  )

  const preview = useDocumentPreview(namespace, project)
  const { data: strategiesData } = useAvailableStrategies(namespace, project)

  const fetchPreview = useCallback(() => {
    if (!fileHash) return

    preview.mutate({
      database,
      file_hash: fileHash,
      dataset_id: datasetId,
      // Only send strategy if explicitly selected
      ...(selectedStrategy && { data_processing_strategy: selectedStrategy }),
      // Only send overrides if explicitly set
      ...(chunkSize !== undefined && { chunk_size: chunkSize }),
      ...(chunkOverlap !== undefined && { chunk_overlap: chunkOverlap }),
      ...(chunkStrategy !== undefined && { chunk_strategy: chunkStrategy }),
    })
  }, [
    database,
    fileHash,
    datasetId,
    selectedStrategy,
    chunkSize,
    chunkOverlap,
    chunkStrategy,
    preview.mutate, // Use stable mutate reference instead of full preview object
  ])

  useEffect(() => {
    if (isOpen && fileHash) {
      fetchPreview()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- fetchPreview intentionally omitted to prevent re-fetch on strategy/override changes
  }, [isOpen, fileHash])

  const handleStrategyChange = (value: string) => {
    // Special value "__default__" means use dataset's default
    setSelectedStrategy(value === '__default__' ? undefined : value)
  }

  return (
    <Dialog open={isOpen} onOpenChange={open => !open && onClose()}>
      <DialogContent
        data-testid="preview-modal"
        className="max-w-6xl max-h-[90vh] overflow-hidden"
      >
        <DialogHeader>
          <DialogTitle>
            Document Preview
            {preview.data?.filename && (
              <span className="ml-2 text-muted-foreground font-normal">
                - {preview.data.filename}
              </span>
            )}
          </DialogTitle>
        </DialogHeader>

        {preview.isPending && (
          <div
            data-testid="preview-loading"
            className="flex items-center justify-center p-8"
          >
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
            <span className="ml-2">Loading preview...</span>
          </div>
        )}

        {preview.isError && (
          <div
            data-testid="preview-error"
            className="p-4 bg-destructive/10 text-destructive rounded"
          >
            Error loading preview: {preview.error?.message || 'Unknown error'}
          </div>
        )}

        {preview.isSuccess && preview.data && (
          <div className="flex flex-col gap-4 h-[600px]">
            {/* Controls and Statistics Row */}
            <div className="flex items-start gap-4 flex-wrap">
              {/* Strategy Selector */}
              {strategiesData?.data_processing_strategies &&
                strategiesData.data_processing_strategies.length > 0 && (
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-muted-foreground whitespace-nowrap">
                      Strategy:
                    </label>
                    <Select
                      value={selectedStrategy ?? '__default__'}
                      onValueChange={handleStrategyChange}
                    >
                      <SelectTrigger className="w-[180px]">
                        <SelectValue placeholder="Dataset default" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__default__">
                          Dataset default
                        </SelectItem>
                        {strategiesData.data_processing_strategies.map(
                          strategy => (
                            <SelectItem key={strategy} value={strategy}>
                              {strategy}
                            </SelectItem>
                          )
                        )}
                      </SelectContent>
                    </Select>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={fetchPreview}
                      disabled={preview.isPending}
                    >
                      <RefreshCw
                        className={`h-4 w-4 ${preview.isPending ? 'animate-spin' : ''}`}
                      />
                    </Button>
                  </div>
                )}

              {/* Inline Statistics */}
              <div
                data-testid="preview-stats"
                className="flex items-center gap-4 px-4 py-2 bg-muted rounded-lg text-sm"
              >
                <span>
                  <strong>{preview.data.total_chunks}</strong> chunks
                </span>
                <span className="text-muted-foreground">|</span>
                <span>
                  Avg:{' '}
                  <strong>{Math.round(preview.data.avg_chunk_size)}</strong>{' '}
                  chars
                </span>
                <span className="text-muted-foreground">|</span>
                <span>
                  Parser: <strong>{preview.data.parser_used}</strong>
                </span>
                <span className="text-muted-foreground">|</span>
                <span>
                  Strategy: <strong>{preview.data.chunk_strategy}</strong>
                </span>
              </div>
            </div>

            {/* Single scrollable container with two columns */}
            <div className="flex-1 min-h-0 overflow-auto border rounded-lg">
              <div className="grid grid-cols-2 gap-4 p-4">
                {/* Original Document Panel */}
                <div>
                  <h3 className="text-sm font-medium mb-2 text-muted-foreground">
                    Original Document
                  </h3>
                  <OriginalDocumentPanel
                    originalText={preview.data.original_text}
                    contentType={preview.data.content_type}
                    filename={preview.data.filename}
                    chunks={preview.data.chunks}
                    chunkOverlap={chunkOverlap ?? preview.data.chunk_overlap}
                    selectedRange={
                      selectedChunkIndex !== null &&
                      preview.data.chunks[selectedChunkIndex]
                        ? {
                            start:
                              preview.data.chunks[selectedChunkIndex]
                                .start_position,
                            end: preview.data.chunks[selectedChunkIndex]
                              .end_position,
                          }
                        : null
                    }
                  />
                </div>

                {/* Chunked Preview Panel */}
                <div>
                  <h3 className="text-sm font-medium mb-2 text-muted-foreground">
                    Chunked Preview
                  </h3>
                  <PreviewPanel
                    originalText={preview.data.original_text}
                    chunks={preview.data.chunks}
                    chunkOverlap={chunkOverlap ?? preview.data.chunk_overlap}
                    selectedChunkIndex={selectedChunkIndex}
                    onChunkClick={setSelectedChunkIndex}
                  />
                </div>
              </div>
            </div>
          </div>
        )}
      </DialogContent>
    </Dialog>
  )
}
