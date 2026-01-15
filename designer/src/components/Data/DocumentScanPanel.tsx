import { useRef, useEffect, useState, useCallback } from 'react'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import type { FileScanResult } from '../../types/datasets'

interface DocumentScanPanelProps {
  /** Whether the panel is open */
  isOpen: boolean
  /** Callback to close the panel */
  onClose: () => void
  /** The filename being viewed */
  fileName: string
  /** The scan result to display */
  scanResult: FileScanResult | null
  /** Whether the scan is currently loading */
  isLoading?: boolean
  /** Error message if scan failed */
  error?: string | null
  /** Callback to retry scanning */
  onRetry?: () => void
}

/**
 * Side panel for viewing parsed document content from OCR scanning.
 * Slides in from the right side of the screen.
 */
export default function DocumentScanPanel({
  isOpen,
  onClose,
  fileName,
  scanResult,
  isLoading = false,
  error = null,
  onRetry,
}: DocumentScanPanelProps) {
  const panelRef = useRef<HTMLDivElement>(null)
  const [copied, setCopied] = useState(false)
  const [selectedPage, setSelectedPage] = useState(0)

  // Reset selected page when scan result changes
  useEffect(() => {
    setSelectedPage(0)
  }, [scanResult])

  // Handle escape key to close
  useEffect(() => {
    if (!isOpen) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isOpen, onClose])

  // Handle click outside to close
  useEffect(() => {
    if (!isOpen) return

    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as HTMLElement
      if (target.classList.contains('scan-panel-overlay')) {
        onClose()
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isOpen, onClose])

  const handleCopyAll = useCallback(async () => {
    if (!scanResult?.pages) return
    const fullText = scanResult.pages
      .map(p => p.text)
      .join('\n\n--- Page Break ---\n\n')
    try {
      await navigator.clipboard.writeText(fullText)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard API may fail due to permissions or insecure context
      // Silently fail - user can manually select and copy text
    }
  }, [scanResult])

  if (!isOpen) return null

  const pages = scanResult?.pages || []
  const currentPage = pages[selectedPage] || pages[0]
  const avgConfidence =
    pages.length > 0
      ? pages.reduce((sum, p) => sum + p.confidence, 0) / pages.length
      : 0

  return (
    <>
      {/* Overlay */}
      <div
        className="scan-panel-overlay absolute inset-0 z-40 bg-gray-200/80 dark:bg-black/60 animate-in fade-in-0 duration-200"
        aria-hidden="true"
      />

      {/* Panel */}
      <div
        ref={panelRef}
        className="absolute top-0 right-0 bottom-0 z-50 w-[480px] max-w-[90%] bg-background border-l border-border shadow-xl animate-in slide-in-from-right duration-200 flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border">
          <div className="flex items-center gap-3 min-w-0">
            <FontIcon type="data" className="w-5 h-5 text-sky-400 flex-shrink-0" />
            <div className="min-w-0">
              <div className="text-sm font-medium truncate">{fileName}</div>
              {pages.length > 0 && (
                <div className="text-xs text-muted-foreground">
                  {pages.length} page{pages.length > 1 ? 's' : ''} •{' '}
                  {(avgConfidence * 100).toFixed(1)}% avg confidence
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center gap-2">
            {pages.length > 0 && (
              <Button
                variant="secondary"
                size="sm"
                onClick={handleCopyAll}
                className="h-8"
              >
                <FontIcon
                  type={copied ? 'checkmark-filled' : 'copy'}
                  className="w-4 h-4 mr-1.5"
                />
                {copied ? 'Copied!' : 'Copy All'}
              </Button>
            )}
            <button
              onClick={onClose}
              className="w-8 h-8 grid place-items-center rounded-md hover:bg-muted text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Close panel"
            >
              <FontIcon type="close" className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-4">
          {isLoading && (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent" />
              <div className="mt-3 text-sm text-muted-foreground">
                Extracting text...
              </div>
            </div>
          )}

          {error && !isLoading && (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="text-center px-6 py-10 rounded-xl border border-amber-500/30 bg-amber-500/10 max-w-sm">
                <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-amber-500/20 border border-amber-500/30">
                  <FontIcon
                    type="alert-triangle"
                    className="w-5 h-5 text-amber-400"
                  />
                </div>
                <div className="text-lg font-medium text-foreground">
                  Scanning Error
                </div>
                <div className="mt-2 text-sm text-amber-400">{error}</div>
                {onRetry && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={onRetry}
                    className="mt-4"
                  >
                    Retry Scan
                  </Button>
                )}
              </div>
            </div>
          )}

          {!isLoading && !error && pages.length === 0 && (
            <div className="flex flex-col items-center justify-center h-full text-center">
              <FontIcon
                type="data"
                className="w-12 h-12 text-muted-foreground/40 mb-4"
              />
              <div className="text-sm font-medium text-foreground">
                No scan results
              </div>
              <div className="text-xs text-muted-foreground mt-1">
                This file has not been scanned yet
              </div>
              {onRetry && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={onRetry}
                  className="mt-4"
                >
                  Scan Document
                </Button>
              )}
            </div>
          )}

          {!isLoading && !error && pages.length > 0 && (
            <>
              {/* Page selector for multi-page documents */}
              {pages.length > 1 && (
                <div className="flex items-center gap-2 mb-4">
                  <span className="text-xs text-muted-foreground">Page:</span>
                  <div className="flex gap-1 flex-wrap">
                    {pages.map((_, idx) => (
                      <button
                        key={idx}
                        onClick={() => setSelectedPage(idx)}
                        className={`px-2 py-0.5 text-xs rounded transition-colors ${
                          idx === selectedPage
                            ? 'bg-primary text-primary-foreground'
                            : 'bg-muted hover:bg-muted/80'
                        }`}
                      >
                        {idx + 1}
                      </button>
                    ))}
                  </div>
                </div>
              )}

              {/* Text content */}
              <div className="rounded-lg border border-border bg-muted/30 p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs text-muted-foreground">
                    Page {selectedPage + 1} •{' '}
                    {currentPage
                      ? (currentPage.confidence * 100).toFixed(1)
                      : 0}
                    % confidence
                  </span>
                </div>
                <div className="whitespace-pre-wrap text-sm leading-relaxed font-mono">
                  {currentPage?.text || '(No text detected)'}
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer with metadata */}
        {scanResult?.scanned_at && (
          <div className="p-3 border-t border-border bg-muted/30 text-xs text-muted-foreground">
            Scanned {new Date(scanResult.scanned_at).toLocaleString()}
            {scanResult.backend && ` using ${scanResult.backend}`}
          </div>
        )}
      </div>
    </>
  )
}
