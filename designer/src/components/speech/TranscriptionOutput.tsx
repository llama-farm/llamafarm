import { useState, useCallback } from 'react'
import { Copy, Check, Download, Globe, Clock, FileText } from 'lucide-react'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import type { TranscriptionResult } from '../../types/ml'

interface TranscriptionOutputProps {
  result: TranscriptionResult | null
  isLoading?: boolean
  error?: string | null
  showTimestamps?: boolean
  className?: string
}

export function TranscriptionOutput({
  result,
  isLoading = false,
  error = null,
  showTimestamps = false,
  className = '',
}: TranscriptionOutputProps) {
  const [copied, setCopied] = useState(false)
  const [showSegments, setShowSegments] = useState(false)

  const handleCopy = useCallback(async () => {
    if (!result?.text) return
    try {
      await navigator.clipboard.writeText(result.text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch {
      // Clipboard access denied
    }
  }, [result])

  const handleDownloadSRT = useCallback(() => {
    if (!result?.segments) return

    const srtContent = result.segments
      .map((seg, i) => {
        const formatSrtTime = (seconds: number) => {
          const h = Math.floor(seconds / 3600)
          const m = Math.floor((seconds % 3600) / 60)
          const s = Math.floor(seconds % 60)
          const ms = Math.floor((seconds % 1) * 1000)
          return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')},${ms.toString().padStart(3, '0')}`
        }
        return `${i + 1}\n${formatSrtTime(seg.start)} --> ${formatSrtTime(seg.end)}\n${seg.text.trim()}\n`
      })
      .join('\n')

    const blob = new Blob([srtContent], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'transcription.srt'
    a.click()
    URL.revokeObjectURL(url)
  }, [result])

  const handleDownloadVTT = useCallback(() => {
    if (!result?.segments) return

    const vttContent = 'WEBVTT\n\n' + result.segments
      .map((seg) => {
        const formatVttTime = (seconds: number) => {
          const h = Math.floor(seconds / 3600)
          const m = Math.floor((seconds % 3600) / 60)
          const s = seconds % 60
          return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toFixed(3).padStart(6, '0')}`
        }
        return `${formatVttTime(seg.start)} --> ${formatVttTime(seg.end)}\n${seg.text.trim()}\n`
      })
      .join('\n')

    const blob = new Blob([vttContent], { type: 'text/vtt' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'transcription.vtt'
    a.click()
    URL.revokeObjectURL(url)
  }, [result])

  // Loading state
  if (isLoading) {
    return (
      <div className={`flex items-center justify-center py-12 ${className}`}>
        <div className="text-center">
          <div className="inline-block h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent mb-3" />
          <div className="text-sm text-muted-foreground">Transcribing...</div>
        </div>
      </div>
    )
  }

  // Error state
  if (error) {
    return (
      <div className={`rounded-xl border border-amber-500/30 bg-amber-500/10 p-6 ${className}`}>
        <div className="text-center">
          <div className="text-lg font-medium text-foreground mb-2">Transcription Error</div>
          <div className="text-sm text-amber-400">{error}</div>
        </div>
      </div>
    )
  }

  // Empty state
  if (!result) {
    return (
      <div className={`flex items-center justify-center py-12 text-muted-foreground ${className}`}>
        <div className="text-center">
          <FileText className="w-10 h-10 mx-auto mb-2 opacity-40" />
          <div className="text-sm">Record or upload audio to see transcription</div>
        </div>
      </div>
    )
  }

  const formatDuration = (seconds: number) => {
    const m = Math.floor(seconds / 60)
    const s = Math.floor(seconds % 60)
    return `${m}:${s.toString().padStart(2, '0')}`
  }

  return (
    <div className={`rounded-xl border border-border bg-card ${className}`}>
      {/* Header with metadata */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-border">
        <div className="flex flex-wrap items-center gap-2">
          {result.language && (
            <Badge className="bg-indigo-500/20 text-indigo-400 border border-indigo-500/30">
              <Globe className="w-3 h-3 mr-1" />
              {result.language.toUpperCase()}
            </Badge>
          )}
          {result.duration && (
            <Badge className="bg-muted text-muted-foreground border border-border">
              <Clock className="w-3 h-3 mr-1" />
              {formatDuration(result.duration)}
            </Badge>
          )}
          {result.confidence !== undefined && (
            <Badge className="bg-muted text-muted-foreground border border-border">
              {(result.confidence * 100).toFixed(0)}% confidence
            </Badge>
          )}
        </div>

        {/* Actions */}
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
          >
            {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
            {copied ? 'Copied' : 'Copy'}
          </Button>
          {result.segments && result.segments.length > 0 && (
            <>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDownloadSRT}
              >
                <Download className="w-4 h-4 mr-1" />
                .srt
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={handleDownloadVTT}
              >
                <Download className="w-4 h-4 mr-1" />
                .vtt
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Main transcription text */}
      <div className="p-4">
        <div className="text-sm leading-relaxed whitespace-pre-wrap">
          {result.text}
        </div>
      </div>

      {/* Segments with timestamps */}
      {showTimestamps && result.segments && result.segments.length > 0 && (
        <div className="border-t border-border">
          <button
            onClick={() => setShowSegments(!showSegments)}
            className="w-full px-4 py-2 text-left text-sm text-muted-foreground hover:bg-muted/50 flex items-center justify-between"
          >
            <span>Word timestamps ({result.segments.length} segments)</span>
            <span>{showSegments ? '−' : '+'}</span>
          </button>
          {showSegments && (
            <div className="px-4 pb-4 max-h-64 overflow-y-auto">
              <div className="space-y-1">
                {result.segments.map((segment) => (
                  <div
                    key={segment.id}
                    className="flex items-start gap-3 text-sm py-1"
                  >
                    <span className="text-xs text-muted-foreground tabular-nums flex-shrink-0 pt-0.5">
                      {formatDuration(segment.start)} - {formatDuration(segment.end)}
                    </span>
                    <span>{segment.text}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
