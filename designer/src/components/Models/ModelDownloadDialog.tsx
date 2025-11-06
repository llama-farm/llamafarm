import { useEffect } from 'react'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from '../ui/dialog'
import { Button } from '../ui/button'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import type { DownloadProgress } from '../../types/model'

interface ModelDownloadDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  modelName: string
  progress: DownloadProgress | null
  error: string | null
  isDownloading: boolean
  onCancel?: () => void
  onComplete?: () => void
}

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let i = Math.floor(Math.log(bytes) / Math.log(1024))
  if (i >= units.length) i = units.length - 1
  const val = bytes / Math.pow(1024, i)
  return `${val.toFixed(i >= 2 ? 1 : 0)} ${units[i]}`
}

function formatSpeed(bytesPerSecond: number): string {
  return `${formatBytes(bytesPerSecond)}/s`
}

function formatTime(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return 'Calculating...'

  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)

  if (hours > 0) {
    return `${hours}h ${minutes}m ${secs}s`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}

function getFileName(path: string): string {
  // Extract filename from path like "model.safetensors.001" or "config.json"
  const parts = path.split('/')
  return parts[parts.length - 1] || path
}

export function ModelDownloadDialog({
  open,
  onOpenChange,
  modelName,
  progress,
  error,
  isDownloading,
  onCancel,
  onComplete,
}: ModelDownloadDialogProps) {
  // Auto-close on completion (after a delay)
  useEffect(() => {
    if (!isDownloading && progress && !error && progress.overallProgress === 100) {
      const timer = setTimeout(() => {
        onComplete?.()
        onOpenChange(false)
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [isDownloading, progress, error, onComplete, onOpenChange])

  const isComplete = !isDownloading && progress && progress.overallProgress === 100

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogTitle>
          {isComplete ? 'Download Complete' : 'Downloading Model'}
        </DialogTitle>
        <DialogDescription>
          <div className="mt-2 flex flex-col gap-4">
            {/* Model Name */}
            <div>
              <div className="text-xs text-muted-foreground mb-1">Model</div>
              <div className="text-sm font-medium text-foreground">
                {modelName}
              </div>
            </div>

            {/* Error Display */}
            {error && (
              <div className="rounded-lg bg-destructive/10 border border-destructive/30 p-3">
                <div className="flex items-start gap-2">
                  <FontIcon
                    type="alert-triangle"
                    className="w-4 h-4 text-destructive flex-shrink-0 mt-0.5"
                  />
                  <div className="flex-1">
                    <div className="text-sm font-medium text-destructive mb-1">
                      Download Failed
                    </div>
                    <div className="text-xs text-destructive/90">{error}</div>
                  </div>
                </div>
              </div>
            )}

            {/* Progress Display */}
            {progress && !error && (
              <>
                {/* Overall Progress Bar */}
                <div>
                  <div className="flex items-center justify-between text-xs mb-2">
                    <span className="text-muted-foreground">
                      Overall Progress
                    </span>
                    <span className="font-medium text-foreground">
                      {progress.overallProgress}%
                    </span>
                  </div>
                  <div className="w-full h-3 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300 ease-out"
                      style={{ width: `${progress.overallProgress}%` }}
                    />
                  </div>
                </div>

                {/* Current File */}
                {progress.currentFile && (
                  <div className="rounded-lg bg-secondary/50 p-3">
                    <div className="flex items-start gap-3">
                      {isDownloading && (
                        <Loader
                          size={16}
                          className="border-primary flex-shrink-0 mt-0.5"
                        />
                      )}
                      {isComplete && (
                        <FontIcon
                          type="checkmark-filled"
                          className="w-4 h-4 text-primary flex-shrink-0 mt-0.5"
                        />
                      )}
                      <div className="flex-1 min-w-0">
                        <div className="text-xs text-muted-foreground mb-1">
                          {isComplete ? 'Completed' : 'Downloading'}
                        </div>
                        <div
                          className="text-sm font-medium text-foreground truncate"
                          title={progress.currentFile}
                        >
                          {getFileName(progress.currentFile)}
                        </div>
                        {progress.currentFileTotal > 0 && (
                          <div className="text-xs text-muted-foreground mt-1">
                            {formatBytes(progress.currentFileDownloaded)} /{' '}
                            {formatBytes(progress.currentFileTotal)}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )}

                {/* Stats Grid */}
                <div className="grid grid-cols-2 gap-3">
                  {/* Files Completed */}
                  <div className="rounded-lg bg-card border border-border p-3">
                    <div className="text-xs text-muted-foreground mb-1">
                      Files
                    </div>
                    <div className="text-lg font-semibold text-foreground">
                      {progress.filesCompleted} / {progress.totalFiles}
                    </div>
                  </div>

                  {/* Download Speed */}
                  <div className="rounded-lg bg-card border border-border p-3">
                    <div className="text-xs text-muted-foreground mb-1">
                      Speed
                    </div>
                    <div className="text-lg font-semibold text-foreground">
                      {formatSpeed(progress.downloadSpeed)}
                    </div>
                  </div>

                  {/* Total Downloaded */}
                  <div className="rounded-lg bg-card border border-border p-3">
                    <div className="text-xs text-muted-foreground mb-1">
                      Downloaded
                    </div>
                    <div className="text-lg font-semibold text-foreground">
                      {formatBytes(progress.overallDownloaded)}
                    </div>
                    {progress.overallTotal > 0 && (
                      <div className="text-xs text-muted-foreground">
                        of {formatBytes(progress.overallTotal)}
                      </div>
                    )}
                  </div>

                  {/* Time Remaining */}
                  <div className="rounded-lg bg-card border border-border p-3">
                    <div className="text-xs text-muted-foreground mb-1">
                      Time Remaining
                    </div>
                    <div className="text-lg font-semibold text-foreground">
                      {isComplete
                        ? 'Done'
                        : formatTime(progress.estimatedTimeRemaining)}
                    </div>
                  </div>
                </div>

                {/* Success Message */}
                {isComplete && (
                  <div className="rounded-lg bg-primary/10 border border-primary/30 p-3">
                    <div className="flex items-center gap-2">
                      <FontIcon
                        type="checkmark-filled"
                        className="w-4 h-4 text-primary"
                      />
                      <div className="text-sm text-foreground">
                        Model downloaded successfully! You can now use it in your
                        project.
                      </div>
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Loading State (no progress yet) */}
            {!progress && !error && isDownloading && (
              <div className="flex flex-col items-center justify-center py-8 gap-3">
                <Loader size={32} className="border-primary" />
                <div className="text-sm text-muted-foreground">
                  Initializing download...
                </div>
              </div>
            )}
          </div>
        </DialogDescription>

        {/* Actions */}
        <div className="flex items-center justify-end gap-2 mt-4">
          {error && (
            <Button
              variant="secondary"
              onClick={() => {
                onOpenChange(false)
              }}
            >
              Close
            </Button>
          )}
          {isDownloading && !error && (
            <Button
              variant="secondary"
              onClick={() => {
                onCancel?.()
                onOpenChange(false)
              }}
            >
              Cancel
            </Button>
          )}
          {isComplete && (
            <Button
              onClick={() => {
                onComplete?.()
                onOpenChange(false)
              }}
            >
              Done
            </Button>
          )}
        </div>
      </DialogContent>
    </Dialog>
  )
}
