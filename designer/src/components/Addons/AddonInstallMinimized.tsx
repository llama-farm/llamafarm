import { Maximize2, Loader2 } from 'lucide-react'

interface AddonInstallMinimizedProps {
  progress: number
  message: string
  addonName: string
  onExpand: () => void
}

/**
 * Minimized installation progress indicator
 *
 * Shows at bottom-right corner when user minimizes the full progress panel.
 * Displays progress bar and current status in compact format.
 */
export function AddonInstallMinimized({
  progress,
  message,
  addonName,
  onExpand,
}: AddonInstallMinimizedProps) {
  return (
    <div
      className="fixed bottom-6 right-6 z-[70] w-80 bg-background border border-border/50 shadow-2xl rounded-xl p-4 animate-in slide-in-from-bottom-4 fade-in-0 duration-300"
      role="status"
      aria-live="polite"
      aria-label="Installation progress indicator"
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 animate-spin text-primary" />
          <p className="text-sm font-semibold text-foreground">
            Installing {addonName}
          </p>
        </div>
        <button
          onClick={onExpand}
          className="rounded-lg p-1.5 opacity-70 hover:opacity-100 hover:bg-accent transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          aria-label="Expand installation progress"
        >
          <Maximize2 className="w-4 h-4" />
        </button>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-muted rounded-full h-2.5 overflow-hidden mb-2.5 border border-border/30 shadow-inner">
        <div
          className="bg-gradient-to-r from-primary to-primary/80 h-2.5 rounded-full transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>

      {/* Status Message */}
      <p className="text-xs text-muted-foreground truncate" title={message}>
        {message}
      </p>

      {/* Progress Percentage */}
      <div className="flex items-center justify-between mt-2">
        <span className="text-xs text-muted-foreground/80">
          {progress}% complete
        </span>
        <span className="text-xs text-muted-foreground/60">
          Est. {Math.max(1, Math.ceil((100 - progress) / 20))} min
        </span>
      </div>
    </div>
  )
}
