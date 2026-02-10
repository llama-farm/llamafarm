import { useState, useEffect } from 'react'
import { Minimize2, X, AlertTriangle, Check, Loader2 } from 'lucide-react'
import { useTaskStatus } from '../../hooks/useAddons'
import { AddonInstallMinimized } from './AddonInstallMinimized'

interface AddonInstallProgressProps {
  taskId: string
  addonName: string
  onComplete: () => void
  onCancel: () => void
}

/**
 * Installation progress panel with minimize capability
 *
 * Shows real-time progress of addon installation with:
 * - Progress bar (0-100%)
 * - Status messages from backend
 * - Error states with retry option
 * - Minimize to bottom-right corner
 */
export function AddonInstallProgress({
  taskId,
  addonName,
  onComplete,
  onCancel,
}: AddonInstallProgressProps) {
  const [minimized, setMinimized] = useState(false)
  const { data: status } = useTaskStatus(taskId)

  // Watch for completion - show success state for 2 seconds before closing
  useEffect(() => {
    if (status?.status === 'completed') {
      // Wait 2 seconds to let user see the success state
      const timer = setTimeout(() => {
        onComplete()
      }, 2000)
      return () => clearTimeout(timer)
    }
  }, [status?.status, onComplete])

  // If minimized, show the minimized indicator
  if (minimized) {
    return (
      <AddonInstallMinimized
        progress={status?.progress || 0}
        message={status?.message || 'Installing...'}
        addonName={addonName}
        onExpand={() => setMinimized(false)}
      />
    )
  }

  // Full progress panel
  return (
    <div
      className="fixed right-0 top-0 bottom-0 w-[440px] z-[70] bg-background border-l border-border/50 shadow-2xl"
      role="dialog"
      aria-modal="true"
      aria-labelledby="progress-panel-title"
    >
      {/* Header */}
      <div className="p-6 border-b border-border/50 bg-gradient-to-b from-muted/30 to-background">
        <div className="flex items-center gap-3 mb-2">
          <h2 id="progress-panel-title" className="text-xl font-semibold text-foreground">
            Installing {addonName}
          </h2>
        </div>
        <p className="text-sm text-muted-foreground">
          {status?.progress === 100 ? 'Installation complete' : 'Please wait while we install the add-ons'}
        </p>
        <button
          onClick={() => setMinimized(true)}
          className="absolute top-5 right-14 rounded-lg p-1.5 opacity-70 hover:opacity-100 hover:bg-accent transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          aria-label="Minimize progress"
        >
          <Minimize2 className="w-5 h-5" />
        </button>
        <button
          onClick={onCancel}
          className="absolute top-5 right-5 rounded-lg p-1.5 opacity-70 hover:opacity-100 hover:bg-accent transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          aria-label="Cancel installation"
        >
          <X className="w-5 h-5" />
        </button>
      </div>

      {/* Progress Content */}
      <div className="p-6 space-y-5">
        {/* Progress Bar */}
        <div className="space-y-3">
          <div className="w-full bg-muted rounded-full h-3 overflow-hidden border border-border/30 shadow-inner">
            <div
              className="bg-gradient-to-r from-primary to-primary/80 h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${status?.progress || 0}%` }}
              role="progressbar"
              aria-valuenow={status?.progress || 0}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-label="Installation progress"
            />
          </div>

          {/* Status Message */}
          <div className="flex items-center gap-2">
            {status?.progress !== undefined && status.progress < 100 && (
              <Loader2 className="w-4 h-4 animate-spin text-primary" />
            )}
            {status?.progress === 100 && (
              <Check className="w-4 h-4 text-green-500" />
            )}
            <p className="text-sm text-foreground font-medium">
              {status?.message || 'Starting installation...'}
            </p>
          </div>

          {/* Progress Percentage */}
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>{status?.progress || 0}% complete</span>
            <span>Est. 3-5 minutes</span>
          </div>
        </div>

        {/* Error State */}
        {status?.error && (
          <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/30 shadow-sm">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-sm font-semibold text-red-600 dark:text-red-400 mb-1">
                  Installation Failed
                </div>
                <div className="text-sm text-red-600 dark:text-red-400 opacity-90 leading-relaxed">
                  {status.error}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Info Box */}
        {!status?.error && (
          <div className="p-4 rounded-lg bg-muted/30 border border-border/50 text-sm">
            <p className="text-foreground/90 leading-relaxed">
              You can minimize this panel and continue working. Services will restart automatically when installation completes.
            </p>
          </div>
        )}

        {/* Progress Steps */}
        <div className="space-y-3 pt-2">
          <div className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
            Installation Steps
          </div>
          <div className="space-y-2.5">
            <ProgressStep
              label="Downloading packages"
              active={status?.progress !== undefined && status.progress > 0 && status.progress < 50}
              completed={status?.progress !== undefined && status.progress >= 50}
            />
            <ProgressStep
              label="Installing packages"
              active={status?.progress !== undefined && status.progress >= 50 && status.progress < 90}
              completed={status?.progress !== undefined && status.progress >= 90}
            />
            <ProgressStep
              label="Restarting services"
              active={status?.progress !== undefined && status.progress >= 90 && status.progress < 100}
              completed={status?.progress === 100}
            />
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="absolute bottom-0 left-0 right-0 p-6 border-t border-border/50 bg-gradient-to-t from-muted/20 to-background">
        {status?.error ? (
          <button
            onClick={onCancel}
            className="w-full py-2.5 px-4 border border-border/50 rounded-lg font-medium hover:bg-accent/50 transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2"
          >
            Close
          </button>
        ) : (
          <button
            onClick={onCancel}
            className="w-full py-2.5 px-4 border border-border/50 rounded-lg font-medium hover:bg-accent/50 transition-all focus:outline-none focus:ring-2 focus:ring-ring focus:ring-offset-2 disabled:opacity-40 disabled:cursor-not-allowed hover:disabled:bg-transparent"
            disabled={status?.progress !== undefined && status.progress > 50}
          >
            Cancel
          </button>
        )}
      </div>
    </div>
  )
}

/**
 * Progress step indicator component
 */
function ProgressStep({
  label,
  active,
  completed,
}: {
  label: string
  active: boolean
  completed: boolean
}) {
  return (
    <div className="flex items-center gap-3">
      <div
        className={`flex items-center justify-center w-6 h-6 rounded-full transition-all ${
          completed
            ? 'bg-green-500/20 border-2 border-green-500'
            : active
            ? 'bg-primary/20 border-2 border-primary'
            : 'bg-muted border-2 border-border'
        }`}
      >
        {completed ? (
          <Check className="w-3 h-3 text-green-500" />
        ) : active ? (
          <Loader2 className="w-3 h-3 text-primary animate-spin" />
        ) : (
          <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/30" />
        )}
      </div>
      <span
        className={`text-sm ${
          completed
            ? 'text-foreground font-medium'
            : active
            ? 'text-foreground font-medium'
            : 'text-muted-foreground'
        }`}
      >
        {label}
      </span>
    </div>
  )
}
