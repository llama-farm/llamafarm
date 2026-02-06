/**
 * Toggle component for switching between batch and streaming modes
 */

import { Badge } from '../ui/badge'
import { cn } from '../../lib/utils'

interface Props {
  isStreamingMode: boolean
  onToggle: (enabled: boolean) => void
  disabled?: boolean
}

function StreamingModeToggle({ isStreamingMode, onToggle, disabled }: Props) {
  return (
    <div className="flex items-center gap-3">
      <button
        type="button"
        role="switch"
        aria-checked={isStreamingMode}
        disabled={disabled}
        onClick={() => onToggle(!isStreamingMode)}
        className={cn(
          'relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2',
          isStreamingMode ? 'bg-primary' : 'bg-muted',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <span
          className={cn(
            'pointer-events-none inline-block h-5 w-5 transform rounded-full bg-white shadow ring-0 transition duration-200 ease-in-out',
            isStreamingMode ? 'translate-x-5' : 'translate-x-0'
          )}
        />
      </button>
      <div className="flex items-center gap-2">
        <span className="text-sm font-medium">Streaming Mode</span>
        {isStreamingMode && (
          <Badge variant="outline" className="text-xs bg-primary/10 text-primary border-primary/20">
            Polars-powered
          </Badge>
        )}
      </div>
    </div>
  )
}

export default StreamingModeToggle
