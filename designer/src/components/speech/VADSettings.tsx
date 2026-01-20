import { Mic } from 'lucide-react'
import { Switch } from '../ui/switch'

interface VADSettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  silenceDuration: number
  onSilenceDurationChange: (duration: number) => void
  className?: string
}

export function VADSettings({
  enabled,
  onEnabledChange,
  silenceDuration,
  onSilenceDurationChange,
  className = '',
}: VADSettingsProps) {
  return (
    <div className={`rounded-lg border border-border bg-card/40 p-3 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Mic className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-sm font-medium">Voice Detection</h3>
        </div>
        <div className="flex items-center gap-2">
          <Switch
            checked={enabled}
            onCheckedChange={onEnabledChange}
            aria-label="Enable auto-detection"
          />
          <span className="text-xs text-muted-foreground">
            {enabled ? 'Auto' : 'Manual'}
          </span>
        </div>
      </div>

      {/* Silence threshold slider - only shown when enabled */}
      <div className={`space-y-2 ${!enabled ? 'opacity-50 pointer-events-none' : ''}`}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 flex-1">
            <span className="text-xs text-muted-foreground whitespace-nowrap">Fast</span>
            <input
              type="range"
              min={0.1}
              max={2.0}
              step={0.1}
              value={silenceDuration}
              onChange={(e) => onSilenceDurationChange(parseFloat(e.target.value))}
              disabled={!enabled}
              className="flex-1 h-1 bg-muted rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-2.5
                [&::-webkit-slider-thumb]:w-2.5
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-primary
                [&::-webkit-slider-thumb]:cursor-pointer
                disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Silence detection threshold"
            />
            <span className="text-xs text-muted-foreground whitespace-nowrap">Patient</span>
          </div>
          <span className="text-xs text-muted-foreground tabular-nums w-10 text-right">
            {silenceDuration.toFixed(1)}s
          </span>
        </div>
        <p className="text-[10px] text-muted-foreground leading-tight">
          {enabled
            ? 'How long to wait after you stop speaking before responding'
            : 'Click stop to end recording manually'}
        </p>
      </div>
    </div>
  )
}
