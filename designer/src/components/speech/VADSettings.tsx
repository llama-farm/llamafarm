import { Mic, HelpCircle } from 'lucide-react'
import { Switch } from '../ui/switch'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip'

interface VADSettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  silenceDuration: number
  onSilenceDurationChange: (duration: number) => void
  /** When true, the entire card is disabled (grayed out) - used when STT is disabled */
  sttDisabled?: boolean
  /** When true, VAD is not available (STT-only mode without conversation) */
  vadNotAvailable?: boolean
  className?: string
}

export function VADSettings({
  enabled,
  onEnabledChange,
  silenceDuration,
  onSilenceDurationChange,
  sttDisabled = false,
  vadNotAvailable = false,
  className = '',
}: VADSettingsProps) {
  // When STT is disabled or VAD not available, the card should be grayed out
  const cardDisabled = sttDisabled || vadNotAvailable

  // Determine tooltip text
  const getTooltipText = () => {
    if (sttDisabled) {
      return 'Enable Speech-to-Text to use voice detection.'
    }
    if (vadNotAvailable) {
      return 'Auto-detection requires conversation mode. In transcription-only mode, tap stop when done speaking.'
    }
    return 'When off, responses start only after you stop recording manually.'
  }

  return (
    <div className={`rounded-lg border border-border bg-card/40 p-3 ${cardDisabled ? 'opacity-50' : ''} ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Mic className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-sm font-medium">Voice Detection</h3>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/60 cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-[220px]">
                <p>{getTooltipText()}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div className={`flex items-center gap-2 ${cardDisabled ? 'pointer-events-none' : ''}`}>
          <Switch
            checked={enabled && !vadNotAvailable}
            onCheckedChange={onEnabledChange}
            disabled={cardDisabled}
            aria-label="Enable auto-detection"
          />
          <span className="text-xs text-muted-foreground">
            {enabled && !vadNotAvailable ? 'Auto' : 'Manual'}
          </span>
        </div>
      </div>

      {/* Silence threshold slider - disabled when VAD is off OR when STT is disabled */}
      <div className={!enabled || cardDisabled ? 'opacity-50 pointer-events-none' : ''}>
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
              disabled={!enabled || cardDisabled}
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
      </div>
    </div>
  )
}
