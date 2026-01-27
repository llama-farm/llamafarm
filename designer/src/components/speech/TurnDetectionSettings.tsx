import { Mic, HelpCircle, ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'
import { Switch } from '../ui/switch'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '../ui/tooltip'

interface TurnDetectionSettingsProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  baseSilenceDuration: number
  onBaseSilenceDurationChange: (duration: number) => void
  thinkingSilenceDuration: number
  onThinkingSilenceDurationChange: (duration: number) => void
  maxSilenceDuration: number
  onMaxSilenceDurationChange: (duration: number) => void
  /** When true, the entire card is disabled (grayed out) - used when STT is disabled */
  sttDisabled?: boolean
  className?: string
}

export function TurnDetectionSettings({
  enabled,
  onEnabledChange,
  baseSilenceDuration,
  onBaseSilenceDurationChange,
  thinkingSilenceDuration,
  onThinkingSilenceDurationChange,
  maxSilenceDuration,
  onMaxSilenceDurationChange,
  sttDisabled = false,
  className = '',
}: TurnDetectionSettingsProps) {
  const [expanded, setExpanded] = useState(false)

  // When STT is disabled, the card should be grayed out
  const cardDisabled = sttDisabled

  // Determine tooltip text
  const getTooltipText = () => {
    if (sttDisabled) {
      return 'Enable Speech-to-Text to use turn detection.'
    }
    return 'Smart turn detection uses linguistic analysis to determine when you\'ve finished speaking. When off, tap stop manually.'
  }

  return (
    <div className={`rounded-lg border border-border bg-card/40 p-3 ${cardDisabled ? 'opacity-50' : ''} ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <Mic className="h-4 w-4 text-muted-foreground" />
          <h3 className="text-sm font-medium">Turn Detection</h3>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/60 cursor-help" />
              </TooltipTrigger>
              <TooltipContent side="top" className="max-w-[250px]">
                <p>{getTooltipText()}</p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
        <div className={`flex items-center gap-2 ${cardDisabled ? 'pointer-events-none' : ''}`}>
          <Switch
            checked={enabled}
            onCheckedChange={onEnabledChange}
            disabled={cardDisabled}
            aria-label="Enable turn detection"
          />
          <span className="text-xs text-muted-foreground">
            {enabled ? 'Auto' : 'Manual'}
          </span>
        </div>
      </div>

      {/* Base silence threshold - the main slider, always visible when enabled */}
      <div className={!enabled || cardDisabled ? 'opacity-50 pointer-events-none' : ''}>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 flex-1">
            <span className="text-xs text-muted-foreground whitespace-nowrap">Fast</span>
            <input
              type="range"
              min={0.1}
              max={2.0}
              step={0.1}
              value={baseSilenceDuration}
              onChange={(e) => onBaseSilenceDurationChange(parseFloat(e.target.value))}
              disabled={!enabled || cardDisabled}
              className="flex-1 h-1 bg-muted rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-2.5
                [&::-webkit-slider-thumb]:w-2.5
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-primary
                [&::-webkit-slider-thumb]:cursor-pointer
                disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label="Base silence threshold"
            />
            <span className="text-xs text-muted-foreground whitespace-nowrap">Patient</span>
          </div>
          <span className="text-xs text-muted-foreground tabular-nums w-10 text-right">
            {baseSilenceDuration.toFixed(1)}s
          </span>
        </div>
      </div>

      {/* Advanced settings toggle */}
      {enabled && !cardDisabled && (
        <button
          onClick={() => setExpanded(!expanded)}
          className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          {expanded ? (
            <ChevronDown className="h-3 w-3" />
          ) : (
            <ChevronRight className="h-3 w-3" />
          )}
          Advanced timing
        </button>
      )}

      {/* Advanced settings - collapsible */}
      {enabled && !cardDisabled && expanded && (
        <div className="mt-3 pt-3 border-t border-border/50 space-y-3">
          {/* Thinking pause slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-muted-foreground">Thinking pause</span>
              <span className="text-xs text-muted-foreground tabular-nums">
                {thinkingSilenceDuration.toFixed(1)}s
              </span>
            </div>
            <input
              type="range"
              min={0.3}
              max={5.0}
              step={0.1}
              value={thinkingSilenceDuration}
              onChange={(e) => onThinkingSilenceDurationChange(parseFloat(e.target.value))}
              className="w-full h-1 bg-muted rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-2.5
                [&::-webkit-slider-thumb]:w-2.5
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-primary
                [&::-webkit-slider-thumb]:cursor-pointer"
              aria-label="Thinking pause duration"
            />
            <p className="text-[10px] text-muted-foreground/70 mt-0.5">
              Wait time when sentence seems incomplete
            </p>
          </div>

          {/* Max silence slider */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-muted-foreground">Max silence</span>
              <span className="text-xs text-muted-foreground tabular-nums">
                {maxSilenceDuration.toFixed(1)}s
              </span>
            </div>
            <input
              type="range"
              min={0.5}
              max={10.0}
              step={0.5}
              value={maxSilenceDuration}
              onChange={(e) => onMaxSilenceDurationChange(parseFloat(e.target.value))}
              className="w-full h-1 bg-muted rounded-full appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:h-2.5
                [&::-webkit-slider-thumb]:w-2.5
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:bg-primary
                [&::-webkit-slider-thumb]:cursor-pointer"
              aria-label="Maximum silence duration"
            />
            <p className="text-[10px] text-muted-foreground/70 mt-0.5">
              Hard timeout regardless of sentence completeness
            </p>
          </div>
        </div>
      )}
    </div>
  )
}
