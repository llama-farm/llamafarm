import { useState, useCallback } from 'react'
import { Play, Pause } from 'lucide-react'
import { Switch } from '../ui/switch'
import { Selector } from '../ui/selector'
import { Button } from '../ui/button'
import { TTS_MODELS, getVoicesForModel } from '../../types/ml'
import type { TTSModel, VoiceClone } from '../../types/ml'

interface TextToSpeechConfigProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  selectedModel: string
  onModelChange: (modelId: string) => void
  selectedVoice: string
  onVoiceChange: (voiceId: string) => void
  speed: number
  onSpeedChange: (speed: number) => void
  models?: TTSModel[]
  customVoices?: VoiceClone[]
  className?: string
}

export function TextToSpeechConfig({
  enabled,
  onEnabledChange,
  selectedModel,
  onModelChange,
  selectedVoice,
  onVoiceChange,
  speed,
  onSpeedChange,
  models = TTS_MODELS,
  customVoices = [],
  className = '',
}: TextToSpeechConfigProps) {
  const [isPreviewing, setIsPreviewing] = useState(false)
  const currentModel = models.find(m => m.id === selectedModel)

  // Get voices for the selected model + custom voices
  const modelVoices = getVoicesForModel(selectedModel)
  const allVoices = [
    ...modelVoices.map(v => ({
      value: v.id,
      label: v.name,
      description: `${v.gender}, ${v.language.toUpperCase()}`,
    })),
    ...customVoices.map(v => ({
      value: v.id,
      label: v.name,
      description: `Custom, ${v.duration}s`,
    })),
  ]

  const handlePreview = useCallback(() => {
    if (isPreviewing) {
      // Stop preview
      setIsPreviewing(false)
      return
    }

    // Start preview
    setIsPreviewing(true)
    // Simulate audio playback duration
    setTimeout(() => {
      setIsPreviewing(false)
    }, 2000)
  }, [isPreviewing])

  return (
    <div className={`rounded-xl border border-border bg-card/40 p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium">Text-to-Speech</h3>
          {currentModel?.supportsVoiceCloning && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/10 text-primary">
              Voice Cloning
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          <Switch
            checked={enabled}
            onCheckedChange={onEnabledChange}
            aria-label="Enable text-to-speech"
          />
          <span className="text-xs text-muted-foreground">
            {enabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      </div>

      {/* Controls */}
      <div className={`space-y-3 ${!enabled ? 'opacity-50 pointer-events-none' : ''}`}>
        {/* Model selector */}
        <Selector
          value={selectedModel}
          options={models.map(m => ({
            value: m.id,
            label: m.name,
            description: m.description || m.size,
          }))}
          onChange={onModelChange}
          label="Model"
          disabled={!enabled}
        />

        {/* Voice selector with preview */}
        <div className="flex gap-2 items-end">
          <div className="flex-1">
            <Selector
              value={selectedVoice}
              options={allVoices}
              onChange={onVoiceChange}
              label="Voice"
              disabled={!enabled}
            />
          </div>
          <Button
            variant="outline"
            size="icon"
            className="h-9 w-9 mb-0.5"
            onClick={handlePreview}
            disabled={!enabled}
            aria-label={isPreviewing ? 'Stop preview' : 'Preview voice'}
          >
            {isPreviewing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
        </div>

        {/* Speed slider */}
        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="text-xs text-muted-foreground">Speed</label>
            <span className="text-xs text-muted-foreground tabular-nums">{speed.toFixed(1)}x</span>
          </div>
          <input
            type="range"
            min={0.5}
            max={2.0}
            step={0.1}
            value={speed}
            onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            disabled={!enabled}
            className="w-full h-1.5 bg-muted rounded-full appearance-none cursor-pointer
              [&::-webkit-slider-thumb]:appearance-none
              [&::-webkit-slider-thumb]:h-3
              [&::-webkit-slider-thumb]:w-3
              [&::-webkit-slider-thumb]:rounded-full
              [&::-webkit-slider-thumb]:bg-primary
              [&::-webkit-slider-thumb]:cursor-pointer
              disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Speech speed"
          />
          <div className="flex justify-between text-[10px] text-muted-foreground">
            <span>0.5x</span>
            <span>1.0x</span>
            <span>2.0x</span>
          </div>
        </div>

        {/* Model info */}
        {currentModel && (
          <div className="text-xs text-muted-foreground p-2 rounded-lg bg-muted/30">
            <span className="font-medium">{currentModel.name}</span>
            <span className="mx-1.5">•</span>
            <span>{currentModel.size}</span>
            {currentModel.description && (
              <>
                <span className="mx-1.5">•</span>
                <span>{currentModel.description}</span>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
