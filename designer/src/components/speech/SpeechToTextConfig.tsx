import { Switch } from '../ui/switch'
import { Selector } from '../ui/selector'
import { Checkbox } from '../ui/checkbox'
import { STT_MODELS, STT_LANGUAGES } from '../../types/ml'
import type { STTModel } from '../../types/ml'

interface SpeechToTextConfigProps {
  enabled: boolean
  onEnabledChange: (enabled: boolean) => void
  selectedModel: string
  onModelChange: (modelId: string) => void
  selectedLanguage: string
  onLanguageChange: (languageCode: string) => void
  wordTimestamps: boolean
  onWordTimestampsChange: (enabled: boolean) => void
  models?: STTModel[]
  className?: string
}

export function SpeechToTextConfig({
  enabled,
  onEnabledChange,
  selectedModel,
  onModelChange,
  selectedLanguage,
  onLanguageChange,
  wordTimestamps,
  onWordTimestampsChange,
  models = STT_MODELS,
  className = '',
}: SpeechToTextConfigProps) {
  const currentModel = models.find(m => m.id === selectedModel)

  return (
    <div className={`rounded-lg border border-border bg-card/40 p-3 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-medium">Speech-to-Text</h3>
        <div className="flex items-center gap-2">
          <Switch
            checked={enabled}
            onCheckedChange={onEnabledChange}
            aria-label="Enable speech-to-text"
          />
          <span className="text-xs text-muted-foreground">
            {enabled ? 'Enabled' : 'Disabled'}
          </span>
        </div>
      </div>

      {/* Controls - more compact */}
      <div className={`space-y-2 ${!enabled ? 'opacity-50 pointer-events-none' : ''}`}>
        {/* Model and Language on same row */}
        <div className="grid grid-cols-2 gap-2">
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
          <Selector
            value={selectedLanguage}
            options={STT_LANGUAGES.map(l => ({
              value: l.code,
              label: l.name,
            }))}
            onChange={onLanguageChange}
            label="Language"
            disabled={!enabled}
          />
        </div>

        {/* Word timestamps and model info on same row */}
        <div className="flex items-center justify-between">
          <label className="flex items-center gap-1.5 cursor-pointer">
            <Checkbox
              checked={wordTimestamps}
              onCheckedChange={(checked) => onWordTimestampsChange(checked === true)}
              disabled={!enabled}
            />
            <span className="text-xs text-muted-foreground">Word timestamps</span>
          </label>
          {currentModel && (
            <span className="text-xs text-muted-foreground">
              {currentModel.size} • {currentModel.description}
            </span>
          )}
        </div>
      </div>
    </div>
  )
}
