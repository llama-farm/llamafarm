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
    <div className={`rounded-xl border border-border bg-card/40 p-4 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-medium">Speech-to-Text</h3>
        </div>
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

      {/* Controls */}
      <div className={`space-y-3 ${!enabled ? 'opacity-50 pointer-events-none' : ''}`}>
        {/* Model selector */}
        <div className="grid grid-cols-2 gap-3">
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

          {/* Language selector */}
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

        {/* Word timestamps toggle */}
        <label className="flex items-center gap-2 cursor-pointer">
          <Checkbox
            checked={wordTimestamps}
            onCheckedChange={(checked) => onWordTimestampsChange(checked === true)}
            disabled={!enabled}
          />
          <span className="text-sm text-muted-foreground">Word timestamps</span>
        </label>

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
