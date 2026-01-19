import { Switch } from '../ui/switch'
import { Selector } from '../ui/selector'
import { Checkbox } from '../ui/checkbox'
import { ModelStatusBadge } from './ModelStatusBadge'
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
          {currentModel && <ModelStatusBadge status={currentModel.status} progress={currentModel.progress} />}
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
              description: `${m.size}${m.status === 'downloading' ? ` (${m.progress}%)` : ''}`,
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

        {/* Download prompt for not-downloaded models */}
        {currentModel && currentModel.status === 'not_downloaded' && enabled && (
          <div className="flex items-center justify-between p-2 rounded-lg bg-muted/50 text-sm">
            <span className="text-muted-foreground">
              Model not downloaded ({currentModel.size})
            </span>
            <button
              className="text-primary hover:underline text-sm"
              onClick={() => {
                // Mock: In real implementation, this would trigger download
                console.log('Download model:', currentModel.id)
              }}
            >
              Download
            </button>
          </div>
        )}

        {/* Download progress for downloading models */}
        {currentModel && currentModel.status === 'downloading' && (
          <div className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Downloading {currentModel.name}...</span>
              <span className="text-muted-foreground">{currentModel.progress}%</span>
            </div>
            <div className="h-1.5 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full bg-primary rounded-full transition-all"
                style={{ width: `${currentModel.progress}%` }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
