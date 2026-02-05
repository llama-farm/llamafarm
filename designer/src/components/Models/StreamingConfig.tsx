/**
 * Collapsible configuration panel for Polars streaming features
 */

import { useState } from 'react'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Checkbox } from '../ui/checkbox'
import { Badge } from '../ui/badge'
import { Selector } from '../ui/selector'
import FontIcon from '../../common/FontIcon'
import { cn } from '../../lib/utils'
import type { AnomalyBackend } from '../../types/ml'

// Backend options - match AnomalyModel.tsx
const BACKEND_OPTIONS = [
  { value: 'ecod', label: 'ECOD (Recommended)' },
  { value: 'hbos', label: 'HBOS' },
  { value: 'copod', label: 'COPOD' },
  { value: 'isolation_forest', label: 'Isolation Forest' },
  { value: 'local_outlier_factor', label: 'Local Outlier Factor' },
  { value: 'loda', label: 'LODA (Streaming)' },
]

export interface StreamingConfigState {
  backend: AnomalyBackend
  minSamples: number
  retrainInterval: number
  windowSize: number
  threshold: number
  contamination: number
  // Polars features
  rollingWindows: number[]
  enableRollingFeatures: boolean
  includeLags: boolean
  lagPeriods: number[]
}

interface Props {
  config: StreamingConfigState
  onChange: (config: StreamingConfigState) => void
  disabled?: boolean
}

function StreamingConfig({ config, onChange, disabled }: Props) {
  const [isPolarsExpanded, setIsPolarsExpanded] = useState(false)

  // Local state for text inputs to allow typing without immediate parsing
  const [rollingWindowsText, setRollingWindowsText] = useState(config.rollingWindows.join(', '))
  const [lagPeriodsText, setLagPeriodsText] = useState(config.lagPeriods.join(', '))

  // Local state for number inputs to allow clearing and retyping
  const [minSamplesText, setMinSamplesText] = useState(String(config.minSamples))
  const [retrainIntervalText, setRetrainIntervalText] = useState(String(config.retrainInterval))
  const [windowSizeText, setWindowSizeText] = useState(String(config.windowSize))
  const [thresholdText, setThresholdText] = useState(String(config.threshold))

  // Calculate total feature count
  const baseFeatures = 4 // temperature, humidity, pressure, motor_rpm
  const rollingFeatureCount = config.enableRollingFeatures
    ? baseFeatures * config.rollingWindows.length * 4 // 4 stats: mean, std, min, max
    : 0
  const lagFeatureCount = config.includeLags
    ? baseFeatures * config.lagPeriods.length
    : 0
  const totalFeatures = baseFeatures + rollingFeatureCount + lagFeatureCount

  const updateConfig = (partial: Partial<StreamingConfigState>) => {
    onChange({ ...config, ...partial })
  }

  const parseNumberArray = (value: string): number[] => {
    return value
      .split(',')
      .map(s => parseInt(s.trim(), 10))
      .filter(n => !isNaN(n) && n > 0)
  }

  const handleRollingWindowsBlur = () => {
    const parsed = parseNumberArray(rollingWindowsText)
    if (parsed.length > 0) {
      updateConfig({ rollingWindows: parsed })
      setRollingWindowsText(parsed.join(', '))
    }
  }

  const handleLagPeriodsBlur = () => {
    const parsed = parseNumberArray(lagPeriodsText)
    if (parsed.length > 0) {
      updateConfig({ lagPeriods: parsed })
      setLagPeriodsText(parsed.join(', '))
    }
  }

  const handleMinSamplesBlur = () => {
    const parsed = parseInt(minSamplesText, 10)
    if (!isNaN(parsed) && parsed >= 10) {
      updateConfig({ minSamples: Math.min(parsed, 500) })
      setMinSamplesText(String(Math.min(parsed, 500)))
    } else {
      setMinSamplesText(String(config.minSamples))
    }
  }

  const handleRetrainIntervalBlur = () => {
    const parsed = parseInt(retrainIntervalText, 10)
    if (!isNaN(parsed) && parsed >= 10) {
      updateConfig({ retrainInterval: Math.min(parsed, 10000) })
      setRetrainIntervalText(String(Math.min(parsed, 10000)))
    } else {
      setRetrainIntervalText(String(config.retrainInterval))
    }
  }

  const handleWindowSizeBlur = () => {
    const parsed = parseInt(windowSizeText, 10)
    if (!isNaN(parsed) && parsed >= 100) {
      updateConfig({ windowSize: Math.min(parsed, 100000) })
      setWindowSizeText(String(Math.min(parsed, 100000)))
    } else {
      setWindowSizeText(String(config.windowSize))
    }
  }

  const handleThresholdBlur = () => {
    const parsed = parseFloat(thresholdText)
    if (!isNaN(parsed) && parsed >= 0.5 && parsed <= 0.99) {
      updateConfig({ threshold: parsed })
      setThresholdText(String(parsed))
    } else {
      setThresholdText(String(config.threshold))
    }
  }

  return (
    <div className="space-y-4">
      {/* Basic detector settings */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label>Backend</Label>
          <Selector
            value={config.backend}
            onChange={(value: string) => updateConfig({ backend: value as AnomalyBackend })}
            options={BACKEND_OPTIONS}
            placeholder="Select backend"
            disabled={disabled}
          />
        </div>
        <div className="space-y-2">
          <Label>Threshold</Label>
          <Input
            type="number"
            value={thresholdText}
            onChange={e => setThresholdText(e.target.value)}
            onBlur={handleThresholdBlur}
            min={0.7}
            max={0.99}
            step={0.05}
            disabled={disabled}
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="space-y-2">
          <Label>Cold start samples</Label>
          <Input
            type="number"
            value={minSamplesText}
            onChange={e => setMinSamplesText(e.target.value)}
            onBlur={handleMinSamplesBlur}
            min={10}
            max={500}
            disabled={disabled}
          />
        </div>
        <div className="space-y-2">
          <Label>Retrain interval</Label>
          <Input
            type="number"
            value={retrainIntervalText}
            onChange={e => setRetrainIntervalText(e.target.value)}
            onBlur={handleRetrainIntervalBlur}
            min={10}
            max={10000}
            disabled={disabled}
          />
        </div>
        <div className="space-y-2">
          <Label>Window size</Label>
          <Input
            type="number"
            value={windowSizeText}
            onChange={e => setWindowSizeText(e.target.value)}
            onBlur={handleWindowSizeBlur}
            min={100}
            max={100000}
            disabled={disabled}
          />
        </div>
      </div>

      {/* Polars Feature Engineering - Collapsible */}
      <div className="border rounded-lg">
        <button
          type="button"
          onClick={() => setIsPolarsExpanded(!isPolarsExpanded)}
          className="w-full flex items-center justify-between p-3 hover:bg-muted/50 transition-colors"
          disabled={disabled}
        >
          <div className="flex items-center gap-2">
            <FontIcon
              type={isPolarsExpanded ? 'chevron-down' : 'arrow-right'}
              className="w-4 h-4 text-muted-foreground shrink-0"
            />
            <span className="font-medium text-sm">Polars Feature Engineering</span>
            {(config.enableRollingFeatures || config.includeLags) && (
              <Badge variant="secondary" className="text-xs">
                {totalFeatures} features
              </Badge>
            )}
          </div>
        </button>

        <div
          className={cn(
            'overflow-hidden transition-all duration-200',
            isPolarsExpanded
              ? 'max-h-[600px] opacity-100'
              : 'max-h-0 opacity-0 pointer-events-none'
          )}
        >
          <div className="p-4 pt-0 space-y-4 border-t">
            <p className="text-xs text-muted-foreground">
              Enhance anomaly detection by computing additional features from your time-series data using Polars DataFrames.
            </p>

            {/* Rolling Window Statistics */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="rolling-features"
                  checked={config.enableRollingFeatures}
                  onCheckedChange={checked =>
                    updateConfig({ enableRollingFeatures: checked === true })
                  }
                  disabled={disabled}
                />
                <Label htmlFor="rolling-features" className="text-sm cursor-pointer">
                  Rolling window statistics
                </Label>
              </div>
              <p className="ml-6 text-xs text-muted-foreground">
                Calculates mean, std, min, and max over sliding windows of recent samples.
                Helps detect gradual drifts, sudden spikes, and volatility changes.
                {config.enableRollingFeatures && config.rollingWindows.length > 0 && (
                  <span className="block mt-1 text-foreground/70">
                    Example: <code className="bg-muted px-1 rounded">temperature_rolling_mean_{config.rollingWindows[0]}</code> =
                    average of last {config.rollingWindows[0]} temperature readings
                  </span>
                )}
              </p>
              {config.enableRollingFeatures && (
                <div className="ml-6 space-y-1">
                  <Label className="text-xs text-muted-foreground">
                    Window sizes (number of samples to look back)
                  </Label>
                  <Input
                    value={rollingWindowsText}
                    onChange={e => setRollingWindowsText(e.target.value)}
                    onBlur={handleRollingWindowsBlur}
                    placeholder="5, 10, 20"
                    className="text-sm"
                    disabled={disabled}
                  />
                  <p className="text-xs text-muted-foreground">
                    Creates {config.rollingWindows.length * 4} features per sensor ({config.rollingWindows.length} windows × 4 stats)
                  </p>
                </div>
              )}
            </div>

            {/* Lag Features */}
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Checkbox
                  id="lag-features"
                  checked={config.includeLags}
                  onCheckedChange={checked => updateConfig({ includeLags: checked === true })}
                  disabled={disabled}
                />
                <Label htmlFor="lag-features" className="text-sm cursor-pointer">
                  Lag features
                </Label>
              </div>
              <p className="ml-6 text-xs text-muted-foreground">
                Includes previous values as features, allowing the model to compare current readings
                against historical values. Useful for detecting sudden changes or cyclic patterns.
                {config.includeLags && config.lagPeriods.length > 0 && (
                  <span className="block mt-1 text-foreground/70">
                    Example: <code className="bg-muted px-1 rounded">temperature_lag_{config.lagPeriods[0]}</code> =
                    temperature from {config.lagPeriods[0]} sample{config.lagPeriods[0] > 1 ? 's' : ''} ago
                  </span>
                )}
              </p>
              {config.includeLags && (
                <div className="ml-6 space-y-1">
                  <Label className="text-xs text-muted-foreground">
                    Lag periods (how many samples back to look)
                  </Label>
                  <Input
                    value={lagPeriodsText}
                    onChange={e => setLagPeriodsText(e.target.value)}
                    onBlur={handleLagPeriodsBlur}
                    placeholder="1, 2, 5"
                    className="text-sm"
                    disabled={disabled}
                  />
                  <p className="text-xs text-muted-foreground">
                    Creates {config.lagPeriods.length} features per sensor
                  </p>
                </div>
              )}
            </div>

            {/* Feature count summary */}
            {(config.enableRollingFeatures || config.includeLags) && (
              <div className="pt-2 border-t">
                <p className="text-xs text-muted-foreground">
                  <span className="font-medium text-foreground">Total features per sample:</span>{' '}
                  {baseFeatures} base + {rollingFeatureCount} rolling + {lagFeatureCount} lag = {totalFeatures} features
                </p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default StreamingConfig
