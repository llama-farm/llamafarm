/**
 * Main container for streaming anomaly detection mode
 * Orchestrates demo playback, configuration, and results display
 */

import { useState, useRef, useCallback, useEffect } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import StreamingConfig, { type StreamingConfigState } from './StreamingConfig'
import StreamingStatusPanel from './StreamingStatusPanel'
import StreamingResultsChart from './StreamingResultsChart'
import {
  STREAMING_DATASETS,
  generateStreamingBatch,
} from './sampleData'
import {
  useStreamAnomaly,
  useDeleteStreamingDetector,
  useResetStreamingDetector,
} from '../../hooks/useMLModels'
import type {
  StreamingAnomalyResult,
  StreamingStatus,
  StreamingAnomalyRequest,
} from '../../types/ml'

// Default streaming configuration
const DEFAULT_CONFIG: StreamingConfigState = {
  backend: 'ecod',
  minSamples: 50,
  retrainInterval: 500, // Higher interval = more stable model (retrains less often)
  windowSize: 1000,
  threshold: 0.9, // Higher threshold = fewer anomalies (top 10% of scores)
  contamination: 0.05, // Lower contamination = expects fewer anomalies in training data
  rollingWindows: [5, 10, 20],
  enableRollingFeatures: false,
  includeLags: false,
  lagPeriods: [1, 2, 5],
}

interface Props {
  modelName: string
}

function StreamingModePanel({ modelName }: Props) {
  const { toast } = useToast()

  // Configuration state
  const [config, setConfig] = useState<StreamingConfigState>(DEFAULT_CONFIG)

  // Streaming state
  const [isPlaying, setIsPlaying] = useState(false)
  const [streamIndex, setStreamIndex] = useState(0)
  const [results, setResults] = useState<StreamingAnomalyResult[]>([])
  const [status, setStatus] = useState<StreamingStatus>('collecting')
  const [modelVersion, setModelVersion] = useState(0)
  const [samplesCollected, setSamplesCollected] = useState(0)
  const [samplesUntilReady, setSamplesUntilReady] = useState(config.minSamples)

  // Manual input state
  const [manualInput, setManualInput] = useState('')

  // Refs
  const playbackRef = useRef<number | null>(null)
  const prevConfigRef = useRef<StreamingConfigState>(config)
  const dataset = STREAMING_DATASETS[0] // Factory sensor stream

  // Mutations
  const streamMutation = useStreamAnomaly()
  const deleteMutation = useDeleteStreamingDetector()
  const resetMutation = useResetStreamingDetector()

  // Detector ID
  const detectorId = `streaming-${modelName || 'demo'}`

  // Calculate recent stats
  const recentResults = results.slice(-50)
  const recentAnomalies = recentResults.filter(r => r.is_anomaly === true).length

  // Stream a batch of data
  const streamBatch = useCallback(
    async (batch: Record<string, number>[]) => {
      const request: StreamingAnomalyRequest = {
        model: detectorId,
        data: batch,
        backend: config.backend,
        min_samples: config.minSamples,
        retrain_interval: config.retrainInterval,
        window_size: config.windowSize,
        threshold: config.threshold,
        contamination: config.contamination,
        ...(config.enableRollingFeatures && { rolling_windows: config.rollingWindows }),
        ...(config.includeLags && {
          include_lags: true,
          lag_periods: config.lagPeriods,
        }),
      }

      try {
        const response = await streamMutation.mutateAsync(request)
        setStatus(response.status)
        setModelVersion(response.model_version)
        setSamplesCollected(response.samples_collected)

        // Update results
        setResults(prev => [...prev, ...response.results])

        // Calculate samples until ready from the last result
        const lastResult = response.results[response.results.length - 1]
        if (lastResult) {
          setSamplesUntilReady(lastResult.samples_until_ready)
        }

        return response
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error'
        toast({
          message: `Streaming Error: ${errorMessage}`,
          variant: 'destructive',
          icon: 'alert-triangle',
        })
        setIsPlaying(false)
        return null
      }
    },
    [config, detectorId, streamMutation, toast]
  )

  // Demo playback tick
  const playbackTick = useCallback(async () => {
    const { data } = generateStreamingBatch(dataset, streamIndex, 5) // 5 samples per tick
    setStreamIndex(prev => prev + 5)
    await streamBatch(data)
  }, [dataset, streamIndex, streamBatch])

  // Start/stop playback
  const togglePlayback = useCallback(() => {
    if (isPlaying) {
      if (playbackRef.current) {
        clearInterval(playbackRef.current)
        playbackRef.current = null
      }
      setIsPlaying(false)
    } else {
      setIsPlaying(true)
      // Run immediately then every 500ms (10 samples/sec)
      playbackTick()
      playbackRef.current = window.setInterval(playbackTick, 500)
    }
  }, [isPlaying, playbackTick])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (playbackRef.current) {
        clearInterval(playbackRef.current)
      }
    }
  }, [])

  // Reset local UI state (without API call)
  const resetLocalState = useCallback(() => {
    setResults([])
    setStreamIndex(0)
    setStatus('collecting')
    setModelVersion(0)
    setSamplesCollected(0)
    setSamplesUntilReady(config.minSamples)
  }, [config.minSamples])

  // Auto-reset detector when config changes
  useEffect(() => {
    const prev = prevConfigRef.current
    const configChanged =
      prev.backend !== config.backend ||
      prev.minSamples !== config.minSamples ||
      prev.retrainInterval !== config.retrainInterval ||
      prev.windowSize !== config.windowSize ||
      prev.threshold !== config.threshold ||
      prev.contamination !== config.contamination ||
      prev.enableRollingFeatures !== config.enableRollingFeatures ||
      prev.includeLags !== config.includeLags ||
      JSON.stringify(prev.rollingWindows) !== JSON.stringify(config.rollingWindows) ||
      JSON.stringify(prev.lagPeriods) !== JSON.stringify(config.lagPeriods)

    if (configChanged && samplesCollected > 0) {
      // Config changed and we have data - delete the old detector so next stream creates a fresh one
      deleteMutation.mutate(detectorId, {
        onSettled: () => {
          resetLocalState()
        },
      })
    }

    prevConfigRef.current = config
  }, [config, detectorId, deleteMutation, resetLocalState, samplesCollected])

  // Send manual data
  const sendManualData = useCallback(async () => {
    if (!manualInput.trim()) return

    const values = manualInput
      .split(',')
      .map(v => parseFloat(v.trim()))
      .filter(v => !isNaN(v))

    if (values.length !== 4) {
      toast({
        message: 'Invalid Input: Please enter 4 comma-separated numbers (temperature, humidity, pressure, motor_rpm)',
        variant: 'destructive',
        icon: 'alert-triangle',
      })
      return
    }

    const data = {
      temperature: values[0],
      humidity: values[1],
      pressure: values[2],
      motor_rpm: values[3],
    }

    await streamBatch([data])
    setManualInput('')
  }, [manualInput, streamBatch, toast])

  // Reset detector
  const handleReset = useCallback(async () => {
    if (isPlaying) {
      togglePlayback()
    }

    try {
      await resetMutation.mutateAsync(detectorId)
      resetLocalState()
      toast({
        message: 'Detector Reset: Streaming detector has been reset to initial state',
        icon: 'checkmark-filled',
      })
    } catch {
      // Detector might not exist yet, that's ok
      resetLocalState()
    }
  }, [detectorId, isPlaying, resetMutation, resetLocalState, toast, togglePlayback])

  // Delete detector
  const handleDelete = useCallback(async () => {
    if (isPlaying) {
      togglePlayback()
    }

    try {
      await deleteMutation.mutateAsync(detectorId)
      resetLocalState()
      toast({
        message: 'Detector Deleted: Streaming detector has been deleted',
        icon: 'checkmark-filled',
      })
    } catch {
      // Detector might not exist, that's ok
    }
  }, [deleteMutation, detectorId, isPlaying, resetLocalState, toast, togglePlayback])

  const isLoading = streamMutation.isPending || resetMutation.isPending || deleteMutation.isPending

  return (
    <div className="space-y-6">
      {/* Configuration Section */}
      <section className="space-y-3">
        <h3 className="text-sm font-medium text-muted-foreground">Detector Settings</h3>
        <StreamingConfig config={config} onChange={setConfig} disabled={isPlaying} />
      </section>

      {/* Status Section */}
      <section className="space-y-3">
        <h3 className="text-sm font-medium text-muted-foreground">Status</h3>
        <StreamingStatusPanel
          status={status}
          modelVersion={modelVersion}
          samplesCollected={samplesCollected}
          samplesUntilReady={samplesUntilReady}
          minSamples={config.minSamples}
          recentAnomalies={recentAnomalies}
          recentTotal={recentResults.length}
        />
      </section>

      {/* Results Chart */}
      <section className="space-y-3">
        <h3 className="text-sm font-medium text-muted-foreground">Detection Results</h3>
        <StreamingResultsChart
          results={results}
          threshold={config.threshold}
          maxResults={100}
        />
      </section>

      {/* Demo & Testing Section */}
      <section className="space-y-4 pt-4 border-t">
        <h3 className="text-sm font-medium text-muted-foreground">Demo & Testing</h3>

        {/* Demo playback button */}
        <div className="flex items-center gap-3">
          <Button
            onClick={togglePlayback}
            disabled={isLoading}
            variant={isPlaying ? 'destructive' : 'default'}
            className="gap-2"
          >
            <FontIcon type={isPlaying ? 'close' : 'arrow-right'} className="w-4 h-4" />
            {isPlaying ? 'Stop Demo' : 'Run Sensor Demo'}
          </Button>
          <span className="text-sm text-muted-foreground">
            Simulates factory sensor data with periodic anomalies
          </span>
        </div>

        {/* Manual input */}
        <div className="space-y-2">
          <Label className="text-xs text-muted-foreground">
            Manual input (temperature, humidity, pressure, motor_rpm)
          </Label>
          <div className="flex gap-2">
            <Input
              value={manualInput}
              onChange={e => setManualInput(e.target.value)}
              placeholder="72.5, 45.2, 1013.5, 3000"
              disabled={isLoading}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  sendManualData()
                }
              }}
            />
            <Button
              onClick={sendManualData}
              disabled={isLoading || !manualInput.trim()}
              variant="secondary"
            >
              Send
            </Button>
          </div>
        </div>

        {/* Detector management */}
        <div className="flex items-center justify-between pt-2">
          <span className="text-xs text-muted-foreground">
            Detector: <code className="bg-muted px-1 rounded">{detectorId}</code>
          </span>
          <div className="flex gap-2">
            <Button
              onClick={handleReset}
              disabled={isLoading || isPlaying}
              variant="outline"
              size="sm"
            >
              Reset
            </Button>
            <Button
              onClick={handleDelete}
              disabled={isLoading || isPlaying}
              variant="outline"
              size="sm"
              className="text-destructive hover:text-destructive"
            >
              Delete
            </Button>
          </div>
        </div>
      </section>
    </div>
  )
}

export default StreamingModePanel
