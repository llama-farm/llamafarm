/**
 * Status panel showing streaming detector state and statistics
 */

import { Badge } from '../ui/badge'
import { cn } from '../../lib/utils'
import type { StreamingStatus } from '../../types/ml'

interface Props {
  status: StreamingStatus
  modelVersion: number
  samplesCollected: number
  samplesUntilReady: number
  minSamples: number
  recentAnomalies?: number
  recentTotal?: number
}

function StreamingStatusPanel({
  status,
  modelVersion,
  samplesCollected,
  samplesUntilReady,
  minSamples,
  recentAnomalies = 0,
  recentTotal = 0,
}: Props) {
  const isReady = status === 'ready' || status === 'retraining'
  const coldStartProgress = Math.min(100, (samplesCollected / minSamples) * 100)
  const anomalyRate = recentTotal > 0 ? ((recentAnomalies / recentTotal) * 100).toFixed(1) : '0.0'

  const getStatusColor = () => {
    switch (status) {
      case 'collecting':
        return 'bg-yellow-500'
      case 'ready':
        return 'bg-green-500'
      case 'retraining':
        return 'bg-blue-500'
      default:
        return 'bg-gray-500'
    }
  }

  const getStatusLabel = () => {
    switch (status) {
      case 'collecting':
        return 'Cold Start'
      case 'ready':
        return 'Ready'
      case 'retraining':
        return 'Retraining'
      default:
        return status
    }
  }

  return (
    <div className="space-y-3">
      {/* Status header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={cn('w-2 h-2 rounded-full animate-pulse', getStatusColor())} />
          <span className="text-sm font-medium">{getStatusLabel()}</span>
          {modelVersion > 0 && (
            <Badge variant="outline" className="text-xs">
              v{modelVersion}
            </Badge>
          )}
        </div>
        <span className="text-sm text-muted-foreground">
          {samplesCollected.toLocaleString()} samples
        </span>
      </div>

      {/* Cold start progress bar */}
      {!isReady && (
        <div className="space-y-1">
          <div className="h-2 bg-muted rounded-full overflow-hidden">
            <div
              className="h-full bg-primary transition-all duration-300 ease-out"
              style={{ width: `${coldStartProgress}%` }}
            />
          </div>
          <p className="text-xs text-muted-foreground">
            {samplesUntilReady} more samples until model training
          </p>
        </div>
      )}

      {/* Stats when ready */}
      {isReady && recentTotal > 0 && (
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="bg-muted/50 rounded p-2">
            <div className="text-lg font-semibold">{recentTotal}</div>
            <div className="text-xs text-muted-foreground">Recent</div>
          </div>
          <div className="bg-muted/50 rounded p-2">
            <div className={cn('text-lg font-semibold', recentAnomalies > 0 && 'text-destructive')}>
              {recentAnomalies}
            </div>
            <div className="text-xs text-muted-foreground">Anomalies</div>
          </div>
          <div className="bg-muted/50 rounded p-2">
            <div className="text-lg font-semibold">{anomalyRate}%</div>
            <div className="text-xs text-muted-foreground">Rate</div>
          </div>
        </div>
      )}
    </div>
  )
}

export default StreamingStatusPanel
