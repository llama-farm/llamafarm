/**
 * CSS-only bar chart showing streaming anomaly detection results
 */

import { cn } from '../../lib/utils'
import type { StreamingAnomalyResult } from '../../types/ml'

interface Props {
  results: StreamingAnomalyResult[]
  threshold: number
  maxResults?: number
}

function StreamingResultsChart({ results, threshold, maxResults = 100 }: Props) {
  // Take the most recent results
  const displayResults = results.slice(-maxResults)

  if (displayResults.length === 0) {
    return (
      <div className="h-32 flex items-center justify-center text-muted-foreground text-sm">
        No results yet
      </div>
    )
  }

  // Find max score for scaling (at least threshold to show the line properly)
  const maxScore = Math.max(
    threshold,
    ...displayResults.map(r => r.score ?? 0)
  )

  return (
    <div className="space-y-2">
      {/* Chart container */}
      <div className="relative h-32 bg-muted/30 rounded border">
        {/* Threshold line */}
        <div
          className="absolute left-0 right-0 border-t-2 border-dashed border-yellow-500/70 z-10"
          style={{ bottom: `${(threshold / maxScore) * 100}%` }}
        >
          <span className="absolute right-1 -top-4 text-[10px] text-yellow-600 bg-background px-1 rounded">
            threshold
          </span>
        </div>

        {/* Bars */}
        <div className="absolute inset-0 flex items-end justify-between p-1">
          {displayResults.map((result, idx) => {
            const score = result.score ?? 0
            const height = maxScore > 0 ? (score / maxScore) * 100 : 0
            const isAnomaly = result.is_anomaly === true

            return (
              <div
                key={`${result.index}-${idx}`}
                className={cn(
                  'flex-1 min-w-[1px] transition-all duration-150',
                  isAnomaly
                    ? 'bg-destructive hover:bg-destructive/80'
                    : 'bg-primary/60 hover:bg-primary/80',
                  result.status === 'collecting' && 'bg-muted'
                )}
                style={{ height: `${Math.max(2, height)}%` }}
                title={`Index: ${result.index}\nScore: ${score.toFixed(4)}\n${isAnomaly ? 'ANOMALY' : 'Normal'}`}
              />
            )
          })}
        </div>
      </div>

      {/* Legend */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-primary/60 rounded-sm" />
            <span>Normal</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="w-3 h-3 bg-destructive rounded-sm" />
            <span>Anomaly</span>
          </div>
        </div>
        <span>Last {displayResults.length} points</span>
      </div>
    </div>
  )
}

export default StreamingResultsChart
