import { useMemo } from 'react'

interface RouteResultProps {
  capability?: string
  confidence?: number
  targetNode?: string
  explanation?: string
  isLoading?: boolean
}

const RouteResult = ({
  capability,
  confidence,
  targetNode,
  explanation,
  isLoading = false,
}: RouteResultProps) => {
  // Color coding based on confidence
  const confidenceColor = useMemo(() => {
    if (!confidence) return 'text-muted-foreground'
    if (confidence >= 0.8) return 'text-green-500 dark:text-green-400'
    if (confidence >= 0.6) return 'text-yellow-500 dark:text-yellow-400'
    return 'text-red-500 dark:text-red-400'
  }, [confidence])

  const confidenceBg = useMemo(() => {
    if (!confidence) return 'bg-muted'
    if (confidence >= 0.8) return 'bg-green-500/10'
    if (confidence >= 0.6) return 'bg-yellow-500/10'
    return 'bg-red-500/10'
  }, [confidence])

  if (isLoading) {
    return (
      <div className="rounded-lg p-6 bg-card border border-border animate-pulse">
        <div className="h-6 bg-muted rounded w-1/3 mb-4"></div>
        <div className="h-4 bg-muted rounded w-2/3 mb-2"></div>
        <div className="h-4 bg-muted rounded w-1/2"></div>
      </div>
    )
  }

  if (!capability) {
    return (
      <div className="rounded-lg p-6 bg-card border border-border">
        <p className="text-muted-foreground text-center">
          No routing result yet. Enter an intent above and click Route.
        </p>
      </div>
    )
  }

  return (
    <div className="rounded-lg p-6 bg-card border border-border space-y-4">
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-2">
          Routing Result
        </h3>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <p className="text-sm text-muted-foreground mb-1">Matched Capability</p>
          <p className="text-base font-medium text-foreground">{capability}</p>
        </div>

        <div>
          <p className="text-sm text-muted-foreground mb-1">Target Node</p>
          <p className="text-base font-medium text-foreground">
            {targetNode || 'Unknown'}
          </p>
        </div>
      </div>

      {confidence !== undefined && (
        <div>
          <p className="text-sm text-muted-foreground mb-2">Confidence Score</p>
          <div className="flex items-center gap-3">
            <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden">
              <div
                className={`h-full ${confidenceBg} transition-all duration-500`}
                style={{ width: `${confidence * 100}%` }}
              />
            </div>
            <span className={`text-lg font-bold ${confidenceColor} min-w-[4rem] text-right`}>
              {(confidence * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      )}

      {explanation && (
        <div>
          <p className="text-sm text-muted-foreground mb-1">Why this match?</p>
          <p className="text-sm text-foreground bg-muted/30 p-3 rounded">
            {explanation}
          </p>
        </div>
      )}
    </div>
  )
}

export default RouteResult
