import type { Detection } from '../../types'
import { getDetectionColor } from '../../types'
import { detectionLabels } from '../../data/mockDetections'

interface DetectionDetailProps {
  detection: Detection
  droneName: string
  onClose: () => void
  onWatchFeed: () => void
  onFlag: () => void
}

export function DetectionDetail({
  detection,
  droneName,
  onClose,
  onWatchFeed,
  onFlag
}: DetectionDetailProps) {
  const color = getDetectionColor(detection.type)
  const label = detectionLabels[detection.type]

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const formatCoords = (lat: number, lng: number) => {
    const latDir = lat >= 0 ? 'N' : 'S'
    const lngDir = lng >= 0 ? 'E' : 'W'
    return `${Math.abs(lat).toFixed(5)}°${latDir}, ${Math.abs(lng).toFixed(5)}°${lngDir}`
  }

  return (
    <div className="absolute inset-0 flex items-center justify-center bg-black/60 z-[2000]">
      <div className="bg-surface-raised rounded-2xl w-80 max-w-[90vw] overflow-hidden shadow-2xl border border-surface-border">
        {/* Header */}
        <div
          className="px-4 py-3 flex items-center justify-between"
          style={{ backgroundColor: color + '20' }}
        >
          <div className="flex items-center gap-2">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="font-semibold text-white">{label} Detected</span>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-full hover:bg-white/10 text-gray-400"
          >
            ✕
          </button>
        </div>

        {/* Image */}
        <div className="relative">
          <img
            src={detection.imageUrl}
            alt={`${label} detection`}
            className="w-full h-48 object-cover bg-surface-overlay"
          />

          {/* Confidence badge */}
          <div
            className="absolute top-2 right-2 px-2 py-1 rounded text-xs font-bold text-white"
            style={{ backgroundColor: color }}
          >
            {Math.round(detection.confidence * 100)}%
          </div>

          {/* Flagged indicator */}
          {detection.flagged && (
            <div className="absolute top-2 left-2 px-2 py-1 rounded text-xs font-bold text-white bg-amber-500 flex items-center gap-1">
              🚩 Flagged
            </div>
          )}
        </div>

        {/* Details */}
        <div className="p-4 space-y-3">
          {/* Time & Drone */}
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Time</span>
            <span className="text-white font-mono">{formatTime(detection.timestamp)}</span>
          </div>

          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Source</span>
            <span className="text-white">{droneName}</span>
          </div>

          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Coordinates</span>
            <span className="text-white font-mono text-xs">
              {formatCoords(detection.position.lat, detection.position.lng)}
            </span>
          </div>

          {/* Actions */}
          <div className="pt-2 flex gap-2">
            <button
              onClick={onWatchFeed}
              className="flex-1 bg-blue-600 hover:bg-blue-500 text-white font-medium py-3 rounded-lg transition-colors"
            >
              Watch Feed
            </button>

            {!detection.flagged && (
              <button
                onClick={onFlag}
                className="px-4 bg-amber-600 hover:bg-amber-500 text-white font-medium py-3 rounded-lg transition-colors"
              >
                🚩 Flag
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
