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
    <div className="absolute inset-0 flex items-center justify-center bg-black/70 z-[2000]" onClick={onClose}>
      <div className="bg-surface-raised rounded-2xl w-[420px] max-w-[92vw] overflow-hidden shadow-2xl border border-surface-border" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div
          className="px-4 py-2 flex items-center justify-between border-b border-surface-border"
          style={{ backgroundColor: color + '10' }}
        >
          <div className="flex items-center gap-2">
            <div
              className="w-2.5 h-2.5 rounded-full"
              style={{ backgroundColor: color }}
            />
            <span className="font-semibold text-text-primary text-sm">{label} Detected</span>
          </div>
          <button
            onClick={onClose}
            className="w-10 h-10 flex items-center justify-center rounded-full hover:bg-surface-overlay text-text-secondary text-lg transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Image */}
        <div className="relative">
          <img
            src={detection.imageUrl}
            alt={`${label} detection`}
            className="w-full h-52 object-cover bg-black"
          />

          {/* Confidence badge */}
          <div className="absolute top-2 right-2 px-2 py-1 rounded text-[10px] font-bold font-mono text-text-primary bg-black/60 border border-surface-border">
            {Math.round(detection.confidence * 100)}%
          </div>
        </div>

        {/* Details */}
        <div className="p-4 space-y-2.5">
          <div className="flex justify-between text-sm">
            <span className="text-text-dim">Time</span>
            <span className="text-text-primary font-mono">{formatTime(detection.timestamp)}</span>
          </div>

          <div className="flex justify-between text-sm">
            <span className="text-text-dim">Source</span>
            <span className="text-text-primary">{droneName}</span>
          </div>

          <div className="flex justify-between text-sm">
            <span className="text-text-dim">Coordinates</span>
            <span className="text-text-primary font-mono text-xs">
              {formatCoords(detection.position.lat, detection.position.lng)}
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
