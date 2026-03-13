import type { Detection } from '../../types'
import { getDetectionColor } from '../../types'
import { detectionIcons } from '../../data/mockDetections'

interface DetectionPinProps {
  detection: Detection
  onClick: () => void
  pulsing?: boolean
}

export function DetectionPin({ detection, onClick, pulsing = false }: DetectionPinProps) {
  const color = getDetectionColor(detection.type)
  const icon = detectionIcons[detection.type]

  return (
    <button
      onClick={onClick}
      className={`
        relative flex items-center justify-center
        w-8 h-8 rounded-full border-2 border-white
        cursor-pointer transition-transform hover:scale-110
        ${pulsing ? 'animate-pulse-detection' : ''}
      `}
      style={{ backgroundColor: color }}
    >
      <span className="text-sm">{icon}</span>

      {/* Flag indicator */}
      {detection.flagged && (
        <span className="absolute -top-2 -right-1 text-xs">🚩</span>
      )}
    </button>
  )
}
