import type { Detection, DetectionType } from '../types'

// Pre-defined detection scenarios for simulation
export interface DetectionScenario {
  type: DetectionType
  confidence: number
  offsetLat: number
  offsetLng: number
  delayMs: number
}

// FAST detection sequence for demo - per PASS2.md timing
// t+0s: Drone launches
// t+5s: First detection (green, animal)
// t+10s: Second detection (yellow, vehicle)
// t+15s: Third detection (red, person - PULSES + alert toast)
// t+30s: Battery drops to 18%
export const detectionScenarios: DetectionScenario[] = [
  // t+5s - Animal (green)
  { type: 'animal', confidence: 0.78, offsetLat: 0.002, offsetLng: 0.003, delayMs: 5000 },

  // t+10s - Vehicle (yellow)
  { type: 'vehicle', confidence: 0.92, offsetLat: 0.003, offsetLng: -0.002, delayMs: 10000 },

  // t+15s - Person (red) - THIS IS THE KEY ONE - triggers alert
  { type: 'person', confidence: 0.87, offsetLat: 0.004, offsetLng: -0.001, delayMs: 15000 },

  // Additional detections for continued engagement
  { type: 'animal', confidence: 0.71, offsetLat: 0.005, offsetLng: 0.002, delayMs: 22000 },
  { type: 'vehicle', confidence: 0.89, offsetLat: 0.001, offsetLng: -0.003, delayMs: 28000 },
]

// Generate a detection from a scenario
export function createDetection(
  scenario: DetectionScenario,
  dronePosition: { lat: number; lng: number },
  droneId: string
): Detection {
  return {
    id: `det-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    type: scenario.type,
    confidence: scenario.confidence,
    position: {
      lat: dronePosition.lat + scenario.offsetLat,
      lng: dronePosition.lng + scenario.offsetLng
    },
    timestamp: new Date(),
    droneId,
    flagged: false,
    imageUrl: getPlaceholderImage(scenario.type)
  }
}

// Placeholder images for detection types
function getPlaceholderImage(type: DetectionType): string {
  const colors = {
    person: '#ef4444',
    vehicle: '#f59e0b',
    animal: '#22c55e'
  }

  const icons: Record<DetectionType, string> = {
    person: '<circle cx="160" cy="100" r="25" fill="' + colors.person + '"/><path d="M160 125 L160 180 M135 150 L185 150 M160 180 L140 220 M160 180 L180 220" stroke="' + colors.person + '" stroke-width="6" fill="none"/>',
    vehicle: '<rect x="110" y="110" width="100" height="50" rx="8" fill="' + colors.vehicle + '"/><circle cx="135" cy="165" r="12" fill="#333"/><circle cx="185" cy="165" r="12" fill="#333"/>',
    animal: '<ellipse cx="160" cy="130" rx="45" ry="25" fill="' + colors.animal + '"/><circle cx="195" cy="115" r="12" fill="' + colors.animal + '"/><line x1="120" y1="155" x2="115" y2="180" stroke="' + colors.animal + '" stroke-width="5"/><line x1="145" y1="155" x2="140" y2="180" stroke="' + colors.animal + '" stroke-width="5"/><line x1="175" y1="155" x2="170" y2="180" stroke="' + colors.animal + '" stroke-width="5"/><line x1="200" y1="155" x2="195" y2="180" stroke="' + colors.animal + '" stroke-width="5"/>'
  }

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="320" height="240" viewBox="0 0 320 240">
      <rect fill="#1a1a1a" width="320" height="240"/>
      <rect fill="${colors[type]}" opacity="0.2" x="80" y="50" width="160" height="140" rx="4"/>
      <rect fill="none" stroke="${colors[type]}" stroke-width="3" x="80" y="50" width="160" height="140" rx="4"/>
      ${icons[type]}
      <text fill="#ffffff" font-family="system-ui" font-size="12" x="160" y="215" text-anchor="middle">${type.toUpperCase()} DETECTED</text>
    </svg>
  `

  return `data:image/svg+xml,${encodeURIComponent(svg)}`
}

// Detection type labels for UI
export const detectionLabels: Record<DetectionType, string> = {
  person: 'Person',
  vehicle: 'Vehicle',
  animal: 'Animal'
}

// Detection type icons (emoji for simplicity)
export const detectionIcons: Record<DetectionType, string> = {
  person: '👤',
  vehicle: '🚗',
  animal: '🦌'
}
