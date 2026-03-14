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

// Simple MGRS-like grid reference for mock data
function toMGRS(lat: number, lng: number): string {
  const zone = Math.floor((lng + 180) / 6) + 1
  const letters = 'CDEFGHJKLMNPQRSTUVWX'
  const letterIdx = Math.min(Math.max(Math.floor((lat + 80) / 8), 0), letters.length - 1)
  const colLetters = 'ABCDEFGHJKLMNPQRSTUVWXYZ'
  const rowLetters = 'ABCDEFGHJKLMNPQRSTUV'
  const col = colLetters[Math.floor(Math.abs(lng * 10)) % colLetters.length]
  const row = rowLetters[Math.floor(Math.abs(lat * 10)) % rowLetters.length]
  const easting = String(Math.floor(Math.abs((lng % 1) * 100000))).padStart(5, '0')
  const northing = String(Math.floor(Math.abs((lat % 1) * 100000))).padStart(5, '0')
  return `${zone}${letters[letterIdx]} ${col}${row} ${easting} ${northing}`
}

// Generate a detection from a scenario
export function createDetection(
  scenario: DetectionScenario,
  dronePosition: { lat: number; lng: number },
  droneId: string
): Detection {
  const position = {
    lat: dronePosition.lat + scenario.offsetLat,
    lng: dronePosition.lng + scenario.offsetLng
  }
  return {
    id: `det-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
    type: scenario.type,
    confidence: scenario.confidence,
    position,
    mgrs: toMGRS(position.lat, position.lng),
    timestamp: new Date(),
    droneId,
    flagged: false,
    imageUrl: getPlaceholderImage(scenario.type)
  }
}

// Night vision style placeholder images
function getPlaceholderImage(type: DetectionType): string {
  // Muted green night-vision palette
  const nvGreen = '#1a3a1a'
  const nvHighlight = '#2d5a2d'
  const nvBright = '#3d7a3d'
  const boxColor = type === 'person' ? '#5a3333' : type === 'vehicle' ? '#5a5033' : '#335a40'
  const boxStroke = type === 'person' ? '#7a4444' : type === 'vehicle' ? '#7a6a44' : '#447a55'

  const shapes: Record<DetectionType, string> = {
    person: `
      <ellipse cx="200" cy="95" rx="14" ry="16" fill="${nvBright}" opacity="0.7"/>
      <rect x="188" y="112" width="24" height="40" rx="3" fill="${nvHighlight}" opacity="0.6"/>
      <rect x="185" y="152" width="12" height="30" rx="2" fill="${nvHighlight}" opacity="0.5"/>
      <rect x="203" y="152" width="12" height="30" rx="2" fill="${nvHighlight}" opacity="0.5"/>
    `,
    vehicle: `
      <rect x="130" y="100" width="100" height="45" rx="6" fill="${nvHighlight}" opacity="0.5"/>
      <rect x="125" y="115" width="110" height="35" rx="4" fill="${nvBright}" opacity="0.3"/>
      <circle cx="150" cy="148" r="8" fill="${nvGreen}" stroke="${nvHighlight}" stroke-width="2"/>
      <circle cx="210" cy="148" r="8" fill="${nvGreen}" stroke="${nvHighlight}" stroke-width="2"/>
    `,
    animal: `
      <ellipse cx="190" cy="130" rx="30" ry="16" fill="${nvHighlight}" opacity="0.5"/>
      <circle cx="215" cy="120" r="8" fill="${nvBright}" opacity="0.5"/>
      <rect x="170" y="146" width="5" height="18" rx="1" fill="${nvHighlight}" opacity="0.4"/>
      <rect x="185" y="146" width="5" height="18" rx="1" fill="${nvHighlight}" opacity="0.4"/>
      <rect x="200" y="146" width="5" height="18" rx="1" fill="${nvHighlight}" opacity="0.4"/>
    `
  }

  // Noise grain pattern
  const grainDots = Array.from({ length: 60 }, () => {
    const x = Math.floor(Math.random() * 400)
    const y = Math.floor(Math.random() * 240)
    const o = (Math.random() * 0.15 + 0.05).toFixed(2)
    return `<circle cx="${x}" cy="${y}" r="1" fill="#4a7a4a" opacity="${o}"/>`
  }).join('')

  const svg = `
    <svg xmlns="http://www.w3.org/2000/svg" width="400" height="240" viewBox="0 0 400 240">
      <rect fill="#0a0f0a" width="400" height="240"/>
      <rect fill="${nvGreen}" opacity="0.15" width="400" height="240"/>
      ${grainDots}
      <line x1="0" y1="60" x2="400" y2="60" stroke="${nvGreen}" stroke-width="0.3" opacity="0.3"/>
      <line x1="0" y1="120" x2="400" y2="120" stroke="${nvGreen}" stroke-width="0.3" opacity="0.3"/>
      <line x1="0" y1="180" x2="400" y2="180" stroke="${nvGreen}" stroke-width="0.3" opacity="0.3"/>
      <line x1="200" y1="0" x2="200" y2="240" stroke="${nvGreen}" stroke-width="0.3" opacity="0.2"/>
      <rect fill="${boxColor}" opacity="0.3" x="120" y="70" width="160" height="120" rx="2"/>
      <rect fill="none" stroke="${boxStroke}" stroke-width="1.5" x="120" y="70" width="160" height="120" rx="2"/>
      ${shapes[type]}
      <text fill="${boxStroke}" font-family="monospace" font-size="10" x="125" y="85" opacity="0.8">TGT</text>
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
