// Drone types
export type DroneStatus = 'ready' | 'armed' | 'flying' | 'returning' | 'stopped' | 'landing' | 'offline'
export type BatteryLevel = 'good' | 'warning' | 'critical'
export type ArmState = 'disarmed' | 'armed'

export interface DronePosition {
  lat: number
  lng: number
}

export interface Drone {
  id: string
  name: string
  battery: number
  status: DroneStatus
  armState: ArmState
  position: DronePosition
  connected: boolean
  homePosition: DronePosition
  altitude: number
  speed: number
  heading: number
}

// Detection types
export type DetectionType = 'person' | 'vehicle' | 'animal'

export interface Detection {
  id: string
  type: DetectionType
  confidence: number
  position: DronePosition
  mgrs: string
  timestamp: Date
  droneId: string
  flagged: boolean
  imageUrl?: string
}

// Search area types
export interface SearchArea {
  id: string
  bounds: {
    northEast: DronePosition
    southWest: DronePosition
  }
  assignedDroneId?: string
}

// View modes — simplified: map is primary, voice log is secondary
export type ViewMode = 'map' | 'voice'

// Alert types
export interface Alert {
  id: string
  type: 'detection' | 'battery' | 'connection' | 'arm'
  message: string
  detectionId?: string
  droneId?: string
  timestamp: Date
  dismissed: boolean
}

// Voice conversation entry
export interface VoiceEntry {
  id: string
  speaker: 'operator' | 'drone' | 'system'
  text: string
  timestamp: Date
  detectionId?: string
  type?: DetectionType
}

// Kill confirmation
export interface KillConfirmation {
  droneId: string
  droneName: string
  active: boolean
}

// Battery helper
export function getBatteryLevel(battery: number): BatteryLevel {
  if (battery > 30) return 'good'
  if (battery > 15) return 'warning'
  return 'critical'
}

// Color helpers
export function getDetectionColor(type: DetectionType): string {
  switch (type) {
    case 'person': return '#994444'
    case 'vehicle': return '#997744'
    case 'animal': return '#449966'
  }
}

export function getDroneColor(index: number): string {
  const colors = ['#3d7cc7', '#1a9e4a', '#c4820a']
  return colors[index % colors.length]
}

export function getStatusColor(status: DroneStatus): string {
  switch (status) {
    case 'flying': return '#1a9e4a'
    case 'armed': return '#c4820a'
    case 'returning': return '#c4820a'
    case 'stopped': return '#c43434'
    case 'offline': return '#4a5868'
    default: return '#6b7a8a'
  }
}
