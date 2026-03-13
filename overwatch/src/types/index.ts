// Drone types
export type DroneStatus = 'ready' | 'flying' | 'returning' | 'stopped' | 'offline'
export type BatteryLevel = 'good' | 'warning' | 'critical'

export interface DronePosition {
  lat: number
  lng: number
}

export interface Drone {
  id: string
  name: string
  battery: number
  status: DroneStatus
  position: DronePosition
  connected: boolean
  homePosition: DronePosition
  altitude: number    // meters AGL
  speed: number       // m/s
  heading: number     // degrees 0-359
}

// Detection types
export type DetectionType = 'person' | 'vehicle' | 'animal'

export interface Detection {
  id: string
  type: DetectionType
  confidence: number
  position: DronePosition
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

// View modes
export type ViewMode = 'fleet' | 'map' | 'feed'

// Alert types
export interface Alert {
  id: string
  type: 'detection' | 'battery' | 'connection'
  message: string
  detectionId?: string
  droneId?: string
  timestamp: Date
  dismissed: boolean
}

// Kill confirmation state
export interface KillConfirmation {
  droneId: string
  droneName: string
  active: boolean
}

// App state
export interface AppState {
  viewMode: ViewMode
  selectedDroneId: string | null
  selectedDetectionId: string | null
  drones: Drone[]
  detections: Detection[]
  searchAreas: SearchArea[]
  alerts: Alert[]
  killConfirmation: KillConfirmation | null
  isDrawingArea: boolean
  pendingLaunch: {
    droneId: string
    searchAreaId: string
  } | null
}

// Utility types
export function getBatteryLevel(battery: number): BatteryLevel {
  if (battery >= 30) return 'good'
  if (battery >= 15) return 'warning'
  return 'critical'
}

export function getDetectionColor(type: DetectionType): string {
  switch (type) {
    case 'person': return '#ef4444'    // Red
    case 'vehicle': return '#f59e0b'   // Yellow/Amber
    case 'animal': return '#22c55e'    // Green
  }
}

export function getStatusColor(status: DroneStatus, battery: number): string {
  if (status === 'offline') return '#666666'
  if (status === 'stopped') return '#ef4444'

  const level = getBatteryLevel(battery)
  switch (level) {
    case 'good': return '#22c55e'
    case 'warning': return '#f59e0b'
    case 'critical': return '#ef4444'
  }
}

// Unique color per drone for map icons and selection highlights
const DRONE_COLORS = ['#22c55e', '#3b82f6', '#a855f7', '#f59e0b', '#ef4444']

export function getDroneColor(droneIndex: number): string {
  return DRONE_COLORS[droneIndex % DRONE_COLORS.length]
}
