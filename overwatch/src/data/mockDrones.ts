import type { Drone } from '../types'

// Fort Liberty, NC area - realistic training ground
const HOME_POSITION = { lat: 35.14, lng: -79.0 }

export const initialDrones: Drone[] = [
  {
    id: 'bird-1',
    name: 'Bird 1',
    battery: 94,
    status: 'ready',
    armState: 'disarmed',
    position: { lat: HOME_POSITION.lat, lng: HOME_POSITION.lng },
    connected: true,
    homePosition: HOME_POSITION,
    altitude: 0,
    speed: 0,
    heading: 0
  },
  {
    id: 'bird-2',
    name: 'Bird 2',
    battery: 87,
    status: 'ready',
    armState: 'disarmed',
    position: { lat: HOME_POSITION.lat + 0.001, lng: HOME_POSITION.lng + 0.001 },
    connected: true,
    homePosition: { lat: HOME_POSITION.lat + 0.001, lng: HOME_POSITION.lng + 0.001 },
    altitude: 0,
    speed: 0,
    heading: 0
  },
]

export const MAP_CENTER = HOME_POSITION
export const MAP_ZOOM = 14
