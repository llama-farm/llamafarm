import type { Drone } from '../types'

// Fort Liberty, NC area - realistic training ground
const HOME_POSITION = { lat: 35.14, lng: -79.0 }

export const initialDrones: Drone[] = [
  {
    id: 'bird-1',
    name: 'Bird 1',
    battery: 94,
    status: 'ready',
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
    battery: 74,
    status: 'ready',
    position: { lat: HOME_POSITION.lat + 0.001, lng: HOME_POSITION.lng + 0.001 },
    connected: true,
    homePosition: { lat: HOME_POSITION.lat + 0.001, lng: HOME_POSITION.lng + 0.001 },
    altitude: 0,
    speed: 0,
    heading: 0
  },
  {
    id: 'bird-3',
    name: 'Bird 3',
    battery: 94,
    status: 'ready',
    position: { lat: HOME_POSITION.lat - 0.001, lng: HOME_POSITION.lng + 0.002 },
    connected: true,
    homePosition: { lat: HOME_POSITION.lat - 0.001, lng: HOME_POSITION.lng + 0.002 },
    altitude: 0,
    speed: 0,
    heading: 0
  },
  {
    id: 'bird-4',
    name: 'Bird 4',
    battery: 94,
    status: 'flying',
    position: { lat: HOME_POSITION.lat + 0.003, lng: HOME_POSITION.lng - 0.002 },
    connected: true,
    homePosition: { lat: HOME_POSITION.lat + 0.002, lng: HOME_POSITION.lng - 0.001 },
    altitude: 120,
    speed: 8.2,
    heading: 45
  },
  {
    id: 'bird-5',
    name: 'Bird 5',
    battery: 4,
    status: 'ready',
    position: { lat: HOME_POSITION.lat - 0.002, lng: HOME_POSITION.lng - 0.001 },
    connected: true,
    homePosition: { lat: HOME_POSITION.lat - 0.002, lng: HOME_POSITION.lng - 0.001 },
    altitude: 0,
    speed: 0,
    heading: 0
  }
]

export const MAP_CENTER = HOME_POSITION
export const MAP_ZOOM = 14
