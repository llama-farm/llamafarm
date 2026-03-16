import { useEffect, useRef, useCallback } from 'react'
import type { Drone, SearchArea, DronePosition } from '../types'
import { detectionScenarios, createDetection } from '../data/mockDetections'

interface SimulationOptions {
  drones: Drone[]
  searchAreas: SearchArea[]
  updateDrone: (droneId: string, updates: Partial<Drone>) => void
  addDetection: (detection: ReturnType<typeof createDetection>) => void
  addBatteryAlert: (droneId: string, battery: number, replacementDroneId?: string) => void
}

export function useSimulation({
  drones,
  searchAreas,
  updateDrone,
  addDetection,
  addBatteryAlert
}: SimulationOptions) {
  const flightTimersRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map())
  const detectionTimeoutsRef = useRef<Map<string, ReturnType<typeof setTimeout>[]>>(new Map())
  const batteryAlertTimeoutRef = useRef<Map<string, ReturnType<typeof setTimeout>>>(new Map())
  const batteryWarningShownRef = useRef<Set<string>>(new Set())

  // Calculate search pattern waypoints for a search area
  const calculateSearchPattern = useCallback((area: SearchArea): DronePosition[] => {
    const { northEast, southWest } = area.bounds
    const waypoints: DronePosition[] = []

    // Lawnmower pattern
    const rows = 4
    const latStep = (northEast.lat - southWest.lat) / rows

    for (let i = 0; i <= rows; i++) {
      const lat = southWest.lat + latStep * i
      if (i % 2 === 0) {
        waypoints.push({ lat, lng: southWest.lng })
        waypoints.push({ lat, lng: northEast.lng })
      } else {
        waypoints.push({ lat, lng: northEast.lng })
        waypoints.push({ lat, lng: southWest.lng })
      }
    }

    return waypoints
  }, [])

  // Start flying a drone along a search pattern
  const startFlight = useCallback((droneId: string, searchAreaId: string) => {
    const area = searchAreas.find(a => a.id === searchAreaId)
    const drone = drones.find(d => d.id === droneId)

    if (!area || !drone) return

    const waypoints = calculateSearchPattern(area)
    let waypointIndex = 0
    let currentPosition = { ...drone.position }

    // Clear any existing timer
    const existingTimer = flightTimersRef.current.get(droneId)
    if (existingTimer) clearInterval(existingTimer)

    // Clear any existing detection timeouts
    const existingTimeouts = detectionTimeoutsRef.current.get(droneId)
    if (existingTimeouts) {
      existingTimeouts.forEach(t => clearTimeout(t))
    }
    detectionTimeoutsRef.current.set(droneId, [])

    // Clear battery alert timeout
    const existingBatteryTimeout = batteryAlertTimeoutRef.current.get(droneId)
    if (existingBatteryTimeout) clearTimeout(existingBatteryTimeout)

    // Movement interval - move towards next waypoint
    // Update at 1Hz (1000ms) — 200ms causes too many re-renders
    const moveInterval = setInterval(() => {
      const targetDrone = drones.find(d => d.id === droneId)
      if (!targetDrone || targetDrone.status !== 'flying') {
        clearInterval(moveInterval)
        flightTimersRef.current.delete(droneId)
        return
      }

      const target = waypoints[waypointIndex]
      const speed = 0.0015 // Adjusted for 1Hz tick rate

      // Calculate direction
      const latDiff = target.lat - currentPosition.lat
      const lngDiff = target.lng - currentPosition.lng
      const distance = Math.sqrt(latDiff * latDiff + lngDiff * lngDiff)

      // Calculate heading from direction of travel
      const heading = (Math.atan2(lngDiff, latDiff) * 180 / Math.PI + 360) % 360

      if (distance < speed) {
        // Reached waypoint, move to next
        waypointIndex = (waypointIndex + 1) % waypoints.length
      } else {
        // Move towards waypoint
        const ratio = speed / distance
        currentPosition = {
          lat: currentPosition.lat + latDiff * ratio,
          lng: currentPosition.lng + lngDiff * ratio
        }
        updateDrone(droneId, {
          position: currentPosition,
          heading: Math.round(heading),
          altitude: 120 + Math.sin(Date.now() / 5000) * 5, // slight altitude variation
          speed: 7.5 + Math.random() * 1.5 // ~7.5-9 m/s
        })
      }

      // Battery drain - 0.1% per 1000ms (~16 min flight time)
      const newBattery = Math.max(0, targetDrone.battery - 0.1)
      updateDrone(droneId, { battery: newBattery })

    }, 1000)

    flightTimersRef.current.set(droneId, moveInterval)

    // Schedule detections based on scenarios
    const timeouts: ReturnType<typeof setTimeout>[] = []
    detectionScenarios.forEach((scenario) => {
      const timeout = setTimeout(() => {
        const currentDrone = drones.find(d => d.id === droneId)
        if (currentDrone && currentDrone.status === 'flying') {
          const detection = createDetection(scenario, currentDrone.position, droneId)
          addDetection(detection)
        }
      }, scenario.delayMs)
      timeouts.push(timeout)
    })
    detectionTimeoutsRef.current.set(droneId, timeouts)

    // Schedule battery alert at t+30s per PASS2.md
    const batteryAlertTimeout = setTimeout(() => {
      const currentDrone = drones.find(d => d.id === droneId)
      if (currentDrone && currentDrone.status === 'flying' && !batteryWarningShownRef.current.has(droneId)) {
        batteryWarningShownRef.current.add(droneId)
        // Set battery to 18% for visual effect
        updateDrone(droneId, { battery: 18 })
        // Find replacement drone
        const replacement = drones.find(d => d.id !== droneId && d.status === 'ready' && d.battery > 50)
        addBatteryAlert(droneId, 18, replacement?.id)
      }
    }, 30000)
    batteryAlertTimeoutRef.current.set(droneId, batteryAlertTimeout)

  }, [drones, searchAreas, calculateSearchPattern, updateDrone, addDetection, addBatteryAlert])

  // Start returning a drone to home
  const startReturn = useCallback((droneId: string) => {
    const drone = drones.find(d => d.id === droneId)
    if (!drone) return

    // Clear flight timer
    const existingTimer = flightTimersRef.current.get(droneId)
    if (existingTimer) clearInterval(existingTimer)

    // Clear detection timeouts
    const existingTimeouts = detectionTimeoutsRef.current.get(droneId)
    if (existingTimeouts) {
      existingTimeouts.forEach(t => clearTimeout(t))
    }

    // Clear battery alert timeout
    const existingBatteryTimeout = batteryAlertTimeoutRef.current.get(droneId)
    if (existingBatteryTimeout) clearTimeout(existingBatteryTimeout)

    let currentPosition = { ...drone.position }
    const target = drone.homePosition

    const returnInterval = setInterval(() => {
      const targetDrone = drones.find(d => d.id === droneId)
      if (!targetDrone || targetDrone.status !== 'returning') {
        clearInterval(returnInterval)
        flightTimersRef.current.delete(droneId)
        return
      }

      const speed = 0.002 // Adjusted for 1Hz tick rate
      const latDiff = target.lat - currentPosition.lat
      const lngDiff = target.lng - currentPosition.lng
      const distance = Math.sqrt(latDiff * latDiff + lngDiff * lngDiff)

      const heading = (Math.atan2(lngDiff, latDiff) * 180 / Math.PI + 360) % 360

      if (distance < speed) {
        // Reached home
        updateDrone(droneId, {
          position: target,
          status: 'ready',
          altitude: 0,
          speed: 0,
          heading: 0
        })
        clearInterval(returnInterval)
        flightTimersRef.current.delete(droneId)
      } else {
        const ratio = speed / distance
        currentPosition = {
          lat: currentPosition.lat + latDiff * ratio,
          lng: currentPosition.lng + lngDiff * ratio
        }
        updateDrone(droneId, {
          position: currentPosition,
          heading: Math.round(heading),
          altitude: 120,
          speed: 10.0 + Math.random() * 1.0
        })
      }
    }, 1000)

    flightTimersRef.current.set(droneId, returnInterval)
  }, [drones, updateDrone])

  // Watch for status changes to trigger simulation
  useEffect(() => {
    drones.forEach(drone => {
      const area = searchAreas.find(a => a.assignedDroneId === drone.id)

      if (drone.status === 'flying' && area && !flightTimersRef.current.has(drone.id)) {
        startFlight(drone.id, area.id)
      }

      if (drone.status === 'returning' && !flightTimersRef.current.has(drone.id)) {
        startReturn(drone.id)
      }
    })
  }, [drones, searchAreas, startFlight, startReturn])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      flightTimersRef.current.forEach(timer => clearInterval(timer))
      flightTimersRef.current.clear()
      detectionTimeoutsRef.current.forEach(timeouts => timeouts.forEach(t => clearTimeout(t)))
      detectionTimeoutsRef.current.clear()
      batteryAlertTimeoutRef.current.forEach(t => clearTimeout(t))
      batteryAlertTimeoutRef.current.clear()
    }
  }, [])

  return {
    startFlight,
    startReturn
  }
}
