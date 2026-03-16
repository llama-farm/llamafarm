import { useState, useCallback } from 'react'
import type { Drone, Detection, SearchArea, Alert, KillConfirmation } from '../types'
import { initialDrones } from '../data/mockDrones'

export function useDrones() {
  const [drones, setDrones] = useState<Drone[]>(initialDrones)
  const [detections, setDetections] = useState<Detection[]>([])
  const [searchAreas, setSearchAreas] = useState<SearchArea[]>([])
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [killConfirmation, setKillConfirmation] = useState<KillConfirmation | null>(null)

  // Update a single drone
  const updateDrone = useCallback((droneId: string, updates: Partial<Drone>) => {
    setDrones(prev => prev.map(d =>
      d.id === droneId ? { ...d, ...updates } : d
    ))
  }, [])

  // Launch a drone to a search area
  const launchDrone = useCallback((droneId: string, searchAreaId: string) => {
    updateDrone(droneId, { status: 'flying', altitude: 120, speed: 8.2, heading: 0 })

    // Assign drone to search area
    setSearchAreas(prev => prev.map(area =>
      area.id === searchAreaId ? { ...area, assignedDroneId: droneId } : area
    ))
  }, [updateDrone])

  // Stop a drone (emergency stop)
  const stopDrone = useCallback((droneId: string) => {
    updateDrone(droneId, { status: 'stopped' })
    setKillConfirmation(null)
  }, [updateDrone])

  // Return drone to launch point
  const returnDrone = useCallback((droneId: string) => {
    updateDrone(droneId, { status: 'returning' })
  }, [updateDrone])

  // Add a new detection
  const addDetection = useCallback((detection: Detection) => {
    setDetections(prev => [...prev, detection])

    // Create alert for person detections
    if (detection.type === 'person') {
      const alert: Alert = {
        id: `alert-${Date.now()}`,
        type: 'detection',
        message: `Person detected — ${Math.round(detection.confidence * 100)}% confidence — tap to view`,
        detectionId: detection.id,
        droneId: detection.droneId,
        timestamp: new Date(),
        dismissed: false
      }
      setAlerts(prev => [...prev, alert])
    }
  }, [])

  // Flag a detection
  const flagDetection = useCallback((detectionId: string) => {
    setDetections(prev => prev.map(d =>
      d.id === detectionId ? { ...d, flagged: true } : d
    ))
  }, [])

  // Add a search area
  const addSearchArea = useCallback((area: SearchArea) => {
    setSearchAreas(prev => [...prev, area])
  }, [])

  // Dismiss an alert
  const dismissAlert = useCallback((alertId: string) => {
    setAlerts(prev => prev.map(a =>
      a.id === alertId ? { ...a, dismissed: true } : a
    ))
  }, [])

  // Request kill confirmation
  const requestKillConfirmation = useCallback((droneId: string) => {
    const drone = drones.find(d => d.id === droneId)
    if (drone) {
      setKillConfirmation({
        droneId,
        droneName: drone.name,
        active: true
      })
    }
  }, [drones])

  // Cancel kill confirmation
  const cancelKillConfirmation = useCallback(() => {
    setKillConfirmation(null)
  }, [])

  // Add low battery alert
  const addBatteryAlert = useCallback((droneId: string, battery: number, replacementDroneId?: string) => {
    const drone = drones.find(d => d.id === droneId)
    if (!drone) return

    const alert: Alert = {
      id: `alert-battery-${Date.now()}`,
      type: 'battery',
      message: replacementDroneId
        ? `${drone.name} — Low Battery (${battery}%) — Send ${drones.find(d => d.id === replacementDroneId)?.name}?`
        : `${drone.name} — Low Battery (${battery}%)`,
      droneId,
      timestamp: new Date(),
      dismissed: false
    }
    setAlerts(prev => [...prev, alert])
  }, [drones])

  // Get active (non-dismissed) alerts
  const activeAlerts = alerts.filter(a => !a.dismissed)

  return {
    drones,
    detections,
    searchAreas,
    removeSearchArea: (id: string) => setSearchAreas(prev => prev.filter(a => a.id !== id)),
    alerts: activeAlerts,
    killConfirmation,
    updateDrone,
    launchDrone,
    stopDrone,
    returnDrone,
    addDetection,
    flagDetection,
    addSearchArea,
    dismissAlert,
    requestKillConfirmation,
    cancelKillConfirmation,
    addBatteryAlert,
    setDrones
  }
}
