import { useState, useCallback } from 'react'
import type { ViewMode, DronePosition, SearchArea, Alert } from './types'
import { useDrones } from './hooks/useDrones'
import { useSimulation } from './hooks/useSimulation'
import { TopBar } from './components/TopBar'
import { FleetStatus } from './components/Fleet/FleetStatus'
import { FleetStrip } from './components/Fleet/FleetStrip'
import { MapView } from './components/Map/MapView'
import { FeedView } from './components/Feed/FeedView'
import { DetectionDetail } from './components/Detection/DetectionDetail'
import { ConfirmSlider } from './components/Controls/ConfirmSlider'
import { LaunchConfirm } from './components/Controls/LaunchConfirm'
import { StoppedOverlay } from './components/Controls/StoppedOverlay'
import { TelemetryHUD } from './components/Map/TelemetryHUD'
import { DetectionFeed } from './components/Detection/DetectionFeed'
import { DrawButton } from './components/Controls/DrawButton'

export default function App() {
  // View state
  const [viewMode, setViewMode] = useState<ViewMode>('fleet')
  const [selectedDroneId, setSelectedDroneId] = useState<string | null>(null)
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(null)
  const [isDrawingArea, setIsDrawingArea] = useState(false)
  const [pendingLaunch, setPendingLaunch] = useState<{ droneId: string; searchAreaId: string } | null>(null)
  const [stoppedDroneId, setStoppedDroneId] = useState<string | null>(null)

  // Drone state management
  const {
    drones,
    detections,
    searchAreas,
    alerts,
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
    addBatteryAlert
  } = useDrones()

  // Simulation
  useSimulation({
    drones,
    searchAreas,
    updateDrone,
    addDetection,
    addBatteryAlert
  })

  // Get selected drone and detection
  const selectedDrone = drones.find(d => d.id === selectedDroneId)
  const selectedDetection = detections.find(d => d.id === selectedDetectionId)
  const stoppedDrone = drones.find(d => d.id === stoppedDroneId)

  // Handle drone selection from fleet
  const handleSelectDrone = useCallback((droneId: string) => {
    setSelectedDroneId(droneId)
    setViewMode('map')
  }, [])

  // Handle draw completion
  const handleDrawComplete = useCallback((bounds: { northEast: DronePosition; southWest: DronePosition }) => {
    setIsDrawingArea(false)

    if (!selectedDroneId) return

    const area: SearchArea = {
      id: `area-${Date.now()}`,
      bounds
    }
    addSearchArea(area)
    setPendingLaunch({ droneId: selectedDroneId, searchAreaId: area.id })
  }, [selectedDroneId, addSearchArea])

  // Handle launch confirmation
  const handleConfirmLaunch = useCallback(() => {
    if (!pendingLaunch) return
    launchDrone(pendingLaunch.droneId, pendingLaunch.searchAreaId)
    setPendingLaunch(null)
  }, [pendingLaunch, launchDrone])

  // Handle detection click
  const handleDetectionClick = useCallback((detectionId: string) => {
    setSelectedDetectionId(detectionId)
  }, [])

  // Handle watch feed
  const handleWatchFeed = useCallback(() => {
    const detection = detections.find(d => d.id === selectedDetectionId)
    if (detection) {
      setSelectedDroneId(detection.droneId)
    }
    setSelectedDetectionId(null)
    setViewMode('feed')
  }, [selectedDetectionId, detections])

  // Handle flag detection
  const handleFlagDetection = useCallback(() => {
    if (selectedDetectionId) {
      flagDetection(selectedDetectionId)
    }
  }, [selectedDetectionId, flagDetection])

  // Handle open stream
  const handleOpenStream = useCallback(() => {
    setViewMode('feed')
  }, [])

  // Handle back to map
  const handleBackToMap = useCallback(() => {
    setViewMode('map')
  }, [])

  // Handle kill button - targets selected drone
  const handleKillButton = useCallback(() => {
    if (selectedDrone && (selectedDrone.status === 'flying' || selectedDrone.status === 'returning')) {
      requestKillConfirmation(selectedDrone.id)
    }
  }, [selectedDrone, requestKillConfirmation])

  // Handle kill confirmation
  const handleConfirmKill = useCallback(() => {
    if (killConfirmation) {
      stopDrone(killConfirmation.droneId)
      setStoppedDroneId(killConfirmation.droneId)
    }
  }, [killConfirmation, stopDrone])

  // Handle return to launch
  const handleReturnToLaunch = useCallback(() => {
    if (stoppedDroneId) {
      returnDrone(stoppedDroneId)
      setStoppedDroneId(null)
    }
  }, [stoppedDroneId, returnDrone])

  // Handle alert action
  const handleAlertAction = useCallback((alert: Alert) => {
    if (alert.type === 'detection' && alert.detectionId) {
      setSelectedDetectionId(alert.detectionId)
    } else if (alert.type === 'battery' && alert.droneId) {
      // Find backup drone and launch it
      const currentDrone = drones.find(d => d.id === alert.droneId)
      const backupDrone = drones.find(d => d.id !== alert.droneId && d.status === 'ready' && d.battery > 50)
      const area = searchAreas.find(a => a.assignedDroneId === alert.droneId)

      if (backupDrone && area) {
        // Return current drone
        if (currentDrone) {
          returnDrone(currentDrone.id)
        }
        // Launch backup
        launchDrone(backupDrone.id, area.id)
        setSelectedDroneId(backupDrone.id)
      }
    }
    dismissAlert(alert.id)
  }, [drones, searchAreas, returnDrone, launchDrone, dismissAlert])

  // Check if selected drone can be killed
  const canKillSelected = selectedDrone?.status === 'flying' || selectedDrone?.status === 'returning'

  return (
    <div className="h-screen flex flex-col bg-surface-base overflow-hidden">
      {/* Top bar */}
      <TopBar connectionStatus="connected" gpsStatus="locked" />

      {/* Main content area */}
      <main className="flex-1 relative overflow-hidden">
        {/* Fleet status view */}
        {viewMode === 'fleet' && (
          <FleetStatus
            drones={drones}
            selectedDroneId={selectedDroneId}
            onSelectDrone={handleSelectDrone}
          />
        )}

        {/* Map view */}
        {viewMode === 'map' && (
          <>
            <MapView
              drones={drones}
              detections={detections}
              searchAreas={searchAreas}
              selectedDroneId={selectedDroneId}
              isDrawingArea={isDrawingArea}
              onDrawComplete={handleDrawComplete}
              onDetectionClick={handleDetectionClick}
            />
            {selectedDrone && (
              <TelemetryHUD drone={selectedDrone} />
            )}
            <DetectionFeed
              detections={detections}
              alerts={alerts}
              selectedDrone={selectedDrone}
              onDetectionClick={handleDetectionClick}
              onAlertAction={handleAlertAction}
              onAlertDismiss={dismissAlert}
              onOpenStream={handleOpenStream}
            />
          </>
        )}

        {/* Feed view */}
        {viewMode === 'feed' && selectedDrone && (
          <FeedView
            drone={selectedDrone}
            droneIndex={drones.findIndex(d => d.id === selectedDroneId)}
            detection={detections.find(d => d.droneId === selectedDroneId)}
            onBackToMap={handleBackToMap}
            onFlag={handleFlagDetection}
          />
        )}
      </main>

      {/* Fleet strip (shown on map and feed views) */}
      {viewMode !== 'fleet' && (
        <FleetStrip
          drones={drones}
          selectedDroneId={selectedDroneId}
          onSelectDrone={(id) => {
            setSelectedDroneId(id)
            if (viewMode === 'feed') {
              setViewMode('map')
            }
          }}
          onKill={handleKillButton}
          killDisabled={!canKillSelected}
          selectedDroneName={canKillSelected ? selectedDrone?.name : undefined}
        />
      )}

      {/* Draw button (map view only, when drone selected) */}
      {viewMode === 'map' && selectedDroneId && (
        <DrawButton
          isDrawing={isDrawingArea}
          disabled={selectedDrone?.status !== 'ready'}
          onClick={() => setIsDrawingArea(!isDrawingArea)}
        />
      )}

      {/* Detection detail modal */}
      {selectedDetection && viewMode === 'map' && (
        <DetectionDetail
          detection={selectedDetection}
          droneName={drones.find(d => d.id === selectedDetection.droneId)?.name || 'Unknown'}
          onClose={() => setSelectedDetectionId(null)}
          onWatchFeed={handleWatchFeed}
          onFlag={handleFlagDetection}
        />
      )}

      {/* Launch confirmation */}
      {pendingLaunch && selectedDrone && (
        <LaunchConfirm
          droneName={selectedDrone.name}
          onConfirm={handleConfirmLaunch}
          onCancel={() => setPendingLaunch(null)}
        />
      )}

      {/* Kill confirmation slider */}
      {killConfirmation && (
        <ConfirmSlider
          label={`STOP ${killConfirmation.droneName}?`}
          onConfirm={handleConfirmKill}
          onCancel={cancelKillConfirmation}
          variant="danger"
        />
      )}

      {/* Stopped drone overlay */}
      {stoppedDrone && (
        <StoppedOverlay
          droneName={stoppedDrone.name}
          onReturnToLaunch={handleReturnToLaunch}
          onDismiss={() => setStoppedDroneId(null)}
        />
      )}
    </div>
  )
}
