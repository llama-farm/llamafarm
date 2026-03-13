import { useState, useCallback, useEffect } from 'react'
import type { ViewMode, DronePosition, SearchArea, Alert } from './types'
import { useDrones } from './hooks/useDrones'
import { useSimulation } from './hooks/useSimulation'
import { useVoice } from './hooks/useVoice'
import { useMGRS } from './hooks/useMGRS'
import { TopBar } from './components/TopBar'
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
import { VoiceBar } from './components/Voice/VoiceBar'
import { VoiceLog } from './components/Voice/VoiceLog'
import { CommsLostBanner } from './components/CommsLostBanner'

export default function App() {
  // View state — map is default, no fleet view
  const [viewMode, setViewMode] = useState<ViewMode>('map')
  const [selectedDroneId, setSelectedDroneId] = useState<string>('bird-1')
  const [selectedDetectionId, setSelectedDetectionId] = useState<string | null>(null)
  const [isDrawingArea, setIsDrawingArea] = useState(false)
  const [pendingLaunch, setPendingLaunch] = useState<{ droneId: string; searchAreaId: string } | null>(null)
  const [stoppedDroneId, setStoppedDroneId] = useState<string | null>(null)
  const [commsConnected, setCommsConnected] = useState(true)
  const [commsRestored, setCommsRestored] = useState(false)

  const { toMGRS } = useMGRS()

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

  // Voice command handler
  const handleVoiceCommand = useCallback((command: string) => {
    const cmd = command.toLowerCase()

    if (cmd.includes('launch') || cmd.includes('go')) {
      voice.speak('Copy, standing by for search area. Draw on map or specify grid.', 'drone')
    } else if (cmd.includes('kill') || cmd.includes('stop')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && (drone.status === 'flying' || drone.status === 'returning')) {
        requestKillConfirmation(drone.id)
        voice.speak('Confirm kill?', 'drone')
      }
    } else if (cmd.includes('confirm')) {
      if (killConfirmation) {
        stopDrone(killConfirmation.droneId)
        setStoppedDroneId(killConfirmation.droneId)
        voice.speak(`${killConfirmation.droneName} stopped. Hovering.`, 'drone')
      }
    } else if (cmd.includes('status')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone) {
        voice.speak(`Battery ${drone.battery}%. Altitude ${drone.altitude} feet. ${drone.status}. ${drone.armState}.`, 'drone')
      }
    } else if (cmd.includes('arm')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && drone.armState === 'disarmed') {
        updateDrone(drone.id, { armState: 'armed', status: 'armed' })
        voice.speak('Armed. Ready for launch.', 'drone')
      }
    } else if (cmd.includes('disarm')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && drone.armState === 'armed' && drone.status !== 'flying') {
        updateDrone(drone.id, { armState: 'disarmed', status: 'ready' })
        voice.speak('Disarmed.', 'drone')
      }
    } else if (cmd.includes('return') || cmd.includes('home') || cmd.includes('rtl')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && drone.status === 'flying') {
        returnDrone(drone.id)
        voice.speak('Roger, returning to launch.', 'drone')
      }
    } else if (cmd.includes('contact') || cmd.includes('detection')) {
      const recent = detections[detections.length - 1]
      if (recent) {
        voice.speak(`Last contact: ${recent.type}, ${recent.confidence}%, grid ${recent.mgrs}`, 'drone')
      } else {
        voice.speak('No contacts.', 'drone')
      }
    } else if (cmd.includes('flag')) {
      const recent = detections[detections.length - 1]
      if (recent && !recent.flagged) {
        flagDetection(recent.id)
        voice.speak('Flagged to commander.', 'drone')
      }
    } else {
      voice.speak('Say again.', 'drone')
    }
  }, [drones, selectedDroneId, detections, killConfirmation, requestKillConfirmation, stopDrone, returnDrone, updateDrone, flagDetection])

  // Voice system
  const voice = useVoice({ onCommand: handleVoiceCommand })

  // Announce new detections
  useEffect(() => {
    if (detections.length > 0) {
      const latest = detections[detections.length - 1]
      // Only announce if it just appeared (within last 2 seconds)
      if (Date.now() - latest.timestamp.getTime() < 2000) {
        voice.announceDetection(latest.type, latest.confidence, latest.mgrs, latest.id)
      }
    }
  }, [detections.length])

  // Simulation
  useSimulation({
    drones,
    searchAreas,
    updateDrone,
    addDetection,
    addBatteryAlert
  })

  const selectedDrone = drones.find(d => d.id === selectedDroneId)
  const selectedDetection = detections.find(d => d.id === selectedDetectionId)
  const stoppedDrone = drones.find(d => d.id === stoppedDroneId)

  // Handle draw completion
  const handleDrawComplete = useCallback((bounds: { northEast: DronePosition; southWest: DronePosition }) => {
    setIsDrawingArea(false)
    if (!selectedDroneId) return
    const area: SearchArea = { id: `area-${Date.now()}`, bounds }
    addSearchArea(area)
    setPendingLaunch({ droneId: selectedDroneId, searchAreaId: area.id })
  }, [selectedDroneId, addSearchArea])

  // Handle launch — arms and launches
  const handleConfirmLaunch = useCallback(() => {
    if (!pendingLaunch) return
    // Arm first, then launch
    updateDrone(pendingLaunch.droneId, { armState: 'armed' })
    launchDrone(pendingLaunch.droneId, pendingLaunch.searchAreaId)
    voice.speak('Armed. Launching search pattern.', 'drone')
    setPendingLaunch(null)
  }, [pendingLaunch, updateDrone, launchDrone, voice])

  const handleDetectionClick = useCallback((detectionId: string) => {
    setSelectedDetectionId(detectionId)
  }, [])

  const handleWatchFeed = useCallback(() => {
    const detection = detections.find(d => d.id === selectedDetectionId)
    if (detection) setSelectedDroneId(detection.droneId)
    setSelectedDetectionId(null)
  }, [selectedDetectionId, detections])

  const handleFlagDetection = useCallback(() => {
    if (selectedDetectionId) {
      flagDetection(selectedDetectionId)
      voice.speak('Flagged to commander.', 'drone')
    }
  }, [selectedDetectionId, flagDetection, voice])

  const handleKillButton = useCallback(() => {
    if (selectedDrone && (selectedDrone.status === 'flying' || selectedDrone.status === 'returning')) {
      requestKillConfirmation(selectedDrone.id)
    }
  }, [selectedDrone, requestKillConfirmation])

  const handleConfirmKill = useCallback(() => {
    if (killConfirmation) {
      stopDrone(killConfirmation.droneId)
      setStoppedDroneId(killConfirmation.droneId)
      // Auto-disarm on stop
      updateDrone(killConfirmation.droneId, { armState: 'disarmed' })
      voice.speak(`${killConfirmation.droneName} stopped. Disarmed. Hovering.`, 'drone')
    }
  }, [killConfirmation, stopDrone, updateDrone, voice])

  const handleReturnToLaunch = useCallback(() => {
    if (stoppedDroneId) {
      returnDrone(stoppedDroneId)
      voice.speak('Returning to launch.', 'drone')
      setStoppedDroneId(null)
    }
  }, [stoppedDroneId, returnDrone, voice])

  const handleAlertAction = useCallback((alert: Alert) => {
    if (alert.type === 'detection' && alert.detectionId) {
      setSelectedDetectionId(alert.detectionId)
    } else if (alert.type === 'battery' && alert.droneId) {
      const backupDrone = drones.find(d => d.id !== alert.droneId && d.status === 'ready' && d.battery > 50)
      const area = searchAreas.find(a => a.assignedDroneId === alert.droneId)
      if (backupDrone && area) {
        returnDrone(alert.droneId)
        updateDrone(backupDrone.id, { armState: 'armed' })
        launchDrone(backupDrone.id, area.id)
        setSelectedDroneId(backupDrone.id)
        voice.speak(`Battery swap. ${backupDrone.name} armed and launching. ${drones.find(d => d.id === alert.droneId)?.name} returning.`, 'drone')
      }
    }
    dismissAlert(alert.id)
  }, [drones, searchAreas, returnDrone, launchDrone, updateDrone, dismissAlert, voice])

  const canKillSelected = selectedDrone?.status === 'flying' || selectedDrone?.status === 'returning'

  return (
    <div className="h-screen flex flex-col bg-surface-base overflow-hidden">
      {/* Top bar */}
      <TopBar
        connectionStatus={commsConnected ? 'connected' : 'disconnected'}
        gpsStatus="locked"
        armState={selectedDrone?.armState || 'disarmed'}
      />

      {/* Comms loss banner */}
      <CommsLostBanner
        connected={commsConnected}
        restored={commsRestored}
        onDismissRestored={() => setCommsRestored(false)}
      />

      {/* Main content */}
      <main className="flex-1 relative overflow-hidden">
        {/* Map view (default) */}
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
              onOpenStream={() => {}}
            />
          </>
        )}

        {/* Voice log view */}
        {viewMode === 'voice' && (
          <VoiceLog
            entries={voice.entries}
            isListening={voice.isListening}
            onToggleMic={voice.toggleListening}
            onBackToMap={() => setViewMode('map')}
          />
        )}
      </main>

      {/* Draw button (map view, drone selected and ready/armed) */}
      {viewMode === 'map' && selectedDroneId && (
        <DrawButton
          isDrawing={isDrawingArea}
          disabled={selectedDrone?.status !== 'ready' && selectedDrone?.status !== 'armed'}
          onClick={() => setIsDrawingArea(!isDrawingArea)}
        />
      )}

      {/* Fleet strip (map view only, minimal for 1:1) */}
      {viewMode === 'map' && (
        <FleetStrip
          drones={drones}
          selectedDroneId={selectedDroneId}
          onSelectDrone={(id) => setSelectedDroneId(id)}
          onKill={handleKillButton}
          killDisabled={!canKillSelected}
          selectedDroneName={canKillSelected ? selectedDrone?.name : undefined}
        />
      )}

      {/* Voice bar (always visible except voice log view) */}
      {viewMode !== 'voice' && (
        <VoiceBar
          isListening={voice.isListening}
          lastMessage={voice.lastMessage}
          onToggleMic={voice.toggleListening}
          onOpenLog={() => setViewMode('voice')}
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

      {/* Launch confirmation (arms + launches) */}
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
