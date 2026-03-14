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
  const [feedCollapsed, setFeedCollapsed] = useState(false)

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

  // Voice command handler — used by both voice recognition and text input
  const handleVoiceCommand = useCallback((command: string) => {
    // Add the operator's message to the log
    voice.addEntry('operator', command)
    const cmd = command.toLowerCase()

    // --- Launch / Search ---
    if (cmd.includes('launch') || cmd.includes('go') || cmd.includes('send') || cmd.includes('survey') || cmd.includes('scan') || cmd.includes('search') || cmd.includes('fly') || cmd.includes('patrol')) {
      voice.speak('Copy. Standing by for search area. Draw on map or specify grid.', 'drone')

    // --- Kill / Stop ---
    } else if (cmd.includes('kill') || cmd.includes('stop all')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && (drone.status === 'flying' || drone.status === 'returning')) {
        requestKillConfirmation(drone.id)
        voice.speak('Copy. Confirm kill?', 'drone')
      } else {
        voice.speak('Copy. No active drone to stop.', 'drone')
      }

    // --- Confirm ---
    } else if (cmd.includes('confirm') || cmd.includes('yes') || cmd.includes('affirmative')) {
      if (killConfirmation) {
        stopDrone(killConfirmation.droneId)
        setStoppedDroneId(killConfirmation.droneId)
        voice.speak(`Roger. ${killConfirmation.droneName} stopped. Hovering.`, 'drone')
      } else {
        voice.speak('Copy. Nothing to confirm.', 'drone')
      }

    // --- Comms check ---
    } else if (cmd.includes('comms') || cmd.includes('mesh status') || cmd.includes('connection')) {
      const connected = drones.filter(d => d.status !== 'offline')
      const offline = drones.filter(d => d.status === 'offline')
      const parts = []
      if (connected.length > 0) parts.push(`${connected.length} drone${connected.length > 1 ? 's' : ''} on mesh: ${connected.map(d => d.name).join(', ')}`)
      if (offline.length > 0) parts.push(`${offline.length} offline: ${offline.map(d => d.name).join(', ')}`)
      voice.speak(`Roger. ${parts.join('. ')}.`, 'drone')

    // --- Battery query (specific drone) ---
    } else if (cmd.includes('battery')) {
      // Check if they asked about a specific drone by number
      const droneMatch = cmd.match(/(?:drone|bird)\s*(\d+)/i)
      if (droneMatch) {
        const drone = drones.find(d => d.name.toLowerCase().includes(droneMatch[1]))
        if (drone) {
          voice.speak(`Roger. ${drone.name}, battery ${drone.battery}%.${drone.battery < 20 ? ' Low battery.' : ''}`, 'drone')
        } else {
          voice.speak('Copy. Drone not found.', 'drone')
        }
      } else {
        // All drones battery
        const report = drones.map(d => `${d.name} ${d.battery}%`).join(', ')
        voice.speak(`Roger. ${report}.`, 'drone')
      }

    // --- Status ---
    } else if (cmd.includes('status') || cmd.includes('sitrep')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone) {
        voice.speak(`Roger. ${drone.name}. Battery ${drone.battery}%. Altitude ${drone.altitude} feet. ${drone.status}. ${drone.armState}.`, 'drone')
      }

    // --- Grid query ---
    } else if (cmd.includes('grid') || cmd.includes('mgrs') || cmd.includes('coordinates') || cmd.includes('position')) {
      const recent = detections[detections.length - 1]
      if (recent) {
        voice.speak(`Roger. Last contact grid ${recent.mgrs}. ${recent.type}, ${Math.round(recent.confidence * 100)}% confidence.`, 'drone')
      } else {
        const drone = drones.find(d => d.id === selectedDroneId)
        if (drone) {
          const grid = toMGRS(drone.position.lat, drone.position.lng)
          voice.speak(`Roger. ${drone.name} position grid ${grid}.`, 'drone')
        }
      }

    // --- Disarm (must be before arm check) ---
    } else if (cmd.includes('disarm')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && drone.armState === 'armed' && drone.status !== 'flying') {
        updateDrone(drone.id, { armState: 'disarmed', status: 'ready' })
        voice.speak('Roger. Disarmed.', 'drone')
      } else if (drone?.status === 'flying') {
        voice.speak('Copy. Cannot disarm while flying. Use kill to stop.', 'drone')
      } else {
        voice.speak('Copy. Already disarmed.', 'drone')
      }

    // --- Arm ---
    } else if (cmd.includes('arm')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && drone.armState === 'disarmed') {
        updateDrone(drone.id, { armState: 'armed', status: 'armed' })
        voice.speak('Roger. Armed. Ready for launch.', 'drone')
      } else if (drone?.armState === 'armed') {
        voice.speak('Copy. Already armed.', 'drone')
      }

    // --- Return / RTL ---
    } else if (cmd.includes('return') || cmd.includes('home') || cmd.includes('rtl') || cmd.includes('come back')) {
      const drone = drones.find(d => d.id === selectedDroneId)
      if (drone && drone.status === 'flying') {
        returnDrone(drone.id)
        voice.speak('Roger. Returning to launch.', 'drone')
      } else {
        voice.speak('Copy. No active flight to return.', 'drone')
      }

    // --- Watch / Track (persistent surveillance) ---
    } else if (cmd.includes('watch') || cmd.includes('track') || cmd.includes('monitor') || cmd.includes('observe') || cmd.includes('eyes on')) {
      voice.speak('Copy. Setting persistent watch on target area. Will alert on movement.', 'drone')

    // --- Count detections ---
    } else if (cmd.includes('how many') || cmd.includes('count')) {
      const people = detections.filter(d => d.type === 'person').length
      const vehicles = detections.filter(d => d.type === 'vehicle').length
      const animals = detections.filter(d => d.type === 'animal').length
      const parts = []
      if (people > 0) parts.push(`${people} person${people > 1 ? 's' : ''}`)
      if (vehicles > 0) parts.push(`${vehicles} vehicle${vehicles > 1 ? 's' : ''}`)
      if (animals > 0) parts.push(`${animals} animal${animals > 1 ? 's' : ''}`)
      voice.speak(parts.length > 0 ? `Roger. ${parts.join(', ')}.` : 'Roger. No contacts.', 'drone')

    // --- Priority contact ---
    } else if (cmd.includes('priority') || cmd.includes('highest') || cmd.includes('urgent')) {
      const sorted = [...detections].sort((a, b) => b.confidence - a.confidence)
      const top = sorted[0]
      if (top) {
        voice.speak(`Roger. Priority contact: ${top.type}, ${Math.round(top.confidence * 100)}% confidence, grid ${top.mgrs}.`, 'drone')
      } else {
        voice.speak('Roger. No contacts.', 'drone')
      }

    // --- Last contact / detection ---
    } else if (cmd.includes('contact') || cmd.includes('detection') || cmd.includes('last')) {
      const recent = detections[detections.length - 1]
      if (recent) {
        voice.speak(`Roger. Last contact: ${recent.type}, ${Math.round(recent.confidence * 100)}%, grid ${recent.mgrs}.`, 'drone')
      } else {
        voice.speak('Roger. No contacts.', 'drone')
      }

    // --- Summarize / Debrief ---
    } else if (cmd.includes('summarize') || cmd.includes('summary') || cmd.includes('debrief') || cmd.includes('what happened')) {
      const totalDetections = detections.length
      const flagged = detections.filter(d => d.flagged).length
      const people = detections.filter(d => d.type === 'person').length
      const vehicles = detections.filter(d => d.type === 'vehicle').length
      const flyingDrones = drones.filter(d => d.status === 'flying').length
      voice.speak(`Roger. Mission summary: ${totalDetections} total contacts — ${people} persons, ${vehicles} vehicles. ${flagged} flagged for review. ${flyingDrones} drone${flyingDrones !== 1 ? 's' : ''} currently airborne.`, 'drone')

    // --- Flag ---
    } else if (cmd.includes('flag') || cmd.includes('mark')) {
      const recent = detections[detections.length - 1]
      if (recent && !recent.flagged) {
        flagDetection(recent.id)
        voice.speak('Roger. Flagged to commander.', 'drone')
      } else if (recent?.flagged) {
        voice.speak('Copy. Already flagged.', 'drone')
      } else {
        voice.speak('Copy. No contact to flag.', 'drone')
      }

    // --- Help ---
    } else if (cmd.includes('help') || cmd.includes('commands')) {
      voice.speak('Commands: launch, arm, disarm, status, kill, return, scan, grid, battery, comms, watch, count, priority, summarize, flag, help.', 'drone')

    // --- Fallback ---
    } else {
      voice.speak('Copy. Command not recognized. Say help for available commands.', 'drone')
    }
  }, [drones, selectedDroneId, detections, killConfirmation, requestKillConfirmation, stopDrone, returnDrone, updateDrone, flagDetection, toMGRS])

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
        viewMode={viewMode}
        onSetView={(v) => setViewMode(v)}
        onToggleArm={() => {
          if (!selectedDrone) return
          if (selectedDrone.armState === 'disarmed') {
            updateDrone(selectedDrone.id, { armState: 'armed', status: 'armed' })
            voice.speak('Armed. Ready for launch.', 'drone')
          } else if (selectedDrone.status !== 'flying' && selectedDrone.status !== 'returning') {
            updateDrone(selectedDrone.id, { armState: 'disarmed', status: 'ready' })
            voice.speak('Disarmed.', 'drone')
          }
        }}
        canDisarm={selectedDrone?.status !== 'flying' && selectedDrone?.status !== 'returning'}
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
          </>
        )}

        {/* Voice log view */}
        {viewMode === 'voice' && (
          <VoiceLog
            entries={voice.entries}
            isListening={voice.isListening}
            feedCollapsed={feedCollapsed}
            onToggleMic={voice.toggleListening}
            onTextCommand={handleVoiceCommand}
          />
        )}

        {/* Detection feed — visible in both map and voice views */}
        <DetectionFeed
          detections={detections}
          alerts={alerts}
          selectedDrone={selectedDrone}
          collapsed={feedCollapsed}
          onToggleCollapsed={() => setFeedCollapsed(c => !c)}
          onDetectionClick={handleDetectionClick}
          onAlertAction={handleAlertAction}
          onAlertDismiss={dismissAlert}
          onOpenStream={() => {}}
        />
      </main>

      {/* Draw button (map view, drone selected and ready/armed) */}
      {viewMode === 'map' && selectedDroneId && (
        <DrawButton
          isDrawing={isDrawingArea}
          disabled={selectedDrone?.status !== 'ready' && selectedDrone?.status !== 'armed'}
          onClick={() => setIsDrawingArea(!isDrawingArea)}
        />
      )}

      {/* Floating KILL button — bottom-right of map area, left of feed */}
      {viewMode === 'map' && canKillSelected && (
        <button
          onClick={handleKillButton}
          className="fixed z-[1100] bg-kill hover:bg-kill-hover text-text-primary font-bold text-xs tracking-wider px-3 py-1.5 rounded border border-kill-hover/30 transition-colors"
          style={{ bottom: '100px', right: '240px' }}
        >
          KILL
        </button>
      )}

      {/* Voice bar (always visible except voice log view) */}
      {viewMode !== 'voice' && (
        <VoiceBar
          isListening={voice.isListening}
          lastMessage={voice.lastMessage}
          onToggleMic={voice.toggleListening}
          onTextCommand={handleVoiceCommand}
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
