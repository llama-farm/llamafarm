import { useState, useCallback, useEffect } from 'react'
import { Panel, PanelGroup, PanelResizeHandle } from 'react-resizable-panels'
import type { ViewMode, DronePosition, SearchArea, Alert } from './types'
import { useDrones } from './hooks/useDrones'
import { useSimulation } from './hooks/useSimulation'
import { useVoice } from './hooks/useVoice'
import { useMGRS } from './hooks/useMGRS'
import { TopBar } from './components/TopBar'
import { MapView } from './components/Map/MapView'
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
import { AltitudeSparkline } from './components/AltitudeSparkline'
import { VolumeControl } from './components/VolumeControl'

// Desktop center panel with Telemetry/Map tabs (like Ace Drone's Spatial/Telemetry/Map)
function DesktopCenterTabs({ selectedDrone, drones, detections, searchAreas, selectedDroneId, isDrawingArea, onDrawComplete, onDetectionClick }: {
  selectedDrone: any, drones: any[], detections: any[], searchAreas: any[], selectedDroneId: string | null,
  isDrawingArea: boolean, onDrawComplete: any, onDetectionClick: any
}) {
  const [tab, setTab] = useState<'telemetry' | 'map'>('telemetry')
  const isActive = selectedDrone?.status === 'flying' || selectedDrone?.status === 'returning'

  const tabCls = (t: string) => `px-4 py-1.5 text-[11px] uppercase tracking-widest transition-colors ${
    tab === t ? 'text-text-primary border-b-2 border-accent' : 'text-text-dim hover:text-text-secondary border-b-2 border-transparent'
  }`

  return (
    <div className="flex-1 flex flex-col min-w-[300px] bg-surface-base">
      {/* Tab bar */}
      <div className="flex items-center border-b border-surface-border bg-surface-raised/40 px-2">
        <button className={tabCls('telemetry')} onClick={() => setTab('telemetry')}>Telemetry</button>
        <button className={tabCls('map')} onClick={() => setTab('map')}>Map</button>
        {selectedDrone && (
          <div className="ml-auto flex items-center gap-3 text-[10px] font-mono text-text-dim pr-2">
            <span>Alt: {isActive ? `${Math.round(selectedDrone.altitude)}m` : '---'}</span>
            <span>Bat: {Math.round(selectedDrone.battery)}%</span>
          </div>
        )}
      </div>

      {/* Tab content */}
      {tab === 'telemetry' && selectedDrone && (
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {/* Big readouts */}
          <div className="grid grid-cols-4 gap-3">
            {[
              { label: 'ALTITUDE', value: isActive ? `${Math.round(selectedDrone.altitude)}` : '---', unit: 'm', color: 'text-accent' },
              { label: 'SPEED', value: isActive ? `${selectedDrone.speed.toFixed(1)}` : '---', unit: 'm/s', color: 'text-text-primary' },
              { label: 'HEADING', value: isActive ? `${String(selectedDrone.heading).padStart(3, '0')}` : '---', unit: '°', color: 'text-text-primary' },
              { label: 'BATTERY', value: `${Math.round(selectedDrone.battery)}`, unit: '%', color: selectedDrone.battery < 20 ? 'text-status-critical' : selectedDrone.battery < 40 ? 'text-status-warning' : 'text-status-good' },
            ].map(t => (
              <div key={t.label} className="bg-surface-raised/60 border border-surface-border rounded-lg px-3 py-3 text-center">
                <div className="text-[9px] text-text-dim uppercase tracking-widest mb-1">{t.label}</div>
                <div className={`font-mono text-2xl font-bold ${t.color} leading-none`}>
                  {t.value}<span className="text-xs text-text-dim ml-0.5">{t.unit}</span>
                </div>
              </div>
            ))}
          </div>

          {/* Battery bar */}
          <div className="bg-surface-raised/60 border border-surface-border rounded-lg px-4 py-3">
            <div className="flex items-center justify-between mb-1.5">
              <span className="text-[9px] text-text-dim uppercase tracking-widest">Battery</span>
              <span className={`text-xs font-mono ${selectedDrone.battery < 20 ? 'text-status-critical' : 'text-text-dim'}`}>{Math.round(selectedDrone.battery)}%</span>
            </div>
            <div className="w-full h-2 bg-surface-overlay rounded-full overflow-hidden">
              <div className={`h-full rounded-full transition-all ${
                selectedDrone.battery < 20 ? 'bg-status-critical' : selectedDrone.battery < 40 ? 'bg-status-warning' : 'bg-status-good'
              }`} style={{ width: `${selectedDrone.battery}%` }} />
            </div>
          </div>

          {/* Elevation graph */}
          {isActive && (
            <div className="bg-surface-raised/60 border border-surface-border rounded-lg px-4 py-3">
              <div className="text-[9px] text-text-dim uppercase tracking-widest mb-2">Elevation Over Time</div>
              <AltitudeSparkline altitude={selectedDrone.altitude} />
            </div>
          )}

          {/* Position info */}
          <div className="grid grid-cols-2 gap-3">
            <div className="bg-surface-raised/60 border border-surface-border rounded-lg px-4 py-3">
              <div className="text-[9px] text-text-dim uppercase tracking-widest mb-1">Position</div>
              <div className="font-mono text-xs text-text-secondary">
                {selectedDrone.position.lat.toFixed(6)}, {selectedDrone.position.lng.toFixed(6)}
              </div>
            </div>
            <div className="bg-surface-raised/60 border border-surface-border rounded-lg px-4 py-3">
              <div className="text-[9px] text-text-dim uppercase tracking-widest mb-1">Status</div>
              <div className="font-mono text-xs text-text-secondary uppercase">{selectedDrone.status} · {selectedDrone.armState}</div>
            </div>
          </div>
        </div>
      )}

      {tab === 'map' && (
        <div className="flex-1 relative overflow-hidden">
          <MapView
            drones={drones}
            detections={detections}
            searchAreas={searchAreas}
            selectedDroneId={selectedDroneId}
            isDrawingArea={isDrawingArea}
            onDrawComplete={onDrawComplete}
            onDetectionClick={onDetectionClick}
                onRemoveZone={undefined}
          />
        </div>
      )}
    </div>
  )
}

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
    addBatteryAlert,
    removeSearchArea
  } = useDrones()

  // Voice command handler — used by both voice recognition and text input
  const handleVoiceCommand = useCallback((command: string) => {
    // Add the operator's message to the log
    voice.addEntry('operator', command)
    const cmd = command.toLowerCase()

    // --- Launch / Search ---
    if (cmd.includes('launch') || cmd.includes('go') || cmd.includes('send') || cmd.includes('survey') || cmd.includes('scan') || cmd.includes('search') || cmd.includes('fly') || cmd.includes('patrol')) {
      // Check for "zone N" or "area N" in command
      const zoneMatch = cmd.match(/(?:zone|area)\s*(\d+)/i)
      if (zoneMatch && selectedDroneId) {
        const zoneIdx = parseInt(zoneMatch[1]) - 1
        if (zoneIdx >= 0 && zoneIdx < searchAreas.length) {
          const area = searchAreas[zoneIdx]
          setPendingLaunch({ droneId: selectedDroneId, searchAreaId: area.id })
          voice.speak(`Copy. Launching to zone ${zoneIdx + 1}.`, 'drone')
        } else {
          voice.speak(`Copy. Zone ${zoneIdx + 1} not found. ${searchAreas.length} zones defined.`, 'drone')
        }
      } else if (searchAreas.length === 0) {
        // Auto-enable draw mode if no zones exist
        setIsDrawingArea(true)
        voice.speak('Copy. Draw search area on map.', 'drone')
      } else if (searchAreas.length === 1 && selectedDroneId) {
        // Only one zone — launch to it automatically
        setPendingLaunch({ droneId: selectedDroneId, searchAreaId: searchAreas[0].id })
        voice.speak('Copy. Launching to zone 1.', 'drone')
      } else {
        voice.speak(`Copy. ${searchAreas.length} zones defined. Specify zone number.`, 'drone')
      }

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
      const droneMatch = cmd.match(/(?:drone|bird)\s*(\d+)/i)
      if (droneMatch) {
        const drone = drones.find(d => d.name.toLowerCase().includes(droneMatch[1]))
        if (drone) {
          voice.speak(`Roger. ${drone.name}, battery ${drone.battery}%.${drone.battery < 20 ? ' Low battery.' : ''}`, 'drone')
        } else {
          voice.speak('Copy. Drone not found.', 'drone')
        }
      } else {
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

    // --- Watch / Track ---
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
  // Only announce high-confidence person/vehicle detections via voice — rest goes to feed only
  useEffect(() => {
    if (detections.length > 0) {
      const latest = detections[detections.length - 1]
      if (Date.now() - latest.timestamp.getTime() < 2000) {
        // Only voice-announce person or vehicle with confidence > 80%
        if ((latest.type === 'person' || latest.type === 'vehicle') && latest.confidence > 0.8) {
          voice.announceDetection(latest.type, latest.confidence, latest.mgrs, latest.id)
        }
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

  // Shared detection feed props
  const feedProps = {
    detections,
    alerts,
    selectedDrone,
    collapsed: feedCollapsed,
    onToggleCollapsed: () => setFeedCollapsed(c => !c),
    onDetectionClick: handleDetectionClick,
    onAlertAction: handleAlertAction,
    onAlertDismiss: dismissAlert,
    onOpenStream: () => {},
  }

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

      {/* ===== MAIN CONTENT ===== */}
      {/* Mobile: single column (existing behavior) */}
      {/* Desktop (lg+): 3-column GCS layout — left instruments, center map, right feed+chat */}
      <main className="flex-1 relative overflow-hidden flex flex-col lg:flex-row">

        {/* ---- DESKTOP LAYOUT: Resizable panels ---- */}
        {/* Columns: Telemetry | Map+Stream (stacked) | Detections */}
        {/* Bottom: Chat (tall) */}
        <div className="hidden lg:flex flex-col flex-1 min-h-0">
          <PanelGroup direction="vertical">
            {/* Top: 3-column area */}
            <Panel defaultSize={70} minSize={40}>
              <PanelGroup direction="horizontal">

                {/* LEFT: Telemetry — its own column */}
                <Panel defaultSize={30} minSize={18}>
                  <div className="h-full flex flex-col bg-surface-base overflow-y-auto border-r border-surface-border">
                    {/* Header */}
                    <div className="px-3 py-2 border-b border-surface-border bg-surface-raised/40 flex items-center justify-between shrink-0">
                      <span className="text-[10px] text-text-dim uppercase tracking-[3px]">Telemetry</span>
                      {selectedDrone && (
                        <span className={`text-[10px] font-mono ${
                          selectedDrone.status === 'flying' ? 'text-status-good' : 'text-text-dim'
                        }`}>
                          {selectedDrone.status === 'flying' ? 'ACTIVE' :
                           selectedDrone.status === 'armed' ? 'STANDBY' :
                           selectedDrone.status === 'returning' ? 'RTL' : 'IDLE'}
                        </span>
                      )}
                    </div>

                    {selectedDrone && (
                      <div className="p-3 space-y-3 flex-1">
                        {/* Pairing */}
                        <div className="flex items-center gap-2 text-[10px]">
                          <div className={`w-1.5 h-1.5 rounded-full ${
                            selectedDrone.status === 'flying' ? 'bg-status-good animate-pulse' :
                            selectedDrone.status === 'armed' ? 'bg-arm-armed' : 'bg-arm-disarmed'
                          }`} />
                          <span className="font-mono font-bold text-xs">{selectedDrone.name}</span>
                          <span className="text-status-good font-mono">PAIRED</span>
                          <span className="text-status-good font-mono">MESH</span>
                        </div>

                        {/* Big readouts */}
                        <div className="grid grid-cols-2 gap-2">
                          {[
                            { label: 'ALT', value: (selectedDrone.status === 'flying' || selectedDrone.status === 'returning') ? `${Math.round(selectedDrone.altitude)}` : '---', unit: 'm', color: 'text-accent' },
                            { label: 'SPD', value: (selectedDrone.status === 'flying' || selectedDrone.status === 'returning') ? `${selectedDrone.speed.toFixed(1)}` : '---', unit: 'm/s', color: 'text-text-primary' },
                            { label: 'HDG', value: (selectedDrone.status === 'flying' || selectedDrone.status === 'returning') ? `${String(selectedDrone.heading).padStart(3, '0')}` : '---', unit: '°', color: 'text-text-primary' },
                            { label: 'BAT', value: `${Math.round(selectedDrone.battery)}`, unit: '%', color: selectedDrone.battery < 20 ? 'text-status-critical' : selectedDrone.battery < 40 ? 'text-status-warning' : 'text-status-good' },
                          ].map(t => (
                            <div key={t.label} className="bg-surface-raised/60 border border-surface-border rounded-lg px-2 py-2 text-center">
                              <div className="text-[8px] text-text-dim uppercase tracking-widest">{t.label}</div>
                              <div className={`font-mono text-xl font-bold ${t.color} leading-tight`}>
                                {t.value}<span className="text-[10px] text-text-dim ml-0.5">{t.unit}</span>
                              </div>
                            </div>
                          ))}
                        </div>

                        {/* Battery bar */}
                        <div className="bg-surface-raised/60 border border-surface-border rounded-lg px-3 py-2">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-[8px] text-text-dim uppercase tracking-widest">Battery</span>
                            <span className={`text-[10px] font-mono ${selectedDrone.battery < 20 ? 'text-status-critical' : 'text-text-dim'}`}>{Math.round(selectedDrone.battery)}%</span>
                          </div>
                          <div className="w-full h-1.5 bg-surface-overlay rounded-full overflow-hidden">
                            <div className={`h-full rounded-full transition-all ${
                              selectedDrone.battery < 20 ? 'bg-status-critical' : selectedDrone.battery < 40 ? 'bg-status-warning' : 'bg-status-good'
                            }`} style={{ width: `${selectedDrone.battery}%` }} />
                          </div>
                        </div>

                        {/* Elevation graph — show empty state when idle */}
                        <div className="bg-surface-raised/60 border border-surface-border rounded-lg px-3 py-2">
                          <div className="text-[8px] text-text-dim uppercase tracking-widest mb-1">Elevation</div>
                          {(selectedDrone.status === 'flying' || selectedDrone.status === 'returning') ? (
                            <AltitudeSparkline altitude={selectedDrone.altitude} />
                          ) : (
                            <div className="h-[32px] flex items-center justify-center">
                              <div className="w-full h-[1px] bg-surface-border relative">
                                <span className="absolute left-0 top-1 text-[8px] text-text-dim/30 font-mono">0m</span>
                                <span className="absolute right-0 top-1 text-[8px] text-text-dim/30 font-mono">ALT</span>
                              </div>
                            </div>
                          )
                          }
                        </div>

                        {/* Position */}
                        <div className="bg-surface-raised/60 border border-surface-border rounded-lg px-3 py-2">
                          <div className="text-[8px] text-text-dim uppercase tracking-widest mb-1">Position</div>
                          <div className="font-mono text-[11px] text-text-secondary">
                            {selectedDrone.position.lat.toFixed(6)}, {selectedDrone.position.lng.toFixed(6)}
                          </div>
                        </div>

                        {/* Controls */}
                        <div className="flex gap-2">
                          <button
                            onClick={() => handleVoiceCommand(selectedDrone.armState === 'armed' ? 'disarm' : 'arm')}
                            disabled={selectedDrone.status === 'flying' || selectedDrone.status === 'returning'}
                            className={`flex-1 py-2 rounded text-[10px] font-bold tracking-wider transition-colors ${
                              selectedDrone.armState === 'armed'
                                ? 'bg-arm-armed/20 text-arm-armed border border-arm-armed/30'
                                : 'bg-surface-overlay text-text-secondary border border-surface-border'
                            } ${selectedDrone.status === 'flying' || selectedDrone.status === 'returning' ? 'opacity-40 cursor-not-allowed' : ''}`}
                          >
                            {selectedDrone.armState === 'armed' ? 'ARMED' : 'ARM'}
                          </button>
                          <button
                            onClick={() => setIsDrawingArea(!isDrawingArea)}
                            disabled={selectedDrone.status !== 'ready' && selectedDrone.status !== 'armed'}
                            className={`flex-1 py-2 rounded text-[10px] font-medium tracking-wider transition-colors ${
                              isDrawingArea ? 'bg-accent/20 text-accent border border-accent/30'
                              : 'bg-surface-overlay text-text-secondary border border-surface-border'
                            } ${selectedDrone.status !== 'ready' && selectedDrone.status !== 'armed' ? 'opacity-40 cursor-not-allowed' : ''}`}
                          >
                            {isDrawingArea ? 'Cancel' : 'Draw Area'}
                          </button>
                          <VolumeControl volume={voice.volume} onVolumeChange={voice.setVolume} muted={voice.muted} onToggleMute={voice.toggleMute} />
                        </div>

                        {/* KILL button — desktop only, in telemetry panel */}
                        {(selectedDrone.status === 'flying' || selectedDrone.status === 'returning') && (
                          <button
                            onClick={handleKillButton}
                            className="w-full py-1.5 rounded bg-red-900/60 hover:bg-red-800/80 active:scale-[0.98] text-red-400 font-bold text-[10px] tracking-widest transition-all border border-red-800/40"
                          >
                            KILL
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                </Panel>

                <PanelResizeHandle className="w-1 bg-surface-border hover:bg-accent/30 transition-colors cursor-col-resize" />

                {/* CENTER: Map + Stream stacked */}
                <Panel defaultSize={45} minSize={25}>
                  <PanelGroup direction="vertical">
                    {/* Map — top portion */}
                    <Panel defaultSize={60} minSize={20}>
                      <div className="h-full relative overflow-hidden">
                        <MapView
                          drones={drones}
                          detections={detections}
                          searchAreas={searchAreas}
                          selectedDroneId={selectedDroneId}
                          isDrawingArea={isDrawingArea}
                          onDrawComplete={handleDrawComplete}
                          onDetectionClick={handleDetectionClick}
                onRemoveZone={removeSearchArea}
                        />
                      </div>
                    </Panel>

                    <PanelResizeHandle className="h-1 bg-surface-border hover:bg-accent/30 transition-colors cursor-row-resize" />

                    {/* Stream — bottom portion, matches mobile style */}
                    <Panel defaultSize={40} minSize={15}>
                      <div className="h-full relative bg-black flex items-center justify-center overflow-hidden">
                        {/* LIVE tag */}
                        <div className="absolute top-2 left-2 z-10 flex items-center gap-1.5">
                          <div className="flex items-center gap-1 bg-red-600/80 px-1.5 py-0.5 rounded text-[9px] font-bold text-white">
                            <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
                            LIVE
                          </div>
                          {selectedDrone && (
                            <span className="text-[9px] font-mono text-white/60">{selectedDrone.name}</span>
                          )}
                        </div>
                        {/* Drone status overlay */}
                        {selectedDrone && (
                          <div className="absolute top-2 right-2 z-10 flex items-center gap-1.5 text-[9px] font-mono">
                            <span className={`px-1.5 py-0.5 rounded ${
                              selectedDrone.status === 'flying' ? 'bg-status-good/20 text-status-good' :
                              selectedDrone.status === 'armed' ? 'bg-arm-armed/20 text-arm-armed' :
                              'bg-white/10 text-white/40'
                            }`}>
                              {selectedDrone.status === 'flying' ? 'FLYING' :
                               selectedDrone.status === 'armed' ? 'ARMED' :
                               selectedDrone.status === 'returning' ? 'RTL' : 'IDLE'}
                            </span>
                          </div>
                        )}
                        {/* Placeholder content */}
                        <div className="text-center">
                          <div className="text-text-dim/20 text-[10px]">No video source connected</div>
                          <div className="text-text-dim/15 text-[9px] mt-0.5">Connect drone camera or sim feed</div>
                        </div>
                      </div>
                    </Panel>
                  </PanelGroup>
                </Panel>

                <PanelResizeHandle className="w-1 bg-surface-border hover:bg-accent/30 transition-colors cursor-col-resize" />

                {/* RIGHT: Detections */}
                <Panel defaultSize={25} minSize={15}>
                  <div className="h-full flex flex-col border-l border-surface-border bg-surface-raised/50 overflow-hidden">
                    <DetectionFeed {...feedProps} layout="desktop" hideStream />
                  </div>
                </Panel>
              </PanelGroup>
            </Panel>

            <PanelResizeHandle className="h-1 bg-surface-border hover:bg-accent/30 transition-colors cursor-row-resize" />

            {/* BOTTOM: Chat — resizable, defaults tall */}
            <Panel defaultSize={30} minSize={12}>
              <div className="h-full flex flex-col border-t border-surface-border bg-surface-bar">
                <VoiceLog
                  entries={voice.entries}
                  isListening={voice.isListening}
                  onToggleMic={voice.toggleListening}
                  onTextCommand={handleVoiceCommand}
                  layout="desktop"
                />
              </div>
            </Panel>
          </PanelGroup>
        </div>

        {/* ---- MOBILE LAYOUT: existing single-column (unchanged) ---- */}
        <div className="flex-1 relative overflow-hidden lg:hidden flex flex-col">
          <div className="flex-1 relative overflow-hidden">
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
                onRemoveZone={removeSearchArea}
              />
              {selectedDrone && (
                <TelemetryHUD drone={selectedDrone} onKill={handleKillButton} canKill={canKillSelected} />
              )}
            </>
          )}

          {/* Voice log view — mobile */}
          {viewMode === 'voice' && (
            <div className="absolute inset-0">
              <VoiceLog
                entries={voice.entries}
                isListening={voice.isListening}
                feedCollapsed={feedCollapsed}
                onToggleMic={voice.toggleListening}
                onTextCommand={handleVoiceCommand}
              />
            </div>
          )}

          {/* Detection feed — mobile */}
          <DetectionFeed {...feedProps} layout="mobile" />

          {/* Draw button — mobile */}
          {viewMode === 'map' && selectedDroneId && (
            <DrawButton
              isDrawing={isDrawingArea}
              disabled={selectedDrone?.status !== 'ready' && selectedDrone?.status !== 'armed'}
              onClick={() => setIsDrawingArea(!isDrawingArea)}
            />
          )}
          </div>
        </div>
      </main>

      {/* Voice bar — MOBILE only */}
      <div className="lg:hidden">
        {viewMode !== 'voice' && (
          <div className="relative">
            <VoiceBar
              isListening={voice.isListening}
              lastMessage={voice.lastMessage}
              onToggleMic={voice.toggleListening}
              onTextCommand={handleVoiceCommand}
            />
            <div className="absolute top-1/2 -translate-y-1/2 right-14 z-[1200]">
              <VolumeControl volume={voice.volume} onVolumeChange={voice.setVolume} muted={voice.muted} onToggleMute={voice.toggleMute} />
            </div>
          </div>
        )}
      </div>

      {/* Floating KILL button removed — kill is in telemetry panel (desktop) and TelemetryHUD (mobile) */}

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
