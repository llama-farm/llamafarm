import { useEffect, useRef, useState } from 'react'
import type { Detection, Drone } from '../../types'
import { getDetectionColor, getDroneColor } from '../../types'
import { detectionLabels } from '../../data/mockDetections'
import { MapContainer, TileLayer, Marker, useMap } from 'react-leaflet'
import L from 'leaflet'
import { MAP_CENTER } from '../../data/mockDrones'

// Mini-map controller that follows the drone
function MiniMapTracker({ position }: { position: { lat: number; lng: number } }) {
  const map = useMap()

  useEffect(() => {
    map.setView([position.lat, position.lng], 16, { animate: true })
  }, [position.lat, position.lng, map])

  return null
}

interface FeedViewProps {
  drone: Drone
  droneIndex: number
  detection?: Detection
  onBackToMap: () => void
  onFlag: () => void
}

export function FeedView({ drone, droneIndex, detection, onBackToMap, onFlag }: FeedViewProps) {
  const [flagged, setFlagged] = useState(false)
  const [showFlagConfirm, setShowFlagConfirm] = useState(false)
  const miniMapRef = useRef<L.Map>(null)

  // Track detection flagged state
  useEffect(() => {
    if (detection?.flagged) {
      setFlagged(true)
    }
  }, [detection?.flagged])

  const handleFlag = () => {
    onFlag()
    setFlagged(true)
    setShowFlagConfirm(true)
    setTimeout(() => setShowFlagConfirm(false), 2000)
  }

  const formatTime = () => {
    const now = new Date()
    return now.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const droneColor = getDroneColor(droneIndex)
  const droneIcon = L.divIcon({
    className: 'drone-marker',
    html: `
      <div style="width: 20px; height: 20px; display: flex; align-items: center; justify-content: center;">
        <svg viewBox="0 0 24 24" fill="${droneColor}" width="20" height="20">
          <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
        </svg>
      </div>
    `,
    iconSize: [20, 20],
    iconAnchor: [10, 10]
  })

  return (
    <div className="relative w-full h-full bg-surface-base flex flex-col view-enter">
      {/* Mock video feed */}
      <div className="flex-1 relative overflow-hidden">
        {/* Video placeholder - simulated feed */}
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
          {/* Scan lines effect */}
          <div
            className="absolute inset-0 opacity-10"
            style={{
              backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)'
            }}
          />

          {/* Simulated aerial view */}
          <svg className="absolute inset-0 w-full h-full opacity-25" viewBox="0 0 200 120" preserveAspectRatio="xMidYMid slice">
            {/* Roads */}
            <line x1="40" y1="0" x2="50" y2="120" stroke="#3a3a3a" strokeWidth="2" />
            <line x1="120" y1="0" x2="110" y2="120" stroke="#3a3a3a" strokeWidth="2" />
            <line x1="0" y1="50" x2="200" y2="55" stroke="#3a3a3a" strokeWidth="2" />
            <line x1="0" y1="80" x2="200" y2="78" stroke="#3a3a3a" strokeWidth="1.5" />
            {/* Buildings */}
            <rect x="55" y="15" width="18" height="25" fill="#2a2a2a" />
            <rect x="80" y="20" width="12" height="18" fill="#2a2a2a" />
            <rect x="130" y="60" width="22" height="16" fill="#2a2a2a" />
            <rect x="25" y="65" width="14" height="20" fill="#2a2a2a" />
            <rect x="155" y="25" width="16" height="12" fill="#2a2a2a" />
            <rect x="70" y="85" width="20" height="14" fill="#2a2a2a" />
            {/* Trees */}
            <circle cx="100" cy="40" r="4" fill="#1a3a1a" />
            <circle cx="160" cy="95" r="5" fill="#1a3a1a" />
            <circle cx="35" cy="35" r="3" fill="#1a3a1a" />
            <circle cx="145" cy="45" r="3.5" fill="#1a3a1a" />
          </svg>

          {/* Detection bounding box overlay */}
          {detection && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div
                className="w-28 h-36 border-2 rounded relative"
                style={{
                  borderColor: getDetectionColor(detection.type),
                  boxShadow: `0 0 20px ${getDetectionColor(detection.type)}40`,
                  animation: 'pulse 2s ease-in-out infinite'
                }}
              >
                {/* Corner brackets */}
                <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2" style={{ borderColor: getDetectionColor(detection.type) }} />
                <div className="absolute -top-1 -right-1 w-4 h-4 border-t-2 border-r-2" style={{ borderColor: getDetectionColor(detection.type) }} />
                <div className="absolute -bottom-1 -left-1 w-4 h-4 border-b-2 border-l-2" style={{ borderColor: getDetectionColor(detection.type) }} />
                <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2" style={{ borderColor: getDetectionColor(detection.type) }} />

                {/* Label */}
                <div
                  className="absolute -top-7 left-0 px-2 py-0.5 text-xs font-bold text-white rounded"
                  style={{ backgroundColor: getDetectionColor(detection.type) }}
                >
                  {detectionLabels[detection.type]} {Math.round(detection.confidence * 100)}%
                </div>
              </div>
            </div>
          )}
        </div>

        {/* HUD overlay - top */}
        <div className="absolute top-0 left-0 right-0 p-3 flex justify-between items-start text-white">
          <div className="bg-black/60 px-3 py-2 rounded-lg text-sm font-mono">
            <div className="font-bold">{drone.name} stream</div>
            <div className="text-xs text-gray-400">{drone.status.toUpperCase()}</div>
          </div>

          {/* LIVE badge */}
          <div className="bg-red-600 px-3 py-1.5 rounded text-sm font-bold live-badge flex items-center gap-1.5">
            <span className="w-2 h-2 bg-white rounded-full"></span>
            LIVE
          </div>

          <div className="bg-black/60 px-3 py-2 rounded-lg text-sm font-mono">
            {formatTime()}
          </div>
        </div>

        {/* HUD overlay - telemetry */}
        <div className="absolute bottom-4 left-3 text-white">
          <div className="bg-black/60 px-3 py-2 rounded-lg space-y-0.5 font-mono text-xs">
            <div>ALT: {drone.altitude > 0 ? `${Math.round(drone.altitude)}m` : '--'}</div>
            <div>SPD: {drone.speed > 0 ? `${drone.speed.toFixed(1)} m/s` : '--'}</div>
            <div>HDG: {drone.heading > 0 ? `${String(Math.round(drone.heading)).padStart(3, '0')}°` : '--'}</div>
            <div className={drone.battery < 20 ? 'text-red-400' : ''}>BAT: {Math.round(drone.battery)}%</div>
          </div>
        </div>

        {/* PIP Mini-map tracking the drone */}
        <button
          onClick={onBackToMap}
          className="absolute bottom-4 right-3 w-36 h-24 rounded-lg overflow-hidden border-2 border-white/30 active:border-white/60 transition-colors group"
        >
          <MapContainer
            ref={miniMapRef}
            center={[drone.position.lat || MAP_CENTER.lat, drone.position.lng || MAP_CENTER.lng]}
            zoom={16}
            className="w-full h-full"
            zoomControl={false}
            attributionControl={false}
            dragging={false}
            scrollWheelZoom={false}
            doubleClickZoom={false}
            touchZoom={false}
          >
            <TileLayer url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png" />
            <MiniMapTracker position={drone.position} />
            <Marker position={[drone.position.lat, drone.position.lng]} icon={droneIcon} />
          </MapContainer>
          <div className="absolute top-1 right-1 bg-black/60 px-1.5 py-0.5 text-[10px] text-white rounded opacity-0 group-hover:opacity-100 transition-opacity">
            MAP
          </div>
        </button>
      </div>

      {/* Bottom controls */}
      <div className="h-16 bg-surface-raised border-t border-surface-border flex items-center justify-center gap-3 px-4 safe-bottom">
        {/* Back to map button */}
        <button
          onClick={onBackToMap}
          className="flex items-center gap-2 bg-surface-overlay active:bg-white/10 text-white px-5 py-2.5 rounded-lg transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
          </svg>
          Map
        </button>

        {/* Flag to Commander button */}
        {!flagged ? (
          <button
            onClick={handleFlag}
            className="flex items-center gap-2 bg-amber-600 active:bg-amber-500 text-white px-5 py-2.5 rounded-lg transition-colors font-medium"
          >
            Flag to Commander
          </button>
        ) : (
          <div className="flex items-center gap-2 bg-amber-600/30 text-amber-400 px-5 py-2.5 rounded-lg">
            Flagged
          </div>
        )}
      </div>

      {/* Flag confirmation toast */}
      {showFlagConfirm && (
        <div className="absolute top-20 left-1/2 -translate-x-1/2 bg-green-500/90 text-white px-6 py-3 rounded-lg font-medium animate-slide-in z-50">
          Flagged to Marco
        </div>
      )}
    </div>
  )
}
