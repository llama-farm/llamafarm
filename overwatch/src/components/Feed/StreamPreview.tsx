import { useState } from 'react'
import type { Drone } from '../../types'

interface StreamPreviewProps {
  drone: Drone
  onOpenStream: () => void
}

export function StreamPreview({ drone, onOpenStream }: StreamPreviewProps) {
  const [collapsed, setCollapsed] = useState(false)

  const isActive = drone.status === 'flying' || drone.status === 'returning'

  if (!isActive) return null

  // Collapsed: just a small pinned tab
  if (collapsed) {
    return (
      <button
        onClick={() => setCollapsed(false)}
        className="absolute bottom-20 right-3 z-[900] bg-surface-raised/90 backdrop-blur-sm border border-surface-border rounded-lg px-2.5 py-2 flex items-center gap-2 hover:bg-white/10 transition-colors"
      >
        <svg className="w-4 h-4 text-red-400" fill="currentColor" viewBox="0 0 24 24">
          <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>
        </svg>
        <span className="text-xs text-gray-400 font-medium">{drone.name}</span>
      </button>
    )
  }

  // Expanded: stream preview thumbnail
  return (
    <div className="absolute bottom-20 right-3 z-[900] w-48 rounded-lg overflow-hidden border border-surface-border shadow-2xl">
      {/* Stream thumbnail - click to go full screen */}
      <button
        onClick={onOpenStream}
        className="relative w-full aspect-video bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 group"
      >
        {/* Scan lines effect */}
        <div
          className="absolute inset-0 opacity-10"
          style={{
            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)'
          }}
        />

        {/* Simulated aerial view */}
        <svg className="absolute inset-0 w-full h-full opacity-30" viewBox="0 0 100 60" preserveAspectRatio="none">
          {/* Roads */}
          <line x1="20" y1="0" x2="25" y2="60" stroke="#3a3a3a" strokeWidth="1.5" />
          <line x1="0" y1="30" x2="100" y2="35" stroke="#3a3a3a" strokeWidth="1.5" />
          {/* Buildings */}
          <rect x="30" y="10" width="12" height="15" fill="#2a2a2a" />
          <rect x="50" y="20" width="8" height="10" fill="#2a2a2a" />
          <rect x="60" y="40" width="15" height="12" fill="#2a2a2a" />
          <rect x="10" y="40" width="10" height="8" fill="#2a2a2a" />
          {/* Trees/vegetation */}
          <circle cx="45" cy="45" r="3" fill="#1a3a1a" />
          <circle cx="75" cy="15" r="4" fill="#1a3a1a" />
          <circle cx="80" cy="50" r="2.5" fill="#1a3a1a" />
        </svg>

        {/* Crosshair */}
        <div className="absolute inset-0 flex items-center justify-center opacity-40">
          <div className="w-8 h-8 relative">
            <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-2 bg-white" />
            <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-2 bg-white" />
            <div className="absolute left-0 top-1/2 -translate-y-1/2 w-2 h-px bg-white" />
            <div className="absolute right-0 top-1/2 -translate-y-1/2 w-2 h-px bg-white" />
          </div>
        </div>

        {/* LIVE badge */}
        <div className="absolute top-1.5 left-1.5 flex items-center gap-1 bg-red-600/90 px-1.5 py-0.5 rounded text-[10px] font-bold text-white">
          <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
          LIVE
        </div>

        {/* Drone name */}
        <div className="absolute top-1.5 right-1.5 bg-black/60 px-1.5 py-0.5 rounded text-[10px] text-white font-mono">
          {drone.name}
        </div>

        {/* Expand icon on hover */}
        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
          <svg className="w-8 h-8 text-white opacity-0 group-hover:opacity-80 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9m11.25-5.25v4.5m0-4.5h-4.5m4.5 0L15 9m-11.25 11.25v-4.5m0 4.5h4.5m-4.5 0L9 15m11.25 5.25v-4.5m0 4.5h-4.5m4.5 0L15 15" />
          </svg>
        </div>
      </button>

      {/* Bottom bar with collapse */}
      <div className="bg-surface-raised/95 backdrop-blur-sm px-2 py-1.5 flex items-center justify-between">
        <span className="text-[10px] text-gray-400 font-mono">
          {drone.altitude > 0 ? `${Math.round(drone.altitude)}m` : '--'} | {drone.speed > 0 ? `${drone.speed.toFixed(1)} m/s` : '--'}
        </span>
        <button
          onClick={() => setCollapsed(true)}
          className="text-gray-500 hover:text-white transition-colors"
          title="Minimize"
        >
          <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 12h-15" />
          </svg>
        </button>
      </div>
    </div>
  )
}
