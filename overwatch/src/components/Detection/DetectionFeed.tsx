import { useState } from 'react'
import type { Detection, Alert, Drone } from '../../types'
import { getDetectionColor } from '../../types'

interface FeedItem {
  id: string
  type: 'detection' | 'alert'
  timestamp: Date
  detection?: Detection
  alert?: Alert
}

interface DetectionFeedProps {
  detections: Detection[]
  alerts: Alert[]
  selectedDrone?: Drone
  onDetectionClick: (detectionId: string) => void
  onAlertAction: (alert: Alert) => void
  onAlertDismiss: (alertId: string) => void
  onOpenStream?: () => void
}

export function DetectionFeed({ detections, alerts, selectedDrone, onDetectionClick, onAlertAction, onAlertDismiss, onOpenStream }: DetectionFeedProps) {
  const [collapsed, setCollapsed] = useState(false)

  // Merge detections and alerts into a single feed sorted by time
  const feedItems: FeedItem[] = [
    ...alerts.map(a => ({ id: `alert-${a.id}`, type: 'alert' as const, timestamp: a.timestamp, alert: a })),
    ...detections.map(d => ({ id: `det-${d.id}`, type: 'detection' as const, timestamp: d.timestamp, detection: d })),
  ].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())

  const activeAlerts = alerts.length
  const isStreaming = selectedDrone && selectedDrone.status !== 'offline'

  const getAlertPriority = (alert: Alert) => {
    switch (alert.type) {
      case 'detection': return { color: '#ef4444', bg: 'bg-red-500/15', border: 'border-l-red-500', label: 'PERSON DETECTED' }
      case 'battery': return { color: '#f59e0b', bg: 'bg-amber-500/15', border: 'border-l-amber-500', label: 'LOW BATTERY' }
      case 'connection': return { color: '#6b7280', bg: 'bg-gray-500/15', border: 'border-l-gray-500', label: 'CONNECTION' }
    }
  }

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'person': return '👤'
      case 'vehicle': return '🚗'
      case 'animal': return '🐾'
      default: return '●'
    }
  }

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })
  }

  return (
    <div className={`absolute top-0 right-0 h-full z-[1000] transition-all duration-300 ${collapsed ? 'w-10' : 'w-64'}`}>
      {/* Toggle button */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="absolute top-3 left-0 -translate-x-full bg-surface-raised/90 backdrop-blur-sm border border-surface-border border-r-0 rounded-l-lg px-1.5 py-2 text-gray-400 hover:text-white z-10"
      >
        <svg className={`w-4 h-4 transition-transform ${collapsed ? 'rotate-180' : ''}`} fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
          <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
        </svg>
      </button>

      {/* Panel */}
      {!collapsed && (
        <div className="h-full bg-surface-raised/90 backdrop-blur-sm border-l border-surface-border flex flex-col">
          {/* Header */}
          <div className="px-3 py-2.5 border-b border-surface-border flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm font-semibold text-white tracking-wide">Live Feed</span>
              {activeAlerts > 0 && (
                <span className="w-4 h-4 rounded-full bg-red-500 flex items-center justify-center text-[10px] text-white font-bold">
                  {activeAlerts}
                </span>
              )}
            </div>
            <span className="text-xs text-gray-500">{feedItems.length}</span>
          </div>

          {/* Feed list */}
          <div className="flex-1 overflow-y-auto">
            {feedItems.length === 0 ? (
              <div className="px-3 py-8 text-center text-gray-500 text-xs">
                No activity yet.
                <br />
                Launch a drone to begin scanning.
              </div>
            ) : (
              feedItems.map(item => {
                if (item.type === 'alert' && item.alert) {
                  const priority = getAlertPriority(item.alert)
                  return (
                    <div
                      key={item.id}
                      className={`w-full px-3 py-2 text-left border-b border-surface-border/50 border-l-2 ${priority.border} ${priority.bg}`}
                    >
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-[10px] font-bold tracking-wider" style={{ color: priority.color }}>
                          {priority.label}
                        </span>
                        <button
                          onClick={() => onAlertDismiss(item.alert!.id)}
                          className="text-gray-500 hover:text-white text-xs"
                        >
                          ✕
                        </button>
                      </div>
                      <p className="text-xs text-gray-300 mb-1.5 leading-tight">{item.alert.message}</p>
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-gray-500 font-mono">{formatTime(item.timestamp)}</span>
                        {item.alert.type === 'detection' && (
                          <button
                            onClick={() => onAlertAction(item.alert!)}
                            className="text-[10px] font-medium px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 transition-colors"
                            style={{ color: priority.color }}
                          >
                            View
                          </button>
                        )}
                        {item.alert.type === 'battery' && (
                          <button
                            onClick={() => onAlertAction(item.alert!)}
                            className="text-[10px] font-medium px-2 py-0.5 rounded bg-white/10 hover:bg-white/20 transition-colors"
                            style={{ color: priority.color }}
                          >
                            Send Backup
                          </button>
                        )}
                      </div>
                    </div>
                  )
                }

                if (item.type === 'detection' && item.detection) {
                  return (
                    <button
                      key={item.id}
                      onClick={() => onDetectionClick(item.detection!.id)}
                      className="w-full px-3 py-2 text-left hover:bg-white/5 border-b border-surface-border/50 transition-colors"
                    >
                      <div className="flex items-center gap-2 mb-0.5">
                        <div
                          className="w-2 h-2 rounded-full shrink-0"
                          style={{ backgroundColor: getDetectionColor(item.detection.type) }}
                        />
                        <span className="text-xs font-medium text-white capitalize">
                          {getTypeIcon(item.detection.type)} {item.detection.type}
                        </span>
                        <span className="text-xs text-gray-500 ml-auto">
                          {Math.round(item.detection.confidence * 100)}%
                        </span>
                      </div>
                      <div className="flex items-center justify-between pl-4">
                        <span className="text-[10px] text-gray-500 font-mono">
                          {formatTime(item.detection.timestamp)}
                        </span>
                        {item.detection.flagged && (
                          <span className="text-[10px] text-red-400">flagged</span>
                        )}
                      </div>
                    </button>
                  )
                }

                return null
              })
            )}
          </div>

          {/* Stream preview - pinned to bottom of panel */}
          {isStreaming && selectedDrone && (
            <div className="border-t border-surface-border shrink-0">
              <button
                onClick={onOpenStream}
                className="relative w-full aspect-[16/9] bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 group"
              >
                {/* Scan lines */}
                <div
                  className="absolute inset-0 opacity-10"
                  style={{
                    backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)'
                  }}
                />

                {/* Simulated aerial view */}
                <svg className="absolute inset-0 w-full h-full opacity-30" viewBox="0 0 100 60" preserveAspectRatio="none">
                  <line x1="20" y1="0" x2="25" y2="60" stroke="#3a3a3a" strokeWidth="1.5" />
                  <line x1="0" y1="30" x2="100" y2="35" stroke="#3a3a3a" strokeWidth="1.5" />
                  <rect x="30" y="10" width="12" height="15" fill="#2a2a2a" />
                  <rect x="50" y="20" width="8" height="10" fill="#2a2a2a" />
                  <rect x="60" y="40" width="15" height="12" fill="#2a2a2a" />
                  <circle cx="45" cy="45" r="3" fill="#1a3a1a" />
                  <circle cx="75" cy="15" r="4" fill="#1a3a1a" />
                </svg>

                {/* Crosshair */}
                <div className="absolute inset-0 flex items-center justify-center opacity-40">
                  <div className="w-6 h-6 relative">
                    <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-1.5 bg-white" />
                    <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-px h-1.5 bg-white" />
                    <div className="absolute left-0 top-1/2 -translate-y-1/2 w-1.5 h-px bg-white" />
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1.5 h-px bg-white" />
                  </div>
                </div>

                {/* Status badge */}
                {(selectedDrone.status === 'flying' || selectedDrone.status === 'returning') ? (
                  <div className="absolute top-1.5 left-1.5 flex items-center gap-1 bg-red-600/90 px-1.5 py-0.5 rounded text-[10px] font-bold text-white">
                    <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                    LIVE
                  </div>
                ) : (
                  <div className="absolute top-1.5 left-1.5 flex items-center gap-1 bg-black/60 px-1.5 py-0.5 rounded text-[10px] font-bold text-gray-400">
                    STANDBY
                  </div>
                )}

                {/* Drone name */}
                <div className="absolute top-1.5 right-1.5 bg-black/60 px-1.5 py-0.5 rounded text-[10px] text-white">
                  {selectedDrone.name} stream
                </div>

                {/* Expand icon on hover */}
                <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                  <svg className="w-6 h-6 text-white opacity-0 group-hover:opacity-80 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v4.5m0-4.5h4.5m-4.5 0L9 9m11.25-5.25v4.5m0-4.5h-4.5m4.5 0L15 9m-11.25 11.25v-4.5m0 4.5h4.5m-4.5 0L9 15m11.25 5.25v-4.5m0 4.5h-4.5m4.5 0L15 15" />
                  </svg>
                </div>
              </button>
            </div>
          )}
        </div>
      )}

      {/* Collapsed state */}
      {collapsed && (
        <div className="h-full bg-surface-raised/90 backdrop-blur-sm border-l border-surface-border flex flex-col items-center pt-12 gap-2">
          {activeAlerts > 0 && (
            <div className="w-5 h-5 rounded-full bg-red-500 flex items-center justify-center animate-pulse">
              <span className="text-[10px] text-white font-bold">{activeAlerts}</span>
            </div>
          )}
          {feedItems.length > 0 && activeAlerts === 0 && (
            <div className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center">
              <span className="text-[10px] text-gray-400 font-bold">{feedItems.length}</span>
            </div>
          )}
          {isStreaming && (
            <div className="w-5 h-5 rounded-full bg-red-600/80 flex items-center justify-center">
              <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 24 24">
                <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>
              </svg>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
