import { useEffect, useState } from 'react'
import type { ArmState, ViewMode } from '../types'

interface TopBarProps {
  connectionStatus: 'connected' | 'degraded' | 'disconnected'
  gpsStatus: 'locked' | 'acquiring' | 'lost'
  armState?: ArmState
  viewMode?: ViewMode
  onSetView?: (view: ViewMode) => void
  onToggleArm?: () => void
  canDisarm?: boolean
}

export function TopBar({ connectionStatus, armState = 'disarmed', viewMode = 'map', onSetView, onToggleArm, canDisarm = true }: TopBarProps) {
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const formatTime = (date: Date) =>
    date.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' })

  const connDot = connectionStatus === 'connected' ? 'bg-status-good' :
    connectionStatus === 'degraded' ? 'bg-status-warning' : 'bg-status-critical'

  const connText = connectionStatus === 'connected' ? 'text-status-good' :
    connectionStatus === 'degraded' ? 'text-status-warning' : 'text-status-critical'

  const tabClass = (tab: ViewMode) =>
    `px-3 text-xs rounded transition-colors flex items-center justify-center ${
      viewMode === tab
        ? 'bg-surface-overlay text-text-primary'
        : 'text-text-dim hover:text-text-secondary'
    }`

  return (
    <header className="h-11 bg-surface-bar border-b border-surface-border flex items-center justify-between pl-3 pr-0 shrink-0 safe-top">
      {/* Left: Title + Time */}
      <div className="flex items-center gap-2.5">
        <span className="font-bold text-text-primary tracking-widest text-sm">ARC</span>
        <span className="text-text-dim font-mono text-xs">{formatTime(currentTime)}</span>
      </div>

      {/* Center: Status cluster */}
      <div className="flex items-center gap-4 text-xs">
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${connDot}`} />
          <span className={`${connText} uppercase`}>
            {connectionStatus === 'connected' ? 'MESH' : connectionStatus === 'disconnected' ? 'NO MESH' : 'DEGRADED'}
          </span>
        </div>
        <button
          onClick={onToggleArm}
          disabled={armState === 'armed' && !canDisarm}
          className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full border transition-colors ${
            armState === 'armed'
              ? 'border-arm-armed/40 bg-arm-armed/10 hover:bg-arm-armed/20'
              : 'border-surface-border bg-surface-raised hover:bg-surface-overlay'
          } ${armState === 'armed' && !canDisarm ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          title={armState === 'armed' ? (canDisarm ? 'Tap to disarm' : 'Cannot disarm while flying') : 'Tap to arm'}
        >
          <div className={`w-2 h-2 rounded-full ${armState === 'armed' ? 'bg-arm-armed' : 'bg-arm-disarmed'}`} />
          <span className={`uppercase text-xs ${armState === 'armed' ? 'text-arm-armed' : 'text-text-dim'}`}>
            {armState}
          </span>
        </button>
      </div>

      {/* Right: View tabs — matches feed panel width (w-56), flush right */}
      <div className="flex items-stretch w-56 shrink-0 self-stretch border-l border-surface-border">
        <button onClick={() => onSetView?.('map')} className={`flex-1 ${tabClass('map')}`}>
          Map
        </button>
        <button onClick={() => onSetView?.('voice')} className={`flex-1 ${tabClass('voice')}`}>
          Chat
        </button>
      </div>
    </header>
  )
}
