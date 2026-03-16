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
  const [confirmingArm, setConfirmingArm] = useState(false)

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  // Auto-dismiss confirm after 3 seconds
  useEffect(() => {
    if (confirmingArm) {
      const t = setTimeout(() => setConfirmingArm(false), 3000)
      return () => clearTimeout(t)
    }
  }, [confirmingArm])

  // Reset confirm state when arm state changes
  useEffect(() => {
    setConfirmingArm(false)
  }, [armState])

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

  const handleArmClick = () => {
    if (armState === 'disarmed') {
      // Arm immediately — no confirm needed
      onToggleArm?.()
      return
    }
    if (!canDisarm) return
    if (confirmingArm) {
      // Second tap = confirm disarm
      onToggleArm?.()
      setConfirmingArm(false)
    } else {
      // First tap = show disarm confirm
      setConfirmingArm(true)
    }
  }

  const pillLabel = confirmingArm ? 'DISARM?' : (armState === 'armed' ? 'ARMED' : 'DISARMED')

  return (
    <header className="h-11 bg-surface-bar border-b border-surface-border flex items-center justify-between pl-3 pr-0 shrink-0 safe-top">
      {/* Left: Title + Time */}
      <div className="flex items-center gap-2.5">
        <span className="font-bold text-text-primary tracking-widest text-sm">ARC</span>
        <span className="text-text-dim font-mono text-xs">{formatTime(currentTime)}</span>
      </div>

      {/* Center: Status cluster */}
      <div className="flex items-center gap-4 text-xs self-center">
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${connDot}`} />
          <span className={`${connText} uppercase`}>
            {connectionStatus === 'connected' ? 'MESH' : connectionStatus === 'disconnected' ? 'NO MESH' : 'DEGRADED'}
          </span>
        </div>
        <button
          onClick={handleArmClick}
          disabled={armState === 'armed' && !canDisarm}
          style={{ height: '26px', maxHeight: '26px', minHeight: '26px', boxSizing: 'border-box' }}
          className={`inline-flex items-center gap-1 px-1.5 rounded-full border text-[10px] leading-none transition-colors outline-none focus:outline-none focus:ring-0 shrink-0 ${
            confirmingArm
              ? (armState === 'armed'
                ? 'border-status-critical/50 bg-status-critical/15 hover:bg-status-critical/25'
                : 'border-arm-armed/50 bg-arm-armed/15 hover:bg-arm-armed/25')
              : armState === 'armed'
                ? 'border-arm-armed/30 bg-arm-armed/10 hover:bg-arm-armed/20'
                : 'border-surface-border bg-surface-raised hover:bg-surface-overlay'
          } ${armState === 'armed' && !canDisarm ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          title={confirmingArm ? 'Tap again to confirm' : (armState === 'armed' ? 'Tap to disarm' : 'Tap to arm')}
        >
          <div className={`w-1.5 h-1.5 rounded-full ${
            confirmingArm ? 'animate-pulse' : ''
          } ${armState === 'armed' ? 'bg-arm-armed' : 'bg-arm-disarmed'}`} />
          <span className={`uppercase font-medium ${
            confirmingArm
              ? (armState === 'armed' ? 'text-status-critical' : 'text-arm-armed')
              : armState === 'armed' ? 'text-arm-armed' : 'text-text-dim'
          }`}>
            {pillLabel}
          </span>
        </button>
      </div>

      {/* Right: View tabs — w-56 on mobile (matches feed), auto on desktop */}
      <div className="flex items-stretch w-56 lg:w-auto shrink-0 self-stretch border-l border-surface-border">
        <button onClick={() => onSetView?.('map')} className={`flex-1 lg:px-4 ${tabClass('map')}`}>
          Map
        </button>
        <button onClick={() => onSetView?.('voice')} className={`flex-1 lg:px-4 ${tabClass('voice')}`}>
          Chat
        </button>
      </div>
    </header>
  )
}
