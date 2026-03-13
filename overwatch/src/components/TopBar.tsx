import { useEffect, useState } from 'react'
import type { ArmState } from '../types'

interface TopBarProps {
  connectionStatus: 'connected' | 'degraded' | 'disconnected'
  gpsStatus: 'locked' | 'acquiring' | 'lost'
  armState?: ArmState
}

export function TopBar({ connectionStatus, armState = 'disarmed' }: TopBarProps) {
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

  return (
    <header className="h-11 bg-surface-bar border-b border-surface-border flex items-center justify-between px-3 shrink-0 safe-top">
      {/* Left: Title */}
      <div className="flex items-center gap-2">
        <span className="font-bold text-text-primary tracking-widest text-sm">ARC</span>
      </div>

      {/* Center: Status cluster */}
      <div className="flex items-center gap-4 text-xs">
        {/* Mesh */}
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${connDot}`} />
          <span className={`${connText} uppercase`}>
            {connectionStatus === 'connected' ? 'MESH' : connectionStatus === 'disconnected' ? 'NO MESH' : 'DEGRADED'}
          </span>
        </div>

        {/* Arm state */}
        <div className="flex items-center gap-1.5">
          <div className={`w-2 h-2 rounded-full ${armState === 'armed' ? 'bg-arm-armed' : 'bg-arm-disarmed'}`} />
          <span className={`uppercase ${armState === 'armed' ? 'text-arm-armed' : 'text-text-dim'}`}>
            {armState}
          </span>
        </div>
      </div>

      {/* Right: Time */}
      <div className="flex items-center">
        <span className="text-text-secondary font-mono text-xs">{formatTime(currentTime)}</span>
      </div>
    </header>
  )
}
