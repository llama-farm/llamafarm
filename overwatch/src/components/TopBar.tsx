import { useEffect, useState } from 'react'

interface TopBarProps {
  connectionStatus: 'connected' | 'degraded' | 'disconnected'
  gpsStatus: 'locked' | 'acquiring' | 'lost'
}

export function TopBar({ connectionStatus }: TopBarProps) {
  const [currentTime, setCurrentTime] = useState(new Date())

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour12: false,
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit'
    })
  }

  const getConnectionColor = () => {
    switch (connectionStatus) {
      case 'connected': return 'text-green-500'
      case 'degraded': return 'text-amber-500'
      case 'disconnected': return 'text-red-500'
    }
  }


  return (
    <header className="h-11 bg-surface-raised border-b border-surface-border flex items-center justify-between px-3 shrink-0 safe-top">
      {/* Left: Logo/Title */}
      <div className="flex items-center gap-2">
        <span className="font-bold text-white tracking-widest text-sm">OVERWATCH</span>
      </div>

      {/* Center: Mesh status */}
      <div className="flex items-center gap-1.5 text-sm">
        <div className={`w-2 h-2 rounded-full ${getConnectionColor().replace('text-', 'bg-')}`} />
        <span className={`${getConnectionColor()} text-xs uppercase`}>
          MESH {connectionStatus}
        </span>
      </div>

      {/* Right: Time */}
      <div className="flex items-center text-sm">
        <span className="text-white font-mono font-medium">{formatTime(currentTime)}</span>
      </div>
    </header>
  )
}
