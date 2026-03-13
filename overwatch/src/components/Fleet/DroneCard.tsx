import type { Drone } from '../../types'
import { getBatteryLevel } from '../../types'

interface DroneCardProps {
  drone: Drone
  selected: boolean
  compact?: boolean
  droneColor?: string
  onClick: () => void
}

function BatteryRing({ battery, size = 27 }: { battery: number; size?: number }) {
  const level = getBatteryLevel(battery)
  const strokeColor = level === 'good' ? '#22c55e' : level === 'warning' ? '#f59e0b' : '#d55e4e'
  const radius = (size - 4) / 2
  const circumference = 2 * Math.PI * radius
  const filled = (battery / 100) * circumference
  const center = size / 2

  return (
    <div className="relative flex items-center gap-1.5">
      <span className="text-xs text-gray-400">{Math.round(battery)}%</span>
      <svg width={size} height={size} className="shrink-0">
        {/* Background ring */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="#2a2a2a"
          strokeWidth={2.5}
        />
        {/* Battery level ring */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={strokeColor}
          strokeWidth={2.5}
          strokeDasharray={`${filled} ${circumference - filled}`}
          strokeDashoffset={circumference / 4}
          strokeLinecap="round"
          transform={`rotate(-90 ${center} ${center})`}
        />
        {/* Lightning bolt */}
        <path
          d="M-2.5,-4 L0.5,-0.5 L-0.5,-0.5 L1.5,4 L-0.5,0.5 L0.5,0.5 Z"
          fill={strokeColor}
          transform={`translate(${center}, ${center}) scale(0.85)`}
        />
      </svg>
    </div>
  )
}

export function DroneCard({ drone, selected, compact = false, droneColor, onClick }: DroneCardProps) {
  const batteryLevel = getBatteryLevel(drone.battery)
  const isFlying = drone.status === 'flying'
  const isNotReady = batteryLevel === 'critical' || drone.status === 'offline' || drone.status === 'stopped'

  const getStatusText = () => {
    if (batteryLevel === 'critical' && drone.status !== 'flying') return 'Not ready'
    switch (drone.status) {
      case 'ready': return 'Ready'
      case 'flying': return 'In flight'
      case 'returning': return 'Returning'
      case 'stopped': return 'Stopped'
      case 'offline': return 'Offline'
    }
  }

  const getStatusColor = () => {
    if (isFlying) return '#66abff'
    if (isNotReady) return '#d55e4e'
    return '#22c55e'
  }

  const getIndicatorColor = () => {
    if (isFlying) return 'bg-[#66abff]'
    if (isNotReady) return 'bg-[#d55e4e]'
    return 'bg-green-500'
  }

  const getConnectionText = () => {
    if (batteryLevel === 'critical' && drone.status !== 'flying') {
      return { text: 'Low battery', color: 'text-[#d55e4e]', icon: 'battery' }
    }
    if (drone.connected) {
      return { text: 'Connected', color: 'text-green-500', icon: 'check' }
    }
    return { text: 'Disconnected', color: 'text-red-500', icon: 'x' }
  }

  const getBatteryBarColor = () => {
    switch (batteryLevel) {
      case 'good': return 'bg-green-500'
      case 'warning': return 'bg-amber-500'
      case 'critical': return 'bg-red-500'
    }
  }

  // Compact mode for fleet strip
  if (compact) {
    const dotColor = getStatusColor()
    return (
      <button
        onClick={onClick}
        className={`
          flex items-center gap-2 px-2.5 py-1.5 rounded-md transition-all
          ${selected
            ? 'bg-white/10'
            : 'bg-surface-overlay hover:bg-white/5'
          }
        `}
        style={selected && droneColor
          ? { boxShadow: `inset 0 0 0 1.5px ${droneColor}` }
          : { boxShadow: 'inset 0 0 0 1px rgba(255,255,255,0.08)' }
        }
      >
        <div
          className="w-2 h-2 rounded-full shrink-0"
          style={{ backgroundColor: dotColor }}
        />
        <span className="text-xs font-medium text-white whitespace-nowrap">
          {drone.name}
        </span>
        <div className="w-5 h-2 rounded-sm overflow-hidden border border-white/20">
          <div
            className={`h-full ${getBatteryBarColor()} transition-all`}
            style={{ width: `${drone.battery}%` }}
          />
        </div>
      </button>
    )
  }

  const conn = getConnectionText()
  const borderClass = isFlying ? 'border-[#66abff]' : 'border-surface-border'

  // Full card mode for fleet status screen
  return (
    <button
      onClick={onClick}
      className={`
        w-full p-4 rounded-xl transition-all text-left
        ${selected
          ? 'bg-white/10 ring-2 ring-white/30'
          : `bg-surface-raised hover:bg-surface-overlay border ${borderClass}`
        }
      `}
    >
      {/* Top row: name + battery ring */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <div className={`w-3 h-3 rounded-full shrink-0 ${getIndicatorColor()}`} />
          <span className="text-lg font-semibold text-white">{drone.name}</span>
        </div>
        <BatteryRing battery={drone.battery} />
      </div>

      {/* Bottom row: connection + status */}
      <div className="flex items-center justify-between">
        <div className={`flex items-center gap-1.5 text-xs ${conn.color}`}>
          {conn.icon === 'check' && (
            <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          )}
          {conn.icon === 'battery' && (
            <svg className="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 16 16">
              <path d="M12 4H2a1 1 0 00-1 1v6a1 1 0 001 1h10a1 1 0 001-1V5a1 1 0 00-1-1zm3 3v2h-1V7h1zM3 6h2v4H3V6z"/>
            </svg>
          )}
          <span>{conn.text}</span>
        </div>
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-400">Status:</span>
          <span className="font-medium" style={{ color: getStatusColor() }}>
            {getStatusText()}
          </span>
        </div>
      </div>
    </button>
  )
}
