import type { Drone } from '../../types'

interface TelemetryHUDProps {
  drone: Drone
  onKill?: () => void
  canKill?: boolean
}

export function TelemetryHUD({ drone, onKill, canKill }: TelemetryHUDProps) {
  const isActive = drone.status === 'flying' || drone.status === 'returning'

  const statusColor = drone.status === 'flying' ? 'bg-status-good'
    : drone.status === 'returning' ? 'bg-status-warning'
    : drone.status === 'armed' ? 'bg-arm-armed'
    : 'bg-arm-disarmed'

  const statusLabel = drone.status === 'flying' ? 'FLY'
    : drone.status === 'returning' ? 'RTL'
    : drone.status === 'armed' ? 'ARM'
    : drone.status === 'stopped' ? 'STP'
    : 'RDY'

  const batPercent = Math.round(drone.battery)
  const batColor = batPercent < 20 ? 'bg-status-critical' : batPercent < 40 ? 'bg-status-warning' : 'bg-status-good'

  return (
    <div className="absolute z-[1000] pointer-events-none" style={{ bottom: '8px', left: '12px' }}>
      <div className="bg-black/75 backdrop-blur-sm border border-surface-border rounded-lg px-2.5 py-2 font-mono text-[11px] text-white min-w-[120px]">
        {/* Drone name + status — single line */}
        <div className="flex items-center gap-1.5 mb-1">
          <div className={`w-1.5 h-1.5 rounded-full ${statusColor}`} />
          <span className="font-bold">{drone.name}</span>
          <span className="text-text-dim text-[9px] ml-auto">{statusLabel}</span>
        </div>

        {/* Telemetry — compact */}
        <div className="text-text-secondary text-[10px] leading-snug">
          <span className="text-text-dim">ALT:</span>{isActive ? `${Math.round(drone.altitude)}m` : '--'}
          {' '}<span className="text-text-dim">SPD:</span>{isActive ? `${drone.speed.toFixed(1)}` : '--'}
          {' '}<span className="text-text-dim">HDG:</span>{isActive ? `${String(drone.heading).padStart(3, '0')}°` : '--'}
        </div>

        {/* Battery — inline */}
        <div className="flex items-center gap-1.5 mt-1">
          <div className="flex-1 h-1 bg-white/10 rounded-full overflow-hidden">
            <div className={`h-full rounded-full ${batColor}`} style={{ width: `${batPercent}%` }} />
          </div>
          <span className={`text-[9px] ${batPercent < 20 ? 'text-status-critical' : 'text-text-dim'}`}>{batPercent}%</span>
          <div className="w-1 h-1 rounded-full bg-status-good" title="ON MESH" />
        </div>

        {/* Kill button — compact */}
        {canKill && onKill && (
          <button
            onClick={onKill}
            className="mt-1.5 w-full py-1 bg-kill hover:bg-kill-hover text-text-primary font-bold text-[10px] tracking-wider rounded border border-kill-hover/30 transition-colors pointer-events-auto"
          >
            KILL
          </button>
        )}
      </div>
    </div>
  )
}
