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

  const statusLabel = drone.status === 'flying' ? 'FLYING'
    : drone.status === 'returning' ? 'RTL'
    : drone.status === 'armed' ? 'ARMED'
    : drone.status === 'stopped' ? 'STOPPED'
    : 'READY'

  const batPercent = Math.round(drone.battery)
  const batColor = batPercent < 20 ? 'bg-status-critical' : batPercent < 40 ? 'bg-status-warning' : 'bg-status-good'

  return (
    <div className="absolute bottom-2 left-3 z-[1000] pointer-events-none">
      <div className="bg-black/75 backdrop-blur-sm border border-surface-border rounded-lg px-3 py-2.5 font-mono text-xs text-white min-w-[140px]">
        {/* Drone name + status */}
        <div className="flex items-center gap-2 mb-1.5 pb-1.5 border-b border-white/10">
          <div className={`w-2 h-2 rounded-full ${statusColor}`} />
          <span className="font-bold text-sm">{drone.name}</span>
          <span className="text-text-dim text-[10px] ml-auto">{statusLabel}</span>
        </div>

        {/* Telemetry */}
        <div className="space-y-0.5 text-text-secondary">
          <div>
            <span className="text-text-dim">ALT: </span>
            {isActive ? `${Math.round(drone.altitude)}m` : '--'}
          </div>
          <div>
            <span className="text-text-dim">SPD: </span>
            {isActive ? `${drone.speed.toFixed(1)} m/s` : '--'}
          </div>
          <div>
            <span className="text-text-dim">HDG: </span>
            {isActive ? `${String(drone.heading).padStart(3, '0')}°` : '--'}
          </div>
        </div>

        {/* Battery bar */}
        <div className="mt-2 pt-1.5 border-t border-white/10">
          <div className="flex items-center justify-between mb-1">
            <span className="text-text-dim">BAT</span>
            <span className={`${batPercent < 20 ? 'text-status-critical' : 'text-text-secondary'}`}>
              {batPercent}%
            </span>
          </div>
          <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${batColor}`}
              style={{ width: `${batPercent}%` }}
            />
          </div>
        </div>

        {/* Mesh status */}
        <div className="mt-1.5 flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full bg-status-good" />
          <span className="text-[10px] text-text-dim">ON MESH</span>
        </div>

        {/* Kill button */}
        {canKill && onKill && (
          <button
            onClick={onKill}
            className="mt-2 w-full py-1.5 bg-kill hover:bg-kill-hover text-text-primary font-bold text-xs tracking-wider rounded border border-kill-hover/30 transition-colors pointer-events-auto"
          >
            KILL
          </button>
        )}
      </div>
    </div>
  )
}
