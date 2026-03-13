import type { Drone } from '../../types'

interface TelemetryHUDProps {
  drone: Drone
}

export function TelemetryHUD({ drone }: TelemetryHUDProps) {
  const isActive = drone.status === 'flying' || drone.status === 'returning'

  return (
    <div className="absolute bottom-2 left-3 z-[1000] pointer-events-none">
      <div className="bg-black/70 backdrop-blur-sm border border-surface-border rounded-lg px-3 py-2 font-mono text-xs text-white">
        <div className="font-bold text-sm mb-1">{drone.name}</div>
        <div className="space-y-0.5 text-gray-300">
          <div>
            <span className="text-gray-500">ALT: </span>
            {isActive ? `${Math.round(drone.altitude)}m` : '--'}
          </div>
          <div>
            <span className="text-gray-500">SPD: </span>
            {isActive ? `${drone.speed.toFixed(1)} m/s` : '--'}
          </div>
          <div>
            <span className="text-gray-500">HDG: </span>
            {isActive ? `${String(drone.heading).padStart(3, '0')}°` : '--'}
          </div>
          <div>
            <span className="text-gray-500">BAT: </span>
            <span className={drone.battery < 20 ? 'text-red-400' : ''}>
              {Math.round(drone.battery)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
