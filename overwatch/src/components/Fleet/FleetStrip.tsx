import type { Drone } from '../../types'
import { getDroneColor } from '../../types'
import { DroneCard } from './DroneCard'
import { KillButton } from '../Controls/KillButton'

interface FleetStripProps {
  drones: Drone[]
  selectedDroneId: string | null
  onSelectDrone: (droneId: string) => void
  onKill: () => void
  killDisabled: boolean
  selectedDroneName?: string
}

export function FleetStrip({ drones, selectedDroneId, onSelectDrone, onKill, killDisabled, selectedDroneName }: FleetStripProps) {
  return (
    <div className="h-12 bg-surface-raised border-t border-surface-border flex items-center px-3 shrink-0 fleet-strip">
      {/* Fleet label */}
      <span className="text-[10px] text-gray-500 uppercase tracking-widest mr-3 shrink-0">Fleet</span>

      {/* Drone cards - horizontal scroll on mobile */}
      <div className="flex items-center gap-1.5 overflow-x-auto flex-1">
        {drones.map((drone, index) => (
          <DroneCard
            key={drone.id}
            drone={drone}
            selected={selectedDroneId === drone.id}
            droneColor={getDroneColor(index)}
            compact
            onClick={() => onSelectDrone(drone.id)}
          />
        ))}
      </div>

      {/* Kill button - right side */}
      <div className="ml-2 shrink-0">
        <KillButton
          onClick={onKill}
          disabled={killDisabled}
          droneName={selectedDroneName}
        />
      </div>
    </div>
  )
}
