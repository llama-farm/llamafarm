import type { Drone } from '../../types'
import { DroneCard } from './DroneCard'

interface FleetStatusProps {
  drones: Drone[]
  selectedDroneId: string | null
  onSelectDrone: (droneId: string) => void
}

export function FleetStatus({ drones, selectedDroneId, onSelectDrone }: FleetStatusProps) {
  return (
    <div className="h-full bg-surface-base flex flex-col px-4 pt-4 view-enter">
      {/* Header */}
      <div className="mb-6 text-center">
        <h1 className="text-xl font-bold text-white mb-1">Fleet Status</h1>
        <p className="text-gray-400 text-sm">Select a drone to deploy, track, or manage</p>
      </div>

      {/* Drone cards - 2 column grid */}
      <div className="flex-1 overflow-y-auto pb-4">
        <div className="grid grid-cols-2 gap-3">
          {drones.map(drone => (
            <DroneCard
              key={drone.id}
              drone={drone}
              selected={selectedDroneId === drone.id}
              onClick={() => onSelectDrone(drone.id)}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
