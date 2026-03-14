import { useEffect, useRef, useState } from 'react'
import { MapContainer, TileLayer, Marker, Rectangle, useMapEvents, useMap } from 'react-leaflet'
import L from 'leaflet'
import type { Drone, Detection, SearchArea, DronePosition } from '../../types'
import { getDetectionColor, getDroneColor } from '../../types'
import { MAP_CENTER, MAP_ZOOM } from '../../data/mockDrones'

// Custom drone icon
const createDroneIcon = (status: string, selected: boolean, droneColor: string) => {
  const color = status === 'offline' ? '#666' : droneColor
  const size = selected ? 32 : 24

  return L.divIcon({
    className: 'drone-marker',
    html: `
      <div style="
        width: ${size}px;
        height: ${size}px;
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <svg viewBox="0 0 24 24" fill="${color}" width="${size}" height="${size}">
          <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
        </svg>
      </div>
    `,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2]
  })
}

// Draw tool component
interface DrawToolProps {
  isDrawing: boolean
  onDrawComplete: (bounds: { northEast: DronePosition; southWest: DronePosition }) => void
}

function DrawTool({ isDrawing, onDrawComplete }: DrawToolProps) {
  const [startPoint, setStartPoint] = useState<L.LatLng | null>(null)
  const [currentRect, setCurrentRect] = useState<L.LatLngBounds | null>(null)
  const map = useMap()

  useMapEvents({
    mousedown(e) {
      if (!isDrawing) return
      setStartPoint(e.latlng)
      map.dragging.disable()
    },
    mousemove(e) {
      if (!isDrawing || !startPoint) return
      const bounds = L.latLngBounds(startPoint, e.latlng)
      setCurrentRect(bounds)
    },
    mouseup(e) {
      if (!isDrawing || !startPoint) return
      map.dragging.enable()

      const bounds = L.latLngBounds(startPoint, e.latlng)
      onDrawComplete({
        northEast: { lat: bounds.getNorth(), lng: bounds.getEast() },
        southWest: { lat: bounds.getSouth(), lng: bounds.getWest() }
      })

      setStartPoint(null)
      setCurrentRect(null)
    }
  })

  if (!currentRect) return null

  return (
    <Rectangle
      bounds={currentRect}
      pathOptions={{
        color: '#16703a',
        weight: 2,
        fillColor: '#16703a',
        fillOpacity: 0.2,
        dashArray: '5, 5'
      }}
    />
  )
}

// Map center/zoom controller
interface MapControllerProps {
  center?: DronePosition
}

function MapController({ center }: MapControllerProps) {
  const map = useMap()

  useEffect(() => {
    if (center) {
      map.setView([center.lat, center.lng], map.getZoom(), { animate: true })
    }
  }, [center, map])

  return null
}

interface MapViewProps {
  drones: Drone[]
  detections: Detection[]
  searchAreas: SearchArea[]
  selectedDroneId: string | null
  isDrawingArea: boolean
  onDrawComplete: (bounds: { northEast: DronePosition; southWest: DronePosition }) => void
  onDetectionClick: (detectionId: string) => void
}

export function MapView({
  drones,
  detections,
  searchAreas,
  selectedDroneId,
  isDrawingArea,
  onDrawComplete,
  onDetectionClick
}: MapViewProps) {
  const mapRef = useRef<L.Map>(null)
  const selectedDrone = drones.find(d => d.id === selectedDroneId)

  // Create detection markers
  const detectionMarkers = detections.map(detection => {
    const icon = L.divIcon({
      className: 'detection-marker',
      html: `
        <div style="
          width: 18px;
          height: 18px;
          border-radius: 50%;
          background-color: ${getDetectionColor(detection.type)};
          border: 1.5px solid ${getDetectionColor(detection.type)}88;
          box-shadow: 0 0 4px ${getDetectionColor(detection.type)}40;
          position: relative;
        ">
        </div>
      `,
      iconSize: [24, 24],
      iconAnchor: [12, 12]
    })

    return (
      <Marker
        key={detection.id}
        position={[detection.position.lat, detection.position.lng]}
        icon={icon}
        eventHandlers={{
          click: () => onDetectionClick(detection.id)
        }}
      />
    )
  })

  return (
    <div className="relative w-full h-full">
      <MapContainer
        ref={mapRef}
        center={[MAP_CENTER.lat, MAP_CENTER.lng]}
        zoom={MAP_ZOOM}
        className="w-full h-full"
        zoomControl={true}
        attributionControl={true}
      >
        {/* Dark tile layer - CartoDB Dark */}
        <TileLayer
          attribution='&copy; <a href="https://carto.com/">CARTO</a>'
          url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        />

        {/* Map controller for centering */}
        <MapController center={selectedDrone?.position} />

        {/* Draw tool */}
        <DrawTool isDrawing={isDrawingArea} onDrawComplete={onDrawComplete} />

        {/* Search areas */}
        {searchAreas.map(area => (
          <Rectangle
            key={area.id}
            bounds={[
              [area.bounds.southWest.lat, area.bounds.southWest.lng],
              [area.bounds.northEast.lat, area.bounds.northEast.lng]
            ]}
            pathOptions={{
              color: '#16703a',
              weight: 2,
              fillColor: '#16703a',
              fillOpacity: 0.1
            }}
          />
        ))}

        {/* Drone markers */}
        {drones.map((drone, index) => (
          <Marker
            key={drone.id}
            position={[drone.position.lat, drone.position.lng]}
            icon={createDroneIcon(drone.status, drone.id === selectedDroneId, getDroneColor(index))}
          />
        ))}

        {/* Detection markers */}
        {detectionMarkers}
      </MapContainer>

      {/* Drawing mode indicator */}
      {isDrawingArea && (
        <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-surface-overlay/90 border border-surface-border text-text-secondary px-4 py-2 rounded-lg text-sm font-medium z-[1000]">
          Click and drag to draw search area
        </div>
      )}
    </div>
  )
}
