// Atmosphere Mesh Types

export interface MeshNode {
  id: string
  name: string
  type: 'gateway' | 'agent' | 'skill' | 'service'
  status: 'online' | 'offline' | 'busy' | 'error'
  capabilities: string[]
  position?: { x: number; y: number }
  sessions?: number
  lastSeen?: string
  metadata?: Record<string, any>
}

export interface MeshConnection {
  source: string
  target: string
  type: 'direct' | 'routed' | 'fallback'
  latency?: number
  active: boolean
  strength?: number
}

export interface MeshIntent {
  id: string
  content: string
  source: string
  target?: string
  route: string[] // node ids in routing path
  timestamp: string
  status: 'pending' | 'routing' | 'completed' | 'failed'
  duration?: number
}

export interface MeshStatus {
  totalNodes: number
  activeNodes: number
  totalConnections: number
  activeIntents: number
  health: 'healthy' | 'degraded' | 'critical'
  uptime?: string
}

export interface TestNode {
  name: string
  type: MeshNode['type']
  capabilities: string[]
}

export interface WebSocketMessage {
  type: 'node_update' | 'connection_update' | 'intent_update' | 'status_update'
  data: any
  timestamp: string
}
