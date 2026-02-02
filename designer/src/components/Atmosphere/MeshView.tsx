import { useState, useEffect, useCallback, useRef } from 'react'
import MeshCanvas from './MeshCanvas'
import TestPanel from './TestPanel'
import axios from 'axios'
import type {
  MeshNode,
  MeshConnection,
  MeshIntent,
  MeshStatus,
  TestNode,
  WebSocketMessage,
} from '../../types/atmosphere'
import { Activity, AlertCircle, Radio } from 'lucide-react'

const API_BASE = '/api/atmosphere' // Adjust to your actual API base

export default function MeshView() {
  const [nodes, setNodes] = useState<MeshNode[]>([])
  const [connections, setConnections] = useState<MeshConnection[]>([])
  const [activeIntents, setActiveIntents] = useState<MeshIntent[]>([])
  const [meshStatus, setMeshStatus] = useState<MeshStatus | null>(null)
  const [selectedNode, setSelectedNode] = useState<MeshNode | null>(null)
  const [isTestPanelOpen, setIsTestPanelOpen] = useState(true)
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const wsRef = useRef<WebSocket | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [dimensions, setDimensions] = useState({ width: 1200, height: 800 })

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    }

    updateDimensions()
    window.addEventListener('resize', updateDimensions)
    return () => window.removeEventListener('resize', updateDimensions)
  }, [])

  // Fetch initial data
  const fetchMeshData = useCallback(async () => {
    try {
      const [nodesRes, statusRes] = await Promise.all([
        axios.get<MeshNode[]>(`${API_BASE}/mesh/nodes`),
        axios.get<MeshStatus>(`${API_BASE}/mesh/status`),
      ])

      setNodes(nodesRes.data)
      setMeshStatus(statusRes.data)
      setError(null)

      // Build connections from nodes (assuming nodes have connection info)
      // Adjust based on your actual API response
      const conns: MeshConnection[] = []
      nodesRes.data.forEach(node => {
        // This is a placeholder - adjust to your actual data structure
        if (node.metadata?.connections) {
          node.metadata.connections.forEach((targetId: string) => {
            conns.push({
              source: node.id,
              target: targetId,
              type: 'direct',
              active: true,
            })
          })
        }
      })
      setConnections(conns)
    } catch (err) {
      console.error('Failed to fetch mesh data:', err)
      setError('Failed to connect to mesh API. Using demo mode.')
      
      // Set up demo data
      const demoNodes: MeshNode[] = [
        {
          id: 'gateway-1',
          name: 'Main Gateway',
          type: 'gateway',
          status: 'online',
          capabilities: ['routing', 'auth'],
          sessions: 3,
        },
        {
          id: 'agent-1',
          name: 'Chat Agent',
          type: 'agent',
          status: 'online',
          capabilities: ['chat', 'conversation'],
          sessions: 1,
        },
        {
          id: 'agent-2',
          name: 'Vision Agent',
          type: 'agent',
          status: 'online',
          capabilities: ['image', 'vision', 'detection'],
          sessions: 0,
        },
        {
          id: 'skill-1',
          name: 'Web Search',
          type: 'skill',
          status: 'online',
          capabilities: ['search', 'web'],
        },
        {
          id: 'service-1',
          name: 'Database',
          type: 'service',
          status: 'online',
          capabilities: ['storage', 'query'],
        },
      ]

      const demoConnections: MeshConnection[] = [
        { source: 'gateway-1', target: 'agent-1', type: 'direct', active: true },
        { source: 'gateway-1', target: 'agent-2', type: 'direct', active: true },
        { source: 'agent-1', target: 'skill-1', type: 'routed', active: true },
        { source: 'agent-1', target: 'service-1', type: 'routed', active: true },
        { source: 'agent-2', target: 'service-1', type: 'routed', active: false },
      ]

      setNodes(demoNodes)
      setConnections(demoConnections)
      setMeshStatus({
        totalNodes: demoNodes.length,
        activeNodes: demoNodes.filter(n => n.status === 'online').length,
        totalConnections: demoConnections.length,
        activeIntents: 0,
        health: 'healthy',
      })
    }
  }, [])

  // Set up WebSocket connection
  useEffect(() => {
    fetchMeshData()

    // Attempt WebSocket connection
    const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = `${wsProtocol}//${window.location.host}${API_BASE}/mesh/ws`

    try {
      const ws = new WebSocket(wsUrl)
      
      ws.onopen = () => {
        console.log('WebSocket connected')
        setIsConnected(true)
      }

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          
          switch (message.type) {
            case 'node_update':
              setNodes(prev => {
                const updated = message.data as MeshNode
                const index = prev.findIndex(n => n.id === updated.id)
                if (index >= 0) {
                  const newNodes = [...prev]
                  newNodes[index] = { ...newNodes[index], ...updated }
                  return newNodes
                }
                return [...prev, updated]
              })
              break

            case 'connection_update':
              setConnections(message.data)
              break

            case 'intent_update':
              const intent = message.data as MeshIntent
              setActiveIntents(prev => {
                const existing = prev.findIndex(i => i.id === intent.id)
                if (existing >= 0) {
                  const newIntents = [...prev]
                  newIntents[existing] = intent
                  return newIntents
                }
                return [...prev, intent]
              })
              
              // Remove completed intents after animation
              if (intent.status === 'completed' || intent.status === 'failed') {
                setTimeout(() => {
                  setActiveIntents(prev => prev.filter(i => i.id !== intent.id))
                }, 3000)
              }
              break

            case 'status_update':
              setMeshStatus(message.data)
              break
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err)
        }
      }

      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setIsConnected(false)
      }

      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setIsConnected(false)
      }

      wsRef.current = ws

      return () => {
        ws.close()
      }
    } catch (err) {
      console.error('Failed to establish WebSocket connection:', err)
      setIsConnected(false)
    }
  }, [fetchMeshData])

  // Handle adding test node
  const handleAddNode = useCallback((testNode: TestNode) => {
    const newNode: MeshNode = {
      id: `test-${Date.now()}`,
      name: testNode.name,
      type: testNode.type,
      status: 'online',
      capabilities: testNode.capabilities,
      sessions: 0,
    }

    setNodes(prev => [...prev, newNode])

    // Auto-connect to gateway if exists
    const gateway = nodes.find(n => n.type === 'gateway')
    if (gateway) {
      setConnections(prev => [
        ...prev,
        {
          source: gateway.id,
          target: newNode.id,
          type: 'direct',
          active: true,
        },
      ])
    }
  }, [nodes])

  // Handle removing node
  const handleRemoveNode = useCallback((nodeId: string) => {
    setNodes(prev => prev.filter(n => n.id !== nodeId))
    setConnections(prev =>
      prev.filter(c => c.source !== nodeId && c.target !== nodeId)
    )
  }, [])

  // Handle sending test intent
  const handleSendIntent = useCallback(async (intent: {
    content: string
    targetCapability?: string
  }) => {
    try {
      const response = await axios.post<MeshIntent>(`${API_BASE}/mesh/intent`, intent)
      
      // Add the intent to active intents
      setActiveIntents(prev => [...prev, response.data])
    } catch (err) {
      console.error('Failed to send intent:', err)
      
      // Simulate intent in demo mode
      const sourceNode = nodes.find(n => n.type === 'gateway') || nodes[0]
      const targetNode = intent.targetCapability
        ? nodes.find(n => n.capabilities.includes(intent.targetCapability!))
        : nodes[Math.floor(Math.random() * nodes.length)]

      if (sourceNode && targetNode) {
        const simulatedIntent: MeshIntent = {
          id: `intent-${Date.now()}`,
          content: intent.content,
          source: sourceNode.id,
          target: targetNode.id,
          route: [sourceNode.id, targetNode.id],
          timestamp: new Date().toISOString(),
          status: 'routing',
        }

        setActiveIntents(prev => [...prev, simulatedIntent])

        // Simulate completion
        setTimeout(() => {
          setActiveIntents(prev =>
            prev.map(i =>
              i.id === simulatedIntent.id
                ? { ...i, status: 'completed' as const, duration: Math.random() * 1000 + 500 }
                : i
            )
          )

          // Remove after animation
          setTimeout(() => {
            setActiveIntents(prev => prev.filter(i => i.id !== simulatedIntent.id))
          }, 3000)
        }, 2000)
      }
    }
  }, [nodes])

  // Handle canvas click (add node at position)
  const handleCanvasClick = useCallback((x: number, y: number) => {
    if (!isTestPanelOpen) return

    const newNode: MeshNode = {
      id: `node-${Date.now()}`,
      name: `Node ${nodes.length + 1}`,
      type: 'agent',
      status: 'online',
      capabilities: ['test'],
      position: { x, y },
    }

    setNodes(prev => [...prev, newNode])
  }, [nodes.length, isTestPanelOpen])

  return (
    <div className="h-full w-full flex flex-col bg-gray-950">
      {/* Header */}
      <div className="bg-gray-900 border-b border-gray-800 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Radio className="text-purple-500" size={28} />
            Atmosphere Mesh
          </h1>
          <div className="flex items-center gap-2 text-sm">
            <div
              className={`w-2 h-2 rounded-full ${
                isConnected ? 'bg-green-500' : 'bg-red-500'
              }`}
            />
            <span className="text-gray-400">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {meshStatus && (
          <div className="flex items-center gap-6 text-sm">
            <div className="flex items-center gap-2">
              <Activity size={16} className="text-blue-400" />
              <span className="text-gray-300">
                {meshStatus.activeNodes}/{meshStatus.totalNodes} nodes
              </span>
            </div>
            <div className="flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  meshStatus.health === 'healthy' ? 'bg-green-500' :
                  meshStatus.health === 'degraded' ? 'bg-yellow-500' :
                  'bg-red-500'
                }`}
              />
              <span className="text-gray-300 capitalize">{meshStatus.health}</span>
            </div>
            <div className="flex items-center gap-2 text-purple-400">
              <span>{activeIntents.length} active intents</span>
            </div>
          </div>
        )}
      </div>

      {/* Error banner */}
      {error && (
        <div className="bg-yellow-900 bg-opacity-50 border-b border-yellow-700 px-6 py-2 flex items-center gap-2 text-yellow-200 text-sm">
          <AlertCircle size={16} />
          <span>{error}</span>
        </div>
      )}

      {/* Main content */}
      <div ref={containerRef} className="flex-1 relative">
        <MeshCanvas
          nodes={nodes}
          connections={connections}
          activeIntents={activeIntents}
          onNodeClick={setSelectedNode}
          onCanvasClick={handleCanvasClick}
          width={dimensions.width}
          height={dimensions.height}
        />

        <TestPanel
          nodes={nodes}
          onAddNode={handleAddNode}
          onRemoveNode={handleRemoveNode}
          onSendIntent={handleSendIntent}
          isOpen={isTestPanelOpen}
          onToggle={() => setIsTestPanelOpen(!isTestPanelOpen)}
        />
      </div>

      {/* Selected node details */}
      {selectedNode && (
        <div className="bg-gray-900 border-t border-gray-800 px-6 py-4">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-white mb-2">
                {selectedNode.name}
              </h3>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Type:</span>
                  <span className="text-white ml-2">{selectedNode.type}</span>
                </div>
                <div>
                  <span className="text-gray-400">Status:</span>
                  <span className="text-white ml-2">{selectedNode.status}</span>
                </div>
                {selectedNode.sessions !== undefined && (
                  <div>
                    <span className="text-gray-400">Active Sessions:</span>
                    <span className="text-white ml-2">{selectedNode.sessions}</span>
                  </div>
                )}
                {selectedNode.lastSeen && (
                  <div>
                    <span className="text-gray-400">Last Seen:</span>
                    <span className="text-white ml-2">
                      {new Date(selectedNode.lastSeen).toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
              {selectedNode.capabilities.length > 0 && (
                <div className="mt-3">
                  <span className="text-gray-400 text-sm">Capabilities:</span>
                  <div className="flex flex-wrap gap-2 mt-1">
                    {selectedNode.capabilities.map(cap => (
                      <span
                        key={cap}
                        className="bg-purple-900 bg-opacity-50 text-purple-200 px-3 py-1 rounded-full text-xs"
                      >
                        {cap}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-white"
            >
              <AlertCircle size={20} />
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
