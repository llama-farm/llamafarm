import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import type { MeshNode, MeshConnection, MeshIntent } from '../../types/atmosphere'

interface MeshCanvasProps {
  nodes: MeshNode[]
  connections: MeshConnection[]
  activeIntents: MeshIntent[]
  onNodeClick?: (node: MeshNode) => void
  onCanvasClick?: (x: number, y: number) => void
  width?: number
  height?: number
}

interface D3Node extends MeshNode {
  x?: number
  y?: number
  vx?: number
  vy?: number
  fx?: number | null
  fy?: number | null
}

interface D3Link {
  source: D3Node
  target: D3Node
  type: MeshConnection['type']
  active: boolean
  latency?: number
  strength?: number
}

export default function MeshCanvas({
  nodes,
  connections,
  activeIntents,
  onNodeClick,
  onCanvasClick,
  width = 1200,
  height = 800,
}: MeshCanvasProps) {
  const svgRef = useRef<SVGSVGElement>(null)
  const simulationRef = useRef<d3.Simulation<D3Node, D3Link> | null>(null)
  const [hoveredNode, setHoveredNode] = useState<string | null>(null)

  // Status color mapping
  const getStatusColor = (status: MeshNode['status']) => {
    const colors = {
      online: '#10b981', // green
      offline: '#6b7280', // gray
      busy: '#f59e0b', // amber
      error: '#ef4444', // red
    }
    return colors[status]
  }

  // Node type sizing
  const getNodeRadius = (type: MeshNode['type']) => {
    const sizes = {
      gateway: 30,
      agent: 24,
      skill: 18,
      service: 20,
    }
    return sizes[type]
  }

  useEffect(() => {
    if (!svgRef.current) return

    const svg = d3.select(svgRef.current)
    svg.selectAll('*').remove()

    // Create container groups
    const container = svg.append('g').attr('class', 'container')
    const linksGroup = container.append('g').attr('class', 'links')
    const nodesGroup = container.append('g').attr('class', 'nodes')
    const labelsGroup = container.append('g').attr('class', 'labels')
    const intentsGroup = container.append('g').attr('class', 'intents')

    // Add zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.3, 3])
      .on('zoom', (event) => {
        container.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Canvas click for adding nodes
    svg.on('click', function (event) {
      if (event.target === svgRef.current && onCanvasClick) {
        const transform = d3.zoomTransform(svgRef.current!)
        const [x, y] = transform.invert(d3.pointer(event))
        onCanvasClick(x, y)
      }
    })

    // Prepare data
    const d3Nodes: D3Node[] = nodes.map(node => ({
      ...node,
      x: node.position?.x ?? width / 2 + (Math.random() - 0.5) * 200,
      y: node.position?.y ?? height / 2 + (Math.random() - 0.5) * 200,
    }))

    const d3Links: D3Link[] = connections.map(conn => {
      const source = d3Nodes.find(n => n.id === conn.source)!
      const target = d3Nodes.find(n => n.id === conn.target)!
      return { ...conn, source, target }
    })

    // Create force simulation
    const simulation = d3.forceSimulation<D3Node>(d3Nodes)
      .force('link', d3.forceLink<D3Node, D3Link>(d3Links)
        .id(d => d.id)
        .distance(150)
        .strength(0.5))
      .force('charge', d3.forceManyBody().strength(-400))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d: any) => getNodeRadius(d.type) + 20))

    simulationRef.current = simulation

    // Draw connections
    const link = linksGroup
      .selectAll<SVGLineElement, D3Link>('line')
      .data(d3Links)
      .join('line')
      .attr('class', 'connection')
      .attr('stroke', d => d.active ? '#3b82f6' : '#4b5563')
      .attr('stroke-width', d => d.active ? 2.5 : 1.5)
      .attr('stroke-opacity', d => d.active ? 0.8 : 0.3)
      .attr('stroke-dasharray', d => d.type === 'fallback' ? '5,5' : 'none')

    // Add connection labels
    const linkLabels = linksGroup
      .selectAll<SVGTextElement, D3Link>('text')
      .data(d3Links.filter(d => d.latency))
      .join('text')
      .attr('class', 'connection-label')
      .attr('font-size', 10)
      .attr('fill', '#9ca3af')
      .text(d => `${d.latency}ms`)

    // Draw nodes
    const node = nodesGroup
      .selectAll<SVGCircleElement, D3Node>('circle')
      .data(d3Nodes)
      .join('circle')
      .attr('class', 'node')
      .attr('r', d => getNodeRadius(d.type))
      .attr('fill', d => getStatusColor(d.status))
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .style('filter', d => d.sessions && d.sessions > 0 ? 'drop-shadow(0 0 8px rgba(59, 130, 246, 0.6))' : 'none')
      .on('click', (event, d) => {
        event.stopPropagation()
        if (onNodeClick) onNodeClick(d)
      })
      .on('mouseenter', (event, d) => {
        setHoveredNode(d.id)
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', getNodeRadius(d.type) * 1.3)
      })
      .on('mouseleave', (event, d) => {
        setHoveredNode(null)
        d3.select(event.currentTarget)
          .transition()
          .duration(200)
          .attr('r', getNodeRadius(d.type))
      })
      .call(d3.drag<SVGCircleElement, D3Node>()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart()
          d.fx = d.x
          d.fy = d.y
        })
        .on('drag', (event, d) => {
          d.fx = event.x
          d.fy = event.y
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0)
          d.fx = null
          d.fy = null
        }))

    // Draw node labels
    const label = labelsGroup
      .selectAll<SVGTextElement, D3Node>('text')
      .data(d3Nodes)
      .join('text')
      .attr('class', 'node-label')
      .attr('text-anchor', 'middle')
      .attr('dy', d => getNodeRadius(d.type) + 16)
      .attr('font-size', 12)
      .attr('font-weight', 500)
      .attr('fill', '#e5e7eb')
      .text(d => d.name)
      .style('pointer-events', 'none')

    // Draw capability badges
    const badges = labelsGroup
      .selectAll<SVGTextElement, D3Node>('text.capability-badge')
      .data(d3Nodes)
      .join('text')
      .attr('class', 'capability-badge')
      .attr('text-anchor', 'middle')
      .attr('dy', d => getNodeRadius(d.type) + 30)
      .attr('font-size', 9)
      .attr('fill', '#9ca3af')
      .text(d => d.capabilities.length ? `${d.capabilities.length} cap` : '')
      .style('pointer-events', 'none')

    // Draw intent routing paths
    const drawIntentPaths = () => {
      intentsGroup.selectAll('*').remove()

      activeIntents.forEach((intent) => {
        if (intent.route.length < 2) return

        const routeNodes = intent.route
          .map(id => d3Nodes.find(n => n.id === id))
          .filter(n => n !== undefined) as D3Node[]

        if (routeNodes.length < 2) return

        // Create path
        const pathData = routeNodes.map((node) => ({
          x: node.x || 0,
          y: node.y || 0,
        }))

        const lineGenerator = d3.line<{ x: number; y: number }>()
          .x(d => d.x)
          .y(d => d.y)
          .curve(d3.curveCardinal.tension(0.5))

        const path = intentsGroup
          .append('path')
          .attr('d', lineGenerator(pathData))
          .attr('fill', 'none')
          .attr('stroke', intent.status === 'completed' ? '#10b981' : '#8b5cf6')
          .attr('stroke-width', 3)
          .attr('stroke-opacity', 0.7)
          .attr('stroke-dasharray', '8,4')

        // Animate the path
        const totalLength = path.node()?.getTotalLength() || 0
        path
          .attr('stroke-dasharray', `${totalLength} ${totalLength}`)
          .attr('stroke-dashoffset', totalLength)
          .transition()
          .duration(2000)
          .ease(d3.easeLinear)
          .attr('stroke-dashoffset', 0)
          .on('end', () => {
            if (intent.status === 'routing') {
              // Keep animating for routing intents
              path
                .transition()
                .duration(2000)
                .ease(d3.easeLinear)
                .attr('stroke-dashoffset', -totalLength)
                .on('end', function repeat() {
                  d3.select(this)
                    .attr('stroke-dashoffset', totalLength)
                    .transition()
                    .duration(2000)
                    .ease(d3.easeLinear)
                    .attr('stroke-dashoffset', 0)
                    .on('end', repeat)
                })
            }
          })

        // Add intent marker at the end
        const endNode = routeNodes[routeNodes.length - 1]
        intentsGroup
          .append('circle')
          .attr('cx', endNode.x || 0)
          .attr('cy', endNode.y || 0)
          .attr('r', 6)
          .attr('fill', intent.status === 'completed' ? '#10b981' : '#8b5cf6')
          .attr('opacity', 0.8)
      })
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x || 0)
        .attr('y1', d => d.source.y || 0)
        .attr('x2', d => d.target.x || 0)
        .attr('y2', d => d.target.y || 0)

      linkLabels
        .attr('x', d => ((d.source.x || 0) + (d.target.x || 0)) / 2)
        .attr('y', d => ((d.source.y || 0) + (d.target.y || 0)) / 2)

      node
        .attr('cx', d => d.x || 0)
        .attr('cy', d => d.y || 0)

      label
        .attr('x', d => d.x || 0)
        .attr('y', d => d.y || 0)

      badges
        .attr('x', d => d.x || 0)
        .attr('y', d => d.y || 0)

      drawIntentPaths()
    })

    return () => {
      simulation.stop()
    }
  }, [nodes, connections, width, height])

  // Update intent paths when activeIntents change
  useEffect(() => {
    if (simulationRef.current && svgRef.current) {
      const svg = d3.select(svgRef.current)
      const intentsGroup = svg.select('.intents')
      
      // Redraw intent paths
      intentsGroup.selectAll('*').remove()
      // Trigger a simulation tick to redraw
      simulationRef.current.alpha(0.1).restart()
    }
  }, [activeIntents])

  return (
    <div className="relative w-full h-full bg-gray-900 rounded-lg overflow-hidden">
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full h-full"
      />
      
      {/* Legend */}
      <div className="absolute top-4 left-4 bg-gray-800 bg-opacity-90 rounded-lg p-4 text-xs text-gray-300">
        <div className="font-semibold mb-2">Status</div>
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-green-500" />
            <span>Online</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-amber-500" />
            <span>Busy</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-red-500" />
            <span>Error</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-gray-500" />
            <span>Offline</span>
          </div>
        </div>
        
        <div className="font-semibold mt-4 mb-2">Node Types</div>
        <div className="space-y-1">
          <div>Gateway: 30px</div>
          <div>Agent: 24px</div>
          <div>Service: 20px</div>
          <div>Skill: 18px</div>
        </div>

        <div className="font-semibold mt-4 mb-2">Interactions</div>
        <div className="space-y-1">
          <div>Drag: Move nodes</div>
          <div>Scroll: Zoom</div>
          <div>Click: Select node</div>
          <div>Click canvas: Add node</div>
        </div>
      </div>

      {/* Hovered node tooltip */}
      {hoveredNode && (
        <div className="absolute top-4 right-4 bg-gray-800 bg-opacity-95 rounded-lg p-4 text-sm text-gray-200 min-w-[200px]">
          {(() => {
            const node = nodes.find(n => n.id === hoveredNode)
            if (!node) return null
            return (
              <>
                <div className="font-semibold text-white mb-2">{node.name}</div>
                <div className="space-y-1 text-xs">
                  <div>Type: <span className="text-blue-400">{node.type}</span></div>
                  <div>Status: <span style={{ color: getStatusColor(node.status) }}>{node.status}</span></div>
                  {node.sessions && node.sessions > 0 && (
                    <div>Sessions: <span className="text-purple-400">{node.sessions}</span></div>
                  )}
                  {node.capabilities.length > 0 && (
                    <div className="mt-2">
                      <div className="font-semibold mb-1">Capabilities:</div>
                      <div className="flex flex-wrap gap-1">
                        {node.capabilities.map(cap => (
                          <span key={cap} className="bg-gray-700 px-2 py-0.5 rounded text-xs">
                            {cap}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </>
            )
          })()}
        </div>
      )}
    </div>
  )
}
