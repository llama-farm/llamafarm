# 🌐 Atmosphere Mesh Visualizer

A real-time, interactive visualization of the Atmosphere distributed agent mesh network.

## Features

### 🎨 Visual Mesh Display
- **Force-directed graph layout** using D3.js
- **Color-coded node status**: Online (green), Busy (amber), Error (red), Offline (gray)
- **Node types with distinct sizing**: Gateway (30px), Agent (24px), Service (20px), Skill (18px)
- **Connection visualization**: Direct, routed, and fallback connections with animation
- **Real-time intent routing**: Watch intents flow through the mesh with animated paths
- **Session tracking**: Visual indicators for active sessions on nodes

### 🧪 Test Mode Panel
- **Add simulated nodes** by clicking on the canvas or using the test panel
- **Configure node capabilities** (comma-separated)
- **Send test intents** with optional capability targeting
- **View active nodes** with status and capabilities
- **Remove nodes** with a single click

### 🔌 API Integration
The component expects these backend endpoints:

```
GET  /api/atmosphere/mesh/nodes     - List all nodes
GET  /api/atmosphere/mesh/status    - Get mesh health status
POST /api/atmosphere/mesh/intent    - Send a test intent
WS   /api/atmosphere/mesh/ws        - WebSocket for real-time updates
```

#### WebSocket Message Format
```typescript
{
  type: 'node_update' | 'connection_update' | 'intent_update' | 'status_update',
  data: /* Type-specific payload */,
  timestamp: string
}
```

### 🎮 Interactions
- **Drag nodes** to reposition them in the graph
- **Zoom/pan** with mouse wheel and drag
- **Click nodes** to view detailed information
- **Click canvas** (in test mode) to add a node at that position
- **Hover nodes** for quick info tooltip

## Usage

Navigate to `/atmosphere` in the designer to view the mesh:

```typescript
// In your router
<Route path="/atmosphere" element={<MeshView />} />
```

## Demo Mode

If the API is unavailable, the component automatically falls back to demo mode with:
- 5 sample nodes (1 gateway, 2 agents, 1 skill, 1 service)
- Pre-configured connections
- Simulated intent routing

## Architecture

### Components
- **MeshView.tsx** - Main container with API integration and WebSocket handling
- **MeshCanvas.tsx** - D3.js force-directed graph visualization
- **TestPanel.tsx** - Test mode controls for simulating nodes and intents

### Types
All types are defined in `types/atmosphere.ts`:
- `MeshNode` - Node in the mesh
- `MeshConnection` - Connection between nodes
- `MeshIntent` - Intent being routed through mesh
- `MeshStatus` - Overall mesh health status
- `WebSocketMessage` - Real-time update messages

## Customization

### Adjust Visualization
Edit `MeshCanvas.tsx` to customize:
- Node colors, sizes, and styling
- Connection appearance
- Force simulation parameters
- Animation timing and effects

### Extend Test Panel
Edit `TestPanel.tsx` to add:
- Additional node configuration options
- Intent scheduling
- Batch operations
- Node grouping

## Performance

The visualization efficiently handles:
- 50+ nodes with smooth animation
- Real-time intent routing
- Multiple simultaneous WebSocket updates
- Responsive canvas resizing

For larger meshes (100+ nodes), consider:
- Implementing canvas clustering
- Adding zoom-based detail levels
- Lazy-loading node metadata

## Future Enhancements

- [ ] Node grouping and hierarchies
- [ ] Historical intent playback
- [ ] Performance metrics overlay
- [ ] Export mesh topology as JSON/SVG
- [ ] Collaborative viewing (multi-user cursors)
- [ ] Search and filter nodes by capability
- [ ] Alert notifications for node failures
