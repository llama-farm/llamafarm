# Atmosphere UI Architecture

## Component Hierarchy

```
MeshView (Container)
├── MeshCanvas (D3 Visualization)
│   ├── Force-directed graph
│   ├── Node rendering
│   ├── Connection lines
│   ├── Intent path animation
│   └── Legend & tooltips
│
└── TestPanel (Controls)
    ├── Add Node form
    ├── Send Intent form
    └── Active Nodes list
```

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│                    MeshView                         │
│  ┌──────────────────────────────────────────────┐  │
│  │  State Management                            │  │
│  │  - nodes: MeshNode[]                         │  │
│  │  - connections: MeshConnection[]             │  │
│  │  - activeIntents: MeshIntent[]               │  │
│  │  - meshStatus: MeshStatus                    │  │
│  └──────────────────────────────────────────────┘  │
│                       │                             │
│         ┌─────────────┴──────────────┐              │
│         ▼                            ▼              │
│  ┌─────────────┐              ┌──────────────┐     │
│  │ MeshCanvas  │              │  TestPanel   │     │
│  │             │              │              │     │
│  │ - Displays  │              │ - Add nodes  │     │
│  │   nodes     │◄─────────────┤ - Send       │     │
│  │ - Shows     │   callbacks  │   intents    │     │
│  │   intents   │              │ - Remove     │     │
│  │ - Animates  │              │   nodes      │     │
│  └─────────────┘              └──────────────┘     │
└─────────────────────────────────────────────────────┘
         │                               │
         ▼                               ▼
    D3.js Force                    User Input
    Simulation                     Events
```

## API Integration

```
┌──────────────┐
│  MeshView    │
└──────┬───────┘
       │
       ├─► REST API
       │   ├─ GET /mesh/nodes      → Initial load
       │   ├─ GET /mesh/status     → Health check
       │   └─ POST /mesh/intent    → Send intent
       │
       └─► WebSocket (/mesh/ws)
           ├─ node_update          → Update node state
           ├─ connection_update    → Refresh connections
           ├─ intent_update        → Track intent lifecycle
           └─ status_update        → Mesh health changes
```

## Event Flow: Adding a Node

```
User clicks "Add Node"
    │
    ▼
TestPanel.handleAddNode()
    │
    ├─ Create MeshNode object
    ├─ Generate unique ID
    └─ Emit to parent
        │
        ▼
MeshView.handleAddNode()
    │
    ├─ Add to nodes state
    ├─ Auto-connect to gateway
    └─ Update connections
        │
        ▼
MeshCanvas re-renders
    │
    ├─ D3 adds new node to simulation
    ├─ Force layout repositions nodes
    └─ New node animates into position
```

## Event Flow: Sending an Intent

```
User clicks "Send Intent"
    │
    ▼
TestPanel.handleSendIntent()
    │
    └─ Emit to parent with content + target capability
        │
        ▼
MeshView.handleSendIntent()
    │
    ├─ POST to /mesh/intent API
    │   │
    │   └─► Backend routes intent
    │       └─► Returns MeshIntent with route[]
    │
    ├─ Add to activeIntents state
    │
    └─ Trigger animation
        │
        ▼
MeshCanvas.drawIntentPaths()
    │
    ├─ Map intent.route to node positions
    ├─ Draw animated path with d3.line()
    ├─ Animate stroke-dashoffset for flow effect
    │
    └─ After completion:
        ├─ Update intent status to "completed"
        └─ Remove after 3s delay
```

## D3.js Force Simulation

```
ForceSimulation
├── Force: Link
│   └── Pulls connected nodes together (distance: 150px)
│
├── Force: Charge
│   └── Repels all nodes (-400 strength)
│
├── Force: Center
│   └── Pulls toward canvas center
│
└── Force: Collision
    └── Prevents node overlap (radius + 20px buffer)

Updates every tick (60fps):
    ├─ Recalculate positions
    ├─ Update node circles (cx, cy)
    ├─ Update connection lines (x1, y1, x2, y2)
    └─ Redraw intent paths
```

## State Management Patterns

### MeshView (Container)
- Owns all mesh data state
- Handles API calls and WebSocket
- Passes data down to children
- Receives callbacks from children

### MeshCanvas (Presentation)
- Receives props, no state management
- Pure D3 visualization
- Emits user interactions via callbacks
- Re-renders on prop changes

### TestPanel (Form Controls)
- Local form state only
- Emits structured data via callbacks
- No direct mesh state access

## Type Safety

All components are fully typed:

```typescript
// Props interfaces ensure type safety
interface MeshCanvasProps {
  nodes: MeshNode[]
  connections: MeshConnection[]
  activeIntents: MeshIntent[]
  onNodeClick?: (node: MeshNode) => void
  onCanvasClick?: (x: number, y: number) => void
}

// Internal D3 types extend base types
interface D3Node extends MeshNode {
  x?: number
  y?: number
  vx?: number
  vy?: number
  fx?: number | null
  fy?: number | null
}
```

## Performance Optimizations

1. **Efficient Re-renders**
   - React.memo() candidates: TestPanel
   - useCallback() for stable references
   - D3 data binding for minimal DOM updates

2. **Simulation Control**
   - Pause simulation when off-screen
   - Lower alpha for fewer ticks
   - Stop on unmount

3. **Intent Cleanup**
   - Remove completed intents after 3s
   - Limit active intents shown (e.g., 20 max)
   - Reuse animation elements

4. **Canvas Sizing**
   - ResizeObserver for container dimensions
   - Debounced resize events
   - SVG viewBox for scaling

## Extensibility Points

### Custom Node Renderers
```typescript
// In MeshCanvas.tsx
const renderCustomNode = (node: MeshNode) => {
  switch (node.type) {
    case 'custom':
      return <CustomSVGElement ... />
    default:
      return <circle ... />
  }
}
```

### Additional Force Types
```typescript
// Add custom forces to simulation
simulation
  .force('boundary', d3.forceBox()...)
  .force('gravity', d3.forceRadial()...)
```

### Plugin System
```typescript
// Define plugin interface
interface MeshPlugin {
  name: string
  onNodeClick?: (node: MeshNode) => void
  onIntentComplete?: (intent: MeshIntent) => void
  render?: () => ReactNode
}
```

## Testing Strategy

### Unit Tests
- Component rendering with mock data
- Callback invocation
- State updates

### Integration Tests
- User interactions (click, drag, zoom)
- API call handling
- WebSocket message processing

### Visual Tests
- Snapshot tests for different mesh configurations
- Animation verification
- Responsive layout checks
