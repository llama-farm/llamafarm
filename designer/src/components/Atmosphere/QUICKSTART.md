# 🚀 Atmosphere UI Quick Start

## Access the UI

Navigate to: **`http://localhost:5173/atmosphere`**

## 5-Minute Demo

### Step 1: Open the Mesh
```
npm run dev
# Navigate to /atmosphere in your browser
```

You'll see:
- 5 demo nodes in a force-directed graph
- Connections between them (lines)
- A test panel on the right
- Status bar at the top

### Step 2: Explore Interactions

**Try these:**
- **Drag a node** → It moves and settles into a new position
- **Zoom in/out** → Mouse wheel or trackpad pinch
- **Hover a node** → See details tooltip on the right
- **Click a node** → See full details at the bottom

### Step 3: Add a Node

**Click "Open Test Panel"** (if closed), then:

1. Enter node name: `"My Test Agent"`
2. Select type: `Agent`
3. Add capabilities: `"chat, vision"`
4. Click **"Add Node"**

✨ Watch it appear and connect to the gateway!

### Step 4: Send an Intent

In the Test Panel:

1. Enter intent: `"Analyze this image"`
2. Select target capability: `vision` (optional)
3. Click **"Send Intent"**

🎬 Watch the animated path flow through the mesh!

## Visual Legend

### Node Colors
- 🟢 **Green** = Online
- 🟡 **Amber** = Busy
- 🔴 **Red** = Error
- ⚫ **Gray** = Offline

### Node Sizes
- **Largest** = Gateway (30px)
- **Large** = Agent (24px)
- **Medium** = Service (20px)
- **Small** = Skill (18px)

### Connection Types
- **Solid blue** = Active direct connection
- **Dashed** = Fallback connection
- **Gray** = Inactive connection

### Intent Animation
- **Purple flowing line** = Intent routing in progress
- **Green flowing line** = Intent completed

## Common Tasks

### Add Multiple Nodes Quickly
Click anywhere on the canvas to add a node at that position (test mode only).

### Connect Nodes
Nodes automatically connect to the nearest gateway. To customize, you'll need to use the API.

### View Node Details
1. Click a node
2. See details in the bottom panel
3. View capabilities, status, and active sessions

### Monitor Mesh Health
Top right corner shows:
- Active/Total nodes
- Overall health (Healthy/Degraded/Critical)
- Active intents count

## Backend Integration

### Required Endpoints

```bash
# List nodes
GET /api/atmosphere/mesh/nodes

# Get mesh status
GET /api/atmosphere/mesh/status

# Send an intent
POST /api/atmosphere/mesh/intent
{
  "content": "Do something",
  "targetCapability": "optional"
}

# WebSocket for updates
WS /api/atmosphere/mesh/ws
```

### WebSocket Messages

The UI subscribes to these events:

```typescript
// Node state changed
{ type: 'node_update', data: MeshNode, timestamp: '...' }

// Connections changed
{ type: 'connection_update', data: MeshConnection[], timestamp: '...' }

// Intent lifecycle
{ type: 'intent_update', data: MeshIntent, timestamp: '...' }

// Mesh health changed
{ type: 'status_update', data: MeshStatus, timestamp: '...' }
```

## Demo Mode

**No backend?** No problem!

The UI automatically falls back to demo mode with:
- 5 pre-configured nodes
- Simulated connections
- Mock intent routing

Everything still works for testing and presentations!

## Troubleshooting

### "Disconnected" indicator
- Backend is not running or WebSocket failed
- UI still works in demo mode
- Check console for connection errors

### Nodes not moving
- Simulation may have settled (this is normal)
- Drag a node to restart physics
- Check browser performance (60fps needed)

### Intent animation not showing
- Make sure intent has a valid route (2+ nodes)
- Check that target nodes exist
- Look for errors in browser console

### Performance issues
- Limit to ~50 nodes for smooth animation
- Close other browser tabs
- Check CPU usage in devtools

## Keyboard Shortcuts (Future)

These could be added:

- `Space` → Pause/resume simulation
- `R` → Reset zoom
- `Escape` → Deselect node
- `Cmd/Ctrl + A` → Select all nodes
- `Delete` → Remove selected node

## Customization

### Change Colors

Edit `MeshCanvas.tsx`:

```typescript
const getStatusColor = (status: MeshNode['status']) => {
  return {
    online: '#YOUR_COLOR',  // Change these
    busy: '#YOUR_COLOR',
    error: '#YOUR_COLOR',
    offline: '#YOUR_COLOR',
  }[status]
}
```

### Adjust Physics

```typescript
const simulation = d3.forceSimulation(nodes)
  .force('link', d3.forceLink(links)
    .distance(200)      // Further apart
    .strength(0.8))     // Stronger pull
  .force('charge', d3.forceManyBody()
    .strength(-600))    // More repulsion
```

### Add Custom Node Types

1. Add to type definition:
```typescript
type: 'gateway' | 'agent' | 'skill' | 'service' | 'custom'
```

2. Add size mapping:
```typescript
const getNodeRadius = (type) => ({
  // ...
  custom: 22
})[type]
```

3. Add icon or custom renderer

## Production Checklist

Before deploying:

- [ ] Backend API endpoints working
- [ ] WebSocket connection stable
- [ ] Authentication/authorization in place
- [ ] Error handling for API failures
- [ ] Rate limiting for test intent sends
- [ ] Monitoring/alerting for mesh issues
- [ ] Performance testing with production node count
- [ ] Mobile responsiveness verified
- [ ] Accessibility audit passed
- [ ] Documentation updated with prod URLs

## Support

**Issues?** Check:

1. Browser console for errors
2. Network tab for API failures
3. `ARCHITECTURE.md` for implementation details
4. `README.md` for feature documentation

**Performance problems?** See ARCHITECTURE.md section on optimizations.

---

**Ready to go!** Start with Step 1 above. 🚀
