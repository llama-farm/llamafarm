# Atmosphere Demo Mode

Interactive testing environment for the Atmosphere mesh network.

## Usage

```bash
atmosphere demo
```

This launches an interactive CLI with a menu-driven interface for testing mesh functionality.

## Features

### Menu Options

**a) Create test mesh** - Initializes a simulated mesh with 3 nodes:
- **CodeBot** - Specializes in code generation and Python expertise
- **VisionBot** - Handles image analysis and object detection
- **ChatBot** - General conversation and Q&A

**b) Add node** - Add a custom node with specific capabilities
- Define node name
- Add multiple capabilities with custom labels
- Auto-connects to existing mesh peers

**c) List nodes** - Display all nodes in the mesh
- Shows node IDs, capabilities, peer count, and status
- Tracks intent handling statistics

**d) Send intent** - Route a test intent through the mesh
- Uses semantic routing to find the best matching node
- Shows routing decisions in real-time
- Displays confidence scores, hops, and latency

**e) Kill node** - Simulate node failure
- Takes a node offline
- Demonstrates fault tolerance
- Gradient tables automatically exclude failed nodes

**f) Revive node** - Bring a failed node back online
- Simulates recovery
- Re-syncs gradient tables
- Demonstrates self-healing

**g) Show routes** - Display gradient table for any node
- View all known capabilities
- See hop counts and latency estimates
- Understand routing topology

**h) Generate token** - Create an invite token
- Simulate mesh join authentication
- Shows token format and metadata

**i) Join mesh** - Simulate a new node joining via token
- Full join workflow simulation
- Demonstrates peer discovery and capability sync

**j) Show intent log** - View routing history
- Last 10 intents with full details
- Performance metrics and routing decisions

**q) Quit** - Exit demo mode

## Architecture

The demo creates a simulated mesh environment with:

- **Semantic Routing**: Intents are vectorized and matched against capability vectors
- **Gradient Tables**: Each node maintains routing information with hop counts and latency
- **Gossip Protocol**: Capability information propagates through the mesh
- **Fault Tolerance**: Nodes can fail and recover, routes automatically adapt

## Example Session

```bash
$ atmosphere demo

🌐 Atmosphere Mesh - Interactive Demo

═══ Demo Menu ═══
a) Create test mesh (3 nodes)
...

Choose option: a

✓ Created mesh 'demo-mesh' with 3 nodes!

┏━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Node     ┃ ID             ┃ Capabilities                ┃ Status ┃
┡━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ CodeBot  │ a1b2c3d4...    │ code_generation, python_... │ 🟢 ... │
│ VisionBot│ e5f6g7h8...    │ image_analysis, object_d... │ 🟢 ... │
│ ChatBot  │ i9j0k1l2...    │ conversation, general_qa    │ 🟢 ... │
└──────────┴────────────────┴─────────────────────────────┴────────┘

Choose option: d

Example intents:
  • 'Write a Python function to sort a list'
  • 'Analyze this image for objects'
  • 'Let's have a conversation about AI'

Enter intent: Write a Python function to calculate fibonacci numbers

🧭 Routing intent...

╭─ 🎯 Routing Result ─╮
│ ✓ Intent routed!    │
│                      │
│ Destination: CodeBot│
│ Capability: code_... │
│ Score: 0.892         │
│ Hops: 0              │
│ Latency: 23.4ms     │
╰──────────────────────╯
```

## Technical Details

### Semantic Routing

Intents are vectorized into a 5-dimensional space:
- `[code, vision, chat, technical, creative]`

Keywords determine the vector weights:
- Code: "code", "function", "python", "algorithm"
- Vision: "image", "photo", "detect", "visual"
- Chat: "chat", "conversation", "discuss"
- Technical: "api", "database", "optimize"
- Creative: "create", "design", "story", "art"

### Gradient Tables

Each node maintains a gradient table mapping capability IDs to:
- **Hops**: Distance to the capability source
- **Via**: Next hop node ID
- **Latency**: Estimated end-to-end latency
- **Vector**: Semantic embedding of the capability

Gradients propagate through the mesh via gossip, similar to BGP routing.

### Fault Handling

When a node fails:
1. It's marked as `alive = False`
2. Gradient tables exclude routes through dead nodes
3. Intents automatically route around failures
4. On revival, gradient tables re-sync

## Requirements

- Python 3.12+
- `rich` library for pretty output
- `numpy` for vector operations

## Use Cases

- **Development**: Test routing logic without deploying real nodes
- **Demos**: Show mesh capabilities to stakeholders
- **Debugging**: Visualize routing decisions and gradient propagation
- **Education**: Learn how distributed intent routing works
- **Testing**: Simulate failure scenarios and edge cases

## Future Enhancements

- [ ] Real-time gossip animation
- [ ] Network latency simulation
- [ ] Capability model loading
- [ ] Interactive routing visualization
- [ ] Export test scenarios
- [ ] Replay logged intents
