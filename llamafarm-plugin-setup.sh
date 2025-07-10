#!/bin/bash

# LlamaFarm Plugin Architecture Setup Script
# Creates an extensible plugin system for easy contributions

set -e

echo "🌾 LlamaFarm Plugin Architecture Setup"
echo "====================================="
echo ""

# Check if we're in the llamafarm-cli directory
if [ ! -f "package.json" ] || [ ! -d "src" ]; then
    echo "❌ Please run this script from the llamafarm-cli root directory"
    exit 1
fi

echo "📦 Creating plugin architecture..."

# Create plugin directories
mkdir -p src/plugins/{fields,equipment,pipes}
mkdir -p src/plugins/fields/{mac,linux,windows,raspberry-pi,jetson}
mkdir -p src/plugins/equipment/{databases,rag-pipelines,model-runtimes,tools}
mkdir -p src/plugins/equipment/databases/{chroma,qdrant,pinecone,weaviate,demo}
mkdir -p src/plugins/equipment/rag-pipelines/{basic,llamaindex,langchain,custom}
mkdir -p src/plugins/equipment/model-runtimes/{ollama,llamacpp,huggingface,demo}
mkdir -p src/plugins/pipes/{websocket,webrtc,sse,http-streaming,demo}

# Create templates directory
mkdir -p templates/plugins/{field,equipment,pipe}

# Create plugin registry
echo "📝 Creating plugin registry..."

cat > src/plugins/registry.ts << 'EOF'
/**
 * LlamaFarm Plugin Registry
 * Central registry for all fields, equipment, and pipes
 */

export interface Plugin {
  name: string;
  type: 'field' | 'equipment' | 'pipe';
  version: string;
  description: string;
  author?: string;
  dependencies?: string[];
  setup: () => Promise<void>;
  teardown?: () => Promise<void>;
}

export interface FieldPlugin extends Plugin {
  type: 'field';
  platform: string;
  constraints: {
    minMemory?: number;
    minDisk?: number;
    requiredTools?: string[];
    supportedModels?: string[];
  };
  optimize: (config: any) => any;
}

export interface EquipmentPlugin extends Plugin {
  type: 'equipment';
  category: 'database' | 'rag-pipeline' | 'model-runtime' | 'tool';
  install: () => Promise<void>;
  configure: (options: any) => Promise<void>;
  test: () => Promise<boolean>;
}

export interface PipePlugin extends Plugin {
  type: 'pipe';
  protocol: string;
  createServer: (options: any) => any;
  createClient: (options: any) => any;
}

class PluginRegistry {
  private plugins: Map<string, Plugin> = new Map();

  register(plugin: Plugin) {
    console.log(`🔌 Registering plugin: ${plugin.name} (${plugin.type})`);
    this.plugins.set(plugin.name, plugin);
  }

  get(name: string): Plugin | undefined {
    return this.plugins.get(name);
  }

  getByType(type: 'field' | 'equipment' | 'pipe'): Plugin[] {
    return Array.from(this.plugins.values()).filter(p => p.type === type);
  }

  async loadAll() {
    // Auto-load all plugins
    const fieldPlugins = await this.loadFields();
    const equipmentPlugins = await this.loadEquipment();
    const pipePlugins = await this.loadPipes();
    
    console.log(`✅ Loaded ${this.plugins.size} plugins`);
  }

  private async loadFields() {
    // Dynamically import all field plugins
    const fields = ['mac', 'linux', 'windows', 'raspberry-pi', 'jetson'];
    for (const field of fields) {
      try {
        const plugin = await import(`./fields/${field}`);
        if (plugin.default) {
          this.register(plugin.default);
        }
      } catch (e) {
        // Field not implemented yet
      }
    }
  }

  private async loadEquipment() {
    // Load equipment plugins
    // Implementation similar to fields
  }

  private async loadPipes() {
    // Load pipe plugins
    // Implementation similar to fields
  }
}

export const registry = new PluginRegistry();
EOF

# Create base plugin classes
echo "🏗️ Creating base plugin classes..."

cat > src/plugins/base.ts << 'EOF'
/**
 * Base classes for LlamaFarm plugins
 */

import { Plugin, FieldPlugin, EquipmentPlugin, PipePlugin } from './registry';

export abstract class BasePlugin implements Plugin {
  abstract name: string;
  abstract type: 'field' | 'equipment' | 'pipe';
  abstract version: string;
  abstract description: string;
  author?: string;
  dependencies?: string[];

  async setup(): Promise<void> {
    console.log(`🌱 Setting up ${this.name}...`);
  }

  async teardown(): Promise<void> {
    console.log(`🧹 Tearing down ${this.name}...`);
  }
}

export abstract class BaseFieldPlugin extends BasePlugin implements FieldPlugin {
  type: 'field' = 'field';
  abstract platform: string;
  abstract constraints: {
    minMemory?: number;
    minDisk?: number;
    requiredTools?: string[];
    supportedModels?: string[];
  };

  abstract optimize(config: any): any;
}

export abstract class BaseEquipmentPlugin extends BasePlugin implements EquipmentPlugin {
  type: 'equipment' = 'equipment';
  abstract category: 'database' | 'rag-pipeline' | 'model-runtime' | 'tool';
  
  abstract install(): Promise<void>;
  abstract configure(options: any): Promise<void>;
  abstract test(): Promise<boolean>;
}

export abstract class BasePipePlugin extends BasePlugin implements PipePlugin {
  type: 'pipe' = 'pipe';
  abstract protocol: string;
  
  abstract createServer(options: any): any;
  abstract createClient(options: any): any;
}
EOF

# Create Mac field plugin (example)
echo "🍎 Creating Mac field plugin..."

cat > src/plugins/fields/mac/index.ts << 'EOF'
import { BaseFieldPlugin } from '../../base';
import * as os from 'os';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class MacFieldPlugin extends BaseFieldPlugin {
  name = 'mac-field';
  version = '1.0.0';
  platform = 'darwin';
  description = 'macOS deployment field with Metal acceleration support';
  author = 'LlamaFarm Team';

  constraints = {
    minMemory: 8, // GB
    minDisk: 20, // GB
    requiredTools: ['xcode-select'],
    supportedModels: ['llama3-8b', 'mixtral-8x7b', 'phi-2', 'codellama-7b']
  };

  async setup(): Promise<void> {
    await super.setup();
    
    // Check system requirements
    const totalMem = os.totalmem() / (1024 ** 3); // Convert to GB
    if (totalMem < this.constraints.minMemory) {
      throw new Error(`Insufficient memory: ${totalMem.toFixed(1)}GB < ${this.constraints.minMemory}GB`);
    }

    // Check for Xcode tools
    try {
      await execAsync('xcode-select -p');
    } catch (e) {
      console.log('⚠️  Xcode Command Line Tools not found');
      console.log('   Install with: xcode-select --install');
    }

    // Check for Metal support
    const arch = os.arch();
    const isAppleSilicon = arch === 'arm64';
    
    console.log(`✅ macOS ${os.release()} detected`);
    console.log(`🖥️  Architecture: ${arch} ${isAppleSilicon ? '(Apple Silicon - Metal enabled)' : '(Intel)'}`);
    console.log(`💾 Memory: ${totalMem.toFixed(1)}GB`);
  }

  optimize(config: any): any {
    // macOS-specific optimizations
    const optimized = { ...config };
    
    // Enable Metal for Apple Silicon
    if (os.arch() === 'arm64') {
      optimized.acceleration = 'metal';
      optimized.computeUnits = 'all'; // CPU + GPU + Neural Engine
    }

    // Optimize thread count
    optimized.threads = os.cpus().length;
    
    // Memory settings
    optimized.memoryMap = true; // Use mmap for large models
    optimized.memoryLock = false; // Don't lock memory on macOS
    
    return optimized;
  }

  async installDependencies(): Promise<void> {
    console.log('📦 Installing macOS dependencies...');
    
    // Check for Homebrew
    try {
      await execAsync('which brew');
      console.log('✅ Homebrew found');
    } catch (e) {
      console.log('⚠️  Homebrew not found. Install from https://brew.sh');
    }
  }
}

export default new MacFieldPlugin();
EOF

# Create demo database equipment plugin
echo "🗄️ Creating demo database plugin..."

cat > src/plugins/equipment/databases/demo/index.ts << 'EOF'
import { BaseEquipmentPlugin } from '../../../base';
import * as fs from 'fs-extra';
import * as path from 'path';

export class DemoDatabasePlugin extends BaseEquipmentPlugin {
  name = 'demo-database';
  version = '1.0.0';
  category: 'database' = 'database';
  description = 'Simple in-memory vector database for demos';
  author = 'LlamaFarm Team';
  
  private data: Map<string, any> = new Map();
  private dbPath: string = '';

  async install(): Promise<void> {
    console.log('📦 Installing demo database...');
    // No real installation needed for demo
    console.log('✅ Demo database ready');
  }

  async configure(options: any): Promise<void> {
    console.log('⚙️  Configuring demo database...');
    
    this.dbPath = options.path || './demo_db';
    await fs.ensureDir(this.dbPath);
    
    // Create config file
    const config = {
      type: 'demo',
      version: this.version,
      settings: {
        maxVectors: options.maxVectors || 10000,
        dimensions: options.dimensions || 384,
        similarity: options.similarity || 'cosine'
      }
    };
    
    await fs.writeJSON(path.join(this.dbPath, 'config.json'), config, { spaces: 2 });
    console.log('✅ Database configured');
  }

  async test(): Promise<boolean> {
    console.log('🧪 Testing demo database...');
    
    // Simple test
    try {
      this.data.set('test', { vector: [0.1, 0.2, 0.3], metadata: { test: true } });
      const result = this.data.get('test');
      this.data.delete('test');
      
      console.log('✅ Database test passed');
      return true;
    } catch (e) {
      console.log('❌ Database test failed:', e);
      return false;
    }
  }

  // Demo database methods
  async addVector(id: string, vector: number[], metadata?: any): Promise<void> {
    this.data.set(id, { vector, metadata, timestamp: Date.now() });
  }

  async search(vector: number[], k: number = 5): Promise<any[]> {
    // Simple cosine similarity search
    const results: any[] = [];
    
    for (const [id, item] of this.data.entries()) {
      const similarity = this.cosineSimilarity(vector, item.vector);
      results.push({ id, similarity, metadata: item.metadata });
    }
    
    return results
      .sort((a, b) => b.similarity - a.similarity)
      .slice(0, k);
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}

export default new DemoDatabasePlugin();
EOF

# Create WebSocket pipe plugin
echo "🔌 Creating WebSocket pipe plugin..."

cat > src/plugins/pipes/websocket/index.ts << 'EOF'
import { BasePipePlugin } from '../../base';
import { WebSocketServer, WebSocket } from 'ws';
import * as http from 'http';

export class WebSocketPipePlugin extends BasePipePlugin {
  name = 'websocket-pipe';
  version = '1.0.0';
  protocol = 'ws';
  description = 'WebSocket communication pipe for real-time chat';
  author = 'LlamaFarm Team';
  dependencies = ['ws'];

  createServer(options: any): any {
    const { port = 8080, onMessage, onConnection } = options;
    
    const server = http.createServer();
    const wss = new WebSocketServer({ server });
    
    wss.on('connection', (ws: WebSocket) => {
      console.log('🔌 New WebSocket connection');
      
      if (onConnection) {
        onConnection(ws);
      }
      
      ws.on('message', (data) => {
        try {
          const message = JSON.parse(data.toString());
          if (onMessage) {
            onMessage(message, ws);
          }
        } catch (e) {
          console.error('Failed to parse message:', e);
        }
      });
      
      ws.on('close', () => {
        console.log('👋 WebSocket connection closed');
      });
      
      ws.on('error', (err) => {
        console.error('WebSocket error:', err);
      });
    });
    
    server.listen(port, () => {
      console.log(`🌐 WebSocket server listening on port ${port}`);
    });
    
    return { server, wss };
  }

  createClient(options: any): any {
    const { url, onMessage, onOpen, onClose, onError } = options;
    
    const ws = new WebSocket(url);
    
    ws.on('open', () => {
      console.log('🔌 Connected to WebSocket server');
      if (onOpen) onOpen();
    });
    
    ws.on('message', (data) => {
      try {
        const message = JSON.parse(data.toString());
        if (onMessage) onMessage(message);
      } catch (e) {
        console.error('Failed to parse message:', e);
      }
    });
    
    ws.on('close', () => {
      console.log('👋 Disconnected from server');
      if (onClose) onClose();
    });
    
    ws.on('error', (err) => {
      console.error('WebSocket error:', err);
      if (onError) onError(err);
    });
    
    return ws;
  }

  // Helper methods
  broadcast(wss: WebSocketServer, message: any): void {
    const data = JSON.stringify(message);
    wss.clients.forEach((client) => {
      if (client.readyState === WebSocket.OPEN) {
        client.send(data);
      }
    });
  }
}

export default new WebSocketPipePlugin();
EOF

# Create plugin templates
echo "📋 Creating plugin templates..."

# Field template
cat > templates/plugins/field/template.ts << 'EOF'
import { BaseFieldPlugin } from '../../base';
import * as os from 'os';

export class {{NAME}}FieldPlugin extends BaseFieldPlugin {
  name = '{{LOWERCASE_NAME}}-field';
  version = '1.0.0';
  platform = '{{PLATFORM}}'; // darwin, linux, win32
  description = '{{DESCRIPTION}}';
  author = '{{AUTHOR}}';

  constraints = {
    minMemory: 8, // GB
    minDisk: 20, // GB
    requiredTools: [], // e.g., ['docker', 'nvidia-smi']
    supportedModels: [] // e.g., ['llama3-8b', 'mixtral-8x7b']
  };

  async setup(): Promise<void> {
    await super.setup();
    
    // Check system requirements
    console.log(`🌾 Setting up ${this.name}...`);
    
    // Add your platform-specific setup here
  }

  optimize(config: any): any {
    // Platform-specific optimizations
    const optimized = { ...config };
    
    // Add your optimizations here
    // e.g., GPU settings, thread count, memory settings
    
    return optimized;
  }
}

export default new {{NAME}}FieldPlugin();
EOF

# Equipment template
cat > templates/plugins/equipment/template.ts << 'EOF'
import { BaseEquipmentPlugin } from '../../../base';

export class {{NAME}}Plugin extends BaseEquipmentPlugin {
  name = '{{LOWERCASE_NAME}}';
  version = '1.0.0';
  category: '{{CATEGORY}}' = '{{CATEGORY}}'; // database, rag-pipeline, model-runtime, tool
  description = '{{DESCRIPTION}}';
  author = '{{AUTHOR}}';
  dependencies = []; // npm packages needed

  async install(): Promise<void> {
    console.log(`📦 Installing ${this.name}...`);
    
    // Installation logic here
    // e.g., download binaries, install npm packages
  }

  async configure(options: any): Promise<void> {
    console.log(`⚙️  Configuring ${this.name}...`);
    
    // Configuration logic here
    // e.g., create config files, set up directories
  }

  async test(): Promise<boolean> {
    console.log(`🧪 Testing ${this.name}...`);
    
    try {
      // Test logic here
      // Return true if tests pass, false otherwise
      
      return true;
    } catch (e) {
      console.error(`Test failed:`, e);
      return false;
    }
  }
}

export default new {{NAME}}Plugin();
EOF

# Pipe template
cat > templates/plugins/pipe/template.ts << 'EOF'
import { BasePipePlugin } from '../../base';

export class {{NAME}}PipePlugin extends BasePipePlugin {
  name = '{{LOWERCASE_NAME}}-pipe';
  version = '1.0.0';
  protocol = '{{PROTOCOL}}'; // ws, http, webrtc, sse
  description = '{{DESCRIPTION}}';
  author = '{{AUTHOR}}';
  dependencies = []; // npm packages needed

  createServer(options: any): any {
    const { port = 8080, ...config } = options;
    
    console.log(`🌐 Creating ${this.protocol} server on port ${port}`);
    
    // Create and return your server instance
    // Handle connections, messages, etc.
  }

  createClient(options: any): any {
    const { url, ...config } = options;
    
    console.log(`🔌 Creating ${this.protocol} client for ${url}`);
    
    // Create and return your client instance
    // Handle connection, messages, etc.
  }
}

export default new {{NAME}}PipePlugin();
EOF

# Create contribution guide
echo "📚 Creating plugin contribution guide..."

cat > src/plugins/CONTRIBUTING.md << 'EOF'
# 🔌 LlamaFarm Plugin Contribution Guide

Thank you for contributing to LlamaFarm's plugin ecosystem! This guide will help you create plugins for fields, equipment, and pipes.

## Plugin Types

### 🌾 Fields (Platform Configurations)
Fields define platform-specific configurations, constraints, and optimizations.

**Examples**: mac, linux, windows, raspberry-pi, jetson

**Location**: `src/plugins/fields/[platform-name]/`

### 🛠️ Equipment (Tools and Services)
Equipment plugins integrate databases, RAG pipelines, model runtimes, and other tools.

**Categories**:
- `database`: Vector databases (ChromaDB, Qdrant, etc.)
- `rag-pipeline`: RAG tools (LlamaIndex, LangChain, etc.)
- `model-runtime`: Model runners (Ollama, llama.cpp, etc.)
- `tool`: Other tools (embedders, tokenizers, etc.)

**Location**: `src/plugins/equipment/[category]/[tool-name]/`

### 🔧 Pipes (Communication Channels)
Pipes handle different communication protocols for streaming data.

**Examples**: websocket, webrtc, sse, http-streaming

**Location**: `src/plugins/pipes/[protocol-name]/`

## Creating a Plugin

### 1. Choose Your Plugin Type

```bash
# For a new field (platform)
cp templates/plugins/field/template.ts src/plugins/fields/my-platform/index.ts

# For new equipment
cp templates/plugins/equipment/template.ts src/plugins/equipment/databases/my-db/index.ts

# For a new pipe
cp templates/plugins/pipe/template.ts src/plugins/pipes/my-protocol/index.ts
```

### 2. Implement Required Methods

#### Field Plugin
```typescript
export class MyFieldPlugin extends BaseFieldPlugin {
  // Required properties
  name = 'my-field';
  platform = 'linux'; // os.platform() value
  constraints = {
    minMemory: 8,
    minDisk: 20,
    requiredTools: ['docker'],
    supportedModels: ['llama3-8b']
  };

  // Required methods
  async setup() { /* Check system requirements */ }
  optimize(config) { /* Return optimized config */ }
}
```

#### Equipment Plugin
```typescript
export class MyEquipmentPlugin extends BaseEquipmentPlugin {
  // Required properties
  name = 'my-database';
  category = 'database';
  
  // Required methods
  async install() { /* Install the tool */ }
  async configure(options) { /* Configure with options */ }
  async test() { /* Test functionality */ }
}
```

#### Pipe Plugin
```typescript
export class MyPipePlugin extends BasePipePlugin {
  // Required properties
  name = 'my-pipe';
  protocol = 'custom';
  
  // Required methods
  createServer(options) { /* Return server instance */ }
  createClient(options) { /* Return client instance */ }
}
```

### 3. Export Default Instance

Always export a default instance:
```typescript
export default new MyPlugin();
```

### 4. Add Tests

Create `src/plugins/[type]/[name]/test.ts`:
```typescript
import plugin from './index';

describe('MyPlugin', () => {
  it('should set up correctly', async () => {
    await expect(plugin.setup()).resolves.not.toThrow();
  });
});
```

### 5. Add Documentation

Create `src/plugins/[type]/[name]/README.md`:
```markdown
# My Plugin Name

Description of what this plugin does.

## Requirements
- List any system requirements
- External dependencies

## Configuration
\```yaml
my-plugin:
  option1: value1
  option2: value2
\```

## Usage
Example of how to use this plugin
```

## Best Practices

1. **Error Handling**: Always handle errors gracefully
2. **Logging**: Use consistent logging with emojis
3. **Dependencies**: Minimize external dependencies
4. **Testing**: Include comprehensive tests
5. **Documentation**: Document all options and usage

## Testing Your Plugin

```bash
# Run all plugin tests
npm test src/plugins

# Test specific plugin
npm test src/plugins/fields/mac

# Manual test
node -e "require('./src/plugins/fields/mac').default.setup()"
```

## Submitting Your Plugin

1. Fork the repository
2. Create feature branch: `git checkout -b plugin/my-awesome-plugin`
3. Implement your plugin following the templates
4. Add tests and documentation
5. Submit PR with description of your plugin

## Plugin Ideas

### Fields Needed
- [ ] Android (Termux)
- [ ] FreeBSD
- [ ] Docker containers
- [ ] Kubernetes pods

### Equipment Needed
- [ ] Databases: Milvus, Faiss, PostgreSQL+pgvector
- [ ] RAG: Haystack, txtai
- [ ] Runtimes: TensorFlow.js, ONNX
- [ ] Tools: Sentence transformers, tokenizers

### Pipes Needed
- [ ] gRPC streaming
- [ ] GraphQL subscriptions
- [ ] MQTT
- [ ] WebTransport

## Questions?

Join our Discord or open an issue! We're here to help make your plugin successful.

Happy farming! 🦙🌾
EOF

# Create plugin loader
echo "🔧 Creating plugin loader..."

cat > src/plugins/loader.ts << 'EOF'
/**
 * Plugin Loader
 * Dynamically loads all plugins at runtime
 */

import { registry } from './registry';
import chalk from 'chalk';

export async function loadPlugins() {
  console.log(chalk.green('\n🔌 Loading LlamaFarm plugins...'));
  
  try {
    await registry.loadAll();
    
    const fields = registry.getByType('field');
    const equipment = registry.getByType('equipment');
    const pipes = registry.getByType('pipe');
    
    console.log(chalk.gray(`\n📊 Loaded plugins:`));
    console.log(chalk.gray(`   🌾 Fields: ${fields.length}`));
    console.log(chalk.gray(`   🛠️  Equipment: ${equipment.length}`));
    console.log(chalk.gray(`   🔧 Pipes: ${pipes.length}`));
    
    // List loaded plugins
    if (fields.length > 0) {
      console.log(chalk.gray(`\n   Fields: ${fields.map(f => f.name).join(', ')}`));
    }
    if (equipment.length > 0) {
      console.log(chalk.gray(`   Equipment: ${equipment.map(e => e.name).join(', ')}`));
    }
    if (pipes.length > 0) {
      console.log(chalk.gray(`   Pipes: ${pipes.map(p => p.name).join(', ')}`));
    }
    
  } catch (error) {
    console.error(chalk.red('❌ Failed to load plugins:'), error);
  }
}

// Auto-load plugins when imported
if (require.main !== module) {
  loadPlugins().catch(console.error);
}
EOF

# Update the plant command to use plugins
echo "🌱 Updating plant command to use plugins..."

cat > src/commands/plant-with-plugins.ts << 'EOF'
import chalk from 'chalk';
import { registry } from '../plugins/registry';
import { FieldPlugin, EquipmentPlugin, PipePlugin } from '../plugins/registry';

export async function plantWithPlugins(model: string, options: any) {
  console.log(chalk.green(`\n🌱 Planting ${model} with plugins...`));
  
  // Load plugins
  await registry.loadAll();
  
  // Get field plugin for current platform
  const platform = options.device || process.platform;
  const fieldPlugin = registry.getByType('field')
    .find(f => (f as FieldPlugin).platform === platform) as FieldPlugin;
    
  if (fieldPlugin) {
    console.log(chalk.cyan(`\n🌾 Using field: ${fieldPlugin.name}`));
    await fieldPlugin.setup();
    const optimizedConfig = fieldPlugin.optimize(options);
    console.log(chalk.gray(`   Optimized for ${platform}`));
  }
  
  // Get database plugin
  const dbName = options.database || 'demo';
  const dbPlugin = registry.get(`${dbName}-database`) as EquipmentPlugin;
  
  if (dbPlugin) {
    console.log(chalk.cyan(`\n🗄️  Using database: ${dbPlugin.name}`));
    await dbPlugin.install();
    await dbPlugin.configure({ path: `./${model}-db` });
    const testPassed = await dbPlugin.test();
    if (!testPassed) {
      console.warn(chalk.yellow('⚠️  Database test failed'));
    }
  }
  
  // Get pipe plugin
  const pipeName = options.pipe || 'websocket';
  const pipePlugin = registry.get(`${pipeName}-pipe`) as PipePlugin;
  
  if (pipePlugin) {
    console.log(chalk.cyan(`\n🔧 Using pipe: ${pipePlugin.name}`));
    // Pipe will be used in the generated server code
  }
  
  console.log(chalk.green('\n✅ Planting complete with plugins!'));
}
EOF

# Create example plugin usage
echo "📝 Creating example plugin usage..."

cat > examples/plugin-usage.ts << 'EOF'
/**
 * Example: Using LlamaFarm Plugins
 */

import { registry } from '../src/plugins/registry';

async function example() {
  // Load all plugins
  await registry.loadAll();
  
  // Use a field plugin
  const macField = registry.get('mac-field');
  if (macField) {
    await macField.setup();
  }
  
  // Use an equipment plugin
  const demoDb = registry.get('demo-database');
  if (demoDb && demoDb.type === 'equipment') {
    await demoDb.install();
    await demoDb.configure({ dimensions: 384 });
  }
  
  // Use a pipe plugin
  const wsPipe = registry.get('websocket-pipe');
  if (wsPipe && wsPipe.type === 'pipe') {
    const server = wsPipe.createServer({
      port: 8080,
      onMessage: (msg: any) => console.log('Received:', msg)
    });
  }
}

example().catch(console.error);
EOF

# Create plugin README
echo "📚 Creating plugin README..."

cat > src/plugins/README.md << 'EOF'
# 🔌 LlamaFarm Plugin System

The LlamaFarm plugin system makes it easy to extend the platform with new capabilities.

## Plugin Types

### 🌾 Fields
Platform-specific configurations and optimizations
- Mac (with Metal support)
- Linux (with CUDA support)
- Windows
- Raspberry Pi
- NVIDIA Jetson

### 🛠️ Equipment
Tools and services that can be integrated
- **Databases**: ChromaDB, Qdrant, Pinecone, Weaviate
- **RAG Pipelines**: LlamaIndex, LangChain, custom
- **Model Runtimes**: Ollama, llama.cpp, HuggingFace
- **Tools**: Embedders, tokenizers, preprocessors

### 🔧 Pipes
Communication channels for different protocols
- WebSocket (real-time bidirectional)
- WebRTC (peer-to-peer)
- Server-Sent Events (one-way streaming)
- HTTP Streaming

## Quick Start

### Using Existing Plugins

```typescript
import { registry } from './plugins/registry';

// Load all plugins
await registry.loadAll();

// Get specific plugin
const macField = registry.get('mac-field');
await macField.setup();
```

### Creating New Plugins

1. Copy the appropriate template from `templates/plugins/`
2. Implement required methods
3. Export default instance
4. Add tests and documentation

See [CONTRIBUTING.md](./CONTRIBUTING.md) for detailed instructions.

## Available Plugins

### Fields
- ✅ Mac (Apple Silicon + Intel)
- 🚧 Linux (coming soon)
- 🚧 Windows (coming soon)
- 🚧 Raspberry Pi (coming soon)

### Equipment
- ✅ Demo Database
- 🚧 ChromaDB (coming soon)
- 🚧 Ollama Runtime (coming soon)
- 🚧 LlamaIndex RAG (coming soon)

### Pipes
- ✅ WebSocket
- 🚧 WebRTC (coming soon)
- 🚧 SSE (coming soon)

## Architecture

```
src/plugins/
├── registry.ts      # Central plugin registry
├── base.ts         # Base classes
├── loader.ts       # Dynamic loader
├── fields/         # Platform plugins
├── equipment/      # Tool plugins
└── pipes/          # Communication plugins
```

Each plugin is self-contained and can be developed independently.

Happy farming! 🦙🌾
EOF

# Update package.json scripts
echo "📦 Updating package.json scripts..."

# Add plugin-related scripts
cat > update-package-scripts.js << 'EOF'
const fs = require('fs');
const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));

// Add new scripts
pkg.scripts['plugins:list'] = 'ts-node src/plugins/loader.ts';
pkg.scripts['plugins:test'] = 'jest src/plugins --coverage';
pkg.scripts['plugin:create'] = 'ts-node scripts/create-plugin.ts';

fs.writeFileSync('package.json', JSON.stringify(pkg, null, 2));
console.log('✅ Updated package.json with plugin scripts');
EOF

node update-package-scripts.js
rm update-package-scripts.js

# Create plugin creation script
echo "🛠️ Creating plugin creation helper..."

cat > scripts/create-plugin.ts << 'EOF'
#!/usr/bin/env ts-node

import * as inquirer from 'inquirer';
import * as fs from 'fs-extra';
import * as path from 'path';
import chalk from 'chalk';

async function createPlugin() {
  console.log(chalk.green('\n🔌 LlamaFarm Plugin Creator\n'));
  
  const answers = await inquirer.prompt([
    {
      type: 'list',
      name: 'type',
      message: 'What type of plugin?',
      choices: [
        { name: '🌾 Field (Platform configuration)', value: 'field' },
        { name: '🛠️  Equipment (Database/Tool/Runtime)', value: 'equipment' },
        { name: '🔧 Pipe (Communication channel)', value: 'pipe' }
      ]
    },
    {
      type: 'input',
      name: 'name',
      message: 'Plugin name (lowercase, hyphens):',
      validate: (input) => /^[a-z0-9-]+$/.test(input) || 'Use lowercase and hyphens only'
    },
    {
      type: 'input',
      name: 'description',
      message: 'Brief description:'
    },
    {
      type: 'input',
      name: 'author',
      message: 'Author name:',
      default: 'Community Contributor'
    }
  ]);
  
  // Additional questions based on type
  if (answers.type === 'field') {
    const fieldAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'platform',
        message: 'Target platform:',
        choices: ['darwin', 'linux', 'win32', 'freebsd', 'android']
      }
    ]);
    Object.assign(answers, fieldAnswers);
  }
  
  if (answers.type === 'equipment') {
    const equipAnswers = await inquirer.prompt([
      {
        type: 'list',
        name: 'category',
        message: 'Equipment category:',
        choices: ['database', 'rag-pipeline', 'model-runtime', 'tool']
      }
    ]);
    Object.assign(answers, equipAnswers);
  }
  
  if (answers.type === 'pipe') {
    const pipeAnswers = await inquirer.prompt([
      {
        type: 'input',
        name: 'protocol',
        message: 'Protocol name:',
        default: answers.name.replace('-pipe', '')
      }
    ]);
    Object.assign(answers, pipeAnswers);
  }
  
  // Create plugin from template
  await createFromTemplate(answers);
  
  console.log(chalk.green(`\n✅ Plugin created successfully!`));
  console.log(chalk.gray(`\n📁 Location: src/plugins/${answers.type}s/${answers.name}/`));
  console.log(chalk.gray(`\n📝 Next steps:`));
  console.log(chalk.gray(`   1. Implement the required methods`));
  console.log(chalk.gray(`   2. Add tests in test.ts`));
  console.log(chalk.gray(`   3. Add documentation in README.md`));
  console.log(chalk.gray(`   4. Test with: npm test src/plugins/${answers.type}s/${answers.name}`));
}

async function createFromTemplate(options: any) {
  const { type, name, description, author } = options;
  
  // Determine paths
  let templatePath = path.join('templates/plugins', type, 'template.ts');
  let targetDir = path.join('src/plugins', `${type}s`);
  
  if (type === 'equipment') {
    targetDir = path.join(targetDir, options.category);
  }
  
  targetDir = path.join(targetDir, name);
  
  // Create directory
  await fs.ensureDir(targetDir);
  
  // Read and process template
  let template = await fs.readFile(templatePath, 'utf8');
  
  // Replace placeholders
  template = template
    .replace(/\{\{NAME\}\}/g, name.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(''))
    .replace(/\{\{LOWERCASE_NAME\}\}/g, name)
    .replace(/\{\{DESCRIPTION\}\}/g, description)
    .replace(/\{\{AUTHOR\}\}/g, author)
    .replace(/\{\{PLATFORM\}\}/g, options.platform || '')
    .replace(/\{\{CATEGORY\}\}/g, options.category || '')
    .replace(/\{\{PROTOCOL\}\}/g, options.protocol || '');
  
  // Write files
  await fs.writeFile(path.join(targetDir, 'index.ts'), template);
  await fs.writeFile(path.join(targetDir, 'README.md'), `# ${name}\n\n${description}\n`);
  await fs.writeFile(path.join(targetDir, 'test.ts'), `import plugin from './index';\n\n// Add tests here\n`);
}

createPlugin().catch(console.error);
EOF

chmod +x scripts/create-plugin.ts

# Final summary
echo ""
echo "✅ LlamaFarm Plugin Architecture created successfully!"
echo ""
echo "📁 Plugin Structure:"
echo "   src/plugins/"
echo "   ├── 📝 registry.ts (plugin registry)"
echo "   ├── 🏗️ base.ts (base classes)"
echo "   ├── 🔧 loader.ts (dynamic loader)"
echo "   ├── 📚 README.md & CONTRIBUTING.md"
echo "   ├── fields/ (platform configs)"
echo "   │   └── mac/ (example implementation)"
echo "   ├── equipment/ (tools & services)"
echo "   │   └── databases/demo/ (example)"
echo "   └── pipes/ (communication)"
echo "       └── websocket/ (example)"
echo ""
echo "🚀 Quick Start:"
echo "   1. List plugins: npm run plugins:list"
echo "   2. Create new plugin: npm run plugin:create"
echo "   3. Test plugins: npm run plugins:test"
echo ""
echo "📋 Examples Created:"
echo "   - Mac field plugin (Metal acceleration)"
echo "   - Demo database plugin (in-memory vectors)"
echo "   - WebSocket pipe plugin (real-time chat)"
echo ""
echo "🌾 Contributors can now easily add:"
echo "   - New platforms (fields)"
echo "   - Databases & tools (equipment)"
echo "   - Communication protocols (pipes)"
echo ""
echo "Happy farming with plugins! 🦙🔌"
EOF