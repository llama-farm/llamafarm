# 🔌 LlamaFarm Plugins

Welcome to the LlamaFarm plugin ecosystem! This is where the community can easily contribute platform support, integrations, and features.

## 🌟 Quick Start for Contributors

1. **Browse existing plugins** to see examples
2. **Copy a template** from `templates/`
3. **Implement your plugin** following the patterns
4. **Submit a PR** with your new plugin!

## 📁 Plugin Structure

```
plugins/
├── fields/                 # 🌾 Platform configurations
│   ├── mac/               # macOS with Metal support
│   ├── linux/             # Linux with CUDA support
│   ├── windows/           # Windows support
│   └── raspberry-pi/      # Raspberry Pi optimizations
│
├── equipment/             # 🛠️ Tools and integrations
│   ├── databases/         # Vector databases
│   │   ├── chroma/       # ChromaDB integration
│   │   ├── qdrant/       # Qdrant integration
│   │   └── demo/         # Example database
│   ├── rag-pipelines/     # RAG tools
│   │   ├── llamaindex/   # LlamaIndex integration
│   │   └── langchain/    # LangChain integration
│   └── model-runtimes/    # Model runners
│       ├── ollama/       # Ollama integration
│       └── llamacpp/     # llama.cpp integration
│
└── pipes/                 # 🔧 Communication channels
    ├── websocket/        # WebSocket real-time
    ├── webrtc/          # WebRTC peer-to-peer
    └── sse/             # Server-sent events
```

## 🚀 Creating a Plugin

### Option 1: Use the Creator Script
```bash
cd plugins
npm run create
```

### Option 2: Copy a Template
```bash
# For a new platform
cp templates/field/template.ts fields/my-platform/index.ts

# For a new database
cp templates/equipment/database-template.ts equipment/databases/my-db/index.ts

# For a new communication pipe
cp templates/pipe/template.ts pipes/my-protocol/index.ts
```

## 📋 Plugin Types Explained

### 🌾 Fields (Platform Configurations)
Fields define how LlamaFarm optimizes for different platforms.

**Examples**: Mac (Metal), Linux (CUDA), Raspberry Pi (ARM)

**Required Methods**:
- `setup()` - Check system requirements
- `optimize()` - Return platform-specific configuration

### 🛠️ Equipment (Tools & Integrations)
Equipment plugins integrate external tools and services.

**Categories**:
- `databases` - Vector databases for RAG
- `rag-pipelines` - RAG processing tools
- `model-runtimes` - Model execution engines
- `tools` - Other utilities

**Required Methods**:
- `install()` - Install the tool
- `configure()` - Set up with options
- `test()` - Verify it works

### 🔧 Pipes (Communication Protocols)
Pipes handle different ways of streaming data between server and client.

**Examples**: WebSocket, WebRTC, Server-Sent Events

**Required Methods**:
- `createServer()` - Create server instance
- `createClient()` - Create client instance

## 🎯 Most Wanted Plugins

### High Priority
- [ ] **Linux Field** - CUDA optimization for NVIDIA GPUs
- [ ] **Windows Field** - Windows-specific optimizations
- [ ] **ChromaDB Equipment** - Production vector database
- [ ] **Ollama Runtime** - Official Ollama integration
- [ ] **WebRTC Pipe** - Peer-to-peer streaming

### Community Wishlist
- [ ] Raspberry Pi 5 optimizations
- [ ] NVIDIA Jetson support
- [ ] Qdrant vector database
- [ ] LlamaIndex RAG pipeline
- [ ] gRPC communication pipe
- [ ] Android/Termux support

## 💡 Plugin Ideas

- **Exotic Platforms**: FreeBSD, OpenBSD, Haiku
- **Cloud Providers**: AWS Lambda, Google Cloud Run
- **Databases**: Milvus, Weaviate, Pinecone
- **Communication**: MQTT, GraphQL subscriptions
- **Tools**: Custom tokenizers, embedding models

## 🤝 Contributing

1. Fork the repository
2. Create your plugin in the appropriate directory
3. Add tests and documentation
4. Submit a PR with:
   - Clear description of what your plugin does
   - Any special requirements
   - Example usage

## 📜 Plugin Guidelines

- **Self-contained**: Each plugin should be independent
- **Well-documented**: Include README.md with examples
- **Tested**: Add test.ts with basic tests
- **Semantic naming**: Use descriptive, lowercase names
- **Error handling**: Fail gracefully with helpful messages

## 🌟 Featured Plugins

### Mac Field Plugin
Optimizes for Apple Silicon with Metal acceleration. Great example of platform detection and hardware optimization.

### Demo Database
Simple in-memory vector database. Perfect template for building real database integrations.

### WebSocket Pipe
Real-time bidirectional communication. Shows how to handle streaming data.

## 📞 Get Help

- Open an issue with the `plugin` label
- Join our Discord (coming soon)
- Check existing plugins for examples

Happy plugin development! 🦙🔌
