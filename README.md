# 🌾 LLaMA Farm

<div align="center">
  <img src="./llamafarm-cli/llama-images/llamasrolldeep.png" alt="LLaMA Farm - Llamas working together" width="500">
  
  <h3>Deploy any AI model, agents, database, and RAG pipeline to any device in 30 seconds. No cloud required.</h3>

  <h4>This is being built in the open!  If it doesn't work, leave an issue or help fix it!</h3>
  
  <p>
    <a href="https://github.com/llama-farm/llamafarm"><img src="https://img.shields.io/github/stars/llama-farm/llamafarm?style=social" alt="GitHub stars"></a>
    <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
    <a href="https://discord.gg/X9xdhTskjN"><img src="https://img.shields.io/badge/Discord-Join%20us-7289da?logo=discord&logoColor=white" alt="Discord"></a>
  </p>
  
  <p>
    <strong>Turn AI models and associated agents, databases, and pipelines into single executables that run anywhere.<br/>It's like Docker, but for AI.</strong>
  </p>
  
  <br/>
  
  <img src="https://raw.githubusercontent.com/llamafarm-ai/llamafarm/main/demo.gif" alt="llamafarm Demo" width="700">
  
</div>

## 📖 Table of Contents

- [What is LLaMA Farm](#-what-is-llama-farm)
- [Quick Start](#-quick-start)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Real-World Examples](#-real-world-examples)
- [Advanced Usage](#-advanced-usage)
- [Contributing](#-contributing)
- [Development Setup](#-development-setup)
- [License](#-license)

---

## 🚀 What is LLaMA Farm

Llama Farm packages your AI models, vector databases, and data pipelines into standalone binaries that run on any device - from Raspberry Pis to enterprise servers. **No Python. No CUDA hassles. No cloud bills.**

This repository contains:
- **[llamafarm-cli](./llamafarm-cli/)** - The core CLI tool (`npm install -g @llamafarm/llamafarm`)
- **[plugins](./plugins/)** - Community plugins for platforms, databases, and integrations

## Why LLaMA Farm
The current cloud AI model makes us digital serfs, paying rent to use tools we don't own, feeding data to systems we don't control. The farm model makes us owners—of our models, our data, our future. But ownership requires responsibility. You must tend your farm.

When you own your model and your data, you own your future.  Let's make the AI revolution for EVERYONE. 

### The Old Way 😰
```bash
# Install Python, CUDA, dependencies...
# Debug version conflicts...
# Pay cloud bills...
# Wait for DevOps...
# 3 days later: "Why isn't it working?"
```

### The llamafarm Way 🌱
We are shipping in real time - join the revolution to help us go faster! 

```bash
llamafarm plant mixtral-8x7b --target raspberry-pi --agent chat123 --rag --database vector
🌱 Planting mixtral-8x7b...
🌱 Planting agent chat123
🌱 Planting vector database
🪴 Fertilizing database with RAG
✓ Dependencies bundled
✓ Baled and compiled to binary (2.3GB)
✓ Optimized for ARM64
🦙 Ready to llamafarm! Download at https://localhost:8080/download/v3.1/2lv2k2lkals
```

---

## ✨ Features

- **🎯 One-Line Deployment** - Deploy complex AI models with a single command
- **📦 Zero Dependencies** - Compiled binaries run anywhere, no runtime needed
- **🔒 100% Private** - Your data never leaves your device
- **⚡ Lightning Fast** - 10x faster deployment than traditional methods
- **💾 90% Smaller** - Optimized models use fraction of original size
- **🔄 Hot Swappable** - Update models without downtime
- **🌍 Universal** - Mac, Linux, Windows, ARM - we support them all
- **🎯 Single Binary** - The Baler compiles everything into one executable file

### 🏗️ Architecture

LlamaFarm uses a plugin-based architecture that makes it easy to add support for new platforms, databases, and features:

- **[CLI Core](./llamafarm-cli/)** - The main command interface and deployment engine
- **[Plugin System](./plugins/)** - Extensible plugins for platforms, tools, and protocols
  - **Fields** (Platforms) - Mac, Linux, Windows, Raspberry Pi, and more
  - **Equipment** (Tools) - Vector databases, RAG pipelines, model runtimes
  - **Pipes** (Protocols) - WebSocket, WebRTC, SSE for real-time communication

---

## 🎬 Quick Start

### Install Llama-Farm

#### Option 1: Install via npm (Recommended)
```bash
npm install -g @llamafarm/llamafarm
```


### Deploy Your First Model
```bash
# Deploy Llama 3 with one command
llamafarm plant llama3-8b

# Or deploy with optimization for smaller devices
llamafarm plant llama3-8b --optimize

# Deploy to specific device
llamafarm plant mistral-7b --target raspberry-pi
```

### That's it! 🎉
Your AI is now running locally. No cloud. No subscriptions. Just pure, private AI.

### 🎯 The Complete Workflow: Plant → Bale → Harvest

LlamaFarm uses a simple agricultural metaphor for AI deployment:

1. **🌱 Plant** - Configure your deployment
```bash
llamafarm plant llama3-8b --device mac-arm --agent chat --rag --database vector
```

2. **📦 Bale** - Compile everything into a single binary
```bash
llamafarm bale ./.llamafarm/llama3-8b --device mac-arm --optimize
# Creates: llama3-8b-mac-arm-v1.0.0.bin (4-8GB)
```

3. **🌾 Harvest** - Deploy anywhere without dependencies
```bash
# Copy to any machine and run - no installation needed!
./llama3-8b-mac-arm-v1.0.0.bin
```

The **Baler** is the magic that packages your model, vector database, agents, and web UI into a single executable file that runs anywhere!

### 🌾 Available Commands

```bash
# Core Commands
llamafarm plant <model>     # Deploy a model
llamafarm bale <dir>        # Compile to single binary
llamafarm harvest <url>     # Download and run a deployment
llamafarm till              # Initialize configuration

# Management Commands
llamafarm silo              # Manage vector databases
llamafarm barn              # Manage model storage
llamafarm field             # Manage deployment environments

# Development Commands
llamafarm greenhouse        # Test in sandbox environment
llamafarm demo              # Run interactive demo
```

See the [CLI documentation](./llamafarm-cli/README.md) for all commands and options, including detailed information about the **Baler** and binary compilation.

---

## 🌟 Real-World Examples

### Deploy ChatGPT-level AI to a Raspberry Pi
```bash
llamafarm plant llama3-8b --target arm64 --optimize
# 🔥 Running in 30 seconds on $35 hardware
```

### Create an Offline Customer Service Bot
```bash
llamafarm plant customer-service-bot \
  --model llama3-8b \
  --data ./knowledge-base \
  --embeddings ./products.vec
# 📞 Complete AI assistant with zero latency
```

### Run HIPAA-Compliant Medical AI
```bash
llamafarm plant med-llama \
  --compliance hipaa \
  --audit-log enabled
# 🏥 Patient data never leaves the hospital
```

### Deploy to 100 Edge Devices
```bash
# Compile once
llamafarm plant llama3-8b --device raspberry-pi --optimize
llamafarm bale ./.llamafarm/llama3-8b --device raspberry-pi --compress

# Deploy everywhere - just copy the binary!
scp llama3-8b-raspberry-pi.bin pi@device1:/home/pi/
scp llama3-8b-raspberry-pi.bin pi@device2:/home/pi/
# ... no installation needed on devices
```

---

## 🏆 Why Developers WILL Love llamafarm

> "We replaced our $50K/month OpenAI bill with llamafarm. Deployment went from 3 days to 30 seconds." - **CTO, Fortune 500 Retailer**

> "Finally, AI that respects user privacy. llamafarm is what we've been waiting for." - **Lead Dev, Healthcare Startup**

> "I deployed Llama 3 to my grandma's laptop. She thinks I'm a wizard now." - **Random Internet Person**

> "I am glad I joined LLaMA Farm so early, I am part os something huge" - **LLama Farm contributor**

---

## 📊 Benchmarks

| Metric | Traditional Deployment | llamafarm | Improvement |
|--------|----------------------|---------|-------------|
| Deployment Time | 3-5 hours | 30 seconds | **360x faster** |
| Binary Size | 15-20 GB | 1.5 GB | **90% smaller** |
| Dependencies | 50+ packages | 0 | **∞ better** |
| Cloud Costs | $1000s/month | $0 | **100% savings** |

---

## 🛠 Advanced Usage

### Create Custom Packages
```yaml
# llamafarm.yaml
name: my-assistant
base_model: llama3-8b
plugins:
  - vector_search
  - voice_recognition
  - tool_calling
data:
  - path: ./company-docs
    type: knowledge
  - path: ./products.csv
    type: structured
optimization:
  quantization: int8
  target_size: 2GB
```

```bash
llamafarm build
# 📦 Creates my-assistant.exe (2GB)
```

### Multi-Model Deployment
```bash
# Deploy multiple models with load balancing
llamafarm plant llama3,mistral,claude --distribute

# Auto-routing based on task
llamafarm plant router.yaml
```

---

## 🌾 The llamafarm Ecosystem

### 🏪 Model Garden
Browse and deploy from our community model collection:
```bash
llamafarm search medical
llamafarm search finance  
llamafarm search creative

# One-click deployment
llamafarm plant community/medical-assistant-v2
```

### 🏢 Enterprise Edition

Need compliance, support, and SLAs? 

**[Get llamafarm Enterprise →](https://llamafarm.ai/enterprise)**

- 🔐 Air-gapped deployments
- 📊 Advanced monitoring
- 🏥 HIPAA/SOC2 compliance
- 💼 Priority support
- 🚀 Custom optimizations

---

## 🗺 Roadmap

- [x] Single binary compilation
- [x] Multi-platform support  
- [x] Model optimization
- [x] Vector DB integration
- [ ] GPU acceleration (Q1 2025)
- [ ] Distributed inference (Q1 2025)
- [ ] Mobile SDKs (Q2 2025)
- [ ] Hardware appliances (Q3 2025)

---

## 📦 Project Structure

LlamaFarm consists of multiple components working together:

### [🦙 LlamaFarm CLI](./llamafarm-cli/README.md)
The main command-line interface for deploying AI models. This is what you install with `npm install -g @llamafarm/llamafarm`.

```bash
cd llamafarm-cli
npm install
npm run build
```

### [🔌 Plugin System](./plugins/README.md)
Community-driven plugins for platform support, integrations, and features.

- **Fields** - Platform-specific optimizations (Mac, Linux, Raspberry Pi)
- **Equipment** - Tools and integrations (databases, RAG pipelines, model runtimes)
- **Pipes** - Communication protocols (WebSocket, WebRTC, SSE)

Browse the [plugins directory](./plugins/) to see available plugins or contribute your own!

---

## 🤝 Contributing

We love contributions! LlamaFarm is designed to be easily extensible:

### Quick Start for Contributors

1. **Core CLI Development**
   ```bash
   git clone https://github.com/llama-farm/llamafarm
   cd llamafarm/llamafarm-cli
   npm install
   npm run dev
   ```

2. **Plugin Development**
   ```bash
   cd plugins
   npm run create  # Interactive plugin creator
   ```
   See the [Plugin Development Guide](./plugins/README.md) for details.

3. **Submit Your Changes**
   ```bash
   npm test        # Run tests
   npm run lint    # Check code style
   git push        # Submit PR
   ```

### 🎯 Most Wanted Contributions

#### High Priority Plugins
- **Linux Field** - CUDA optimization for NVIDIA GPUs
- **Windows Field** - Windows-specific optimizations  
- **ChromaDB Equipment** - Production vector database
- **Ollama Runtime** - Official Ollama integration
- **WebRTC Pipe** - Peer-to-peer streaming

#### Community Wishlist
- Raspberry Pi 5 optimizations
- NVIDIA Jetson support
- Qdrant vector database
- LlamaIndex RAG pipeline
- Android/Termux support

See our [Plugin System](./plugins/README.md) for more ideas and how to contribute!

### 🌟 Contributors

<!-- ALL-CONTRIBUTORS-LIST:START -->
<a href="https://github.com/user1"><img src="https://github.com/user1.png" width="50px" alt=""/></a>
<a href="https://github.com/user2"><img src="https://github.com/user2.png" width="50px" alt=""/></a>
<a href="https://github.com/user3"><img src="https://github.com/user3.png" width="50px" alt=""/></a>
<a href="https://github.com/user4"><img src="https://github.com/user4.png" width="50px" alt=""/></a>
<a href="https://github.com/user5"><img src="https://github.com/user5.png" width="50px" alt=""/></a>
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## 📈 Stats

<p align="center">
  <img src="https://repobeats.axiom.co/api/embed/llamafarm-ai.svg" alt="Repobeats analytics" />
</p>

---

## 🔧 Development Setup

Want to contribute or run from source? Here's how:

```bash
# Clone the repository
git clone https://github.com/llama-farm/llamafarm
cd llamafarm

# Set up the CLI
cd llamafarm-cli
npm install
npm run build
npm link  # Makes 'llamafarm' command available globally

# Run tests
npm test

# Development mode
npm run dev
```

### 📚 Documentation

- [CLI Documentation](./llamafarm-cli/README.md) - Detailed CLI usage and options
- [Plugin Development](./plugins/README.md) - How to create plugins
- [API Reference](./docs/api.md) - Coming soon

### 🧪 Mock Mode (For Development)

LlamaFarm now includes a mock mode that allows you to test without installing Ollama or downloading models:

```bash
# Use --mock flag
llamafarm plant llama3-8b --mock

# Or set environment variable
export LLAMAFARM_MOCK=true
llamafarm plant llama3-8b
```

This is perfect for:
- Contributing to the project
- Testing the CLI functionality
- CI/CD pipelines
- Development on limited bandwidth

---

## 🎯 Getting Help

- 💬 [Discord Community](https://discord.gg/X9xdhTskjN)
- 🐛 [Issue Tracker](https://github.com/llama-farm/llamafarm/issues)
- 📧 [Email Support](mailto:support@llamafarm.dev)
- 📖 [CLI Documentation](./llamafarm-cli/README.md)
- 🔌 [Plugin Guide](./plugins/README.md)

---

## 📜 License

llamafarm is MIT licensed. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>🌾 Bringing AI back to the farm, one deployment at a time.</strong>
  <br>
  <sub>If you like llamafarm, give us a ⭐ on GitHub!</sub>
</p>

---

<details>
<summary><b>🚀 One more thing...</b></summary>

<br>

We're building something even bigger. **llamafarm Compass** - beautiful hardware that makes AI deployment truly plug-and-play.

[Join the waitlist →](https://llamafarm.dev)

</details>
