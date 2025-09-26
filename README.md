# 🦙 LlamaFarm - Build Powerful AI Locally, Deploy Anywhere

<div align="center">
  <img src="docs/images/rocket-llama.png" alt="Llama Building a Rocket" width="400">

  **The Complete AI Development Framework - From Local Prototypes to Production Systems**

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Go 1.19+](https://img.shields.io/badge/go-1.19+-00ADD8.svg)](https://golang.org/dl/)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
  [![Discord](https://img.shields.io/discord/1234567890?color=7289da&logo=discord&logoColor=white)](https://discord.gg/8kH9AmQpSa)


  [🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🏗️ Architecture](#-architecture) • [🤝 Contributing](#-contributing)

</div>

---

## 🚀 What is LlamaFarm?

> **🚧 Building in the Open:** We're actively developing LlamaFarm and not everything is working yet. Join us as we build the future of local-first AI development! Check our [roadmap](#-roadmap) to see what's coming and how you can contribute.

### Why LlamaFarm?

The AI revolution should be accessible to everyone, not just ML experts and big tech companies. We believe you shouldn't need a PhD to build powerful AI applications - just a CLI, your config files, and your data. Too many teams are stuck between expensive cloud APIs that lock you in, or complex open-source tools that require months of ML expertise to productionize. LlamaFarm changes this: full control and production-ready AI with simple commands and YAML configs. No machine learning degree required - if you can write config files and run CLI commands, you can build sophisticated AI systems. Build locally with your data, maintain complete control over costs, and deploy anywhere from your laptop to the cloud - all with the same straightforward interface.

LlamaFarm is a comprehensive, modular framework for building AI Projects that run locally, collaborate, and deploy anywhere. We provide battle-tested components for RAG systems, vector databases, model management, prompt engineering, and soon fine-tuning - all designed to work seamlessly together or independently.

We're not local-only zealots - use cloud APIs where they make sense for your needs - llamafarm helps with that! But we believe the real value in the AI economy comes from building something uniquely yours, not just wrapping another UI around GPT-5. True innovation happens when you can train on your proprietary data, fine-tune for your specific use cases, and maintain full control over your AI stack. LlamaFarm gives you the tools to create differentiated AI products that your competitors can't simply copy by calling the same API.


LlamaFarm is a **comprehensive, modular AI framework** that gives you complete control over your AI stack. Unlike cloud-only solutions, we provide:

- **🏠 Local-First Development** - Build and test entirely on your machine
- **🔧 Production-Ready Components** - Battle-tested modules that scale from laptop to cluster
- **🎯 Strategy/config-Based Configuration** - Smart defaults with infinite customization
- **🚀 Deploy Anywhere** - Same code runs locally, on-premise, or in any cloud

### 🎭 Perfect For

- **Developers** who want to build AI applications without vendor lock-in
- **Teams** needing cost control and data privacy
- **Enterprises** requiring scalable, secure AI infrastructure
- **Researchers** experimenting with cutting-edge techniques

---

## 🏗️ Core Components

LlamaFarm is built as a modular system where each component can be used independently or orchestrated together for powerful AI applications.

### ⚙️ System Components

#### **🚀 Runtime**
The execution environment that orchestrates all components and manages the application lifecycle.
- **Process Management**: Handles component initialization and shutdown
- **API/Access Layer**: Send queries to /chat, data to /data, and get full results with ease.
- **Resource Allocation**: Manages memory, CPU, and GPU resources efficiently
- **Service Discovery**: Automatically finds and connects components
- **Health Monitoring**: Tracks component status and performance metrics
- **Error Recovery**: Automatic restart and fallback mechanisms

#### **📦 Deployer**
Zero-configuration deployment system that works from local development to production clusters.
- **Environment Detection**: Automatically adapts to local, Docker, or cloud environments
- **Configuration Management**: Handles environment variables and secrets securely
- **Scaling**: Horizontal and vertical scaling based on load
- **Load Balancing**: Distributes requests across multiple instances
- **Rolling Updates**: Zero-downtime deployments with automatic rollback

### 🧠 AI Components

#### **🔍 Data Pipeline (RAG)**
Complete document processing and retrieval system for building knowledge-augmented applications.
- **Document Ingestion**: Parse 15+ formats (PDF, Word, Excel, HTML, Markdown, etc.)
- **Smart Extraction**: Extract entities, keywords, statistics without LLMs
- **Vector Storage**: Integration with 8+ vector databases (Chroma, Pinecone, FAISS, etc.)
- **Hybrid Search**: Combine semantic, keyword, and metadata-based retrieval
- **Chunking Strategies**: Adaptive chunking based on document type and use case
- **Incremental Updates**: Efficiently update knowledge base without full reprocessing

#### **🤖 Models**
Unified interface for all LLM operations with enterprise-grade features.
- **Multi-Provider Support**: 25+ providers (OpenAI, Anthropic, Google, Ollama, etc.)
- **Automatic Failover**: Seamless fallback between providers when errors occur
- **Fine-Tuning Pipeline**: Train custom models on your data *(Coming Q2 2025)*
- **Cost Optimization**: Route queries to cheapest capable model
- **Load Balancing**: Distribute across multiple API keys and endpoints
- **Response Caching**: Intelligent caching to reduce API costs
- **Model Configuration**: Per-model temperature, token limits, and parameters

#### **📝 Prompts**
Enterprise prompt management system with version control and A/B testing.
- **Template Library**: 20+ pre-built templates for common use cases
- **Dynamic Variables**: Jinja2 templating with type validation (roadmap)
- **Strategy Selection**: Automatically choose best template based on context
- **Version Control**: Track prompt changes and performance over time (roadmap)
- **A/B Testing**: Compare prompt variations with built-in analytics (roadmap)
- **Chain-of-Thought**: Built-in support for reasoning chains
- **Multi-Agent**: Coordinate multiple specialized prompts (roadmap)

### 🔄 How Components Work Together

1. **User Request** → Runtime receives and validates the request
2. **Context Retrieval** → Data Pipeline searches relevant documents
3. **Prompt Selection** → Prompts system chooses optimal template
4. **Model Execution** → Models component handles LLM interaction with automatic failover
5. **Response Delivery** → Runtime returns formatted response to user

Each component is independent but designed to work seamlessly together through standardized interfaces.


---

## 🚀 Quick Start

### Installation

```bash
curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash
```

Or, to start components manually for development:

```bash
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

npm install -g nx
nx init --useDotNxInstallation --interactive=false

# Start both server and RAG worker (required for full functionality)
# Option 1: Use the convenience script
nx dev

# Option 2: Run in separate terminals
# Terminal 1:
nx start rag
# Terminal 2:
nx start server
```

### 🎯 Getting Started

> **💡 Important:** All our demos use the **REAL CLI** and **REAL configuration system** - what you see in the demos is exactly how you'll use LlamaFarm in production!

For the best experience getting started with LlamaFarm, we recommend exploring our component documentation and running the interactive demos:

#### 📚 RAG System (Document Processing & Retrieval)
- **[Read the RAG Documentation](rag/README.md)** - Complete guide to document ingestion, embedding, and retrieval
- **Run the Interactive Demos:**
  ```bash
  cd rag
  uv sync

  # Interactive setup wizard - guides you through configuration
  uv run python setup_demo.py

  # Or try specific demos with the real CLI:
  uv run python cli.py demo research_papers    # Academic paper analysis
  uv run python cli.py demo customer_support   # Support ticket processing
  uv run python cli.py demo code_analysis      # Source code understanding

  # Use your own documents:
  uv run python cli.py ingest ./your-docs/ --strategy research
  uv run python cli.py search "your query here" --top-k 5
  ```

#### 🤖 Models (LLM Management & Optimization)
- **[Read the Models Documentation](models/README.md)** - Multi-provider support, fallback strategies, and cost optimization
- **Run the Interactive Demos:**
  ```bash
  cd models
  uv sync

  # Try our showcase demos:
  uv run python demos/demo1_cloud_fallback.py  # Automatic provider fallback
  uv run python demos/demo2_multi_model.py     # Smart model routing
  uv run python demos/demo3_training.py        # Fine-tuning pipeline (preview)

  # Or use the real CLI directly:
  uv run python cli.py chat --strategy balanced "Explain quantum computing"
  uv run python cli.py chat --primary gpt-4 --fallback claude-3 "Write a haiku"

  # Test with your own config:
  uv run python cli.py setup your-strategy.yaml --verify
  uv run python cli.py demo your-strategy
  ```

#### 📝 Prompts (Coming Soon)
The prompts system is under active development. For now, explore the template system:
```bash
cd prompts
uv sync
uv run python -m prompts.cli template list  # View available templates
uv run python -m prompts.cli execute "Your task" --template research
```

### 🎮 Try It Live with the LlamaFarm CLI

#### Building the CLI Locally

If you're working with the latest changes that haven't been released yet, you can build and run the CLI locally:

```bash
# Prerequisites: Go 1.19+ must be installed

# Build the CLI binary
cd cli && go build -o lf . && cd ..

# Create a symlink for easy access (optional)
ln -sf cli/lf lf

# Now you can run the CLI as ./lf from the project root
./lf version  # Should show "LlamaFarm CLI vdev"

# To rebuild after making changes to the CLI code:
cd cli && go build -o lf . && cd ..
```

#### Complete RAG Pipeline Example
```bash
# Using the locally built CLI
./lf version  # Verify it's working

# Create and populate a dataset
./lf datasets add my-docs -s universal_processor -b main_database
./lf datasets ingest my-docs examples/rag_pipeline/sample_files/research_papers/*.txt
./lf datasets ingest my-docs examples/rag_pipeline/sample_files/fda/*.pdf
./lf datasets process my-docs

# Query your documents
./lf rag query --database main_database "What is transformer architecture?"
./lf rag query --database main_database --top-k 10 "What FDA submissions are discussed?"

# Chat with RAG augmentation (default behavior)
./lf run --database main_database "Explain neural scaling laws"
./lf run --database main_database --debug "What is BLA 761248?"

# Chat without RAG (LLM only)
./lf run --no-rag "What is machine learning?"
```

#### Component-Specific Examples

**RAG System:**
```bash
cd rag
uv run python cli.py demo research_papers
uv run python cli.py ingest ./your-docs/ --strategy research
uv run python cli.py search "your query" --top-k 5
```

**Models System:**
```bash
cd models
uv run python demos/demo1_cloud_fallback.py
uv run python cli.py chat --strategy balanced "Explain quantum computing"
```

**Prompts System:**
```bash
cd prompts
uv run python -m prompts.cli template list
uv run python -m prompts.cli execute "Your task" --template research
```

---

## 🎯 Configuration System

LlamaFarm uses a **strategy-based configuration** system that adapts to your use case:

### Strategy Configuration Example


```yaml
# config/strategies.yaml
strategies:
  research:
    rag:
      embedder: "sentence-transformers"
      chunk_size: 512
      overlap: 50
      retrievers:
        - type: "hybrid"
          weights: {dense: 0.7, sparse: 0.3}
    models:
      primary: "gpt-4"
      fallback: "claude-3-opus"
      temperature: 0.3
    prompts:
      template: "academic_research"
      style: "formal"
      citations: true

  customer_support:
    rag:
      embedder: "openai"
      chunk_size: 256
      retrievers:
        - type: "similarity"
          top_k: 3
    models:
      primary: "gpt-3.5-turbo"
      temperature: 0.7
    prompts:
      template: "conversational"
      style: "friendly"
      include_context: true

```

### Using Strategies

```bash
# Apply strategy across all components
export LLAMAFARM_STRATEGY=research

# Or specify per command
uv run python rag/cli.py ingest docs/ --strategy research
uv run python models/cli.py chat --strategy customer_support "Help me with my order"
```

---

## 📚 Documentation

### 📖 Comprehensive Guides

| Component | Description | Documentation |
|-----------|-------------|---------------|
| **RAG System** | Document processing, embedding, retrieval | [📚 RAG Guide](rag/README.md) |
| **Models** | LLM providers, management, optimization | [🤖 Models Guide](models/README.md) |
| **Prompts** | Templates, strategies, evaluation | [📝 Prompts Guide](prompts/README.md) |
| **CLI** | Command-line tools and utilities | [⚡ CLI Reference](cli/README.md) |
| **API** | REST API services | [🔌 API Docs](docs/api/README.md) |

### 🎓 Tutorials

- [Building Your First RAG Application](docs/tutorials/first-rag-app.md)
- [Setting Up Local Models with Ollama](docs/tutorials/local-models.md)
- [Advanced Prompt Engineering](docs/tutorials/prompt-engineering.md)
- [Deploying to Production](docs/tutorials/deployment.md)
- [Cost Optimization Strategies](docs/tutorials/cost-optimization.md)

### 🔧 Examples

Check out our [examples/](examples/) directory for complete working applications:
- 📚 Knowledge Base Assistant
- 💬 Customer Support Bot
- 📊 Document Analysis Pipeline
- 🔍 Semantic Search Engine
- 🤖 Multi-Agent System

---

## 🚢 Deployment Options

### Local Development
```bash
# Run with hot-reload
uv run python main.py --dev

# Or use Docker
docker-compose up -d
```

### Production Deployment

```yaml
# docker-compose.prod.yml
version: '3.8'
services:
  llamafarm:
    image: llamafarm/llamafarm:latest
    environment:
      - STRATEGY=production
      - WORKERS=4
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    ports:
      - "8000:8000"
    deploy:
      replicas: 3
      resources:
        limits:
          memory: 4G
```

### Cloud Deployment

- **AWS**: ECS, Lambda, SageMaker
- **GCP**: Cloud Run, Vertex AI
- **Azure**: Container Instances, ML Studio
- **Self-Hosted**: Kubernetes, Docker Swarm

See [deployment guide](docs/deployment/) for detailed instructions.

---

## 🛠️ Advanced Features

### 🔄 Pipeline Composition


```python
from llamafarm import Pipeline, RAG, Models, Prompts

# Create a complete AI pipeline
pipeline = Pipeline(strategy="research")
  .add(RAG.ingest("documents/"))
  .add(Prompts.select_template())
  .add(Models.generate())
  .add(RAG.store_results())

# Execute with monitoring
results = pipeline.run(
    query="What are the implications?",
    monitor=True,
    cache=True
)
```

### 🎯 Custom Strategies

```python
from llamafarm.strategies import Strategy

class MedicalStrategy(Strategy):
    """Custom strategy for medical document analysis"""

    def configure_rag(self):
        return {
            "extractors": ["medical_entities", "dosages", "symptoms"],
            "embedder": "biobert",
            "chunk_size": 256
        }

    def configure_models(self):
        return {
            "primary": "med-palm-2",
            "temperature": 0.1,
            "require_citations": True
        }
```

### 📊 Monitoring & Analytics

```python
from llamafarm.monitoring import Monitor

monitor = Monitor()
monitor.track_usage()
monitor.analyze_costs()
monitor.export_metrics("prometheus")
```

---

## 🌍 Community & Ecosystem

### 🤝 Contributing

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for:
- 🐛 Reporting bugs
- 💡 Suggesting features
- 🔧 Submitting PRs
- 📚 Improving docs

### 🏆 Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/BobbyRadford">
          <img src="https://avatars.githubusercontent.com/u/6943982?v=4?v=4&s=100" width="100px;" alt="Bobby Radford"/>
          <br />
          <sub><b>Bobby Radford</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=BobbyRadford" title="Code">💻</a> <a href="#maintenance" title="Maintenance">🚧</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/mhamann">
          <img src="https://avatars.githubusercontent.com/u/130131?v=4?v=4&s=100" width="100px;" alt="Matt Hamann"/>
          <br />
          <sub><b>Matt Hamann</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=mhamann" title="Code">💻</a> <a href="#maintenance" title="Maintenance">🚧</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/rachradulo">
          <img src="https://avatars.githubusercontent.com/u/128095403?v=4?v=4&s=100" width="100px;" alt="Rachel Orrino"/>
          <br />
          <sub><b>Rachel Orrino</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=rachradulo" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/rgthelen">
          <img src="https://avatars.githubusercontent.com/u/10455926?v=4?v=4&s=100" width="100px;" alt="Rob Thelen"/>
          <br />
          <sub><b>Rob Thelen</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=rgthelen" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/rachmlenig">
          <img src="https://avatars.githubusercontent.com/u/106166434?v=4?v=4&s=100" width="100px;" alt="Racheal Ochalek"/>
          <br />
          <sub><b>Racheal Ochalek</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=rachmlenig" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/apps/github-actions">
          <img src="https://avatars.githubusercontent.com/in/15368?v=4?v=4&s=100" width="100px;" alt="github-actions[bot]"/>
          <br />
          <sub><b>github-actions[bot]</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=github-actions[bot]" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/davon-davis">
          <img src="https://avatars.githubusercontent.com/u/77517056?v=4?v=4&s=100" width="100px;" alt="Davon Davis"/>
          <br />
          <sub><b>Davon Davis</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=davon-davis" title="Code">💻</a>
      </td>
    </tr>
    <tr>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/naaa760">
          <img src="https://avatars.githubusercontent.com/u/182316592?v=4?v=4&s=100" width="100px;" alt="Neha Prasad"/>
          <br />
          <sub><b>Neha Prasad</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=naaa760" title="Code">💻</a>
      </td>
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

### 🔗 Integration Partners

- **Vector DBs**: ChromaDB, Pinecone, Weaviate, Qdrant, FAISS
- **LLM Providers**: OpenAI, Anthropic, Google, Cohere, Together, Groq
- **Deployment**: Docker, Kubernetes, AWS, GCP, Azure
- **Monitoring**: Prometheus, Grafana, DataDog, New Relic

---

## 🚦 Roadmap

### ✅ Released
- RAG System with 10+ parsers and 5+ extractors
- 25+ LLM provider integrations
- 20+ prompt templates with strategies
- CLI tools for all components
- Docker deployment support

### 🚀 Coming Soon
- **Full Runtime System** - Complete orchestration layer for managing all components with health monitoring, resource allocation, and automatic recovery
- **Production Deployer** - Zero-configuration deployment from local development to cloud with automatic scaling and load balancing
- **Fine-tuning Pipeline** - Train custom models on your data with integrated evaluation and deployment
- **Web UI Dashboard** - Visual interface for monitoring, configuration, and management
- **Enhanced CLI** - Unified command interface across all components

### 🚧 In Progress
- **Fine-tuning pipeline** *(Looking for contributors with ML experience)*
- **Advanced caching system** *(Redis/Memcached integration - 40% complete)*
- **GraphRAG implementation** *(Design phase - [Join discussion](https://github.com/llama-farm/llamafarm/discussions))*
- **Multi-modal support** *(Vision models integration - Early prototype)*
- **Agent orchestration** *(LangGraph integration planned)*

### 📅 Planned (late-2025)
- **AutoML for strategy optimization** *(Q4 2025 - Seeking ML engineers)*
- **Distributed training** *(Q4 2025 - Partnership opportunities welcome)*
- **Edge deployment** *(Q4 2025 - IoT and mobile focus)*
- **Mobile SDKs** *(iOS/Android - Looking for mobile developers)*
- **Web UI dashboard** *(Q4 2025 - React/Vue developers needed)*

### 🤝 Want to Contribute?
We're actively looking for contributors in these areas:
- 🧠 **Machine Learning**: Fine-tuning, distributed training
- 📱 **Mobile Development**: iOS/Android SDKs
- 🎨 **Frontend**: Web UI dashboard
- 🔍 **Search**: GraphRAG and advanced retrieval
- 📚 **Documentation**: Tutorials and examples


---

## 📄 License

LlamaFarm is MIT licensed. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

LlamaFarm stands on the shoulders of giants:

- 🦜 [LangChain](https://github.com/hwchase17/langchain) - LLM orchestration inspiration
- 🤗 [Transformers](https://github.com/huggingface/transformers) - Model implementations
- 🎯 [ChromaDB](https://github.com/chroma-core/chroma) - Vector database excellence
- 🚀 [uv](https://github.com/astral-sh/uv) - Lightning-fast package management

See [CREDITS.md](CREDITS.md) for complete acknowledgments.

---

<div align="center">
  <h3>🦙 Ready to Build Production AI?</h3>
  <p>Join thousands of developers building with LlamaFarm</p>
  <p>
    <a href="https://github.com/llama-farm/llamafarm">⭐ Star on GitHub</a> •
    <a href="https://discord.gg/https://discord.gg/8kH9AmQpSa">💬 Join Discord</a> •
    <a href="https://docs.llamafarm.dev">📚 Read Docs</a> •

  </p>
  <br>
  <p><i>Build locally. Deploy anywhere. Own your AI.</i></p>
</div>
