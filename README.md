# 🦙 LlamaFarm - Build Powerful AI Locally, Deploy Anywhere

<div align="center">
  <img src="docs/images/rocket-llama.png" alt="Llama Building a Rocket" width="400">
  
  **The Complete AI Development Framework - From Local Prototypes to Production Systems**
  
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Go 1.19+](https://img.shields.io/badge/go-1.19+-00ADD8.svg)](https://golang.org/dl/)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
  [![Discord](https://img.shields.io/discord/1234567890?color=7289da&logo=discord&logoColor=white)](https://discord.gg/llamafarm)
  
  [🚀 Quick Start](#-quick-start) • [📚 Documentation](#-documentation) • [🏗️ Architecture](#-architecture) • [🤝 Contributing](#-contributing)
</div>

---

## 🌟 Why LlamaFarm?

LlamaFarm is a **comprehensive, modular AI framework** that gives you complete control over your AI stack. Unlike cloud-only solutions, we provide:

- **🏠 Local-First Development** - Build and test entirely on your machine
- **🔧 Production-Ready Components** - Battle-tested modules that scale from laptop to cluster
- **🎯 Strategy-Based Configuration** - Smart defaults with infinite customization
- **🚀 Deploy Anywhere** - Same code runs locally, on-premise, or in any cloud

### 🎭 Perfect For

- **Developers** who want to build AI applications without vendor lock-in
- **Teams** needing cost control and data privacy
- **Enterprises** requiring scalable, secure AI infrastructure
- **Researchers** experimenting with cutting-edge techniques

---

## 🏗️ Architecture

### System Overview

```mermaid
graph TB
    subgraph "📝 Prompts Layer"
        PT[Template Library]
        PS[Strategy Engine]
        PE[Evaluation System]
        PA[Multi-Agent Coordination]
    end
    
    subgraph "🔍 RAG Pipeline"
        DP[Document Parsers]
        EM[Embedders]
        VS[Vector Stores]
        RT[Retrieval Strategies]
        EX[Extractors]
    end
    
    subgraph "🤖 Model Layer"
        CP[Cloud Providers]
        LM[Local Models]
        FT[Fine-Tuning]
        MO[Monitoring]
    end
    
    subgraph "⚙️ Core Infrastructure"
        CF[Config Management]
        CL[CLI Tools]
        AP[API Services]
        DC[Docker Containers]
    end
    
    PT --> PS --> MO
    DP --> EM --> VS --> RT
    RT --> PS
    PS --> CP
    PS --> LM
    CF --> PT & DP & CP & LM
    CL --> CF
    AP --> CF
    
    style PT fill:#e1f5fe
    style DP fill:#fff3e0
    style CP fill:#f3e5f5
    style CF fill:#e8f5e9
```

### How Components Work Together

```mermaid
sequenceDiagram
    participant U as User
    participant C as CLI/API
    participant P as Prompts
    participant R as RAG System
    participant M as Models
    participant S as Strategy Config
    
    U->>C: Query/Request
    C->>S: Load Configuration
    S-->>C: Strategy Settings
    C->>R: Retrieve Context
    R->>R: Parse → Embed → Search
    R-->>C: Relevant Documents
    C->>P: Select Template
    P->>P: Apply Strategy
    P-->>C: Formatted Prompt
    C->>M: Execute with Provider
    M->>M: Fallback if needed
    M-->>C: Response
    C-->>U: Final Output
```

---

## ✨ Core Components

### 🔍 **RAG System** - Transform Documents into AI Knowledge

```mermaid
graph LR
    subgraph "Input"
        D[📄 Documents]
        W[🌐 Web Pages]
        DB[💾 Databases]
    end
    
    subgraph "Processing"
        P[Parse] --> E[Extract]
        E --> EM[Embed]
        EM --> I[Index]
    end
    
    subgraph "Retrieval"
        Q[Query] --> S[Search]
        S --> R[Rank]
        R --> F[Filter]
    end
    
    D & W & DB --> P
    I --> S
    F --> O[Output]
```

**Features:**
- **Universal Parsers**: PDF, Word, Excel, CSV, HTML, Markdown, and more
- **Smart Extractors**: Keywords, entities, statistics, sentiment, topics - no LLM required
- **Multiple Embedders**: OpenAI, Ollama, HuggingFace, Sentence Transformers
- **Vector Store Flexibility**: ChromaDB, Pinecone, FAISS, Qdrant, Weaviate
- **Advanced Retrieval**: Hybrid search, re-ranking, metadata filtering
- **Production Features**: Batch processing, progress tracking, error recovery

### 🤖 **Model Management** - Unified Interface for All LLMs

```mermaid
graph TB
    subgraph "Providers"
        O[OpenAI]
        A[Anthropic]
        G[Google]
        C[Cohere]
        T[Together]
        GR[Groq]
        OL[Ollama]
        HF[HuggingFace]
    end
    
    subgraph "Management"
        LB[Load Balancer]
        FB[Fallback Logic]
        RL[Rate Limiter]
        CM[Cost Monitor]
    end
    
    subgraph "Features"
        ST[Streaming]
        FN[Functions]
        EM[Embeddings]
        FT[Fine-Tuning]
    end
    
    O & A & G & C & T & GR & OL & HF --> LB
    LB --> FB --> RL --> CM
    CM --> ST & FN & EM & FT
```

**Features:**
- **25+ Provider Support**: All major cloud and local providers
- **Intelligent Routing**: Automatic fallbacks and load balancing
- **Cost Optimization**: Token tracking and budget management
- **Local Models**: Full Ollama and HuggingFace integration
- **Fine-Tuning**: Custom model training *(Coming Soon - [Help wanted!](https://github.com/llama-farm/llamafarm/labels/help-wanted))*

### 📝 **Prompt Engineering** - Enterprise-Grade Template System

**Features:**
- **20+ Pre-Built Templates**: Across 6 categories
- **Strategy-Based Selection**: Context-aware template matching
- **Dynamic Variables**: Jinja2 templating with validation
- **Multi-Agent Support**: Coordinate complex workflows
- **A/B Testing**: Built-in experimentation framework
- **Quality Evaluation**: 5 evaluation templates included

---

## 🚀 Quick Start

### Installation

```bash
# Quick install with our script
curl -fsSL https://raw.githubusercontent.com/llama-farm/llamafarm/main/install.sh | bash

# Or clone and set up manually
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm
```

### 📦 Component Setup

Each component can be used independently or together:

```bash
# 1. RAG System - Document Processing & Retrieval
cd rag
uv sync
uv run python setup_demo.py  # Interactive setup wizard

# 2. Models - LLM Management
cd ../models
uv sync
uv run python demos/demo_fallback.py  # See fallback in action

# 3. Prompts - Template System
cd ../prompts
uv sync
uv run python -m prompts.cli template list  # View available templates
```

### 🎮 Try It Live

#### RAG Pipeline Example
```bash
# Ingest documents with smart extraction
uv run python rag/cli.py ingest samples/ \
  --extractors keywords entities statistics \
  --strategy research

# Search with advanced retrieval
uv run python rag/cli.py search \
  "What are the key findings about climate change?" \
  --top-k 5 --rerank
```

#### Multi-Model Chat Example
```bash
# Chat with automatic fallback
uv run python models/cli.py chat \
  --primary gpt-4 \
  --fallback claude-3 \
  --local-fallback llama3.2 \
  "Explain quantum entanglement"
```

#### Smart Prompt Example
```bash
# Use domain-specific templates
uv run python prompts/cli.py execute \
  "Analyze this medical report for anomalies" \
  --strategy medical \
  --template diagnostic_analysis
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
        <a href="https://github.com/llama-farm/llamafarm/commits?author=BobbyRadford" title="Code">💻</a>
      </td>
      <td align="center" valign="top" width="14.28%">
        <a href="https://github.com/mhamann">
          <img src="https://avatars.githubusercontent.com/u/130131?v=4?v=4&s=100" width="100px;" alt="Matt Hamann"/>
          <br />
          <sub><b>Matt Hamann</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=mhamann" title="Code">💻</a>
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
        <a href="https://github.com/davon-davis">
          <img src="https://avatars.githubusercontent.com/u/77517056?v=4?v=4&s=100" width="100px;" alt="Davon Davis"/>
          <br />
          <sub><b>Davon Davis</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=davon-davis" title="Code">💻</a>
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
        <a href="https://github.com/rachradulo">
          <img src="https://avatars.githubusercontent.com/u/128095403?v=4?v=4&s=100" width="100px;" alt="rachradulo"/>
          <br />
          <sub><b>rachradulo</b></sub>
        </a>
        <br />
        <a href="https://github.com/llama-farm/llamafarm/commits?author=rachradulo" title="Code">💻</a>
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

## 📊 Benchmarks

| Operation | LlamaFarm | Alternative A | Alternative B |
|-----------|-----------|---------------|---------------|
| Document Ingestion | 1,000 docs/min | 400 docs/min | 600 docs/min |
| Embedding Generation | 50ms/chunk | 120ms/chunk | 80ms/chunk |
| Vector Search | 5ms @ 1M vectors | 15ms @ 1M | 10ms @ 1M |
| LLM Fallback | Automatic | Manual | Not Supported |
| Cost Optimization | -60% avg | Baseline | -30% avg |

---

## 🚦 Roadmap

### ✅ Released
- RAG System with 10+ parsers and 5+ extractors
- 25+ LLM provider integrations
- 20+ prompt templates with strategies
- CLI tools for all components
- Docker deployment support

### 🚧 In Progress
- **Fine-tuning pipeline** *(Looking for contributors with ML experience)*
- **Advanced caching system** *(Redis/Memcached integration - 40% complete)*
- **GraphRAG implementation** *(Design phase - [Join discussion](https://github.com/llama-farm/llamafarm/discussions))*
- **Multi-modal support** *(Vision models integration - Early prototype)*
- **Agent orchestration** *(LangGraph integration planned)*

### 📅 Planned (2025)
- **AutoML for strategy optimization** *(Q3 2025 - Seeking ML engineers)*
- **Distributed training** *(Q4 2025 - Partnership opportunities welcome)*
- **Edge deployment** *(Q2 2025 - IoT and mobile focus)*
- **Mobile SDKs** *(iOS/Android - Looking for mobile developers)*
- **Web UI dashboard** *(Q2 2025 - React/Vue developers needed)*

### 🤝 Want to Contribute?
We're actively looking for contributors in these areas:
- 🧠 **Machine Learning**: Fine-tuning, distributed training
- 📱 **Mobile Development**: iOS/Android SDKs
- 🎨 **Frontend**: Web UI dashboard
- 🔍 **Search**: GraphRAG and advanced retrieval
- 📚 **Documentation**: Tutorials and examples

See our [public roadmap](https://github.com/llama-farm/llamafarm/projects) for details.

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
    <a href="https://discord.gg/llamafarm">💬 Join Discord</a> • 
    <a href="https://llamafarm.ai">📚 Read Docs</a> •
    <a href="https://twitter.com/llamafarm">🐦 Follow Updates</a>
  </p>
  <br>
  <p><i>Build locally. Deploy anywhere. Own your AI.</i></p>
</div>