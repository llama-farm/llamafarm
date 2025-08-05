# 🦙 LlamaFarm - Build Powerful AI Locally, Deploy Anywhere

<div align="center">
  <img src="docs/images/rocket-llama.png" alt="Llama Building a Rocket" width="400">

  **Empowering developers to build production-ready AI applications with complete local control**

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
  [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
  [![Discord](https://img.shields.io/discord/1234567890?color=7289da&logo=discord&logoColor=white)](https://discord.gg/llamafarm)

   [Getting Started](#-getting-started) • [Features](#-features) • [Contributing](#-contributing)
</div>

---

## 🚀 What is LlamaFarm?

LlamaFarm is a comprehensive, modular framework for building AI Projects that run locally, collaborate, and deploy anywhere. We provide battle-tested components for RAG systems, vector databases, model management, prompt engineering, and soon fine-tuning - all designed to work seamlessly together or independently.

### 🎯 Our Mission

We believe AI development should be:
- **Local First**: Full control over your data and models
- **Production Ready**: Built for scale from day one
- **Developer Friendly**: Configuration over code, with sensible defaults
- **Modular**: Use what you need, ignore what you don't
- **Open**: No vendor lock-in, works with any provider

## 🏗️ Building in the Open

We're building LlamaFarm in public! Join us:
- 🐛 [Report bugs](https://github.com/llama-farm/llamafarm/issues)
- 💡 [Request features](https://github.com/llama-farm/llamafarm/discussions)
- 🤝 [Contribute code](CONTRIBUTING.md)
- 💬 [Join our Discord](https://discord.gg/llamafarm)

---

## 🚀 Quick Start

### Install the CLI

Get started with LlamaFarm in seconds:

```bash
curl -fsSL https://raw.githubusercontent.com/llamafarm/llamafarm/main/install.sh | bash
```

After installation, verify it works:
```bash
lf version
lf help
```

For detailed installation options and troubleshooting, see the [Installation Guide](INSTALL.md).

### Your First Project

```bash
# Initialize a new project
lf init my-ai-project
cd my-ai-project

# Start the designer interface
lf designer start
```

---

## ✨ Features

### 🔍 RAG (Retrieval-Augmented Generation)
*Transform any document into AI-accessible knowledge*

- **📄 Universal Document Support**: Parse PDFs, CSVs, Word docs, web pages, and more
- **🧩 Modular Pipeline**: Mix and match parsers, embedders, and vector stores
- **🎯 Smart Retrieval**: 5+ retrieval strategies including hybrid search and re-ranking
- **🔌 Database Agnostic**: Works with ChromaDB, Pinecone, Weaviate, Qdrant, and more
- **📊 Local Extractors**: 5 built-in extractors for metadata enrichment without LLMs
- **⚡ Production Ready**: Batch processing, error handling, and progress tracking

**Quick Example:**
```bash
# Ingest documents with smart extraction
uv run python rag/cli.py ingest documents/ --extractors keywords entities statistics

# Search with advanced retrieval
uv run python rag/cli.py search "How does the authentication system work?" --top-k 5
```

[Learn more about RAG →](rag/README.md)

### 🤖 Model Management
*Run and manage AI models locally or in the cloud*

- **🌍 Multi-Provider Support**: OpenAI, Anthropic, Google, Cohere, Together, Groq, Ollama, HuggingFace
- **💰 Cost Optimization**: Automatic provider fallbacks and smart routing
- **📊 Usage Tracking**: Monitor tokens, costs, and performance
- **🔄 Load Balancing**: Distribute requests across multiple providers
- **🎛️ Fine Control**: Rate limiting, retry logic, and timeout management
- **🏠 Local Models**: Full support for Ollama and HuggingFace models

**Quick Example:**
```yaml
# config/models.yaml
providers:
  primary:
    provider: "openai"
    model: "gpt-4o-mini"
    fallback_to: "local_llama"

  local_llama:
    provider: "ollama"
    model: "llama3.2"
    temperature: 0.7
```

[Learn more about Models →](models/README.md)

### 📝 Prompt Engineering
*Enterprise-grade prompt management system*

- **📚 20+ Templates**: Pre-built templates for common use cases
- **🧠 Smart Selection**: Context-aware template selection
- **🔄 A/B Testing**: Built-in experimentation framework
- **🎯 6 Template Categories**: Basic, Chat, Few-shot, Advanced, Domain-specific, Agentic
- **🤝 Multi-Agent Support**: Coordinate multiple AI agents
- **📊 Evaluation Tools**: 5 evaluation templates for quality assessment

**Quick Example:**
```bash
# List all templates
uv run python prompts/cli.py template list

# Execute with smart selection
uv run python prompts/cli.py execute "Analyze this medical report" --domain medical

# Evaluate responses
uv run python prompts/cli.py evaluate "AI response text" --template llm_judge
```

[Learn more about Prompts →](prompts/README.md)

### 🎓 Fine-Tuning (Coming Soon!)
*Train custom models on your data*

- **🔧 Local Training**: Fine-tune models on your hardware
- **☁️ Cloud Training**: Integration with major training platforms
- **📊 Dataset Management**: Tools for data preparation and validation
- **🎯 Task-Specific Models**: Optimize for your specific use case
- **📈 Training Analytics**: Monitor loss, accuracy, and other metrics

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- Optional: Ollama for local models
- Optional: Docker for containerized deployment

### Quick Install

```bash
# Clone the repository
git clone https://github.com/llama-farm/llamafarm.git
cd llamafarm

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Set up RAG system
cd rag
uv sync
./setup_and_demo.sh

# Set up Models system
cd ../models
uv sync
./setup_and_demo.sh

# Set up Prompts system
cd ../prompts
uv sync
./setup_and_demo.sh
```

### 🎮 Try It Out

```bash
# RAG: Ingest and search documents
cd rag
uv run python cli.py ingest samples/documents.pdf
uv run python cli.py search "What are the key findings?"

# Models: Chat with multiple providers
cd ../models
uv run python cli.py chat --provider openai "Explain quantum computing"
uv run python cli.py chat --provider ollama "Write a Python function"

# Prompts: Use intelligent templates
cd ../prompts
uv run python -m prompts.cli execute "Compare solar vs wind energy" \
  --template comparative_analysis
```

---

## 📚 Documentation

### Component Guides
- 📖 [RAG System Guide](rag/README.md) - Document processing and retrieval
- 🤖 [Models Guide](models/README.md) - Model management and providers
- 📝 [Prompts Guide](prompts/README.md) - Prompt engineering and templates

### Tutorials
- 🎓 [Building Your First RAG App](docs/tutorials/first-rag-app.md)
- 🔧 [Setting Up Local Models](docs/tutorials/local-models.md)
- 🎯 [Prompt Engineering Best Practices](docs/tutorials/prompt-engineering.md)

### API Reference
- 🔌 [RAG API](docs/api/rag.md)
- 🤖 [Models API](docs/api/models.md)
- 📝 [Prompts API](docs/api/prompts.md)

---

## 🛠️ Architecture

LlamaFarm follows a modular, configuration-driven architecture:

```
llamafarm/
├── rag/              # Document processing and retrieval
│   ├── core/         # Base classes and interfaces
│   ├── parsers/      # Document parsers (PDF, CSV, etc.)
│   ├── embedders/    # Text embedding models
│   ├── stores/       # Vector database integrations
│   └── retrieval/    # Retrieval strategies
│
├── models/           # Model management
│   ├── providers/    # LLM provider integrations
│   ├── config/       # Configuration system
│   ├── monitoring/   # Usage tracking and analytics
│   └── optimization/ # Cost and performance optimization
│
├── prompts/          # Prompt engineering
│   ├── templates/    # Prompt template library
│   ├── strategies/   # Template selection strategies
│   ├── evaluation/   # Response evaluation tools
│   └── agents/       # Multi-agent coordination
│
└── training/         # Fine-tuning (coming soon)
    ├── datasets/     # Dataset management
    ├── trainers/     # Training implementations
    └── evaluation/   # Model evaluation
```

---

## 🌟 Why LlamaFarm?

### For Developers
- **🏠 Local First**: Run everything on your machine, no API keys required
- **🔧 Hackable**: Clean, modular code that's easy to understand and extend
- **📦 Batteries Included**: Pre-built components for common use cases
- **🎯 Production Ready**: Built with scale, monitoring, and reliability in mind

### For Teams
- **💰 Cost Control**: Optimize spending with multi-provider support
- **🔒 Data Privacy**: Keep sensitive data on-premise
- **🚀 Fast Iteration**: Hot-reload configs, no redeploys needed
- **📊 Full Visibility**: Built-in monitoring and analytics

### For Enterprises
- **🏢 Multi-Tenant**: Isolated environments for different teams
- **🔐 Security First**: SOC2-ready with audit logging
- **📈 Scalable**: From laptop to cluster without code changes
- **🤝 Vendor Neutral**: No lock-in, works with any provider

---

## 🤝 Contributing

<div align="center">
  <img src="docs/images/iron-workers-llama.png" alt="Iron Worker Llamas" width="400">

  **Join our herd of contributors building the future of local AI!**
</div>

We love contributions! Whether you're fixing bugs, adding features, or improving documentation, we'd love to have you aboard.

### How to Contribute
1. 🍴 Fork the repository
2. 🌿 Create a feature branch (`git checkout -b feature/amazing-feature`)
3. 💻 Make your changes
4. ✅ Run tests (`uv run pytest`)
5. 📝 Commit your changes (`git commit -m 'Add amazing feature'`)
6. 🚀 Push to the branch (`git push origin feature/amazing-feature`)
7. 🎉 Open a Pull Request

See our [Contributing Guide](CONTRIBUTING.md) for more details.

### Good First Issues
- 🏷️ [good-first-issue](https://github.com/llama-farm/llamafarm/labels/good-first-issue)
- 📚 [documentation](https://github.com/llama-farm/llamafarm/labels/documentation)
- 🧪 [testing](https://github.com/llama-farm/llamafarm/labels/testing)

---

## 👥 Contributors

Thanks to all our amazing contributors who make LlamaFarm possible!

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/contributor1"><img src="https://avatars.githubusercontent.com/contributor1?v=4?s=100" width="100px;" alt="Contributor 1"/><br /><sub><b>Contributor 1</b></sub></a><br /><a href="https://github.com/llama-farm/llamafarm/commits?author=contributor1" title="Code">💻</a></td>
      <!-- Add more contributors here -->
    </tr>
  </tbody>
</table>
<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

---

## 🙏 Open Source Credits

LlamaFarm is built on the shoulders of giants. Special thanks to:

### Core Dependencies
- 🦜 [LangChain](https://github.com/hwchase17/langchain) - LLM orchestration
- 🤗 [Transformers](https://github.com/huggingface/transformers) - Model library
- 🎯 [ChromaDB](https://github.com/chroma-core/chroma) - Vector database
- 📊 [Pandas](https://github.com/pandas-dev/pandas) - Data manipulation
- 🔥 [PyTorch](https://github.com/pytorch/pytorch) - Deep learning

### Development Tools
- 🚀 [uv](https://github.com/astral-sh/uv) - Fast Python package management
- 🧪 [pytest](https://github.com/pytest-dev/pytest) - Testing framework
- 📝 [Pydantic](https://github.com/pydantic/pydantic) - Data validation
- 🎨 [Rich](https://github.com/Textualize/rich) - Beautiful terminal output

See [CREDITS.md](CREDITS.md) for a complete list.

---

## 📄 License

LlamaFarm is MIT licensed. See [LICENSE](LICENSE) for details.

---



## 💬 Community

Join the LlamaFarm community:

- 💬 [Discord Server](https://discord.gg/llamafarm) - Chat with the community


---

<div align="center">
  <p>
    <b>Ready to farm some AI? 🦙🚜</b>
  </p>
  <p>
    <a href="https://github.com/llama-farm/llamafarm">⭐ Star us on GitHub</a> •
    <a href="https://discord.gg/llamafarm">💬 Join Discord</a> •
  </p>
</div>