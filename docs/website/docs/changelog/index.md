# Changelog

Stay up to date with the latest features, improvements, and fixes in LlamaFarm.

---

## Latest Release

<details open>
<summary><strong>v0.0.28</strong> — 2026-03-05</summary>
<think>

</think>

## LlamaFarm 0.0.28: Building Blocks for Smarter AI Workflows

This release focuses on expanding the capabilities of LlamaFarm, making it easier to build, deploy, and manage AI models with greater flexibility and reliability. We’ve introduced new tools, improved performance, and fixed issues that were impacting the user experience.

---

### New Features & Enhancements

**Deploy Models with Ease**  
We've added a new `deploy` command to the CLI and a bundled packaging system, making it simple to package and deploy models as self-contained units. This is especially useful for sharing models with teams or deploying in production environments.

**Visual Designer for Deployment**  
A new Bundle UI has been added to the Designer, allowing users to create and manage deployment workflows visually. This makes it easier to build complex model pipelines without needing to write code.

**ML Addons for Specialized Tasks**  
We've introduced several new ML addons, including support for time series analysis, drift detection, and CatBoost. These can be easily enabled or disabled, giving users the flexibility to tailor their AI workflows to specific needs.

**Enhanced Log Support for GGUF Models**  
Users can now access log probabilities for GGUF chat completions, which is helpful for understanding model behavior and improving the quality of generated responses.

**Improved Server-Side KV Cache**  
We’ve implemented a more efficient server-side key-value cache that supports multi-turn chaining and pre-warming, which helps maintain performance during long conversations with large models.

**Built-in Tools System**  
A new tools system has been added to the server, including a `tasks` tool. This allows users to run custom functions directly within the AI platform, opening up new possibilities for automation and integration.

**Structured Output Support**  
Models can now return structured outputs, which is especially useful for applications that require precise data formatting, such as data pipelines or API integrations.

**Vision API Improvements**  
The vision API has seen significant improvements, including better evaluation pipelines and object tracking. This makes it easier to build and test computer vision models within LlamaFarm.

**Vision UI for Designer**  
A new Vision UI has been added to the Designer, allowing users to build and manage vision workflows visually, including detection, classification, and training.

**Vision MVP with Basic Functionality**  
We've launched a vision MVP that includes core capabilities like detection, classification, training, and feedback loops. This provides a solid foundation for building more advanced vision models.

---

### Bug Fixes & Stability Improvements

**CI Process Optimization**  
We fixed an issue where the prose changelog was causing an ARG_MAX overflow in the CI pipeline, making the build process more reliable.

**Addon Registry Integration**  
The addon registry is now embedded into the binary for released builds, ensuring that models and tools are available without needing to download external packages.

**Content Budget Calculations**  
We've improved the math behind content budget calculations, ensuring that model usage is tracked accurately and efficiently.

**Remote Access for Designer**  
Users can now access the Designer remotely, which is especially useful for collaborative workflows or when working with headless environments.

**Improved Addon Bundling**  
We’ve fixed an issue where base-install dependencies were being included in addon wheel bundles, ensuring that addons are self-contained and easier to manage.

**Better Error Handling for Timeseries**  
If a timeseries backend is unavailable, the system now returns a 422 error instead of a 500, which helps users understand and resolve issues more quickly.

---

### Other Updates

We've also completed the release process for version 0.0.28, ensuring that everything is ready for users to try out and provide feedback.

---

LlamaFarm 0.0.28 is a major step forward in making AI development more intuitive, efficient, and powerful. Whether you're building models, managing workflows, or exploring new capabilities like vision, there's something here to help you get more done. Let us know what you think!

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.28)**

</details>

<details>
<summary><strong>v0.0.27</strong> — 2026-02-16</summary>

**LlamaFarm 0.0.27: Addons, Smarter RAG, and Runtime Resilience**

This release introduces the addons system, smarter RAG defaults, and significant runtime stability improvements.

### New Features

#### Addons System

LlamaFarm now supports addons — modular extensions you can install and enable to expand your platform's capabilities. The Designer includes a polished UX for browsing, installing, and managing addons, with sequential installation and auto-enable on install for a smooth experience.

#### Per-Model RAG Defaults

You can now configure default RAG retrieval strategies on a per-model basis. This means different models can automatically use the retrieval settings that work best for them — no manual configuration needed each time.

#### RAG Source Chunks in Test Outputs

The Designer now shows RAG source chunks directly in test outputs, so you can see exactly which documents your model is referencing. Great for debugging retrieval quality and understanding model responses.

#### Cascading Data Processing Strategies

The server now supports cascading default data processing strategies, making it easier to set up sensible defaults that flow through your entire pipeline.

#### Anomaly Detection Documentation

Comprehensive docs, use-cases, and a full demo for the anomaly detection feature introduced in v0.0.24 — making it much easier to get started with outlier detection.

### Infrastructure

- **Binary component builds** for faster CI and distribution
- **Server port change** — default port moved from 8000 to 14345 to avoid conflicts

### Bug Fixes

- **Smart GPU allocation** — prevents multi-model OOM crashes by intelligently managing GPU memory across loaded models
- **Event loop protection** — model loading in the Universal Runtime no longer blocks the event loop, improving responsiveness during heavy loads
- **API system prompts** — fixed a bug where API-provided system prompts were being overridden by config-level system prompts
- **Designer improvements** — better delete UX, ghost project handling, fixed 404 on train button, improved onboarding checklist updates after demo project conversion
- **Audio error handling** — improved error handling in the Designer for audio features

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.27)**

</details>

<details>
<summary><strong>v0.0.26</strong> — 2026-01-27</summary>

**LlamaFarm 0.0.26: Smarter, Faster, and More Accessible**

This release brings a range of improvements to make LlamaFarm more intuitive, efficient, and accessible across different platforms and use cases.

### New Features and Enhancements

#### Reusability and Configuration Improvements

We've introduced reusable components in the configuration system, allowing you to define and reuse common settings across different parts of your application. This makes managing complex configurations much simpler and reduces duplication.

#### Enhanced RAG Capabilities

**Universal RAG** - We've added zero-config default strategies that work out of the box for most use cases. No more complex setup required to get started with retrieval-augmented generation.

**Document Preview** - You can now preview documents with strategy selection directly in the Designer, making it easier to understand how your RAG pipeline processes different file types.

#### Dataset Management

New sample datasets for gardening and home repair scenarios help you get started quickly with realistic data. Plus, datasets now auto-process on upload, eliminating manual processing steps.

#### Developer Experience

**Dynamic Value Substitution** - Prompts and tools now support dynamic variable substitution, making your configurations more flexible and powerful.

**Service Status Panel** - A new status panel in the Designer header gives you real-time visibility into your LlamaFarm services, so you know exactly what's running.

#### Audio and Speech

This release introduces a full-duplex speech reasoning pipeline with audio processing capabilities in the Universal Runtime. Build voice-enabled AI applications with ease.

#### Cross-Platform Support

- **Desktop App Improvements** - Better splash screen UX and enhanced cross-platform support
- **Intel Mac Support** - Added support for Intel Macs (x86_64) with PyTorch 2.2.2
- **Jetson/Tegra Optimization** - Improved CUDA optimization and unified memory GPU support

### Bug Fixes

- Fixed dev builds stopping running services
- Resolved sample project creation failures
- Fixed chat input clearing during streaming
- Improved error display and Service Status panel reliability

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.26)**

</details>

---

## Recent Releases

<details>
<summary><strong>v0.0.25</strong> — 2026-01-14</summary>

**LlamaFarm 0.0.25: Native Tool Calling and Developer Productivity**

This release focuses on improving the developer experience with better tooling, native tool calling support, and automatic file processing capabilities.

### New Features

#### Native Tool Calling

The Universal Runtime now supports native tool calling, enabling your AI models to interact with external tools and APIs more efficiently. This is a major step forward for building agentic AI applications that can take actions in the real world.

#### Automatic File Processing

Files uploaded to datasets now process automatically, eliminating the manual processing step and streamlining your workflow. Just upload and go.

#### Enhanced Designer Development Tools

The Designer now includes comprehensive API call logging in the dev tools panel, making it easier to debug and understand how your application communicates with the backend. See every request and response in real-time.

#### Streaming Model Downloads

Embedding model downloads now use SSE streaming, providing real-time progress updates so you always know exactly what's happening during long downloads.

#### Extended Testing Capabilities

The test space now includes support for anomaly detection and classifier tests, giving you more ways to validate your AI models before deployment.

### Bug Fixes

- Fixed config validation error output for clearer debugging
- Resolved install and run failures on Windows with NVIDIA GPUs
- Removed parser fallback to prevent unexpected behavior
- Enabled offline GGUF model loading for air-gapped environments

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.25)**

</details>

<details>
<summary><strong>v0.0.24</strong> — 2026-01-06</summary>

**LlamaFarm 0.0.24: Anomaly Detection**

This release introduces anomaly detection capabilities to help identify outliers and unusual patterns in your data.

### New Features

#### Anomaly Detection

The Universal Runtime now supports anomaly detection with configurable normalization methods for scoring. Whether you're monitoring for fraud, equipment failures, or data quality issues, LlamaFarm can now help identify when something doesn't look right.

#### Designer UX for Anomaly Detection

The Designer includes a new interface for configuring and testing anomaly detection models, making it easy to set up detection pipelines and visualize results.

### Bug Fixes

- Fixed anomaly and classifier UX issues in the Designer for smoother workflows

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.24)**

</details>

<details>
<summary><strong>v0.0.23</strong> — 2025-12-20</summary>

**LlamaFarm 0.0.23: Stability Improvements**

A focused stability release addressing a critical logging issue in the Universal Runtime.

### Bug Fixes

- Fixed broken pipe errors caused by problematic logging in the Universal Runtime, improving reliability for long-running inference tasks

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.23)**

</details>

<details>
<summary><strong>v0.0.22</strong> — 2025-12-19</summary>

**LlamaFarm 0.0.22: Inference Fix**

A quick bug fix release addressing an issue with logits processor handling.

### Bug Fixes

- Fixed logits_processor to be passed as callable instead of list, resolving inference issues with certain model configurations

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.22)**

</details>

<details>
<summary><strong>v0.0.21</strong> — 2025-12-19</summary>

**LlamaFarm 0.0.21: Specialized ML Models and Vision API**

A feature-packed holiday release bringing specialized ML models, vision capabilities, and major Designer enhancements.

### New Features

#### Specialized ML Models

Added support for OCR, document extraction, and anomaly detection models in the Universal Runtime. These specialized models expand what you can build with LlamaFarm beyond text generation - now you can extract text from images, parse documents, and detect anomalies.

#### Vision API

New vision router and model versioning for ML endpoints, enabling image understanding capabilities in your applications. Build apps that can see and understand visual content.

#### Designer Improvements

- **Santa's Holiday Helper Demo** - A festive demo project to help new users get started
- **Enhanced RAG UX** - Improved retrieval strategy settings in test chat
- **Data Enhancements** - Better tools for managing your datasets
- **Global Project Listing** - Easily see all your projects in one place

#### Cross-Platform Support

Native llama-cpp bindings now included for all platforms, and Windows builds correctly include the `.exe` extension for seamless installation.

### Bug Fixes

- Fixed upgrade failures on Linux
- Ensured multi-arch Linux builds work correctly
- Fixed model unload cleanup and OpenAI message validation
- Removed console log spam in Designer

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.21)**

</details>

---

## Older Releases

<details>
<summary><strong>View all releases</strong></summary>

<details>
<summary><strong>v0.0.20</strong> — 2025-12-10</summary>

**Auto-Start Services, RAG Stats, and Reliability Improvements**

### New Features

- **Auto-Start Service Flag** - Services can now start automatically when you run LlamaFarm
- **More GGUF Download Options** - More quantization options for model downloads in Designer
- **RAG Database Listing** - List all documents in your RAG databases
- **RAG Statistics** - View detailed stats about your RAG setup
- **Chunk Cleanup** - Automatically remove database chunks when files are deleted
- **Data Processing Control** - Start and stop data processing from the API

### Bug Fixes

- Fixed first-run startup failures for new users
- Improved path resolution with `~` expansion
- Better process manager locking to prevent conflicts
- Fixed upgrade hang caused by process stop deadlock
- Prevented storage of failed vectors in RAG

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.20)**

</details>

<details>
<summary><strong>v0.0.19</strong> — 2025-12-03</summary>

**Automatic Model Downloads, Custom RAG Queries, and Reasoning Models**

### New Features

- **Automatic Model Download Management** - Models download automatically when needed
- **Custom RAG Queries** - Send custom RAG queries through the chat/completions endpoint
- **Thinking/Reasoning Model Support** - Support for models that show their reasoning process
- **Database CRUD API** - Full create, read, update, delete operations for databases
- **Better Day-2 UX** - Improved experience for returning users
- **Disk Space Checking** - Check available disk space before downloading models
- **GGUF Model Listing** - Browse available GGUF models for download

### Bug Fixes

- Fixed datasets endpoint trailing slash requirement
- Improved cross-filesystem data moves
- Fixed PDF parsing issues in RAG
- Addressed demo timeout issues

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.19)**

</details>

<details>
<summary><strong>v0.0.18</strong> — 2025-11-25</summary>

**Code Signing and Advanced RAG Retrieval**

### New Features

- **Signed Apps** - Windows and Mac apps are now code-signed for easier installation
- **Advanced RAG Retrieval** - Cross-encoder reranking and multi-turn RAG for better search results

### Bug Fixes

- Ensured service logs are always enabled

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.18)**

</details>

<details>
<summary><strong>v0.0.17</strong> — 2025-11-24</summary>

**Bug Fixes and Documentation**

### Bug Fixes

- Fixed empty prompts array for new projects
- Added troubleshooting documentation
- Fixed HuggingFace progress bar crashes

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.17)**

</details>

<details>
<summary><strong>v0.0.16</strong> — 2025-11-23</summary>

**CLI Packaging Fix**

### Bug Fixes

- Fixed CLI packaging issues

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.16)**

</details>

<details>
<summary><strong>v0.0.15</strong> — 2025-11-22</summary>

**Desktop App Launch and GGUF Model Support**

### New Features

- **Desktop App** - Full Electron desktop app with auto-updates and polished UI
- **GGUF Model Support** - Run quantized GGUF models in the Universal Runtime
- **Demo Project System** - Interactive demo projects to help new users get started
- **Universal Event Logging** - Comprehensive observability across the platform
- **Enhanced Tool Calling** - Improved tool calling capabilities
- **Project Cloning** - Create new projects from existing ones

### Bug Fixes

- Fixed upgrade failures on Unix-like systems
- Improved RAG integration and chat context management
- Fixed database tab switching in Designer
- Better dataset validation and status display

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.15)**

</details>

<details>
<summary><strong>v0.0.14</strong> — 2025-11-13</summary>

**Database Strategies and RAG Improvements**

### New Features

- RAG query stats showing size information
- Database embedding and retrieval strategies in Designer

### Bug Fixes

- Fixed build chat errors
- Fixed file drop dataset selection

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.14)**

</details>

<details>
<summary><strong>v0.0.13</strong> — 2025-11-11</summary>

**Version Number Fix**

### Bug Fixes

- CLI now displays correct version number

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.13)**

</details>

<details>
<summary><strong>v0.0.12</strong> — 2025-11-11</summary>

**Project Management and Config Editor**

### New Features

- **Delete Projects** - Remove projects from CLI and API
- **Config Editor Enhancements** - Copy button, search, anchor points, unsaved changes prompts
- **Embedding Strategies API** - Configure embedding strategies via API
- **MCP Server Config** - Add MCP server configuration to runtime
- **Project Context Provider** - Better project context management

**[Full Changelog →](https://github.com/llama-farm/llamafarm/releases/tag/v0.0.12)**

</details>

<details>
<summary><strong>v0.0.11 and earlier</strong></summary>

For releases v0.0.11 and earlier, please see the [full changelog on GitHub](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md).

</details>

</details>

---

## About These Release Notes

These release notes are generated from our conventional commit history. For the complete structured changelog with commit links and PR references, see the [CHANGELOG.md](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md) on GitHub.

## Stay Updated

- **GitHub Releases**: [github.com/llama-farm/llamafarm/releases](https://github.com/llama-farm/llamafarm/releases)
- **Reddit**: [r/LlamaFarm](https://www.reddit.com/r/LlamaFarm/)
- **Discord**: [Join our community](https://discord.gg/jtChvg8T)
