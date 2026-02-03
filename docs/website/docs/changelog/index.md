# Changelog

Stay up to date with the latest features, improvements, and fixes in LlamaFarm.

Our releases follow [Semantic Versioning](https://semver.org/) and use [Conventional Commits](https://www.conventionalcommits.org/) for clear, structured change tracking.

---

## Recent Releases

<details open>
<summary><strong>v0.0.26 (2026-01-27)</strong> - Latest</summary>

<br/>

*Released on 2026-01-27*

LlamaFarm 0.0.26: Smarter, Faster, and More Accessible

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

**Desktop App Improvements** - Better splash screen UX and enhanced cross-platform support for Windows, macOS, and Linux.

**Intel Mac Support** - Added support for Intel Macs (x86_64) with PyTorch 2.2.2, ensuring LlamaFarm works on older hardware.

**Jetson/Tegra Optimization** - Improved CUDA optimization and unified memory GPU support for NVIDIA Jetson and Tegra platforms, perfect for edge AI deployments.

### Bug Fixes

- Fixed an issue where dev builds would stop running services
- Resolved sample project creation failures related to PyTorch memory settings
- Fixed chat input clearing during streaming
- Improved error message display and overflow handling in the Designer
- Enhanced Service Status panel reliability and accessibility
- Restricted synchronous inference to Jetson/Tegra platforms where it's needed

### Developer Notes

This release includes numerous internal improvements to make LlamaFarm more maintainable and easier to extend. We've refined our testing infrastructure and improved error handling across the platform.

---

**Full Changelog**: [v0.0.26](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md#0026)


</details>

<details>
<summary><strong>v0.0.25 (2026-01-14)</strong></summary>



*Released on 2026-01-14*

This release focuses on improving the developer experience with better tooling, native tool calling support, and automatic file processing capabilities.

### New Features

#### Native Tool Calling

The Universal Runtime now supports native tool calling, enabling your AI models to interact with external tools and APIs more efficiently. This opens up new possibilities for building agentic AI applications.

#### Automatic File Processing

Files uploaded to datasets now process automatically, eliminating the manual processing step and streamlining your workflow.

#### Enhanced Designer Development Tools

The Designer now includes comprehensive API call logging in the dev tools, making it easier to debug and understand how your application communicates with the backend.

#### Streaming Model Downloads

Embedding model downloads now use SSE streaming, providing real-time progress updates so you know exactly what's happening during long downloads.

#### Extended Testing Capabilities

The test space now includes support for anomaly detection and classifier tests, giving you more ways to validate your AI models.

### Bug Fixes

- Fixed config validation error output for clearer debugging
- Resolved install and run failures on Windows with NVIDIA GPUs
- Removed parser fallback to prevent unexpected behavior
- Moved dependencies to main package and enabled offline GGUF model loading

### What's Coming Next

We're continuing to improve the platform's ease of use and expanding support for more specialized ML tasks. Stay tuned for more updates!

---

**Full Changelog**: [v0.0.25](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md#0025)


</details>

<details>
<summary><strong>View More Releases...</strong></summary>

<br/>

### v0.0.24 (2026-01-06)

This release introduces anomaly detection capabilities to help identify outliers in your data.

#### Anomaly Detection

The Universal Runtime now supports anomaly detection with configurable normalization methods for scoring. The Designer includes a new UX for configuring and testing anomaly detection models, making it easy to identify unusual patterns in your datasets.

**Full Changelog**: [v0.0.24](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md#0024)

---

### v0.0.23 (2025-12-20)

A stability release fixing a logging issue that could cause broken pipe errors in the runtime.

**Full Changelog**: [v0.0.23](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md#0023)

---

### v0.0.22 (2025-12-19)

Bug fix release addressing an issue with logits processor handling in the Universal Runtime.

**Full Changelog**: [v0.0.22](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md#0022)

---

### v0.0.21 (2025-12-19)

A feature-packed holiday release bringing specialized ML models and major Designer enhancements.

#### Specialized ML Models

Added support for OCR, document extraction, and anomaly detection models in the Universal Runtime. These specialized models expand what you can build with LlamaFarm beyond text generation.

#### Vision API

New vision router and model versioning for ML endpoints, enabling image understanding capabilities in your applications.

#### Designer Improvements

- **Santa's Holiday Helper Demo** - A festive demo project to help you get started
- **Enhanced RAG UX** - Improved retrieval strategy settings in test chat
- **Data Enhancements** - Better tools for managing your datasets

#### Cross-Platform Support

Native llama-cpp bindings now included for all platforms, and Windows builds correctly include the `.exe` extension.

**Full Changelog**: [v0.0.21](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md#0021)

---

**Full History**: [CHANGELOG.md](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md)

</details>

---

## About These Release Notes

These human-readable release notes are automatically generated from our conventional commit history using LlamaFarm's own AI capabilities. Each release note focuses on explaining the value and impact of changes for end users, rather than technical implementation details.

For the complete structured changelog with commit links, see the [CHANGELOG.md](https://github.com/llama-farm/llamafarm/blob/main/CHANGELOG.md) in the repository.

## Stay Updated

- **GitHub Releases**: [github.com/llama-farm/llamafarm/releases](https://github.com/llama-farm/llamafarm/releases)
- **Reddit**: [r/LlamaFarm](https://www.reddit.com/r/LlamaFarm/)
- **Discord**: [Join our community](https://discord.gg/jtChvg8T)
