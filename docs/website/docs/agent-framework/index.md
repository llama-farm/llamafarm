# Agent Framework

The LlamaFarm Agent Framework allows you to write **Active Agents** and **Inline Tools** in pure Python, running natively within the LlamaFarm platform.

Unlike traditional setups where you wrap your code in complex servers (like MCP) or manage your own Docker containers, the Agent Framework provides a **"Magic Runtime"** that automatically discovers and manages your code.

## Key Features

- **Zero-Boilerplate Tools**: Define tools with a simple `@tool` decorator.
- **Active Agents**: Create long-running background agents (`class MyAgent(Agent)`) that can monitor, poll, and act autonomously.
- **Unified Runtime**: Runs your Tools and Agents side-by-side in a high-performance asyncio environment.
- **Real AI Integration**: Built-in `self.client` provides direct, authenticated access to LlamaFarm's Universal Runtime (Inference, Training, RAG).

## Use Cases

1.  **Intelligent Monitoring**: Watch data streams, detect anomalies using AI, and trigger alerts (e.g., SRE Agents, Bio-Sentinels).
2.  **Cron Jobs on Steroids**: Run periodic tasks that have access to LLM reasoning.
3.  **Data Pipelines**: Fetch data from external APIs and pipe it into LlamaFarm datasets automatically.

## Next Steps

- [Quickstart](quickstart.md): Write your first Tool in 5 minutes.
- [Building Agents](agents.md): Learn how to build active monitoring agents.
- [Architecture](architecture.md): Understand how the Magic Runtime works.
