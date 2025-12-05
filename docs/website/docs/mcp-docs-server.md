---
title: Documentation MCP Server
sidebar_position: 15
---

# Documentation MCP Server

LlamaFarm includes an MCP server that exposes the documentation to AI models, allowing them to search, read, and navigate the docs programmatically.

## Why Use the MCP Server?

- **AI-Assisted Development**: Let Claude, GPT, or other AI assistants query LlamaFarm docs directly
- **IDE Integration**: Use with Cursor, Claude Code, or other MCP-enabled tools
- **Accurate Answers**: AI gets real-time access to current documentation
- **Context Retrieval**: Models can search for relevant configuration examples

## Quick Setup

### 1. Build the Server

```bash
cd docs/website/mcp
npm install
npm run build
```

### 2. Configure Your AI Tool

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "llamafarm-docs": {
      "command": "node",
      "args": ["/path/to/llamafarm/docs/website/mcp/dist/index.js"]
    }
  }
}
```

**Claude Code** (`.mcp.json` in project root):

```json
{
  "mcpServers": {
    "llamafarm-docs": {
      "command": "node",
      "args": ["docs/website/mcp/dist/index.js"]
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "llamafarm-docs": {
      "type": "stdio",
      "command": "node",
      "args": ["docs/website/mcp/dist/index.js"]
    }
  }
}
```

## Available Tools

### list_docs

List all available documentation files:

```
Tool: list_docs
Arguments: { "category": "rag" }  // Optional filter
```

Returns file paths, titles, and descriptions.

### read_doc

Read a specific documentation file:

```
Tool: read_doc
Arguments: { "path": "rag/parsers.md" }
```

Returns the full markdown content.

### search_docs

Search across all documentation:

```
Tool: search_docs
Arguments: {
  "query": "PDFParser configuration",
  "max_results": 5
}
```

Returns matching files with line numbers and context.

### get_toc

Get the documentation structure:

```
Tool: get_toc
Arguments: {}
```

Returns a hierarchical table of contents.

## Example Interactions

Once configured, you can ask your AI assistant questions like:

> "Search the LlamaFarm docs for how to configure CrossEncoderRerankedStrategy"

The AI will use the `search_docs` tool to find relevant documentation, then `read_doc` to get the full content.

> "List all the RAG documentation pages"

The AI will call `list_docs` with `category: "rag"`.

> "Show me the parsers reference documentation"

The AI will call `read_doc` with `path: "rag/parsers.md"`.

## Using with LlamaFarm

You can also configure LlamaFarm itself to use the docs MCP server:

```yaml
mcp:
  servers:
    - name: llamafarm-docs
      transport: stdio
      command: node
      args:
        - docs/website/mcp/dist/index.js

runtime:
  models:
    - name: assistant
      provider: ollama
      model: llama3.1:8b
      mcp_servers: [llamafarm-docs]
```

This allows LlamaFarm's chat agents to query the documentation when answering questions.

## Development

To modify the MCP server:

```bash
cd docs/website/mcp

# Development mode with hot reload
npm run dev

# Inspect with MCP Inspector
npm run inspect

# Build for production
npm run build
```

## How It Works

1. The server scans `docs/website/docs/` for all markdown files
2. Extracts titles and descriptions from frontmatter
3. Provides full-text search with relevance scoring
4. Exposes docs as both MCP tools and resources

## Source Code

See `docs/website/mcp/` for the full implementation.
