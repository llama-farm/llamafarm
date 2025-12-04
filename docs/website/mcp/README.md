# LlamaFarm Documentation MCP Server

This MCP (Model Context Protocol) server exposes the LlamaFarm documentation to AI models, allowing them to search, read, and navigate the docs programmatically.

## Overview

The MCP server provides these tools for AI assistants:

| Tool | Description |
|------|-------------|
| `list_docs` | List all documentation files with paths and titles |
| `read_doc` | Read the full content of a specific documentation file |
| `search_docs` | Search for content across all documentation files |
| `get_toc` | Get the table of contents / navigation structure |

## Installation

```bash
cd docs/website/mcp
npm install
npm run build
```

## Usage

### With Claude Desktop

Add to your Claude Desktop configuration (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

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

### With Claude Code

Add to your project's `.mcp.json`:

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

### With Cursor

Add to `.cursor/mcp.json`:

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

### With LlamaFarm (as an MCP client)

Add to your `llamafarm.yaml`:

```yaml
mcp:
  servers:
    - name: llamafarm-docs
      transport: stdio
      command: node
      args:
        - docs/website/mcp/dist/index.js
```

Then reference it in your model config:

```yaml
runtime:
  models:
    - name: assistant
      provider: ollama
      model: llama3.1:8b
      mcp_servers: [llamafarm-docs]
```

## Development

```bash
# Run in development mode (with hot reload)
npm run dev

# Inspect with MCP Inspector
npm run inspect

# Build for production
npm run build
```

## Tool Examples

### list_docs

List all available documentation:

```json
{
  "name": "list_docs",
  "arguments": {}
}
```

Filter by category:

```json
{
  "name": "list_docs",
  "arguments": {
    "category": "rag"
  }
}
```

### read_doc

Read a specific documentation file:

```json
{
  "name": "read_doc",
  "arguments": {
    "path": "rag/parsers.md"
  }
}
```

### search_docs

Search for content:

```json
{
  "name": "search_docs",
  "arguments": {
    "query": "PDFParser chunk_size",
    "max_results": 5
  }
}
```

### get_toc

Get documentation structure:

```json
{
  "name": "get_toc",
  "arguments": {}
}
```

## How It Works

1. **File Discovery**: The server recursively scans the `docs/` directory for markdown files
2. **Metadata Extraction**: Titles and descriptions are extracted from frontmatter or headings
3. **Search**: Full-text search across all files with relevance scoring
4. **Security**: Path traversal is prevented by validating all paths stay within the docs directory

## Resources

The server also exposes documentation as MCP resources with URIs like `docs://rag/parsers.md`, allowing clients that prefer the resources API to access docs that way.

## Architecture

```
docs/website/mcp/
├── src/
│   └── index.ts      # Main MCP server implementation
├── dist/             # Compiled JavaScript (after build)
├── package.json
├── tsconfig.json
└── README.md
```

The server uses:
- `@modelcontextprotocol/sdk` - Official MCP SDK for TypeScript
- `StdioServerTransport` - Standard I/O transport for local connections

## License

Apache-2.0 - Same as LlamaFarm
