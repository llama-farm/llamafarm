#!/usr/bin/env node
/**
 * LlamaFarm Documentation MCP Server
 *
 * This MCP server exposes the LlamaFarm documentation to AI models,
 * allowing them to search, read, and navigate the docs programmatically.
 *
 * Tools provided:
 * - list_docs: List all documentation files with their paths and titles
 * - read_doc: Read the full content of a specific documentation file
 * - search_docs: Search for content across all documentation files
 * - get_toc: Get the table of contents / navigation structure
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  ListResourcesRequestSchema,
  ReadResourceRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";
import * as fs from "fs";
import * as path from "path";

// Path to documentation directory (relative to this script)
const DOCS_ROOT = path.resolve(import.meta.dirname, "../../docs");

interface DocFile {
  path: string;
  relativePath: string;
  title: string;
  description?: string;
}

interface SearchResult {
  file: string;
  title: string;
  matches: { line: number; content: string }[];
  score: number;
}

/**
 * Recursively find all markdown files in the docs directory
 */
function findMarkdownFiles(dir: string, basePath: string = ""): DocFile[] {
  const files: DocFile[] = [];

  if (!fs.existsSync(dir)) {
    return files;
  }

  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    const relativePath = path.join(basePath, entry.name);

    if (entry.isDirectory()) {
      files.push(...findMarkdownFiles(fullPath, relativePath));
    } else if (entry.name.endsWith(".md") || entry.name.endsWith(".mdx")) {
      const content = fs.readFileSync(fullPath, "utf-8");
      const title = extractTitle(content) || entry.name.replace(/\.mdx?$/, "");
      const description = extractDescription(content);

      files.push({
        path: fullPath,
        relativePath,
        title,
        description,
      });
    }
  }

  return files;
}

/**
 * Extract title from markdown frontmatter or first heading
 */
function extractTitle(content: string): string | null {
  // Check frontmatter
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
  if (frontmatterMatch) {
    const titleMatch = frontmatterMatch[1].match(/^title:\s*["']?(.+?)["']?\s*$/m);
    if (titleMatch) {
      return titleMatch[1];
    }
  }

  // Check first heading
  const headingMatch = content.match(/^#\s+(.+)$/m);
  if (headingMatch) {
    return headingMatch[1];
  }

  return null;
}

/**
 * Extract description from frontmatter or first paragraph
 */
function extractDescription(content: string): string | undefined {
  const frontmatterMatch = content.match(/^---\n([\s\S]*?)\n---/);
  if (frontmatterMatch) {
    const descMatch = frontmatterMatch[1].match(/^description:\s*["']?(.+?)["']?\s*$/m);
    if (descMatch) {
      return descMatch[1];
    }
  }
  return undefined;
}

/**
 * Search for a query across all documentation files
 */
function searchDocs(query: string, maxResults: number = 10): SearchResult[] {
  const files = findMarkdownFiles(DOCS_ROOT);
  const results: SearchResult[] = [];
  const queryLower = query.toLowerCase();
  const queryTerms = queryLower.split(/\s+/).filter((t) => t.length > 2);

  for (const file of files) {
    const content = fs.readFileSync(file.path, "utf-8");
    const lines = content.split("\n");
    const matches: { line: number; content: string }[] = [];
    let score = 0;

    // Check title match (higher weight)
    if (file.title.toLowerCase().includes(queryLower)) {
      score += 10;
    }

    // Check content matches
    for (let i = 0; i < lines.length; i++) {
      const lineLower = lines[i].toLowerCase();
      if (lineLower.includes(queryLower)) {
        matches.push({ line: i + 1, content: lines[i].trim() });
        score += 2;
      } else {
        // Check individual terms
        for (const term of queryTerms) {
          if (lineLower.includes(term)) {
            score += 0.5;
          }
        }
      }
    }

    if (score > 0) {
      results.push({
        file: file.relativePath,
        title: file.title,
        matches: matches.slice(0, 5), // Limit matches per file
        score,
      });
    }
  }

  // Sort by score descending
  results.sort((a, b) => b.score - a.score);

  return results.slice(0, maxResults);
}

/**
 * Read a specific documentation file
 */
function readDoc(filePath: string): string | null {
  // Normalize and validate path
  const normalizedPath = path.normalize(filePath);
  const fullPath = path.join(DOCS_ROOT, normalizedPath);

  // Security: ensure path is within docs directory
  if (!fullPath.startsWith(DOCS_ROOT)) {
    return null;
  }

  if (!fs.existsSync(fullPath)) {
    return null;
  }

  return fs.readFileSync(fullPath, "utf-8");
}

/**
 * Get table of contents structure
 */
function getTableOfContents(): object {
  const files = findMarkdownFiles(DOCS_ROOT);
  const toc: Record<string, any> = {};

  for (const file of files) {
    const parts = file.relativePath.split(path.sep);
    let current = toc;

    for (let i = 0; i < parts.length - 1; i++) {
      const part = parts[i];
      if (!current[part]) {
        current[part] = { _files: [] };
      }
      current = current[part];
    }

    const fileName = parts[parts.length - 1];
    if (!current._files) {
      current._files = [];
    }
    current._files.push({
      name: fileName,
      title: file.title,
      path: file.relativePath,
    });
  }

  return toc;
}

// Create the MCP server
const server = new Server(
  {
    name: "llamafarm-docs",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
      resources: {},
    },
  }
);

// Register tools
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "list_docs",
        description:
          "List all available LlamaFarm documentation files. Returns file paths, titles, and descriptions.",
        inputSchema: {
          type: "object",
          properties: {
            category: {
              type: "string",
              description:
                "Optional category to filter by (e.g., 'rag', 'configuration', 'cli')",
            },
          },
        },
      },
      {
        name: "read_doc",
        description:
          "Read the full content of a specific documentation file. Use list_docs first to find available files.",
        inputSchema: {
          type: "object",
          properties: {
            path: {
              type: "string",
              description:
                "Relative path to the documentation file (e.g., 'rag/parsers.md')",
            },
          },
          required: ["path"],
        },
      },
      {
        name: "search_docs",
        description:
          "Search across all LlamaFarm documentation for specific content, configuration options, or features.",
        inputSchema: {
          type: "object",
          properties: {
            query: {
              type: "string",
              description: "Search query (e.g., 'PDFParser', 'ChromaStore', 'embedder configuration')",
            },
            max_results: {
              type: "number",
              description: "Maximum number of results to return (default: 10)",
            },
          },
          required: ["query"],
        },
      },
      {
        name: "get_toc",
        description:
          "Get the documentation table of contents / navigation structure. Useful for understanding what documentation is available.",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

// Handle tool calls
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "list_docs": {
      const files = findMarkdownFiles(DOCS_ROOT);
      let filtered = files;

      if (args?.category) {
        const category = String(args.category).toLowerCase();
        filtered = files.filter((f) =>
          f.relativePath.toLowerCase().includes(category)
        );
      }

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(
              filtered.map((f) => ({
                path: f.relativePath,
                title: f.title,
                description: f.description,
              })),
              null,
              2
            ),
          },
        ],
      };
    }

    case "read_doc": {
      const filePath = String(args?.path || "");
      const content = readDoc(filePath);

      if (!content) {
        return {
          content: [
            {
              type: "text",
              text: `Error: Documentation file not found: ${filePath}`,
            },
          ],
          isError: true,
        };
      }

      return {
        content: [
          {
            type: "text",
            text: content,
          },
        ],
      };
    }

    case "search_docs": {
      const query = String(args?.query || "");
      const maxResults = Number(args?.max_results) || 10;

      if (!query) {
        return {
          content: [
            {
              type: "text",
              text: "Error: Search query is required",
            },
          ],
          isError: true,
        };
      }

      const results = searchDocs(query, maxResults);

      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(results, null, 2),
          },
        ],
      };
    }

    case "get_toc": {
      const toc = getTableOfContents();
      return {
        content: [
          {
            type: "text",
            text: JSON.stringify(toc, null, 2),
          },
        ],
      };
    }

    default:
      return {
        content: [
          {
            type: "text",
            text: `Unknown tool: ${name}`,
          },
        ],
        isError: true,
      };
  }
});

// Also expose docs as resources (optional, for clients that prefer resources)
server.setRequestHandler(ListResourcesRequestSchema, async () => {
  const files = findMarkdownFiles(DOCS_ROOT);

  return {
    resources: files.map((f) => ({
      uri: `docs://${f.relativePath}`,
      name: f.title,
      description: f.description,
      mimeType: "text/markdown",
    })),
  };
});

server.setRequestHandler(ReadResourceRequestSchema, async (request) => {
  const uri = request.params.uri;
  const match = uri.match(/^docs:\/\/(.+)$/);

  if (!match) {
    throw new Error(`Invalid resource URI: ${uri}`);
  }

  const content = readDoc(match[1]);

  if (!content) {
    throw new Error(`Resource not found: ${uri}`);
  }

  return {
    contents: [
      {
        uri,
        mimeType: "text/markdown",
        text: content,
      },
    ],
  };
});

// Start the server
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("LlamaFarm Docs MCP Server running on stdio");
}

main().catch(console.error);
