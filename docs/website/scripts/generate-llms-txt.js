#!/usr/bin/env node
/**
 * Generate llms.txt and llms-full.txt for AI crawlers
 *
 * llms.txt - Concise index with links to documentation sections
 * llms-full.txt - Complete documentation in a single file
 *
 * See https://llmstxt.org for specification
 */

const fs = require('fs');
const path = require('path');

const DOCS_DIR = path.join(__dirname, '../docs');
const STATIC_DIR = path.join(__dirname, '../static');
const BASE_URL = 'https://docs.llamafarm.dev';

// Document order based on sidebar structure (most important first)
const DOC_ORDER = [
  'intro.md',
  'quickstart/index.md',
  'concepts/index.md',
  'troubleshooting/index.md',
  'desktop-app/index.md',
  'designer/index.md',
  'designer/features.md',
  'designer/development.md',
  'cli/index.md',
  'cli/lf-init.md',
  'cli/lf-start.md',
  'cli/lf-chat.md',
  'cli/lf-models.md',
  'cli/lf-datasets.md',
  'cli/lf-rag.md',
  'cli/lf-projects.md',
  'cli/lf-services.md',
  'cli/lf-version.md',
  'configuration/index.md',
  'configuration/example-configs.md',
  'api/index.md',
  'rag/index.md',
  'rag/databases.md',
  'rag/embedders.md',
  'rag/parsers.md',
  'rag/extractors.md',
  'rag/retrieval-strategies.md',
  'rag/advanced-retrieval.md',
  'models/index.md',
  'models/specialized-ml.md',
  'models/anomaly-detection.md',
  'prompts/index.md',
  'mcp/index.md',
  'deployment/index.md',
  'use-cases/index.md',
  'use-cases/pharmaceutical-fda.md',
  'examples/index.md',
  'examples/medical-records-helper.md',
  'extending/index.md',
  'contributing/index.md',
];

// Section descriptions for llms.txt
const SECTION_DESCRIPTIONS = {
  'intro.md': 'Introduction and overview of LlamaFarm',
  'quickstart/index.md': 'Get started in 5 minutes',
  'concepts/index.md': 'Core architecture and concepts',
  'troubleshooting/index.md': 'Common issues and solutions',
  'desktop-app/index.md': 'Desktop application guide',
  'designer/index.md': 'Designer web UI overview',
  'cli/index.md': 'CLI command reference',
  'configuration/index.md': 'Configuration file reference (llamafarm.yaml)',
  'api/index.md': 'REST API reference',
  'rag/index.md': 'RAG (Retrieval-Augmented Generation) guide',
  'rag/databases.md': 'Vector database configuration',
  'rag/embedders.md': 'Embedding model configuration',
  'rag/parsers.md': 'Document parser configuration',
  'rag/retrieval-strategies.md': 'Retrieval strategy options',
  'models/index.md': 'Model providers and runtime configuration',
  'models/specialized-ml.md': 'Classifiers, NER, OCR, reranking',
  'models/anomaly-detection.md': 'Anomaly detection backends',
  'prompts/index.md': 'Prompt templates and management',
  'mcp/index.md': 'Model Context Protocol (MCP) tools',
  'deployment/index.md': 'Production deployment guide',
  'use-cases/index.md': 'Industry use cases',
  'examples/index.md': 'Example projects and tutorials',
  'extending/index.md': 'Extending LlamaFarm with custom components',
  'contributing/index.md': 'Contributing to LlamaFarm',
};

function getTitle(content, filename) {
  // Try to extract title from frontmatter or first heading
  const frontmatterMatch = content.match(/^---[\s\S]*?title:\s*["']?([^"'\n]+)["']?[\s\S]*?---/);
  if (frontmatterMatch) {
    return frontmatterMatch[1].trim();
  }

  const headingMatch = content.match(/^#\s+(.+)$/m);
  if (headingMatch) {
    return headingMatch[1].trim();
  }

  // Fallback to filename
  return path.basename(filename, '.md').replace(/-/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function cleanContent(content, skipFirstHeading = false) {
  // Remove frontmatter
  content = content.replace(/^---[\s\S]*?---\n*/m, '');

  // Remove Docusaurus admonitions but keep content
  content = content.replace(/:::(info|tip|note|warning|danger|caution)\s*(.*?)\n([\s\S]*?):::/g, (match, type, title, body) => {
    const prefix = type.toUpperCase();
    return title ? `**${prefix}: ${title}**\n${body}` : `**${prefix}:**\n${body}`;
  });

  // Remove import statements
  content = content.replace(/^import\s+.*$/gm, '');

  // Remove JSX/MDX components with inline styles (like the download buttons)
  content = content.replace(/<div[^>]*style=\{\{[\s\S]*?\}\}[^>]*>[\s\S]*?<\/div>/g, '');
  content = content.replace(/<a[^>]*style=\{\{[\s\S]*?\}\}[^>]*>[\s\S]*?<\/a>/g, '');

  // Remove JSX/MDX components (keep simple markdown)
  content = content.replace(/<[A-Z][a-zA-Z]*[^>]*>[\s\S]*?<\/[A-Z][a-zA-Z]*>/g, '');
  content = content.replace(/<[A-Z][a-zA-Z]*[^>]*\/>/g, '');

  // Optionally remove the first h1 heading (since we add it ourselves)
  if (skipFirstHeading) {
    content = content.replace(/^#\s+[^\n]+\n*/m, '');
  }

  // Clean up multiple blank lines
  content = content.replace(/\n{3,}/g, '\n\n');

  return content.trim();
}

function docPathToUrl(docPath) {
  // Convert file path to URL
  let url = docPath.replace(/\.md$/, '').replace(/\/index$/, '');
  if (url === 'intro') url = '';
  return `${BASE_URL}/docs/${url}`;
}

function generateLlmsTxt() {
  const lines = [
    '# LlamaFarm',
    '',
    '> Edge AI platform for running LLMs, RAG, classifiers, and document processing locally. No cloud required. Your data never leaves your device.',
    '',
    'LlamaFarm provides enterprise AI capabilities on your own hardware: RAG (retrieval-augmented generation), custom text classifiers (SetFit with 8-16 training examples), anomaly detection (Isolation Forest, One-Class SVM, LOF, Autoencoders), OCR/document extraction (Surya, EasyOCR, PaddleOCR), and multi-model runtime support (Ollama, OpenAI-compatible, Universal Runtime).',
    '',
    '## Quick Start',
    '',
    '- [Quickstart](https://docs.llamafarm.dev/docs/quickstart): Install CLI and run your first RAG query in 5 minutes',
    '- [Core Concepts](https://docs.llamafarm.dev/docs/concepts): Architecture overview - Server, RAG Worker, Universal Runtime',
    '',
    '## Common API Patterns',
    '',
    'Base URL: `http://localhost:8000`',
    '',
    '### Chat Completions (OpenAI-compatible)',
    '',
    '```bash',
    'POST /v1/projects/{namespace}/{project}/chat/completions',
    '',
    'curl -X POST http://localhost:8000/v1/projects/default/my-project/chat/completions \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "messages": [{"role": "user", "content": "Hello"}],',
    '    "stream": false,',
    '    "rag_enabled": true,',
    '    "database": "main_db",',
    '    "rag_top_k": 5',
    '  }\'',
    '```',
    '',
    'Key request fields:',
    '- `messages`: Array of {role, content} - required',
    '- `stream`: true for SSE streaming, false for complete response',
    '- `rag_enabled`: true/false to enable RAG retrieval',
    '- `database`: Which RAG database to query',
    '- `rag_top_k`: Number of documents to retrieve (default: 5)',
    '- `rag_score_threshold`: Minimum similarity score (0-1)',
    '- `model`: Select specific model from config',
    '- `think`: true to enable chain-of-thought reasoning',
    '- `thinking_budget`: Max tokens for thinking (default: 1024)',
    '',
    'Session headers:',
    '- `X-Session-ID`: Reuse session for conversation history',
    '- `X-No-Session`: Stateless mode, no history saved',
    '',
    '### Create RAG Database',
    '',
    '```bash',
    'POST /v1/projects/{namespace}/{project}/rag/databases',
    '',
    'curl -X POST http://localhost:8000/v1/projects/default/my-project/rag/databases \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "name": "my_database",',
    '    "type": "ChromaStore",',
    '    "embedding_strategies": [{',
    '      "name": "default",',
    '      "type": "UniversalEmbedder",',
    '      "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"}',
    '    }],',
    '    "retrieval_strategies": [{',
    '      "name": "semantic",',
    '      "type": "BasicSimilarityStrategy",',
    '      "config": {"top_k": 5}',
    '    }]',
    '  }\'',
    '```',
    '',
    '### Create Dataset',
    '',
    '```bash',
    'POST /v1/projects/{namespace}/{project}/datasets',
    '',
    'curl -X POST http://localhost:8000/v1/projects/default/my-project/datasets \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "name": "research_papers",',
    '    "database": "main_db",',
    '    "data_processing_strategy": "default"',
    '  }\'',
    '```',
    '',
    '### Upload File to Dataset',
    '',
    '```bash',
    'POST /v1/projects/{namespace}/{project}/datasets/{dataset}/data',
    '',
    'curl -X POST http://localhost:8000/v1/projects/default/my-project/datasets/research_papers/data \\',
    '  -F "file=@document.pdf"',
    '```',
    '',
    '### Process Dataset (Ingest into RAG)',
    '',
    '```bash',
    'POST /v1/projects/{namespace}/{project}/datasets/{dataset}/actions',
    '',
    'curl -X POST http://localhost:8000/v1/projects/default/my-project/datasets/research_papers/actions \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{"action": "ingest"}\'',
    '```',
    '',
    '### Query RAG Directly',
    '',
    '```bash',
    'POST /v1/projects/{namespace}/{project}/rag/query',
    '',
    'curl -X POST http://localhost:8000/v1/projects/default/my-project/rag/query \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "query": "What are the key findings?",',
    '    "database": "main_db",',
    '    "top_k": 5',
    '  }\'',
    '```',
    '',
    '### Train Classifier (SetFit few-shot)',
    '',
    '```bash',
    'POST /v1/ml/classifier/fit',
    '',
    'curl -X POST http://localhost:8000/v1/ml/classifier/fit \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "model": "ticket-router",',
    '    "training_data": [',
    '      {"text": "Cannot log in", "label": "auth"},',
    '      {"text": "Password reset", "label": "auth"},',
    '      {"text": "Charged twice", "label": "billing"},',
    '      {"text": "Refund request", "label": "billing"}',
    '    ]',
    '  }\'',
    '```',
    '',
    '### Classify Text',
    '',
    '```bash',
    'POST /v1/ml/classifier/predict',
    '',
    'curl -X POST http://localhost:8000/v1/ml/classifier/predict \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "model": "ticket-router",',
    '    "texts": ["I forgot my password", "Need invoice copy"]',
    '  }\'',
    '```',
    '',
    '### Anomaly Detection',
    '',
    '```bash',
    '# Train on normal data',
    'POST /v1/ml/anomaly/fit',
    '',
    'curl -X POST http://localhost:8000/v1/ml/anomaly/fit \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "model": "latency-monitor",',
    '    "backend": "isolation_forest",',
    '    "data": [[50], [55], [48], [52], [51]]',
    '  }\'',
    '',
    '# Detect anomalies',
    'POST /v1/ml/anomaly/detect',
    '',
    'curl -X POST http://localhost:8000/v1/ml/anomaly/detect \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "model": "latency-monitor",',
    '    "data": [[52], [500], [49]],',
    '    "threshold": 0.5',
    '  }\'',
    '```',
    '',
    'Backends: `isolation_forest`, `one_class_svm`, `local_outlier_factor`, `autoencoder`',
    '',
    '### OCR (Text Extraction)',
    '',
    '```bash',
    'POST /v1/vision/ocr',
    '',
    'curl -X POST http://localhost:8000/v1/vision/ocr \\',
    '  -F "file=@document.pdf" \\',
    '  -F "model=surya"',
    '```',
    '',
    'OCR backends: `surya` (best), `easyocr`, `paddleocr`, `tesseract`',
    '',
    '### Embeddings',
    '',
    '```bash',
    'POST /v1/embeddings',
    '',
    'curl -X POST http://localhost:8000/v1/embeddings \\',
    '  -H "Content-Type: application/json" \\',
    '  -d \'{',
    '    "input": ["Hello world", "How are you?"],',
    '    "model": "sentence-transformers/all-MiniLM-L6-v2"',
    '  }\'',
    '```',
    '',
    '## Configuration',
    '',
    '- [Configuration Guide](https://docs.llamafarm.dev/docs/configuration): Complete llamafarm.yaml schema reference',
    '',
    '## RAG (Retrieval-Augmented Generation)',
    '',
    '- [RAG Overview](https://docs.llamafarm.dev/docs/rag): End-to-end document ingestion and querying',
    '- [Databases](https://docs.llamafarm.dev/docs/rag/databases): ChromaStore, QdrantStore vector databases',
    '- [Embedders](https://docs.llamafarm.dev/docs/rag/embedders): HuggingFace, Ollama, OpenAI, SentenceTransformer, Universal',
    '- [Parsers](https://docs.llamafarm.dev/docs/rag/parsers): PDF, CSV, Excel, DOCX, Markdown, Text parsers',
    '- [Retrieval Strategies](https://docs.llamafarm.dev/docs/rag/retrieval-strategies): Vector, Hybrid, BM25, Reranked, Graph retrieval',
    '',
    '## ML Capabilities',
    '',
    '- [Specialized ML](https://docs.llamafarm.dev/docs/models/specialized-ml): Classifiers, NER, OCR, reranking',
    '- [Anomaly Detection](https://docs.llamafarm.dev/docs/models/anomaly-detection): Train and detect with multiple backends',
    '- [Models & Runtime](https://docs.llamafarm.dev/docs/models): Universal Runtime, Ollama, OpenAI provider configuration',
    '',
    '## CLI Commands',
    '',
    '- `lf init <name>` - Create new project with llamafarm.yaml',
    '- `lf start` - Start all services (Server, Universal Runtime)',
    '- `lf chat` - Interactive chat with optional RAG',
    '- `lf datasets create -s <strategy> -b <db> <name>` - Create dataset',
    '- `lf datasets upload <name> <files>` - Upload files to dataset',
    '- `lf datasets process <name>` - Process/ingest dataset',
    '- `lf rag query --database <db> "query"` - Query RAG database',
    '',
    '## Optional',
    '',
    '- [MCP (Tools)](https://docs.llamafarm.dev/docs/mcp): Model Context Protocol for AI access to filesystems, databases',
    '- [Examples](https://docs.llamafarm.dev/docs/examples): FDA RAG assistant, Medical Records helper demos',
    '- [Extending](https://docs.llamafarm.dev/docs/extending): Add custom parsers, embedders, providers',
    '- [Deployment](https://docs.llamafarm.dev/docs/deployment): Production deployment with systemd, PM2, launchd',
    '- [Troubleshooting](https://docs.llamafarm.dev/docs/troubleshooting): Common issues and solutions',
    '',
  ];

  return lines.join('\n');
}

function generateLlmsFullTxt() {
  const lines = [
    '# LlamaFarm - Complete Documentation',
    '',
    '> Edge AI platform for running LLMs, RAG, classifiers, and document processing locally. No cloud required. Your data never leaves your device.',
    '',
    'This file contains the complete LlamaFarm documentation for AI consumption.',
    '',
    '---',
    '',
  ];

  let docsProcessed = 0;
  let totalLines = 0;

  for (const docPath of DOC_ORDER) {
    const fullPath = path.join(DOCS_DIR, docPath);
    if (fs.existsSync(fullPath)) {
      const content = fs.readFileSync(fullPath, 'utf-8');
      const cleaned = cleanContent(content, true);
      const title = getTitle(content, docPath);

      lines.push(`# ${title}`);
      lines.push('');
      lines.push(cleaned);
      lines.push('');
      lines.push('---');
      lines.push('');

      docsProcessed++;
      totalLines += cleaned.split('\n').length;
    }
  }

  // Also include any docs not in DOC_ORDER
  const allDocs = getAllMarkdownFiles(DOCS_DIR);
  for (const docPath of allDocs) {
    const relativePath = path.relative(DOCS_DIR, docPath).replace(/\\/g, '/');
    if (!DOC_ORDER.includes(relativePath)) {
      const content = fs.readFileSync(docPath, 'utf-8');
      const cleaned = cleanContent(content, true);
      const title = getTitle(content, relativePath);

      lines.push(`# ${title}`);
      lines.push('');
      lines.push(cleaned);
      lines.push('');
      lines.push('---');
      lines.push('');

      docsProcessed++;
      totalLines += cleaned.split('\n').length;
    }
  }

  console.log(`Processed ${docsProcessed} documents (~${totalLines} content lines)`);

  return lines.join('\n');
}

function getAllMarkdownFiles(dir, files = []) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      getAllMarkdownFiles(fullPath, files);
    } else if (entry.name.endsWith('.md')) {
      files.push(fullPath);
    }
  }
  return files;
}

function main() {
  console.log('Generating llms.txt files...');

  // Ensure static directory exists
  if (!fs.existsSync(STATIC_DIR)) {
    fs.mkdirSync(STATIC_DIR, { recursive: true });
  }

  // Generate llms.txt (concise index)
  const llmsTxt = generateLlmsTxt();
  const llmsTxtPath = path.join(STATIC_DIR, 'llms.txt');
  fs.writeFileSync(llmsTxtPath, llmsTxt);
  console.log(`Written: ${llmsTxtPath} (${llmsTxt.split('\n').length} lines)`);

  // Generate llms-full.txt (complete docs)
  const llmsFullTxt = generateLlmsFullTxt();
  const llmsFullTxtPath = path.join(STATIC_DIR, 'llms-full.txt');
  fs.writeFileSync(llmsFullTxtPath, llmsFullTxt);
  console.log(`Written: ${llmsFullTxtPath} (${llmsFullTxt.split('\n').length} lines)`);

  console.log('Done!');
}

main();
