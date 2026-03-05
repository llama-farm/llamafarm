import type { SidebarsConfig } from '@docusaurus/plugin-content-docs'

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */
const sidebars: SidebarsConfig = {
  tutorialSidebar: [
    {
      type: 'category',
      label: 'Start Here',
      link: { type: 'doc', id: 'intro' },
      items: [
        { type: 'doc', id: 'quickstart/index', label: 'Quickstart' },
        { type: 'doc', id: 'concepts/index', label: 'Core Concepts' },
        { type: 'doc', id: 'troubleshooting/index', label: 'Troubleshooting & FAQ' },
      ],
    },
    {
      type: 'category',
      label: 'Desktop App',
      link: { type: 'doc', id: 'desktop-app/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'Designer Web UI',
      link: { type: 'doc', id: 'designer/index' },
      items: [
        { type: 'doc', id: 'designer/features', label: 'Features Overview' },
        { type: 'doc', id: 'designer/getting-started', label: 'Getting Started' },
        { type: 'doc', id: 'designer/dashboard', label: 'Dashboard' },
        { type: 'doc', id: 'designer/prompts', label: 'Prompts' },
        { type: 'doc', id: 'designer/data-management', label: 'Data Management' },
        { type: 'doc', id: 'designer/databases', label: 'Databases (RAG)' },
        { type: 'doc', id: 'designer/models', label: 'Models' },
        { type: 'doc', id: 'designer/chat-and-testing', label: 'Chat & Testing' },
        { type: 'doc', id: 'designer/vision', label: 'Vision' },
        { type: 'doc', id: 'designer/speech', label: 'Speech' },
        { type: 'doc', id: 'designer/ml-training', label: 'ML Training' },
        { type: 'doc', id: 'designer/addons', label: 'Addons' },
        { type: 'doc', id: 'designer/samples-and-demos', label: 'Samples & Demos' },
        { type: 'doc', id: 'designer/config-editor', label: 'Config Editor' },
        { type: 'doc', id: 'designer/development', label: 'Development' },
      ],
    },
    {
      type: 'category',
      label: 'CLI Reference',
      link: { type: 'doc', id: 'cli/index' },
      items: [
        { type: 'doc', id: 'cli/lf-init', label: 'lf init' },
        { type: 'doc', id: 'cli/lf-start', label: 'lf start' },
        { type: 'doc', id: 'cli/lf-chat', label: 'lf chat' },
        { type: 'doc', id: 'cli/lf-models', label: 'lf models' },
        { type: 'doc', id: 'cli/lf-datasets', label: 'lf datasets' },
        { type: 'doc', id: 'cli/lf-rag', label: 'lf rag' },
        { type: 'doc', id: 'cli/lf-projects', label: 'lf projects' },
        { type: 'doc', id: 'cli/lf-services', label: 'lf services' },
        { type: 'doc', id: 'cli/lf-version', label: 'lf version' },
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      link: { type: 'doc', id: 'configuration/index' },
      items: [
        { type: 'doc', id: 'configuration/example-configs', label: 'Example configs' },
      ],
    },
    {
      type: 'category',
      label: 'API Reference',
      link: { type: 'doc', id: 'api/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'RAG',
      link: { type: 'doc', id: 'rag/index' },
      items: [
        { type: 'doc', id: 'rag/databases', label: 'Databases (Vector Stores)' },
        { type: 'doc', id: 'rag/embedders', label: 'Embedders' },
        { type: 'doc', id: 'rag/parsers', label: 'Parsers' },
        { type: 'doc', id: 'rag/extractors', label: 'Extractors' },
        { type: 'doc', id: 'rag/retrieval-strategies', label: 'Retrieval Strategies' },
        { type: 'doc', id: 'rag/advanced-retrieval', label: 'Advanced Retrieval' },
      ],
    },
    {
      type: 'category',
      label: 'Models & Runtime',
      link: { type: 'doc', id: 'models/index' },
      items: [
        { type: 'doc', id: 'models/specialized-ml', label: 'Specialized ML Models' },
        { type: 'doc', id: 'models/vision', label: 'Vision Pipeline' },
        { type: 'doc', id: 'models/anomaly-detection', label: 'Anomaly Detection' },
        { type: 'doc', id: 'models/ml-addons', label: 'ML Addons' },
        { type: 'doc', id: 'models/kv-cache', label: 'KV Cache (Prompt Caching)' },
        { type: 'doc', id: 'models/vision-eval', label: 'Vision Evaluation' },
        { type: 'doc', id: 'models/vision-tracking', label: 'Vision Object Tracking' },
      ],
    },
    {
      type: 'category',
      label: 'Prompts',
      link: { type: 'doc', id: 'prompts/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'MCP (Tools)',
      link: { type: 'doc', id: 'mcp/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'Deployment',
      link: { type: 'doc', id: 'deployment/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'Industry Use Cases',
      link: { type: 'doc', id: 'use-cases/index' },
      items: [
        { type: 'doc', id: 'use-cases/pharmaceutical-fda', label: 'Pharmaceutical & FDA' },
      ],
    },
    {
      type: 'category',
      label: 'Examples',
      link: { type: 'doc', id: 'examples/index' },
      items: [
        { type: 'doc', id: 'examples/medical-records-helper', label: '🏥 Medical Records Helper' },
      ],
    },
    {
      type: 'category',
      label: 'Extending',
      link: { type: 'doc', id: 'extending/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'Contributing',
      link: { type: 'doc', id: 'contributing/index' },
      items: [],
    },
    {
      type: 'doc',
      id: 'changelog/index',
      label: 'Changelog',
    },
  ],
}

export default sidebars
