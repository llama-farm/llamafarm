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
      label: 'Overview',
      link: { type: 'doc', id: 'intro' },
      items: [
        {
          type: 'doc',
          id: 'overview/getting-started',
          label: 'Getting started',
        },
        { type: 'doc', id: 'overview/features', label: 'Features' },
        { type: 'doc', id: 'overview/architecture', label: 'Architecture' },
        { type: 'doc', id: 'overview/why-llamafarm', label: 'Why LlamaFarm' },
        { type: 'doc', id: 'overview/contributing', label: 'Contributing' },
        { type: 'doc', id: 'overview/community', label: 'Community' },
        { type: 'doc', id: 'overview/license', label: 'License' },
        { type: 'doc', id: 'overview/credits', label: 'Credits' },
      ],
    },
    {
      type: 'category',
      label: 'RAG system',
      link: { type: 'doc', id: 'rag/index' },
      items: [
        { type: 'doc', id: 'rag/quick-start', label: 'Quick start' },
        {
          type: 'doc',
          id: 'rag/local-only-extractors',
          label: 'Local only extractors',
        },
        { type: 'doc', id: 'rag/architecture', label: 'Architecture' },
        { type: 'doc', id: 'rag/data-sources', label: 'Data sources' },
        {
          type: 'doc',
          id: 'rag/stores-and-indexing',
          label: 'Stores & indexing',
        },
        {
          type: 'doc',
          id: 'rag/retrieval-strategies',
          label: 'Retrieval strategies',
        },
      ],
    },
    {
      type: 'category',
      label: 'Prompts',
      link: { type: 'doc', id: 'prompts/index' },
      items: [
        { type: 'doc', id: 'prompts/templates', label: 'Templates' },
        { type: 'doc', id: 'prompts/evaluation', label: 'Evaluation' },
      ],
    },
    {
      type: 'category',
      label: 'Models',
      link: { type: 'doc', id: 'models/index' },
      items: [
        { type: 'doc', id: 'models/providers', label: 'Providers' },
        { type: 'doc', id: 'models/adapters', label: 'Adapters' },
      ],
    },
    {
      type: 'category',
      label: 'Configuration',
      link: { type: 'doc', id: 'configuration/index' },
      items: [
        { type: 'doc', id: 'configuration/example-configs', label: 'Examples' },
      ],
    },
    {
      type: 'category',
      label: 'Deployment',
      link: { type: 'doc', id: 'deployment/index' },
      items: [
        {
          type: 'doc',
          id: 'deployment/docker-compose',
          label: 'Docker Compose',
        },
        { type: 'doc', id: 'deployment/kubernetes', label: 'Kubernetes' },
      ],
    },
  ],
}

export default sidebars
