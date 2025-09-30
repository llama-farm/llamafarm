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
      label: 'CLI Reference',
      link: { type: 'doc', id: 'cli/index' },
      items: [
        { type: 'doc', id: 'cli/lf-init', label: 'lf init' },
        { type: 'doc', id: 'cli/lf-start', label: 'lf start' },
        { type: 'doc', id: 'cli/lf-chat', label: 'lf chat' },
        { type: 'doc', id: 'cli/lf-datasets', label: 'lf datasets' },
        { type: 'doc', id: 'cli/lf-rag', label: 'lf rag' },
        { type: 'doc', id: 'cli/lf-projects', label: 'lf projects' },
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
      label: 'RAG',
      link: { type: 'doc', id: 'rag/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'Models & Runtime',
      link: { type: 'doc', id: 'models/index' },
      items: [],
    },
    {
      type: 'category',
      label: 'Prompts',
      link: { type: 'doc', id: 'prompts/index' },
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
      label: 'Examples',
      link: { type: 'doc', id: 'examples/index' },
      items: [],
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
  ],
}

export default sidebars
