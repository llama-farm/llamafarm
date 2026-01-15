/**
 * Demo project configurations
 * Each demo is fully self-contained with config and files
 */

import type { ProjectType } from '../types/onboarding'

export interface DemoFile {
  path: string
  filename: string
  type: string
}

export interface DemoConfig {
  id: string
  name: string
  displayName: string
  description: string
  icon: string
  category: string
  estimatedTime: string

  // Which project types this demo is suitable for
  projectTypes: ProjectType[]

  // Paths relative to /demo-files/ (optional for inline data demos like classifier/anomaly)
  configPath?: string
  files?: DemoFile[]

  // Dataset info from config (optional for model-based demos)
  datasetName?: string

  // Sample questions to try (optional)
  sampleQuestions?: string[]

  // For model-based demos (classifier, anomaly) - links directly to model page with sample data
  modelType?: 'classifier' | 'anomaly'
  sampleDataId?: string // ID to pass to the model page to load sample data
}

export const AVAILABLE_DEMOS: DemoConfig[] = [
  {
    id: 'llama-encyclopedia',
    name: 'llama-expert',
    displayName: 'Llama & Alpaca Encyclopedia',
    description: 'Chat with a comprehensive encyclopedia about llama and alpaca care, breeding, health, and fiber production.',
    icon: '🦙',
    category: 'Agriculture & Animal Husbandry',
    estimatedTime: '~30 seconds',
    projectTypes: ['doc-qa', 'exploring'],

    configPath: '/demo-files/llama/llamafarm.yaml',
    files: [
      {
        path: '/demo-files/llama/llamas.md',
        filename: 'llamas.md',
        type: 'text/markdown'
      }
    ],

    datasetName: 'llama_encyclopedia',

    sampleQuestions: [
      'What are the key differences between llamas and alpacas?',
      'How do I tell if my alpaca is experiencing heat stress?',
      'What should I feed a pregnant female alpaca?',
      'Explain the difference between Huacaya and Suri fiber',
      'How do I train a llama for pack work?',
      'What are the signs of meningeal worm in camelids?'
    ]
  },

  {
    id: 'santa-helper',
    name: 'santa-helper',
    displayName: "Santa's Holiday Helper 🎄",
    description: "Ho ho ho! Chat with Santa about gift ideas, holiday traditions, festive recipes, and making Christmas magical!",
    icon: '🎅',
    category: 'Holiday & Seasonal',
    estimatedTime: '~30 seconds',
    projectTypes: ['doc-qa', 'exploring'],

    configPath: '/demo-files/santa/llamafarm.yaml',
    files: [
      {
        path: '/demo-files/santa/santa-knowledge.md',
        filename: 'santa-knowledge.md',
        type: 'text/markdown'
      }
    ],

    datasetName: 'santa_knowledge',

    sampleQuestions: [
      'What are the best gifts for a 10-year-old who loves science?',
      'How do I make the perfect hot cocoa for Christmas Eve?',
      'What are some fun Christmas traditions from around the world?',
      'Give me creative stocking stuffer ideas for teens',
      'How do I keep my Christmas tree fresh all season?',
      'What are some easy holiday cookies kids can help bake?'
    ]
  },

  // Classifier sample datasets
  {
    id: 'sentiment-classifier',
    name: 'sentiment',
    displayName: 'Sentiment Analysis',
    description: '3 classes, 200 examples - Classify text as positive, negative, or neutral',
    icon: '😊',
    category: 'Text Classification',
    estimatedTime: '~2 minutes',
    projectTypes: ['classifier'],
    modelType: 'classifier',
    sampleDataId: 'sentiment',
  },
  {
    id: 'expense-classifier',
    name: 'expense',
    displayName: 'Expense Reports',
    description: '5 classes, 200 examples - Categorize expense descriptions',
    icon: '💰',
    category: 'Text Classification',
    estimatedTime: '~2 minutes',
    projectTypes: ['classifier'],
    modelType: 'classifier',
    sampleDataId: 'expense',
  },

  // Anomaly detection sample datasets
  {
    id: 'fridge-temp-anomaly',
    name: 'fridge-temp',
    displayName: 'Fridge Temperature Data',
    description: 'Numeric, 1 column - Detect temperature anomalies',
    icon: '🌡️',
    category: 'Anomaly Detection',
    estimatedTime: '~1 minute',
    projectTypes: ['anomaly'],
    modelType: 'anomaly',
    sampleDataId: 'fridge-temp',
  },
  {
    id: 'biometric-anomaly',
    name: 'biometric',
    displayName: 'Biometric Data',
    description: 'Numeric, 5 columns - Monitor health metrics for outliers',
    icon: '❤️',
    category: 'Anomaly Detection',
    estimatedTime: '~1 minute',
    projectTypes: ['anomaly'],
    modelType: 'anomaly',
    sampleDataId: 'biometric',
  },
  {
    id: 'build-status-anomaly',
    name: 'build-status',
    displayName: 'Build Statuses',
    description: 'Text, 1 column - Detect unusual CI/CD patterns',
    icon: '🔧',
    category: 'Anomaly Detection',
    estimatedTime: '~1 minute',
    projectTypes: ['anomaly'],
    modelType: 'anomaly',
    sampleDataId: 'build-status',
  },
  {
    id: 'support-ticket-anomaly',
    name: 'support-ticket',
    displayName: 'Support Ticket Data',
    description: 'Text, 5 columns - Find unusual support patterns',
    icon: '🎫',
    category: 'Anomaly Detection',
    estimatedTime: '~1 minute',
    projectTypes: ['anomaly'],
    modelType: 'anomaly',
    sampleDataId: 'support-ticket',
  },

  // Easy to add more demos:
  // {
  //   id: 'legal-contracts',
  //   name: 'contract-analyzer',
  //   displayName: 'Legal Contract Analyzer',
  //   description: 'Analyze and understand complex legal contracts',
  //   icon: '⚖️',
  //   category: 'Legal',
  //   estimatedTime: '~45 seconds',
  //   configPath: '/demo-files/legal/llamafarm.yaml',
  //   files: [
  //     { path: '/demo-files/legal/sample-contract.pdf', filename: 'sample-contract.pdf', type: 'application/pdf' }
  //   ],
  //   datasetName: 'legal_contracts',
  //   sampleQuestions: [
  //     'What are the key terms of this contract?',
  //     'What are my obligations under this agreement?'
  //   ]
  // }
]

export function getDemoById(id: string): DemoConfig | undefined {
  return AVAILABLE_DEMOS.find(demo => demo.id === id)
}

export function getDemosByProjectType(projectType: ProjectType | null): DemoConfig[] {
  if (!projectType) return AVAILABLE_DEMOS
  return AVAILABLE_DEMOS.filter(demo => demo.projectTypes.includes(projectType))
}

/**
 * Type guard to check if a demo is a file-based demo (RAG/doc-qa)
 * These demos have configPath, files, and datasetName
 */
export interface FileBasedDemo extends DemoConfig {
  configPath: string
  files: DemoFile[]
  datasetName: string
  sampleQuestions: string[]
}

export function isFileBasedDemo(demo: DemoConfig): demo is FileBasedDemo {
  return !!demo.configPath && !!demo.files && !!demo.datasetName
}

/**
 * Get only file-based demos (for Data page import)
 */
export function getFileBasedDemos(): FileBasedDemo[] {
  return AVAILABLE_DEMOS.filter(isFileBasedDemo)
}

/**
 * Type guard to check if a demo is a model-based demo (classifier/anomaly)
 * These demos have modelType and sampleDataId
 */
export interface ModelBasedDemo extends DemoConfig {
  modelType: 'classifier' | 'anomaly'
  sampleDataId: string
}

export function isModelBasedDemo(demo: DemoConfig): demo is ModelBasedDemo {
  return !!demo.modelType && !!demo.sampleDataId
}

/**
 * Check if a project is a demo project (created via DemoModal)
 * Demo projects are stored in localStorage to persist across refreshes
 */
export function isDemoProject(projectName: string | null): boolean {
  if (!projectName) return false
  try {
    const demoProjects = JSON.parse(localStorage.getItem('lf_demo_projects') || '[]')
    return demoProjects.includes(projectName)
  } catch {
    return false
  }
}

/**
 * Get the demo config for a demo project by matching project name pattern
 * Demo projects are named like "llama-expert-1", "santa-helper-2", etc.
 */
export function getDemoConfigForProject(projectName: string | null): FileBasedDemo | undefined {
  if (!projectName) return undefined

  // Check if this is a demo project first
  if (!isDemoProject(projectName)) return undefined

  // Extract base name (e.g., "llama-expert" from "llama-expert-1")
  const baseName = projectName.replace(/-\d+$/, '')

  // Find the demo config by name
  return getFileBasedDemos().find(demo => demo.name === baseName)
}
