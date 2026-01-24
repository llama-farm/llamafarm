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
    displayName: "Santa's Holiday Helper",
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

{
    id: 'gardening-guide',
    name: 'gardening-guide',
    displayName: 'US Gardening Guide',
    description: 'Get personalized gardening advice by zone - what to plant, when to plant, pest help, and troubleshooting for vegetables, flowers, and containers.',
    icon: '🌱',
    category: 'Home & Garden',
    estimatedTime: '~30 seconds',
    projectTypes: ['doc-qa', 'exploring'],

    configPath: '/demo-files/gardening/llamafarm.yaml',
    files: [
      { path: '/demo-files/gardening/gardening-guide.md', filename: 'gardening-guide.md', type: 'text/markdown' }
    ],

    datasetName: 'gardening_knowledge',

    sampleQuestions: [
      'What vegetables should I plant in Atlanta in March?',
      'My tomato leaves have brown spots with yellow halos - what is it?',
      'How do I start a compost pile?',
      'What can I grow on a shady apartment balcony?',
      'When should I plant garlic in Zone 5?',
      'Why are my cucumber leaves covered in white powder?',
      'What flowers will bloom all summer in full sun?',
      'How often should I water tomatoes in containers?'
    ]
  },

  {
    id: 'home-repair-helper',
    name: 'home-repair-helper',
    displayName: 'Home Repair Helper',
    description: 'Get help with common home repairs - plumbing, electrical, drywall, painting, appliance troubleshooting, and when to call a pro.',
    icon: '🛠️',
    category: 'Home & Garden',
    estimatedTime: '~30 seconds',
    projectTypes: ['doc-qa', 'exploring'],

    configPath: '/demo-files/home-repairs/llamafarm.yaml',
    files: [
      { path: '/demo-files/home-repairs/home-repair-guide.md', filename: 'home-repair-guide.md', type: 'text/markdown' }
    ],

    datasetName: 'home_repair_knowledge',

    sampleQuestions: [
      'My toilet keeps running - how do I fix it?',
      'Is it safe to replace an electrical outlet myself?',
      'How do I patch a hole in drywall?',
      'My dishwasher isn\'t draining',
      'What should I do to prepare my house for winter?',
      'My garbage disposal hums but won\'t spin',
      'How do I fix a door that sticks?',
      'What tools does every homeowner need?'
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
    sampleQuestions: [
      'I absolutely love this product! Best purchase ever.',
      'This is the worst experience I have ever had.',
    ],
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
    sampleQuestions: [
      'Uber ride to client meeting downtown - $24.50',
      'Team lunch at Chipotle - $87.30',
    ],
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
    sampleQuestions: [
      '37.2',
      '58.5',
    ],
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
    sampleQuestions: [
      '72, 120, 80, 98.6, 95',
      '145, 180, 110, 103.2, 88',
    ],
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
    sampleQuestions: [
      'build passed - all 847 tests green',
      'CRITICAL: deployment failed - database connection timeout after 3 retries',
    ],
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
    sampleQuestions: [
      'password reset, low, web, 2 hours, resolved',
      'URGENT system down, critical, phone, 72 hours, open',
    ],
  },
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
  return !!demo.configPath && !!demo.files && !!demo.datasetName && !!demo.sampleQuestions
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
 * Remove a project from the demo projects list
 * Used when "Build your own" is clicked to convert a demo project to a regular project
 */
export function removeDemoProject(projectName: string | null): void {
  if (!projectName) return
  try {
    const demoProjects = JSON.parse(localStorage.getItem('lf_demo_projects') || '[]')
    const filtered = demoProjects.filter((name: string) => name !== projectName)
    localStorage.setItem('lf_demo_projects', JSON.stringify(filtered))
  } catch {
    // Ignore errors
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
