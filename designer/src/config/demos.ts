/**
 * Demo project configurations
 * Each demo is fully self-contained with config and files
 */

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

  // Paths relative to /demo-files/
  configPath: string
  files: DemoFile[]

  // Dataset info from config
  datasetName: string

  // Sample questions to try
  sampleQuestions: string[]
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
