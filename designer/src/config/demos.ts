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

  {
    id: 'santa-helper',
    name: 'santa-helper',
    displayName: "Santa's Holiday Helper",
    description: "Ho ho ho! Chat with Santa about gift ideas, holiday traditions, festive recipes, and making Christmas magical!",
    icon: '🎅',
    category: 'Holiday & Seasonal',
    estimatedTime: '~30 seconds',

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
    icon: '🔧',
    category: 'Home & Garden',
    estimatedTime: '~30 seconds',

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
