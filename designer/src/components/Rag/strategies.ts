export type RagStrategy = {
  id: string
  name: string
  description: string
  isDefault: boolean
  datasetsUsing: number
  configName?: string // Original config name for API calls (e.g., "universal_processor")
}

// No hardcoded strategies - everything comes from config
export const defaultStrategies: RagStrategy[] = []
