export type RagStrategy = {
  id: string
  name: string
  description: string
  isDefault: boolean
  datasetsUsing: number
}

export const defaultStrategies: RagStrategy[] = [
  {
    id: 'processing-universal',
    name: 'Universal document processor',
    description:
      'Unified processor for PDFs, Word docs, CSVs, Markdown, and text files.',
    isDefault: true,
    datasetsUsing: 2,
  },
]
