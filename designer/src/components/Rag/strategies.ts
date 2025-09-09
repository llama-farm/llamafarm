export type RagStrategy = {
  id: string
  name: string
  description: string
  isDefault: boolean
  datasetsUsing: number
}

export const defaultStrategies: RagStrategy[] = [
  {
    id: 'pdf-simple',
    name: 'PDF Simple',
    description: 'Description saying what good for',
    isDefault: true,
    datasetsUsing: 2,
  },
  {
    id: 'pdf-complex',
    name: 'PDF Complex',
    description: 'Description saying what good for',
    isDefault: true,
    datasetsUsing: 0,
  },
  {
    id: 'csv-extract',
    name: 'CSV extract',
    description: 'Description saying what good for',
    isDefault: true,
    datasetsUsing: 0,
  },
  {
    id: 'chat-extract-1',
    name: 'Chat extract',
    description: 'Description saying what good for',
    isDefault: true,
    datasetsUsing: 0,
  },
  {
    id: 'json-extract-1',
    name: 'JSON extract',
    description: 'Description saying what good for',
    isDefault: true,
    datasetsUsing: 0,
  },
  {
    id: 'chat-extract-2',
    name: 'Chat extract',
    description: 'Description saying what good for',
    isDefault: true,
    datasetsUsing: 0,
  },
  {
    id: 'json-extract-2',
    name: 'JSON extract',
    description: 'Description saying what good for',
    isDefault: false,
    datasetsUsing: 2,
  },
]
