// Utilities for working with prompt sets in the native YAML structure

export type PromptRole = 'system' | 'assistant' | 'user'

export interface PromptItem {
  role: PromptRole
  content: string
}

export interface RawPromptMessage {
  role?: string
  content: string
}

export interface PromptSet {
  name: string
  items: PromptItem[]
}

export function parsePromptSets(
  prompts: Array<{ name: string; messages: RawPromptMessage[] }> | undefined
): PromptSet[] {
  if (!Array.isArray(prompts) || prompts.length === 0) {
    return [
      {
        name: 'Default',
        items: [],
      },
    ]
  }

  const sets: PromptSet[] = prompts.map(promptSet => {
    const items: PromptItem[] = (promptSet.messages || []).map(msg => ({
      role: (msg.role || 'system') as PromptRole,
      content: msg.content || '',
    }))

    return {
      name: promptSet.name,
      items,
    }
  })

  return sets.length > 0 ? sets : [{ name: 'Default', items: [] }]
}

export function serializePromptSets(
  sets: PromptSet[]
): Array<{ name: string; messages: RawPromptMessage[] }> {
  return sets.map(set => ({
    name: set.name,
    messages: set.items.map(item => ({
      role: item.role,
      content: item.content,
    })),
  }))
}
