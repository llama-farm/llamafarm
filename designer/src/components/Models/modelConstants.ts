export interface ModelVariant {
  id: number
  label: string
  parameterSize?: string
  downloadSize?: string
  modelIdentifier: string
}

export interface LocalModelGroup {
  id: number
  name: string
  baseModelId: string // Base model ID without quantization (e.g., "unsloth/Qwen3-1.7B-GGUF")
  defaultQuantization: string // Default quantization to show (e.g., "Q4_K_M")
  variants: ModelVariant[]
}

// Recommended quantizations with descriptions
export const recommendedQuantizations: Record<
  string,
  { quantization: string; description: string }
> = {
  'unsloth/Qwen3-1.7B-GGUF': {
    quantization: 'Q5_K_M',
    description: 'Best balance of speed + accuracy.',
  },
  'unsloth/granite-4.0-h-1b-GGUF': {
    quantization: 'Q5_K_M',
    description:
      'Granite benefits a lot from higher precision; Q5 is the sweet spot.',
  },
  'unsloth/Llama-3.2-1B-Instruct-GGUF': {
    quantization: 'Q5_K_M',
    description:
      'Best general choice — higher quality than Q4 without being huge.',
  },
  'unsloth/gpt-oss-20b-GGUF': {
    quantization: 'Q4_K_M',
    description:
      'This model already runs fast; Q4 keeps it snappy without big quality loss.',
  },
  'unsloth/gemma-3-4b-it-GGUF': {
    quantization: 'Q4_K_M',
    description:
      'Gemma performs well at Q4; Q5 is good but not required unless added.',
  },
}

export const localGroups: LocalModelGroup[] = [
  {
    id: 1,
    name: 'Qwen3',
    baseModelId: 'unsloth/Qwen3-1.7B-GGUF',
    defaultQuantization: 'Q4_K_M',
    variants: [
      {
        id: 11,
        label: 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M',
        modelIdentifier: 'unsloth/Qwen3-1.7B-GGUF:Q4_K_M',
      },
    ],
  },
  {
    id: 2,
    name: 'IBM Granite',
    baseModelId: 'unsloth/granite-4.0-h-1b-GGUF',
    defaultQuantization: 'Q5_K_M',
    variants: [
      {
        id: 21,
        label: 'unsloth/granite-4.0-h-1b-GGUF:Q5_K_M',
        modelIdentifier: 'unsloth/granite-4.0-h-1b-GGUF:Q5_K_M',
      },
    ],
  },
  {
    id: 3,
    name: 'Llama 3.2',
    baseModelId: 'unsloth/Llama-3.2-1B-Instruct-GGUF',
    defaultQuantization: 'Q5_K_M',
    variants: [
      {
        id: 31,
        label: 'unsloth/Llama-3.2-1B-Instruct-GGUF:Q5_K_M',
        modelIdentifier: 'unsloth/Llama-3.2-1B-Instruct-GGUF:Q5_K_M',
      },
    ],
  },
  {
    id: 4,
    name: 'GPT-OSS',
    baseModelId: 'unsloth/gpt-oss-20b-GGUF',
    defaultQuantization: 'Q4_K_M',
    variants: [
      {
        id: 41,
        label: 'unsloth/gpt-oss-20b-GGUF:Q4_K_M',
        modelIdentifier: 'unsloth/gpt-oss-20b-GGUF:Q4_K_M',
      },
    ],
  },
  {
    id: 5,
    name: 'Gemma 3',
    baseModelId: 'unsloth/gemma-3-4b-it-GGUF',
    defaultQuantization: 'Q4_K_M',
    variants: [
      {
        id: 51,
        label: 'unsloth/gemma-3-4b-it-GGUF:Q4_K_M',
        modelIdentifier: 'unsloth/gemma-3-4b-it-GGUF:Q4_K_M',
      },
    ],
  },
]

