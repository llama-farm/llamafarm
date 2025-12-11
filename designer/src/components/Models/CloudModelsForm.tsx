import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import type { InferenceModel } from './types'

interface CloudModelsFormProps {
  onAddModel: (m: InferenceModel, promptSets?: string[]) => void
  onGoToProject: () => void
  promptSetNames: string[]
}

export function CloudModelsForm({
  onAddModel,
  onGoToProject,
  promptSetNames: _promptSetNames,
}: CloudModelsFormProps) {
  const providerOptions = [
    'OpenAI',
    'Anthropic',
    'Google',
    'Cohere',
    'Mistral',
    'Azure OpenAI',
    'Groq',
    'Together',
    'AWS Bedrock',
    'Ollama (remote)',
  ] as const
  type Provider = (typeof providerOptions)[number]
  const modelMap: Record<Provider, string[]> = {
    OpenAI: ['GPT-4.1', 'GPT-4.1-mini', 'o3-mini', 'GPT-4o'],
    Anthropic: ['Claude 3.5 Sonnet', 'Claude 3 Haiku'],
    Google: ['Gemini 2.0 Flash', 'Gemini 1.5 Pro'],
    Cohere: ['Command R', 'Command R+'],
    Mistral: ['Mistral Large', 'Mixtral 8x7B'],
    'Azure OpenAI': ['GPT-4.1', 'GPT-4o'],
    Groq: ['Llama 3 70B', 'Mixtral 8x7B'],
    Together: ['Llama 3 8B', 'Qwen2-72B'],
    'AWS Bedrock': ['Claude 3 Sonnet', 'Llama 3 8B Instruct'],
    'Ollama (remote)': ['llama3.1:8b', 'qwen2.5:7b'],
  }

  const [provider, setProvider] = useState<Provider>('OpenAI')
  const [model, setModel] = useState<string>(modelMap['OpenAI'][0])
  const [customModel, setCustomModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [maxTokens, setMaxTokens] = useState<number | null>(null)
  const [baseUrl, setBaseUrl] = useState('')
  const [submitState, setSubmitState] = useState<
    'idle' | 'loading' | 'success'
  >('idle')

  const modelsForProvider = [...modelMap[provider], 'Custom']
  const canAdd =
    model === 'Custom'
      ? apiKey.trim().length > 0 || baseUrl.trim().length > 0
      : apiKey.trim().length > 0

  const handleAddCloud = () => {
    if (!canAdd || submitState === 'loading') return
    const name = model === 'Custom' ? customModel || 'Custom model' : `${model}`
    setSubmitState('loading')
    onAddModel({
      id: `cloud-${provider}-${name}`.toLowerCase().replace(/\s+/g, '-'),
      name,
      meta: `Added on ${new Date().toLocaleDateString()}`,
      badges: ['Cloud'],
      status: 'ready',
      provider,
      apiKey: apiKey.trim() || undefined,
      baseUrl: baseUrl.trim() || undefined,
      maxTokens: maxTokens || undefined,
    })
    setTimeout(() => {
      setSubmitState('success')
      setTimeout(() => {
        setSubmitState('idle')
        onGoToProject()
      }, 500)
    }, 800)
  }

  return (
    <div className="w-full rounded-lg border border-border p-4 md:p-6 flex flex-col gap-4">
      <div className="flex flex-col gap-2">
        <Label className="text-xs text-muted-foreground">
          Select cloud provider
        </Label>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="w-full h-9 rounded-md border border-border bg-background px-3 text-left flex items-center justify-between">
              <span>{provider}</span>
              <FontIcon type="chevron-down" className="w-4 h-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-64">
            {providerOptions.map(p => (
              <DropdownMenuItem
                key={p}
                className="w-full justify-start text-left"
                onClick={() => {
                  setProvider(p)
                  setModel(modelMap[p][0])
                }}
              >
                {p}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="flex flex-col gap-2">
        <Label className="text-xs text-muted-foreground">Select model</Label>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="w-full h-9 rounded-md border border-border bg-background px-3 text-left flex items-center justify-between">
              <span>{model}</span>
              <FontIcon type="chevron-down" className="w-4 h-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent className="w-64 max-h-64 overflow-auto">
            {modelsForProvider.map(m => (
              <DropdownMenuItem
                key={m}
                className="w-full justify-start text-left"
                onClick={() => setModel(m)}
              >
                {m}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        {model === 'Custom' && (
          <Input
            placeholder="Enter model name/id"
            value={customModel}
            onChange={e => setCustomModel(e.target.value)}
            className="h-9"
          />
        )}
      </div>

      <div className="flex flex-col gap-2">
        <Label className="text-xs text-muted-foreground">API Key</Label>
        <div className="relative">
          <Input
            type={showApiKey ? 'text' : 'password'}
            placeholder="enter here"
            value={apiKey}
            onChange={e => setApiKey(e.target.value)}
            className="h-9 pr-9"
          />
          <button
            type="button"
            className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
            onClick={() => setShowApiKey(v => !v)}
            aria-label={showApiKey ? 'Hide API key' : 'Show API key'}
          >
            <FontIcon
              type={showApiKey ? 'eye-off' : 'eye'}
              className="w-4 h-4"
            />
          </button>
        </div>
        <div className="text-xs text-muted-foreground">
          Your API key can be found in your {provider} account settings
        </div>
      </div>

      {model === 'Custom' && (
        <div className="flex flex-col gap-2">
          <Label className="text-xs text-muted-foreground">
            Base URL override (optional)
          </Label>
          <Input
            placeholder="https://api.example.com"
            value={baseUrl}
            onChange={e => setBaseUrl(e.target.value)}
            className="h-9"
          />
          <div className="text-xs text-muted-foreground">
            Use to point to a proxy or self-hosted endpoint.
          </div>
        </div>
      )}

      <div className="flex flex-col gap-2">
        <Label className="text-xs text-muted-foreground">
          Max tokens (optional)
        </Label>
        <div className="flex items-center gap-2">
          <div className="flex-1 text-sm px-3 py-2 rounded-md border border-border bg-background">
            {maxTokens === null ? 'n / a' : maxTokens}
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              className="h-8 w-8"
              onClick={() =>
                setMaxTokens(prev => (prev ? Math.max(prev - 500, 0) : null))
              }
            >
              –
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-8 w-8"
              onClick={() => setMaxTokens(prev => (prev ? prev + 500 : 500))}
            >
              +
            </Button>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <Button
          onClick={handleAddCloud}
          disabled={!canAdd || submitState === 'loading'}
        >
          {submitState === 'loading' && (
            <span className="mr-2 inline-flex">
              <Loader
                size={14}
                className="border-blue-400 dark:border-blue-100"
              />
            </span>
          )}
          {submitState === 'success' && (
            <span className="mr-2 inline-flex">
              <FontIcon type="checkmark-filled" className="w-4 h-4" />
            </span>
          )}
          Add new Cloud model to project
        </Button>
      </div>
    </div>
  )
}

