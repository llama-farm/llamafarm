import { useEffect, useState } from 'react'
import { Button } from '../ui/button'
import PageActions from '../common/PageActions'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { Input } from '../ui/input'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'
import { Label } from '../ui/label'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject, useUpdateProject } from '../../hooks/useProjects'
import { Checkbox } from '../ui/checkbox'
import { parsePromptSets } from '../../utils/promptSets'
import { useModeWithReset } from '../../hooks/useModeWithReset'

interface TabBarProps {
  activeTab: string
  onChange: (tabId: string) => void
  tabs: { id: string; label: string }[]
}

function TabBar({ activeTab, onChange, tabs }: TabBarProps) {
  return (
    <div className="w-full flex items-end gap-1 border-b border-border">
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={`px-3 py-2 -mb-[1px] border-b-2 transition-colors text-sm rounded-t-md ${
            activeTab === tab.id
              ? 'border-primary text-foreground'
              : 'border-transparent text-muted-foreground hover:text-foreground'
          }`}
          onClick={() => onChange(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

type ModelStatus = 'ready' | 'downloading'

interface InferenceModel {
  id: string
  name: string
  modelIdentifier?: string
  meta: string
  badges: string[]
  isDefault?: boolean
  status?: ModelStatus
}

interface ModelCardProps {
  model: InferenceModel
  onMakeDefault?: () => void
  onDelete?: () => void
  promptSetNames: string[]
  selectedPromptSets: string[]
  onTogglePromptSet: (name: string, checked: boolean | string) => void
  onClearPromptSets: () => void
}

function ModelCard({
  model,
  onMakeDefault,
  onDelete,
  promptSetNames,
  selectedPromptSets,
  onTogglePromptSet,
  onClearPromptSets,
}: ModelCardProps) {
  return (
    <div className="w-full bg-card rounded-lg border border-border flex flex-col gap-3 p-4 relative">
      <div className="absolute top-2 right-2">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30">
              <FontIcon type="overflow" className="w-4 h-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-[10rem] w-[10rem]">
            {!model.isDefault && (
              <DropdownMenuItem onClick={onMakeDefault}>
                Make default
              </DropdownMenuItem>
            )}
            <DropdownMenuItem
              className="text-destructive focus:text-destructive"
              onClick={onDelete}
            >
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 md:items-center gap-3 w-full">
        <div>
          <div className="text-sm text-muted-foreground mb-1">
            {model.modelIdentifier || model.name}
          </div>

          <div className="flex items-center gap-2 mb-2">
            <div className="text-lg font-medium">{model.name}</div>
            {model.isDefault && (
              <div className="text-[10px] leading-4 rounded-xl px-2 py-0.5 bg-teal-600 text-teal-50 dark:bg-teal-400 dark:text-teal-900">
                Default
              </div>
            )}
          </div>

          <div className="text-sm text-muted-foreground mb-3">{model.meta}</div>

          <div className="flex flex-row gap-2 mb-2">
            {model.badges.map((b, i) => (
              <div
                key={`${b}-${i}`}
                className="text-xs text-primary-foreground bg-primary rounded-xl px-3 py-0.5"
              >
                {b}
              </div>
            ))}
          </div>

          {model.status === 'downloading' ? (
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Loader
                size={16}
                className="border-blue-400 dark:border-blue-100"
              />
              Downloading...
            </div>
          ) : null}
        </div>
        {/* Prompt sets multi-select column */}
        <div className="mt-3 md:mt-0 md:justify-self-end w-full flex flex-col md:pl-4">
          <div className="text-xs text-muted-foreground mb-1">Prompt sets</div>
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="w-full h-8 rounded-md border border-border bg-background px-3 text-left flex items-center justify-between mr-6 md:mr-8">
                <span className="truncate text-sm flex items-center gap-2">
                  {selectedPromptSets.length > 0 ? (
                    <>
                      <span className="inline-flex items-center px-2 py-0.5 text-[10px] rounded-full bg-secondary text-secondary-foreground">
                        {selectedPromptSets.length}
                      </span>
                      <span className="truncate">
                        {selectedPromptSets.join(', ')}
                      </span>
                    </>
                  ) : (
                    'All sets'
                  )}
                </span>
                <FontIcon type="chevron-down" className="w-4 h-4" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent className="w-64 max-h-64 overflow-auto">
              {promptSetNames.map(name => (
                <DropdownMenuItem
                  key={name}
                  className="w-full justify-start text-left"
                  onSelect={e => e.preventDefault()}
                >
                  <label className="flex items-center gap-2 w-full">
                    <Checkbox
                      checked={selectedPromptSets.includes(name)}
                      onCheckedChange={v => onTogglePromptSet(name, v)}
                    />
                    <span className="text-sm">{name}</span>
                  </label>
                </DropdownMenuItem>
              ))}
              <div className="h-px bg-border my-1" />
              <DropdownMenuItem onClick={onClearPromptSets}>
                <span className="text-xs text-muted-foreground">
                  Clear selection (All sets)
                </span>
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  )
}

function ProjectInferenceModels({
  models,
  onMakeDefault,
  onDelete,
  getSelected,
  promptSetNames,
  onToggle,
  onClear,
}: {
  models: InferenceModel[]
  onMakeDefault: (id: string) => void
  onDelete: (id: string) => void
  getSelected: (id: string) => string[]
  promptSetNames: string[]
  onToggle: (id: string, name: string, checked: boolean | string) => void
  onClear: (id: string) => void
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-1 gap-2 mb-6">
      {models.map(m => (
        <ModelCard
          key={m.id}
          model={m}
          onMakeDefault={() => onMakeDefault(m.id)}
          onDelete={() => onDelete(m.id)}
          promptSetNames={promptSetNames}
          selectedPromptSets={getSelected(m.id)}
          onTogglePromptSet={(name, checked) => onToggle(m.id, name, checked)}
          onClearPromptSets={() => onClear(m.id)}
        />
      ))}
    </div>
  )
}

function CloudModelsForm({
  onAddModel,
  onGoToProject,
  promptSetNames: _promptSetNames,
}: {
  onAddModel: (m: InferenceModel, promptSets?: string[]) => void
  onGoToProject: () => void
  promptSetNames: string[]
}) {
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
    const providerLabel = provider
    setSubmitState('loading')
    onAddModel({
      id: `cloud-${provider}-${name}`.toLowerCase().replace(/\s+/g, '-'),
      name,
      meta: `Added on ${new Date().toLocaleDateString()}`,
      badges: ['Cloud', providerLabel],
      status: 'ready',
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

function AddOrChangeModels({
  onAddModel,
  onGoToProject,
  promptSetNames,
}: {
  onAddModel: (m: InferenceModel, promptSets?: string[]) => void
  onGoToProject: () => void
  promptSetNames: string[]
}) {
  const [sourceTab, setSourceTab] = useState<'local' | 'cloud'>('local')
  const [query, setQuery] = useState('')
  const [expandedGroupId, setExpandedGroupId] = useState<number | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [pendingVariant, setPendingVariant] = useState<ModelVariant | null>(
    null
  )
  const [submitState, setSubmitState] = useState<
    'idle' | 'loading' | 'success'
  >('idle')
  const [modelName, setModelName] = useState('')
  const [modelDescription, setModelDescription] = useState('')
  const [selectedPromptSets, setSelectedPromptSets] = useState<string[]>([])

  interface ModelVariant {
    id: number
    label: string
    parameterSize: string
    downloadSize: string
  }

  interface LocalModelGroup {
    id: number
    name: string
    parameterSummary: string
    downloadSummary: string
    variants: ModelVariant[]
  }

  const localGroups: LocalModelGroup[] = [
    {
      id: 1,
      name: 'deepseek-r1',
      parameterSummary: '1b, 7b, 70b, 100b',
      downloadSummary: '4.5–45 GB',
      variants: [
        {
          id: 11,
          label: 'deepseek-r1,1b',
          parameterSize: '1b',
          downloadSize: '4.5 GB',
        },
        {
          id: 12,
          label: 'deepseek-r1,7b',
          parameterSize: '7b',
          downloadSize: '13 GB',
        },
        {
          id: 13,
          label: 'deepseek-r1,70b',
          parameterSize: '70b',
          downloadSize: '25 GB',
        },
        {
          id: 14,
          label: 'deepseek-r1,100b',
          parameterSize: '100b',
          downloadSize: '45 GB',
        },
      ],
    },
    {
      id: 2,
      name: 'tinyllama',
      parameterSummary: '1.1b',
      downloadSummary: '1–2 GB',
      variants: [
        {
          id: 21,
          label: 'tinyllama,1.1b',
          parameterSize: '1.1b',
          downloadSize: '1.6 GB',
        },
      ],
    },
    {
      id: 3,
      name: 'mistral',
      parameterSummary: '7b, 8x7b, 22b',
      downloadSummary: '2.5–12 GB',
      variants: [
        {
          id: 31,
          label: 'mistral,7b',
          parameterSize: '7b',
          downloadSize: '2.5 GB',
        },
        {
          id: 32,
          label: 'mistral,8x7b',
          parameterSize: '8x7b',
          downloadSize: '8.0 GB',
        },
        {
          id: 33,
          label: 'mistral,22b',
          parameterSize: '22b',
          downloadSize: '12 GB',
        },
      ],
    },
    {
      id: 4,
      name: 'qwen2.5',
      parameterSummary: '1.5b, 7b, 32b, 72b',
      downloadSummary: '3.4–20 GB',
      variants: [
        {
          id: 41,
          label: 'qwen2.5,1.5b',
          parameterSize: '1.5b',
          downloadSize: '3.4 GB',
        },
        {
          id: 42,
          label: 'qwen2.5,7b',
          parameterSize: '7b',
          downloadSize: '7 GB',
        },
        {
          id: 43,
          label: 'qwen2.5,32b',
          parameterSize: '32b',
          downloadSize: '14 GB',
        },
        {
          id: 44,
          label: 'qwen2.5,72b',
          parameterSize: '72b',
          downloadSize: '20 GB',
        },
      ],
    },
    {
      id: 5,
      name: 'llama3.2',
      parameterSummary: '1b, 3b, 11b',
      downloadSummary: '2–8 GB',
      variants: [
        {
          id: 51,
          label: 'llama3.2,1b',
          parameterSize: '1b',
          downloadSize: '2 GB',
        },
        {
          id: 52,
          label: 'llama3.2,3b',
          parameterSize: '3b',
          downloadSize: '3.5 GB',
        },
        {
          id: 53,
          label: 'llama3.2,11b',
          parameterSize: '11b',
          downloadSize: '8 GB',
        },
      ],
    },
    {
      id: 6,
      name: 'llama3.1',
      parameterSummary: '8b, 70b',
      downloadSummary: '4–42 GB',
      variants: [
        {
          id: 61,
          label: 'llama3.1,8b',
          parameterSize: '8b',
          downloadSize: '4.1 GB',
        },
        {
          id: 62,
          label: 'llama3.1,70b',
          parameterSize: '70b',
          downloadSize: '42 GB',
        },
      ],
    },
    {
      id: 7,
      name: 'phi-3',
      parameterSummary: '3.8b, 14b',
      downloadSummary: '2.8–10 GB',
      variants: [
        {
          id: 71,
          label: 'phi-3,3.8b',
          parameterSize: '3.8b',
          downloadSize: '2.8 GB',
        },
        {
          id: 72,
          label: 'phi-3,14b',
          parameterSize: '14b',
          downloadSize: '10 GB',
        },
      ],
    },
    {
      id: 8,
      name: 'codellama',
      parameterSummary: '7b, 13b, 34b',
      downloadSummary: '7–24 GB',
      variants: [
        {
          id: 81,
          label: 'codellama,7b',
          parameterSize: '7b',
          downloadSize: '7 GB',
        },
        {
          id: 82,
          label: 'codellama,13b',
          parameterSize: '13b',
          downloadSize: '13 GB',
        },
        {
          id: 83,
          label: 'codellama,34b',
          parameterSize: '34b',
          downloadSize: '24 GB',
        },
      ],
    },
  ]

  const filteredGroups = localGroups.filter(g =>
    [g.name, g.parameterSummary].some(v =>
      v.toLowerCase().includes(query.toLowerCase())
    )
  )

  return (
    <div className="rounded-xl border border-border bg-card p-4 md:p-6 flex flex-col gap-4">
      <div className="text-sm text-muted-foreground">
        Add a new model provider or switch which models are enabled for this
        project.
      </div>

      {/* Source switcher */}
      <div className="w-full flex items-center">
        <div className="flex w-full max-w-xl rounded-lg overflow-hidden border border-border">
          <button
            className={`flex-1 h-10 text-sm ${
              sourceTab === 'local'
                ? 'bg-primary text-primary-foreground'
                : 'text-foreground hover:bg-secondary/80'
            }`}
            onClick={() => setSourceTab('local')}
            aria-pressed={sourceTab === 'local'}
          >
            Local models
          </button>
          <button
            className={`flex-1 h-10 text-sm ${
              sourceTab === 'cloud'
                ? 'bg-primary text-primary-foreground'
                : 'text-foreground hover:bg-secondary/80'
            }`}
            onClick={() => setSourceTab('cloud')}
            aria-pressed={sourceTab === 'cloud'}
          >
            Cloud models
          </button>
        </div>
      </div>

      {/* Search - only show for local models */}
      {sourceTab === 'local' && (
        <div className="relative w-full">
          <FontIcon
            type="search"
            className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2"
          />
          <Input
            placeholder="Search local options"
            value={query}
            onChange={e => setQuery(e.target.value)}
            className="pl-9 h-10"
          />
        </div>
      )}

      {/* Table */}
      {sourceTab === 'local' && (
        <div className="w-full overflow-hidden rounded-lg border border-border">
          <div className="grid grid-cols-12 items-center bg-secondary text-secondary-foreground text-xs px-3 py-2">
            <div className="col-span-6">Model</div>
            <div className="col-span-3">Parameter size</div>
            <div className="col-span-2 text-right pr-4 sm:pr-10">
              Download size
            </div>
            <div className="col-span-1" />
          </div>
          {filteredGroups.map(group => {
            const isOpen = expandedGroupId === group.id
            return (
              <div key={group.id} className="border-t border-border">
                <div
                  className="grid grid-cols-12 items-center px-3 py-3 text-sm cursor-pointer hover:bg-accent/40"
                  onClick={() =>
                    setExpandedGroupId(prev =>
                      prev === group.id ? null : group.id
                    )
                  }
                >
                  <div className="col-span-6 flex items-center gap-2">
                    <FontIcon
                      type="chevron-down"
                      className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    />
                    <span className="truncate">{group.name}</span>
                  </div>
                  <div className="col-span-3 text-xs">
                    {group.parameterSummary}
                  </div>
                  <div className="col-span-2 text-xs text-right pr-4 sm:pr-10">
                    <span className="inline-block min-w-[3.5rem] truncate">
                      {group.downloadSummary}
                    </span>
                  </div>
                  <div className="col-span-1" />
                </div>
                {isOpen && (
                  <div className="px-3 pb-2">
                    {group.variants.map(variant => (
                      <div
                        key={variant.id}
                        className="grid grid-cols-12 items-center px-3 py-3 text-sm rounded-md hover:bg-accent/40"
                      >
                        <div className="col-span-6 flex items-center text-muted-foreground">
                          <span className="inline-block w-4" />
                          <span className="ml-2 truncate">{variant.label}</span>
                        </div>
                        <div className="col-span-3 text-xs">
                          {variant.parameterSize}
                        </div>
                        <div className="col-span-2 flex items-center justify-end pr-4 sm:pr-10">
                          <div className="text-xs text-muted-foreground min-w-[3.5rem] text-right whitespace-nowrap">
                            {variant.downloadSize}
                          </div>
                        </div>
                        <div className="col-span-1 flex items-center justify-end pr-2">
                          <Button
                            size="sm"
                            className="h-8 px-3"
                            onClick={() => {
                              setPendingVariant(variant)
                              setConfirmOpen(true)
                            }}
                          >
                            Add
                          </Button>
                        </div>
                      </div>
                    ))}
                    <div className="flex justify-end pr-3">
                      <button
                        className="text-xs text-muted-foreground hover:text-foreground"
                        onClick={() => setExpandedGroupId(null)}
                      >
                        Hide
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
      {sourceTab === 'cloud' && (
        <div className="flex flex-col gap-4">
          <div className="flex items-start gap-3 p-3 rounded-md bg-secondary/40 border border-border">
            <p className="text-xs text-muted-foreground">
              Cloud model options coming soon!
            </p>
          </div>
          <div className="relative">
            <div className="opacity-40 pointer-events-none">
              <CloudModelsForm
                onAddModel={onAddModel}
                onGoToProject={onGoToProject}
                promptSetNames={promptSetNames}
              />
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="bg-background/80 backdrop-blur-sm rounded-lg px-6 py-3 border border-border shadow-lg">
                <div className="text-sm font-medium">Coming soon</div>
              </div>
            </div>
          </div>
        </div>
      )}

      <Dialog
        open={confirmOpen}
        onOpenChange={open => {
          setConfirmOpen(open)
          if (!open) {
            setSubmitState('idle')
            setPendingVariant(null)
            setModelName('')
            setModelDescription('')
            setSelectedPromptSets([])
          }
        }}
      >
        <DialogContent>
          <DialogTitle>Download and add this model?</DialogTitle>
          <DialogDescription>
            {pendingVariant ? (
              <div className="mt-2 flex flex-col gap-3">
                <p className="text-sm">
                  You are about to download and add
                  <span className="mx-1 font-medium text-foreground">
                    {pendingVariant.label}
                  </span>
                  to your project.
                </p>

                <div>
                  <label
                    className="text-xs text-muted-foreground"
                    htmlFor="model-name"
                  >
                    Name
                  </label>
                  <input
                    id="model-name"
                    type="text"
                    placeholder="Enter model name"
                    value={modelName}
                    onChange={e => setModelName(e.target.value)}
                    className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                  />
                </div>

                <div>
                  <label
                    className="text-xs text-muted-foreground"
                    htmlFor="model-description"
                  >
                    Description
                  </label>
                  <textarea
                    id="model-description"
                    rows={2}
                    placeholder="Enter model description"
                    value={modelDescription}
                    onChange={e => setModelDescription(e.target.value)}
                    className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                  />
                </div>

                <div>
                  <label
                    className="text-xs text-muted-foreground mb-1 block"
                    htmlFor="prompt-sets-trigger"
                  >
                    Prompt sets
                  </label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button
                        id="prompt-sets-trigger"
                        className="w-full h-9 rounded-lg border border-input bg-background px-3 text-left flex items-center justify-between"
                      >
                        <span className="truncate text-sm flex items-center gap-2">
                          {selectedPromptSets.length > 0 ? (
                            <>
                              <span className="inline-flex items-center px-2 py-0.5 text-[10px] rounded-full bg-secondary text-secondary-foreground">
                                {selectedPromptSets.length}
                              </span>
                              <span className="truncate">
                                {selectedPromptSets.join(', ')}
                              </span>
                            </>
                          ) : (
                            <span className="text-muted-foreground">
                              All sets
                            </span>
                          )}
                        </span>
                        <FontIcon type="chevron-down" className="w-4 h-4" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent className="w-64 max-h-64 overflow-auto">
                      {promptSetNames.map(name => (
                        <DropdownMenuItem
                          key={name}
                          className="w-full justify-start text-left"
                          onSelect={e => e.preventDefault()}
                        >
                          <label className="flex items-center gap-2 w-full">
                            <Checkbox
                              checked={selectedPromptSets.includes(name)}
                              onCheckedChange={v => {
                                if (v) {
                                  setSelectedPromptSets(prev => [...prev, name])
                                } else {
                                  setSelectedPromptSets(prev =>
                                    prev.filter(s => s !== name)
                                  )
                                }
                              }}
                            />
                            <span className="text-sm">{name}</span>
                          </label>
                        </DropdownMenuItem>
                      ))}
                      <div className="h-px bg-border my-1" />
                      <DropdownMenuItem
                        onClick={() => setSelectedPromptSets([])}
                      >
                        <span className="text-xs text-muted-foreground">
                          Clear selection
                        </span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-muted-foreground">Provider</div>
                  <div>Ollama</div>
                  <div className="text-muted-foreground">Parameter size</div>
                  <div>{pendingVariant.parameterSize}</div>
                  <div className="text-muted-foreground">Download size</div>
                  <div>{pendingVariant.downloadSize}</div>
                </div>
              </div>
            ) : null}
          </DialogDescription>
          <DialogFooter>
            <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
              Cancel
            </Button>
            <Button
              disabled={submitState === 'loading' || !modelName.trim()}
              onClick={() => {
                if (!pendingVariant) return
                // Show download and add a placeholder card with user-entered data
                onAddModel(
                  {
                    id: `dl-${pendingVariant.id}`,
                    name: modelName.trim(),
                    modelIdentifier: pendingVariant.label,
                    meta: modelDescription.trim() || 'Downloading…',
                    badges: ['Local', 'Ollama'],
                    status: 'downloading',
                  },
                  selectedPromptSets.length > 0 ? selectedPromptSets : undefined
                )
                setSubmitState('loading')
                setTimeout(() => {
                  setSubmitState('success')
                  setTimeout(() => {
                    setConfirmOpen(false)
                    onGoToProject()
                    setSubmitState('idle')
                  }, 600)
                }, 1000)
              }}
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
              Download and add
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

function TrainingData() {
  return (
    <div className="rounded-xl border border-border bg-card p-10 flex items-center justify-center">
      <div className="text-sm text-muted-foreground">
        Training data features coming soon.
      </div>
    </div>
  )
}

const Models = () => {
  const activeProject = useActiveProject()
  const { data: projectResponse } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )
  const updateProject = useUpdateProject()
  const [activeTab, setActiveTab] = useState('project')
  const [mode, setMode] = useModeWithReset('designer')
  const [projectModels, setProjectModels] = useState<InferenceModel[]>([])
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
  const [modelToDelete, setModelToDelete] = useState<string | null>(null)

  // Load models from config
  useEffect(() => {
    if (!projectResponse?.project?.config?.runtime?.models) {
      setProjectModels([])
      return
    }

    const runtimeModels = projectResponse.project.config.runtime.models
    const defaultModelName =
      projectResponse.project.config.runtime.default_model

    const mappedModels: InferenceModel[] = runtimeModels.map((model: any) => {
      const name: string =
        (model && (model.name || model.model)) || 'unnamed-model'
      const provider: string =
        typeof model?.provider === 'string' ? model.provider : ''
      const providerBadge = provider
        ? provider.charAt(0).toUpperCase() + provider.slice(1)
        : 'Unknown'
      const localityBadge = provider
        ? provider === 'ollama'
          ? 'Local'
          : 'Cloud'
        : 'Unknown'

      return {
        id: name,
        name,
        modelIdentifier: typeof model?.model === 'string' ? model.model : '',
        meta: (model && model.description) || 'Model from config',
        badges: [localityBadge, providerBadge],
        isDefault: name === defaultModelName,
        status: 'ready' as ModelStatus,
      }
    })

    setProjectModels(mappedModels)
  }, [projectResponse])

  const addProjectModel = async (m: InferenceModel, promptSets?: string[]) => {
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !projectResponse?.project?.config
    )
      return

    // Add to local state first for immediate UI feedback
    setProjectModels(prev => {
      if (prev.some(x => x.id === m.id)) return prev
      return [...prev, m]
    })

    // Add to config
    const currentConfig = projectResponse.project.config
    const runtimeModels = currentConfig.runtime?.models || []

    const newModel = {
      name: m.name,
      description: m.meta === 'Downloading…' ? '' : m.meta,
      provider: 'ollama',
      model: m.modelIdentifier || m.name,
      base_url: 'http://localhost:11434',
      prompt_format: 'unstructured',
      provider_config: {},
      prompts: promptSets || [],
    }

    const updatedModels = [...runtimeModels, newModel]

    const nextConfig = {
      ...currentConfig,
      runtime: {
        ...currentConfig.runtime,
        models: updatedModels,
      },
    }

    try {
      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: nextConfig },
      })
    } catch (error) {
      console.error('Failed to add model to config:', error)
      // Rollback local optimistic update
      setProjectModels(prev => prev.filter(x => x.id !== m.id))
    }

    if (m.status === 'downloading') {
      const addedId = m.id
      setTimeout(() => {
        setProjectModels(prev =>
          prev.map(x =>
            x.id === addedId
              ? {
                  ...x,
                  status: 'ready',
                  meta:
                    x.meta === 'Downloading…'
                      ? `Added on ${new Date().toLocaleDateString()}`
                      : x.meta,
                }
              : x
          )
        )
      }, 10000)
    }
  }

  const makeDefault = async (id: string) => {
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !projectResponse?.project?.config
    )
      return

    const currentConfig = projectResponse.project.config
    const nextConfig = {
      ...currentConfig,
      runtime: {
        ...currentConfig.runtime,
        default_model: id,
      },
    }

    try {
      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: nextConfig },
      })
      setProjectModels(prev =>
        prev.map(m => ({ ...m, isDefault: m.id === id }))
      )
    } catch (error) {
      console.error('Failed to set default model:', error)
    }
  }

  const deleteModel = (id: string) => {
    setModelToDelete(id)
    setDeleteConfirmOpen(true)
  }

  const confirmDeleteModel = async () => {
    if (
      !modelToDelete ||
      !activeProject?.namespace ||
      !activeProject?.project ||
      !projectResponse?.project?.config
    )
      return

    const currentConfig = projectResponse.project.config
    const runtime = currentConfig.runtime || {}
    const runtimeModels = runtime.models || []

    // Remove the model from config
    const updatedModels = runtimeModels.filter(
      (m: any) => m.name !== modelToDelete
    )

    // If deleting the default model, clear the default
    const newDefaultModel =
      runtime.default_model === modelToDelete
        ? undefined
        : runtime.default_model

    const nextConfig = {
      ...currentConfig,
      runtime: {
        ...runtime,
        models: updatedModels,
        default_model: newDefaultModel,
      },
    }

    // Optimistically update UI
    const prevModels = projectModels
    const prevMap = modelSetMap
    setProjectModels(prev => prev.filter(x => x.id !== modelToDelete))
    const optimisticMap = { ...modelSetMap }
    delete optimisticMap[modelToDelete]
    setModelSetMap(optimisticMap)

    try {
      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: nextConfig },
      })
      setDeleteConfirmOpen(false)
      setModelToDelete(null)
    } catch (error) {
      console.error('Failed to delete model:', error)
      // Rollback optimistic updates
      setProjectModels(prevModels)
      setModelSetMap(prevMap)
    }
  }

  // Prompt set assignment per model (loaded from config)
  const loadMapFromConfig = (): Record<string, string[]> => {
    if (!projectResponse?.project?.config?.runtime?.models) return {}

    const modelPromptsMap: Record<string, string[]> = {}
    const runtimeModels = projectResponse.project.config.runtime.models

    runtimeModels.forEach((model: any) => {
      if (model.name && model.prompts && Array.isArray(model.prompts)) {
        modelPromptsMap[model.name] = model.prompts
      }
    })

    return modelPromptsMap
  }

  const [modelSetMap, setModelSetMap] = useState<Record<string, string[]>>({})

  const promptSetNames = (() => {
    const prompts = projectResponse?.project?.config?.prompts as
      | Array<{
          name: string
          messages: Array<{ role?: string; content: string }>
        }>
      | undefined
    const sets = parsePromptSets(prompts)
    return sets.map((s: { name: string }) => s.name)
  })()

  // Load model-to-prompts mapping from config
  useEffect(() => {
    setModelSetMap(loadMapFromConfig())
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [projectResponse])

  const getSelectedFor = (id: string): string[] => modelSetMap[id] || []

  const toggleFor = async (
    id: string,
    name: string,
    checked: boolean | string
  ) => {
    const prevMap = { ...modelSetMap }
    const updatedMap = { ...modelSetMap }
    const cur = new Set(updatedMap[id] || [])
    if (checked) cur.add(name)
    else cur.delete(name)
    const arr = Array.from(cur)
    if (arr.length === 0) delete updatedMap[id]
    else updatedMap[id] = arr

    setModelSetMap(updatedMap)

    // Write to config
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !projectResponse?.project?.config
    )
      return

    const currentConfig = projectResponse.project.config
    const runtimeModels = currentConfig.runtime?.models || []

    const updatedModels = runtimeModels.map((model: any) => {
      if (model.name === id) {
        return {
          ...model,
          prompts: updatedMap[id] || [],
        }
      }
      return model
    })

    const nextConfig = {
      ...currentConfig,
      runtime: {
        ...currentConfig.runtime,
        models: updatedModels,
      },
    }

    try {
      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: nextConfig },
      })
    } catch (error) {
      console.error('Failed to update model prompt sets:', error)
      // Rollback on failure
      setModelSetMap(prevMap)
    }
  }

  const clearFor = async (id: string) => {
    const prevMap = { ...modelSetMap }
    const updatedMap = { ...modelSetMap }
    delete updatedMap[id]

    setModelSetMap(updatedMap)

    // Write to config
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !projectResponse?.project?.config
    )
      return

    const currentConfig = projectResponse.project.config
    const runtimeModels = currentConfig.runtime?.models || []

    const updatedModels = runtimeModels.map((model: any) => {
      if (model.name === id) {
        return {
          ...model,
          prompts: [],
        }
      }
      return model
    })

    const nextConfig = {
      ...currentConfig,
      runtime: {
        ...currentConfig.runtime,
        models: updatedModels,
      },
    }

    try {
      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: nextConfig },
      })
    } catch (error) {
      console.error('Failed to clear model prompt sets:', error)
      // Rollback on failure
      setModelSetMap(prevMap)
    }
  }

  return (
    <div
      className={`h-full w-full flex flex-col ${mode === 'designer' ? 'gap-3 pb-32' : ''}`}
    >
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-2xl">
          {mode === 'designer' ? 'Models' : 'Config editor'}
        </h2>
        <PageActions mode={mode} onModeChange={setMode} />
      </div>

      {mode !== 'designer' ? (
        <div className="flex-1 min-h-0 overflow-hidden pb-6">
          <ConfigEditor className="h-full" />
        </div>
      ) : (
        <>
          <TabBar
            activeTab={activeTab}
            onChange={setActiveTab}
            tabs={[
              { id: 'project', label: 'Project inference models' },
              { id: 'manage', label: 'Add or change models' },
              { id: 'training', label: 'Training data' },
            ]}
          />

          {activeTab === 'project' &&
            (projectModels.length === 0 ? (
              <div className="w-full h-full flex items-center justify-center">
                <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40 max-w-md">
                  <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
                    <FontIcon type="model" className="w-6 h-6 text-primary" />
                  </div>
                  <div className="text-lg font-medium text-foreground mb-2">
                    No models yet
                  </div>
                  <div className="text-sm text-muted-foreground mb-6">
                    Add your first model to start building. You can add local
                    Ollama models or configure cloud providers.
                  </div>
                  <Button
                    onClick={() => setActiveTab('manage')}
                    className="w-full sm:w-auto"
                  >
                    Add models
                  </Button>
                </div>
              </div>
            ) : (
              <ProjectInferenceModels
                models={projectModels}
                onMakeDefault={makeDefault}
                onDelete={deleteModel}
                getSelected={getSelectedFor}
                promptSetNames={promptSetNames}
                onToggle={toggleFor}
                onClear={clearFor}
              />
            ))}
          {activeTab === 'manage' && (
            <AddOrChangeModels
              onAddModel={addProjectModel}
              onGoToProject={() => setActiveTab('project')}
              promptSetNames={promptSetNames}
            />
          )}
          {activeTab === 'training' && <TrainingData />}
        </>
      )}

      {/* Inline multi-select on cards replaces separate dialog */}

      {/* Delete confirmation dialog */}
      <Dialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogTitle>Delete model</DialogTitle>
          <div className="text-sm text-muted-foreground">
            Are you sure you want to delete this model? This will remove it from
            your project configuration.
          </div>
          <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2">
            <div />
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => {
                  setDeleteConfirmOpen(false)
                  setModelToDelete(null)
                }}
                type="button"
              >
                Cancel
              </button>
              <button
                className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
                onClick={confirmDeleteModel}
                type="button"
              >
                Delete
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Models
