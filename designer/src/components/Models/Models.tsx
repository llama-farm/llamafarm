import { useState } from 'react'
import { Button } from '../ui/button'
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
  meta: string
  badges: string[]
  isDefault?: boolean
  status?: ModelStatus
}

interface ModelCardProps {
  model: InferenceModel
  onMakeDefault?: () => void
  onDelete?: () => void
}

function ModelCard({ model, onMakeDefault, onDelete }: ModelCardProps) {
  return (
    <div className="w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative">
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

      <div className="flex items-center gap-2">
        <div className="text-sm">{model.name}</div>
        {model.isDefault && (
          <div className="text-[10px] leading-4 rounded-xl px-2 py-0.5 bg-teal-600 text-teal-50 dark:bg-teal-400 dark:text-teal-900">
            Default
          </div>
        )}
      </div>
      <div className="text-xs text-muted-foreground">{model.meta}</div>
      <div className="flex flex-row gap-2">
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
          <Loader size={16} className="border-blue-400 dark:border-blue-100" />
          Downloading...
        </div>
      ) : (
        <div className="text-xs text-muted-foreground">
          more info here in a line
        </div>
      )}
    </div>
  )
}

function ProjectInferenceModels({
  models,
  onMakeDefault,
  onDelete,
}: {
  models: InferenceModel[]
  onMakeDefault: (id: string) => void
  onDelete: (id: string) => void
}) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-6">
      {models.map(m => (
        <ModelCard
          key={m.id}
          model={m}
          onMakeDefault={() => onMakeDefault(m.id)}
          onDelete={() => onDelete(m.id)}
        />
      ))}
    </div>
  )
}

function CloudModelsForm({
  onAddModel,
  onGoToProject,
}: {
  onAddModel: (m: InferenceModel) => void
  onGoToProject: () => void
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
}: {
  onAddModel: (m: InferenceModel) => void
  onGoToProject: () => void
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

      {/* Search */}
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

      {/* Table */}
      {sourceTab === 'local' && (
        <div className="w-full overflow-hidden rounded-lg border border-border">
          <div className="grid grid-cols-12 items-center bg-secondary text-secondary-foreground text-xs px-3 py-2">
            <div className="col-span-6">Model</div>
            <div className="col-span-3">Parameter size</div>
            <div className="col-span-2 text-right pr-10">Download size</div>
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
                  <div className="col-span-2 text-xs text-right pr-10">
                    {group.downloadSummary}
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
                        <div className="col-span-2 flex items-center justify-end pr-10">
                          <div className="text-xs text-muted-foreground min-w-[3.5rem] text-right">
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
        <CloudModelsForm
          onAddModel={onAddModel}
          onGoToProject={onGoToProject}
        />
      )}

      <Dialog
        open={confirmOpen}
        onOpenChange={open => {
          setConfirmOpen(open)
          if (!open) {
            setSubmitState('idle')
            setPendingVariant(null)
          }
        }}
      >
        <DialogContent>
          <DialogTitle>Download and add this model?</DialogTitle>
          <DialogDescription>
            {pendingVariant ? (
              <div className="mt-2 text-sm">
                You are about to download and add
                <span className="mx-1 font-medium text-foreground">
                  {pendingVariant.label}
                </span>
                to your project.
                <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
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
              disabled={submitState === 'loading'}
              onClick={() => {
                if (!pendingVariant) return
                // Show download and add a placeholder card
                onAddModel({
                  id: `dl-${pendingVariant.id}`,
                  name:
                    pendingVariant.label.split(',')[0] ?? pendingVariant.label,
                  meta: 'Downloading…',
                  badges: ['Local', 'Ollama'],
                  status: 'downloading',
                })
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
  const [activeTab, setActiveTab] = useState('project')
  const [projectModels, setProjectModels] = useState<InferenceModel[]>([
    {
      id: 'tinyllama',
      name: 'TinyLlama',
      meta: 'Updated on 8/23/25',
      badges: ['Local', 'Ollama'],
      isDefault: true,
    },
    {
      id: 'gpt5',
      name: 'GPT5',
      meta: 'Updated on 8/23/25',
      badges: ['Cloud', 'OpenAI'],
    },
  ])

  const addProjectModel = (m: InferenceModel) => {
    setProjectModels(prev => {
      if (prev.some(x => x.id === m.id)) return prev
      return [...prev, m]
    })
    if (m.status === 'downloading') {
      const addedId = m.id
      setTimeout(() => {
        setProjectModels(prev =>
          prev.map(x =>
            x.id === addedId
              ? {
                  ...x,
                  status: 'ready',
                  meta: `Added on ${new Date().toLocaleDateString()}`,
                }
              : x
          )
        )
      }, 10000)
    }
  }

  const makeDefault = (id: string) => {
    setProjectModels(prev => prev.map(m => ({ ...m, isDefault: m.id === id })))
  }

  const deleteModel = (id: string) => {
    setProjectModels(prev => prev.filter(m => m.id !== id))
  }

  return (
    <div className="h-full w-full flex flex-col gap-3">
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-2xl">Models</h2>
        <Button variant="outline" size="sm" disabled>
          Deploy
        </Button>
      </div>

      <TabBar
        activeTab={activeTab}
        onChange={setActiveTab}
        tabs={[
          { id: 'project', label: 'Project inference models' },
          { id: 'manage', label: 'Add or change models' },
          { id: 'training', label: 'Training data' },
        ]}
      />

      {activeTab === 'project' && (
        <ProjectInferenceModels
          models={projectModels}
          onMakeDefault={makeDefault}
          onDelete={deleteModel}
        />
      )}
      {activeTab === 'manage' && (
        <AddOrChangeModels
          onAddModel={addProjectModel}
          onGoToProject={() => setActiveTab('project')}
        />
      )}
      {activeTab === 'training' && <TrainingData />}
    </div>
  )
}

export default Models
