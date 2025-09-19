import { useMemo, useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { useToast } from '../ui/toast'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'
import { getClientSideSecret } from '../../utils/crypto'

// Helper for symmetric AES encryption (same as edit page)
async function encryptAPIKey(apiKey: string, secret: string) {
  const enc = new TextEncoder()
  const keyMaterial = await window.crypto.subtle.importKey(
    'raw',
    enc.encode(secret),
    { name: 'PBKDF2' },
    false,
    ['deriveKey']
  )
  const salt = window.crypto.getRandomValues(new Uint8Array(16))
  const key = await window.crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt: salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    keyMaterial,
    { name: 'AES-GCM', length: 256 },
    false,
    ['encrypt']
  )
  const iv = window.crypto.getRandomValues(new Uint8Array(12))
  const ciphertext = await window.crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    enc.encode(apiKey)
  )
  const base64 = (ab: ArrayBuffer) =>
    window.btoa(String.fromCharCode(...new Uint8Array(ab)))
  return JSON.stringify({
    salt: base64(salt.buffer),
    iv: base64(iv.buffer),
    data: base64(ciphertext),
  })
}

function AddEmbeddingStrategy() {
  const navigate = useNavigate()
  const { toast } = useToast()

  // Name
  const [name, setName] = useState('New embedding strategy')

  // Editable settings shown at top
  const [baseUrl, setBaseUrl] = useState('http://localhost:11434')
  const [dimension, setDimension] = useState<number>(768)
  const [batchSize, setBatchSize] = useState<number>(16)
  const [timeoutSec, setTimeoutSec] = useState<number>(60)
  const [autoPull, setAutoPull] = useState<boolean>(true)

  // Selection UI state (reusing the edit page’s structure)
  const [sourceTab, setSourceTab] = useState<'local' | 'cloud'>('local')
  const [query, setQuery] = useState('')
  const [expandedGroupId, setExpandedGroupId] = useState<number | null>(null)

  type Variant = {
    id: string
    label: string
    dim: string
    quality: string
    download: string
  }
  type LocalGroup = {
    id: number
    name: string
    dim: string
    quality: string
    ramVram: string
    download: string
    variants: Variant[]
  }

  const localGroups: LocalGroup[] = [
    {
      id: 1,
      name: 'BAAI — BGE (English)',
      dim: '384–1024',
      quality: 'General',
      ramVram: '≥4–8GB',
      download: '120–650MB',
      variants: [
        {
          id: 'bge-small-en-v1.5',
          label: 'bge-small-en-v1.5',
          dim: '384',
          quality: 'General',
          download: '120MB',
        },
        {
          id: 'bge-base-en-v1.5',
          label: 'bge-base-en-v1.5',
          dim: '768',
          quality: 'General',
          download: '340MB',
        },
        {
          id: 'bge-large-en-v1.5',
          label: 'bge-large-en-v1.5',
          dim: '1024',
          quality: 'General',
          download: '650MB',
        },
      ],
    },
    {
      id: 2,
      name: 'BAAI — bge-m3 (multilingual/multi-function)',
      dim: '1024',
      quality: 'Multilingual',
      ramVram: '≥8GB',
      download: '900MB',
      variants: [
        {
          id: 'bge-m3',
          label: 'bge-m3',
          dim: '1024',
          quality: 'Multilingual',
          download: '900MB',
        },
      ],
    },
    {
      id: 3,
      name: 'intfloat — E5 v2',
      dim: '768–1024',
      quality: 'General',
      ramVram: '≥6–8GB',
      download: '400–800MB',
      variants: [
        {
          id: 'e5-base-v2',
          label: 'e5-base-v2',
          dim: '768',
          quality: 'General',
          download: '400MB',
        },
        {
          id: 'e5-large-v2',
          label: 'e5-large-v2',
          dim: '1024',
          quality: 'General',
          download: '800MB',
        },
      ],
    },
    {
      id: 4,
      name: 'sentence-transformers — all-MiniLM-L6-v2',
      dim: '384',
      quality: 'Fast',
      ramVram: '≥4GB',
      download: '60–120MB',
      variants: [
        {
          id: 'all-MiniLM-L6-v2',
          label: 'all-MiniLM-L6-v2',
          dim: '384',
          quality: 'Fast',
          download: '60MB',
        },
      ],
    },
  ]

  const filteredGroups: LocalGroup[] = useMemo(() => {
    if (!query.trim()) return localGroups
    const q = query.toLowerCase()
    return localGroups
      .map(g => ({
        ...g,
        variants: g.variants.filter(v => v.label.toLowerCase().includes(q)),
      }))
      .filter(
        g =>
          g.name.toLowerCase().includes(q) ||
          (g.variants && g.variants.length > 0)
      )
  }, [query])

  const providerOptions = [
    'OpenAI',
    'Google',
    'Cohere',
    'Azure OpenAI',
    'AWS Bedrock',
    'Ollama (remote)',
  ] as const
  type Provider = (typeof providerOptions)[number]
  const modelMap: Record<Provider, string[]> = {
    OpenAI: ['text-embedding-3-large', 'text-embedding-3-small', 'Custom'],
    Google: ['text-embedding-004', 'Custom'],
    Cohere: ['embed-english-v3.0', 'embed-multilingual-v3.0', 'Custom'],
    'Azure OpenAI': [
      'text-embedding-3-large',
      'text-embedding-3-small',
      'Custom',
    ],
    'AWS Bedrock': [
      'cohere.embed-english-v3',
      'amazon.titan-embed-text-v2:0',
      'Custom',
    ],
    'Ollama (remote)': ['nomic-embed-text', 'bge-m3', 'Custom'],
  }

  const [provider, setProvider] = useState<Provider>('Ollama (remote)')
  const [model, setModel] = useState<string>('nomic-embed-text')
  const [customModel, setCustomModel] = useState('')
  const modelsForProvider = useMemo(() => modelMap[provider], [provider])

  // Cloud-specific state
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [openaiOrg, setOpenaiOrg] = useState('')
  const [openaiMaxRetries, setOpenaiMaxRetries] = useState(3)
  const [azureDeployment, setAzureDeployment] = useState('')
  const [azureResource, setAzureResource] = useState('')
  const [azureApiVersion, setAzureApiVersion] = useState('')
  const [vertexProjectId, setVertexProjectId] = useState('')
  const [vertexLocation, setVertexLocation] = useState('')
  const [vertexEndpoint, setVertexEndpoint] = useState('')
  const [bedrockRegion, setBedrockRegion] = useState('')

  // Meta for dimension defaults
  const embeddingMeta: Record<string, { dim: string; tokens: string }> = {
    'text-embedding-3-large': { dim: '3072', tokens: '8192' },
    'text-embedding-3-small': { dim: '1536', tokens: '8192' },
    'text-embedding-004': { dim: '768', tokens: '8192' },
    'embed-english-v3.0': { dim: '1024', tokens: '5120' },
    'embed-multilingual-v3.0': { dim: '1024', tokens: '5120' },
    'mistral-embed': { dim: '1024', tokens: '8192' },
    'nomic-embed-text': { dim: '768', tokens: '8192' },
    'BAAI/bge-large-en-v1.5': { dim: '1024', tokens: '8192' },
    'intfloat/e5-large-v2': { dim: '1024', tokens: '8192' },
    'cohere.embed-english-v3': { dim: '1024', tokens: '5120' },
    'amazon.titan-embed-text-v2:0': { dim: '1024', tokens: '8192' },
    'bge-m3': { dim: '1024', tokens: '8192' },
    'bge-small-en-v1.5': { dim: '384', tokens: '8192' },
    'bge-base-en-v1.5': { dim: '768', tokens: '8192' },
    'e5-base-v2': { dim: '768', tokens: '8192' },
    'all-MiniLM-L6-v2': { dim: '384', tokens: '8192' },
  }

  // Selected model for saving (set via 'Use' / 'Use cloud model')
  const [selected, setSelected] = useState<{
    runtime: 'Local' | 'Cloud'
    provider: string
    modelId: string
  } | null>(null)

  // Update defaults when provider/model changes
  useEffect(() => {
    const key = model === 'Custom' ? customModel.trim() : model
    const meta = key ? embeddingMeta[key] : undefined
    if (meta?.dim) {
      const d = Number(String(meta.dim).replace(/[^0-9]/g, ''))
      if (!Number.isNaN(d)) setDimension(d)
    }
    if (provider === 'Ollama (remote)') {
      setBaseUrl(prev => (prev ? prev : 'http://localhost:11434'))
    }
  }, [provider, model, customModel])

  // Confirm modal state
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [makeDefault, setMakeDefault] = useState(false)

  // Re-embed modal state
  const [reembedOpen, setReembedOpen] = useState(false)

  const isModelChosen =
    model === 'Custom' ? customModel.trim().length > 0 : !!model

  // Handlers to select models (no modal here)
  const selectLocal = (v: Variant) => {
    const meta = embeddingMeta[v.id]
    if (meta?.dim) setDimension(Number(String(meta.dim).replace(/[^0-9]/g, '')))
    if (!baseUrl.trim()) setBaseUrl('http://localhost:11434')
    setSelected({ runtime: 'Local', provider: 'Ollama', modelId: v.id })
  }

  const selectCloud = () => {
    const key = model === 'Custom' ? customModel.trim() : model
    const meta = key ? embeddingMeta[key] : undefined
    if (meta?.dim) setDimension(Number(String(meta.dim).replace(/[^0-9]/g, '')))
    setSelected({ runtime: 'Cloud', provider, modelId: key || 'Custom' })
  }

  // Derived summary
  const summaryProvider =
    selected?.runtime === 'Local' ? 'Ollama' : selected?.provider || null
  const summaryModel = selected?.modelId || null
  const summaryLocation = (() => {
    try {
      if (!baseUrl) return null
      const u = new URL(baseUrl)
      return `${u.hostname}${u.port ? `:${u.port}` : ''}`
    } catch {
      return baseUrl || null
    }
  })()

  // Validation for saving cloud strategies
  const validateCloud = (): string | null => {
    if (!selected || selected.runtime !== 'Cloud') return null
    if (!apiKey.trim()) return 'API key is required'
    if (provider === 'Azure OpenAI') {
      if (!azureDeployment.trim()) return 'Deployment name is required'
      if (!azureResource.trim()) return 'Endpoint/Resource name is required'
    }
    if (provider === 'Google') {
      if (!vertexProjectId.trim()) return 'Project ID is required'
      if (!vertexLocation.trim()) return 'Location/Region is required'
    }
    if (provider === 'AWS Bedrock') {
      if (!bedrockRegion.trim()) return 'Region is required'
    }
    return null
  }

  const createStrategy = (
    runtime: 'Local' | 'Cloud',
    providerLabel: string,
    chosenModel: string,
    encryptedApiKey?: string
  ) => {
    const EMB_LIST_KEY = 'lf_project_embeddings'
    const raw = localStorage.getItem(EMB_LIST_KEY)
    const list = raw ? JSON.parse(raw) : []
    const slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
    const id = `emb-${slug || Date.now()}`
    const exists = Array.isArray(list) && list.some((e: any) => e?.id === id)
    const finalId = exists ? `emb-${slug}-${Date.now()}` : id
    const entry = {
      id: finalId,
      name: name.trim(),
      isDefault: list.length === 0,
      enabled: true,
    }
    const nextList = Array.isArray(list) ? [...list, entry] : [entry]
    localStorage.setItem(EMB_LIST_KEY, JSON.stringify(nextList))

    const cfg: any = {
      runtime: runtime === 'Local' ? 'local' : 'cloud',
      provider: runtime === 'Local' ? 'local' : providerLabel,
      modelId: chosenModel,
      baseUrl: baseUrl.trim() || undefined,
      dimension,
      batchSize,
      timeout: timeoutSec,
      auto_pull: runtime === 'Local' ? autoPull : undefined,
      similarity: 'cosine',
    }

    if (runtime === 'Cloud') {
      if (providerLabel === 'OpenAI') {
        cfg.organization = openaiOrg.trim() || undefined
        cfg.maxRetries = openaiMaxRetries
      }
      if (providerLabel === 'Azure OpenAI') {
        cfg.deployment = azureDeployment.trim() || undefined
        cfg.endpoint = azureResource.trim() || undefined
        cfg.apiVersion = azureApiVersion.trim() || undefined
      }
      if (providerLabel === 'Google') {
        cfg.projectId = vertexProjectId.trim() || undefined
        cfg.endpoint = vertexEndpoint.trim() || undefined
        cfg.region = vertexLocation.trim() || undefined
      }
      if (providerLabel === 'AWS Bedrock') {
        cfg.region = bedrockRegion.trim() || undefined
      }
      if (encryptedApiKey) cfg.apiKey = encryptedApiKey
    }

    localStorage.setItem(
      `lf_strategy_embedding_config_${finalId}`,
      JSON.stringify(cfg)
    )
    localStorage.setItem(`lf_strategy_embedding_model_${finalId}`, chosenModel)

    if (makeDefault) {
      try {
        const curRaw = localStorage.getItem(EMB_LIST_KEY)
        const curList = curRaw ? JSON.parse(curRaw) : []
        const updated = curList.map((e: any) => ({
          ...e,
          isDefault: e.id === finalId,
        }))
        localStorage.setItem(EMB_LIST_KEY, JSON.stringify(updated))
      } catch {}
    }

    return finalId
  }

  const finalizeAndRedirect = () => {
    toast({ message: 'Embedding strategy created', variant: 'default' })
    navigate('/chat/rag')
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-40">
      <div className="flex items-center justify-between mb-1 md:mb-3">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/rag')}
          >
            RAG
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">Add embedding strategy</span>
        </nav>
        <div className="flex items-center gap-3">
          <div className="text-xs text-muted-foreground max-w-[50vw] truncate">
            {summaryProvider && summaryModel ? (
              <>
                <span className="text-foreground">{summaryProvider}</span>
                <span className="mx-1">•</span>
                <span className="font-mono">{summaryModel}</span>
                {summaryLocation ? (
                  <>
                    <span className="mx-1">•</span>
                    <span>{summaryLocation}</span>
                  </>
                ) : null}
              </>
            ) : (
              'No model selected yet'
            )}
          </div>
          <Button
            onClick={() => {
              if (!selected) return
              if (selected.runtime === 'Cloud') {
                const err = validateCloud()
                if (err) {
                  toast({ message: err, variant: 'destructive' })
                  return
                }
              }
              setConfirmOpen(true)
            }}
            disabled={!selected || name.trim().length === 0}
          >
            Save strategy
          </Button>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-card p-4 md:p-6 flex flex-col gap-4">
        {/* Name */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">
              Strategy name
            </Label>
            <Input
              value={name}
              onChange={e => setName(e.target.value)}
              placeholder="Enter a name"
              className="h-9"
            />
          </div>
        </div>

        {/* Settings at top */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">Base URL</Label>
            <Input
              placeholder="http://localhost:11434"
              value={baseUrl}
              onChange={e => setBaseUrl(e.target.value)}
              className="h-9"
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">Dimension</Label>
            <Input
              type="number"
              value={dimension}
              onChange={e =>
                setDimension(Math.max(1, Number(e.target.value || 768)))
              }
              className="h-9"
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">Batch size</Label>
            <Input
              type="number"
              value={batchSize}
              onChange={e =>
                setBatchSize(
                  Math.min(512, Math.max(1, Number(e.target.value || 16)))
                )
              }
              className="h-9"
            />
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">
              Timeout (sec)
            </Label>
            <Input
              type="number"
              value={timeoutSec}
              onChange={e =>
                setTimeoutSec(
                  Math.min(600, Math.max(10, Number(e.target.value || 60)))
                )
              }
              className="h-9"
            />
          </div>
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">
              Auto-pull model
            </Label>
            <button
              className="h-9 rounded-md border border-border bg-background px-3 text-left"
              onClick={() => setAutoPull(v => !v)}
            >
              {autoPull ? 'Enabled' : 'Disabled'}
            </button>
          </div>
        </div>

        {/* Selection UI */}
        <div className="flex items-center justify-between">
          <div className="text-sm text-muted-foreground">
            Select the model you would like to use for this strategy.
          </div>
          <div />
        </div>

        <div className="w-full flex items-center">
          <div className="flex w-full max-w-3xl rounded-lg overflow-hidden border border-border">
            <button
              className={`flex-1 h-10 text-sm ${sourceTab === 'local' ? 'bg-primary text-primary-foreground' : 'text-foreground hover:bg-secondary/80'}`}
              onClick={() => setSourceTab('local')}
              aria-pressed={sourceTab === 'local'}
            >
              Local models
            </button>
            <button
              className={`flex-1 h-10 text-sm ${sourceTab === 'cloud' ? 'bg-primary text-primary-foreground' : 'text-foreground hover:bg-secondary/80'}`}
              onClick={() => setSourceTab('cloud')}
              aria-pressed={sourceTab === 'cloud'}
            >
              Cloud models
            </button>
          </div>
        </div>

        {sourceTab === 'local' && (
          <>
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
            <div className="w-full overflow-hidden rounded-lg border border-border">
              <div className="grid grid-cols-12 items-center bg-secondary text-secondary-foreground text-xs px-3 py-2">
                <div className="col-span-4">Model</div>
                <div className="col-span-2">dim</div>
                <div className="col-span-2">Quality</div>
                <div className="col-span-2">Download</div>
                <div className="col-span-1">RAM/VRAM</div>
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
                      <div className="col-span-4 flex items-center gap-2">
                        <FontIcon
                          type="chevron-down"
                          className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                        />
                        <span className="truncate font-medium">
                          {group.name}
                        </span>
                      </div>
                      <div className="col-span-2 text-xs text-muted-foreground">
                        {group.dim}
                      </div>
                      <div className="col-span-2 text-xs text-muted-foreground">
                        {group.quality}
                      </div>
                      <div className="col-span-2 text-xs text-muted-foreground">
                        {group.download}
                      </div>
                      <div className="col-span-1 text-xs text-muted-foreground">
                        {group.ramVram}
                      </div>
                      <div className="col-span-1" />
                    </div>
                    {group.variants && isOpen && (
                      <div className="px-3 pb-2">
                        {group.variants.map(v => {
                          const isUsing =
                            selected?.runtime === 'Local' &&
                            selected?.modelId === v.id
                          return (
                            <div
                              key={v.id}
                              className="grid grid-cols-12 items-center px-3 py-3 text-sm rounded-md hover:bg-accent/40"
                            >
                              <div className="col-span-4 flex items-center text-muted-foreground">
                                <span className="inline-block w-4" />
                                <span className="ml-2 font-mono text-xs truncate">
                                  {v.label}
                                </span>
                              </div>
                              <div className="col-span-2 text-xs text-muted-foreground">
                                {v.dim}
                              </div>
                              <div className="col-span-2 text-xs text-muted-foreground">
                                {v.quality}
                              </div>
                              <div className="col-span-2 text-xs text-muted-foreground">
                                {group.download}
                              </div>
                              <div className="col-span-1 text-xs text-muted-foreground">
                                {group.ramVram}
                              </div>
                              <div className="col-span-1 flex items-center justify-end pr-2">
                                <Button
                                  size="sm"
                                  className="h-8 px-3"
                                  onClick={() => selectLocal(v)}
                                >
                                  {isUsing ? (
                                    <span className="inline-flex items-center gap-1">
                                      <FontIcon
                                        type="checkmark-filled"
                                        className="w-4 h-4"
                                      />{' '}
                                      Using
                                    </span>
                                  ) : (
                                    'Use'
                                  )}
                                </Button>
                              </div>
                            </div>
                          )
                        })}
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
          </>
        )}

        {sourceTab === 'cloud' && (
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
              <Label className="text-xs text-muted-foreground">
                Select embedding model
              </Label>
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

            {/* Provider-specific fields */}
            {provider === 'OpenAI' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Base URL (optional)
                  </Label>
                  <Input
                    placeholder="https://api.openai.com"
                    value={baseUrl}
                    onChange={e => setBaseUrl(e.target.value)}
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Organization (optional)
                  </Label>
                  <Input
                    placeholder="org_xxx"
                    value={openaiOrg}
                    onChange={e => setOpenaiOrg(e.target.value)}
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Timeout (sec)
                  </Label>
                  <Input
                    type="number"
                    value={timeoutSec}
                    onChange={e => setTimeoutSec(Number(e.target.value || 60))}
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Max retries
                  </Label>
                  <Input
                    type="number"
                    value={openaiMaxRetries}
                    onChange={e =>
                      setOpenaiMaxRetries(Number(e.target.value || 3))
                    }
                    className="h-9"
                  />
                </div>
              </div>
            )}

            {provider === 'Azure OpenAI' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Deployment name
                  </Label>
                  <Input
                    placeholder="my-embed-deployment"
                    value={azureDeployment}
                    onChange={e => setAzureDeployment(e.target.value)}
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Endpoint/Resource name
                  </Label>
                  <Input
                    placeholder="my-azure-openai"
                    value={azureResource}
                    onChange={e => setAzureResource(e.target.value)}
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    API version (optional)
                  </Label>
                  <Input
                    placeholder="2024-02-15-preview"
                    value={azureApiVersion}
                    onChange={e => setAzureApiVersion(e.target.value)}
                    className="h-9"
                  />
                </div>
              </div>
            )}

            {provider === 'Google' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Project ID
                  </Label>
                  <Input
                    placeholder="my-gcp-project"
                    value={vertexProjectId}
                    onChange={e => setVertexProjectId(e.target.value)}
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Location/Region
                  </Label>
                  <Input
                    placeholder="us-central1"
                    value={vertexLocation}
                    onChange={e => setVertexLocation(e.target.value)}
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Endpoint (optional)
                  </Label>
                  <Input
                    placeholder="optional override"
                    value={vertexEndpoint}
                    onChange={e => setVertexEndpoint(e.target.value)}
                    className="h-9"
                  />
                </div>
              </div>
            )}

            {provider === 'AWS Bedrock' && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Region
                  </Label>
                  <Input
                    placeholder="us-east-1"
                    value={bedrockRegion}
                    onChange={e => setBedrockRegion(e.target.value)}
                    className="h-9"
                  />
                </div>
              </div>
            )}

            {/* API key common for cloud providers */}
            {provider !== 'Ollama (remote)' && (
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
              </div>
            )}

            <div className="flex justify-end">
              <Button
                disabled={
                  !isModelChosen ||
                  (provider !== 'Ollama (remote)' && apiKey.trim().length === 0)
                }
                onClick={selectCloud}
              >
                {selected?.runtime === 'Cloud' &&
                selected?.modelId ===
                  (model === 'Custom' ? customModel.trim() : model) ? (
                  <span className="inline-flex items-center gap-1">
                    <FontIcon type="checkmark-filled" className="w-4 h-4" />{' '}
                    Using
                  </span>
                ) : (
                  'Use cloud model'
                )}
              </Button>
            </div>
          </div>
        )}

        {/* Confirm selection modal with Make default */}
        <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
          <DialogContent>
            <DialogTitle>Save this embedding strategy?</DialogTitle>
            <DialogDescription>
              {selected && (
                <div className="mt-2 text-sm">
                  You are selecting
                  <span className="mx-1 font-medium text-foreground">
                    {selected.modelId}
                  </span>
                  using
                  <span className="mx-1 font-medium text-foreground">
                    {summaryProvider}
                  </span>
                  .
                  <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                    <div className="text-muted-foreground">Runtime</div>
                    <div>{selected.runtime}</div>
                    <div className="text-muted-foreground">
                      Vector dimension (d)
                    </div>
                    <div>{dimension}</div>
                    {summaryLocation ? (
                      <>
                        <div className="text-muted-foreground">Location</div>
                        <div>{summaryLocation}</div>
                      </>
                    ) : null}
                  </div>
                  <div className="mt-3 flex items-center gap-2 text-xs">
                    <input
                      id="make-default"
                      type="checkbox"
                      checked={makeDefault}
                      onChange={e => setMakeDefault(e.target.checked)}
                    />
                    <label htmlFor="make-default">Make default strategy</label>
                  </div>
                </div>
              )}
            </DialogDescription>
            <DialogFooter>
              <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
                Cancel
              </Button>
              <Button
                onClick={async () => {
                  if (!selected) return
                  let encryptedKey: string | undefined = undefined
                  try {
                    if (selected.runtime === 'Cloud' && apiKey.trim()) {
                      encryptedKey = await encryptAPIKey(
                        apiKey.trim(),
                        getClientSideSecret()
                      )
                    }
                  } catch (e) {
                    toast({
                      message: 'Failed to encrypt API key',
                      variant: 'destructive',
                    })
                    return
                  }
                  createStrategy(
                    selected.runtime,
                    summaryProvider || 'Provider',
                    selected.modelId,
                    encryptedKey
                  )
                  setConfirmOpen(false)
                  if (makeDefault) {
                    setReembedOpen(true)
                  } else {
                    finalizeAndRedirect()
                  }
                }}
              >
                {selected?.runtime === 'Local'
                  ? 'Download and save strategy'
                  : 'Save strategy'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Re-embed confirmation modal */}
        <Dialog open={reembedOpen} onOpenChange={setReembedOpen}>
          <DialogContent>
            <DialogTitle>Re-embed project data?</DialogTitle>
            <DialogDescription>
              To keep your project running smoothly, this change requires
              re-embedding project data. Would you like to proceed now?
            </DialogDescription>
            <DialogFooter>
              <Button
                variant="destructive"
                onClick={() => finalizeAndRedirect()}
              >
                I'll do it later
              </Button>
              <Button onClick={() => finalizeAndRedirect()}>
                Yes, proceed with re-embed
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}

export default AddEmbeddingStrategy
