import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { useToast } from '../ui/toast'
import { Label } from '../ui/label'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import Loader from '../../common/Loader'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'

// Helper for symmetric AES encryption using Web Crypto API
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
    {
      name: 'AES-GCM',
      iv: iv,
    },
    key,
    enc.encode(apiKey)
  )
  // encode salt, iv, ciphertext as base64 for storage
  function base64(arrayBuffer: ArrayBuffer) {
    return window.btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)))
  }
  return JSON.stringify({
    salt: base64(salt.buffer),
    iv: base64(iv.buffer),
    data: base64(ciphertext),
  })
}

function ChangeEmbeddingModel() {
  const navigate = useNavigate()
  const { strategyId } = useParams()
  const { toast } = useToast()

  const strategyName = useMemo(() => {
    if (!strategyId) return 'Strategy'
    return strategyId
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
  }, [strategyId])

  const [currentModel, setCurrentModel] = useState<string>(
    'text-embedding-3-large'
  )

  useEffect(() => {
    try {
      if (!strategyId) return
      const storedCfg = localStorage.getItem(
        `lf_strategy_embedding_config_${strategyId}`
      )
      if (storedCfg) {
        const parsed = JSON.parse(storedCfg)
        if (parsed?.modelId) setCurrentModel(parsed.modelId)
      }
      const storedModel = localStorage.getItem(
        `lf_strategy_embedding_model_${strategyId}`
      )
      if (storedModel) setCurrentModel(storedModel)
    } catch {}
  }, [strategyId])

  // UI state (same structure as original component)
  const [sourceTab, setSourceTab] = useState<'local' | 'cloud'>('local')
  const [query, setQuery] = useState('')
  const [expandedGroupId, setExpandedGroupId] = useState<number | null>(null)
  const [isApplying, setIsApplying] = useState<string | null>(null)

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

  const filteredProviderOptions = providerOptions.filter(
    p => (modelMap as any)[p]?.length > 0
  )

  const [provider, setProvider] = useState<Provider>('OpenAI')
  const [model, setModel] = useState<string>(modelMap['OpenAI'][0])
  const [customModel, setCustomModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [baseUrl, setBaseUrl] = useState('')
  const [maxTokens, setMaxTokens] = useState<number | null>(null)
  const [submitState, setSubmitState] = useState<
    'idle' | 'loading' | 'success'
  >('idle')
  const [hasPickedModel, setHasPickedModel] = useState(false)
  const [batchSize, setBatchSize] = useState<number>(64)
  const [errors, setErrors] = useState<Record<string, string>>({})
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [confirmCtx, setConfirmCtx] = useState<{
    runtime: 'Local' | 'Cloud'
    provider: string
    modelId: string
    dimension: number
    maxInputTokens: number
    similarity: 'cosine'
    downloadSize?: string
    ramHint?: string
    region?: string
  } | null>(null)
  const [pendingLocalModelId, setPendingLocalModelId] = useState<string | null>(
    null
  )
  const [azureDeployment, setAzureDeployment] = useState('')
  const [azureResource, setAzureResource] = useState('')
  const [azureApiVersion, setAzureApiVersion] = useState('')
  const [vertexProjectId, setVertexProjectId] = useState('')
  const [vertexLocation, setVertexLocation] = useState('')
  const [vertexEndpoint, setVertexEndpoint] = useState('')
  const [bedrockRegion, setBedrockRegion] = useState('')

  const modelsForProvider = [...modelMap[provider]]
  const isModelChosen =
    model === 'Custom' ? customModel.trim().length > 0 : !!model
  const hasApiAuth = apiKey.trim().length > 0
  const providerRequiredOk =
    (provider !== 'Azure OpenAI' ||
      (azureDeployment.trim() && azureResource.trim())) &&
    (provider !== 'Google' ||
      (vertexProjectId.trim() && vertexLocation.trim())) &&
    (provider !== 'AWS Bedrock' || bedrockRegion.trim().length > 0)
  const canApply = isModelChosen && hasApiAuth && providerRequiredOk

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
    'gte-base': { dim: '768', tokens: '8192' },
    'gte-large': { dim: '1024', tokens: '8192' },
    'jina-embeddings-v2-small-en': { dim: '384', tokens: '8192' },
    'jina-embeddings-v2-base-en': { dim: '768', tokens: '8192' },
  }
  const selectedKey = model === 'Custom' ? customModel.trim() : model
  const meta = hasPickedModel ? embeddingMeta[selectedKey] : undefined

  const persistForStrategy = (payload: any) => {
    if (!strategyId) return
    try {
      localStorage.setItem(
        'lf_last_embedding_provider_config',
        JSON.stringify(payload)
      )
      localStorage.setItem(
        `lf_strategy_embedding_config_${strategyId}`,
        JSON.stringify(payload)
      )
      if (payload?.modelId) {
        localStorage.setItem(
          `lf_strategy_embedding_model_${strategyId}`,
          payload.modelId
        )
      }
      if (typeof window !== 'undefined') {
        try {
          // Notify listeners (e.g., StrategyView) of model change
          window.dispatchEvent(
            new CustomEvent('lf:strategyEmbeddingUpdated', {
              detail: { strategyId, modelId: payload?.modelId },
            })
          )
        } catch {}
      }
    } catch {}
  }

  const handleApplyCloud = async () => {
    if (!strategyId || submitState === 'loading') return
    const nextErrors: Record<string, string> = {}
    if (model === 'Custom' && !customModel.trim())
      nextErrors.customModel = 'Enter a custom model id'
    if (!apiKey.trim()) nextErrors.apiKey = 'API key is required'
    if (provider === 'Azure OpenAI') {
      if (!azureDeployment.trim())
        nextErrors.azureDeployment = 'Deployment name is required'
      if (!azureResource.trim())
        nextErrors.azureResource = 'Endpoint/Resource name is required'
    }
    if (provider === 'Google') {
      if (!vertexProjectId.trim())
        nextErrors.vertexProjectId = 'Project ID is required'
      if (!vertexLocation.trim())
        nextErrors.vertexLocation = 'Location/Region is required'
    }
    if (provider === 'AWS Bedrock') {
      if (!bedrockRegion.trim()) nextErrors.bedrockRegion = 'Region is required'
    }
    setErrors(nextErrors)
    if (Object.keys(nextErrors).length > 0) return
    if (!canApply) return
    const chosen = model === 'Custom' ? customModel.trim() : model

    const metaVals = embeddingMeta[chosen] || { dim: '1536', tokens: '8192' }
    const payload: any = {
      runtime: 'cloud',
      provider,
      modelId: chosen,
      apiKey: undefined, // will be set after encryption below
      region:
        provider === 'AWS Bedrock'
          ? bedrockRegion.trim() || undefined
          : undefined,
      endpoint:
        provider === 'Azure OpenAI'
          ? azureResource.trim() || undefined
          : provider === 'Google'
            ? vertexEndpoint.trim() || undefined
            : undefined,
      projectId:
        provider === 'Google' ? vertexProjectId.trim() || undefined : undefined,
      deployment:
        provider === 'Azure OpenAI'
          ? azureDeployment.trim() || undefined
          : undefined,
      apiVersion:
        provider === 'Azure OpenAI'
          ? azureApiVersion.trim() || undefined
          : undefined,
      batchSize,
      dimension: Number(metaVals.dim) || 0,
      maxInputTokens: Number(metaVals.tokens) || 0,
      similarity: 'cosine',
    }

    // Encrypt the apiKey before storage; use a static project-level secret or derive from e.g. strategyId, or use non-sensitive fallback if none available.
    const secret = strategyId || 'default-project-secret' // Should be rotated/secured in production
    if (apiKey.trim()) {
      payload.apiKey = await encryptAPIKey(apiKey.trim(), secret)
    } else {
      payload.apiKey = undefined
    }

    persistForStrategy(payload)

    setSubmitState('loading')
    setTimeout(() => {
      setSubmitState('success')
      setCurrentModel(chosen)
      setTimeout(() => {
        toast({
          message: `Embedding model set to ${chosen}`,
          variant: 'default',
        })
        navigate(`/chat/rag/${strategyId}`)
        setSubmitState('idle')
      }, 500)
    }, 800)
  }

  const applyEmbedding = (modelId: string) => {
    if (!strategyId) return
    setIsApplying(modelId)
    persistForStrategy({ runtime: 'local', provider: 'local', modelId })
    setTimeout(() => {
      setCurrentModel(modelId)
      toast({
        message: `Embedding model set to ${modelId}`,
        variant: 'default',
      })
      navigate(`/chat/rag/${strategyId}`)
    }, 400)
  }

  const openConfirmLocal = (group: any, variant: Variant) => {
    const metaVals = embeddingMeta[variant.id] || {
      dim: group.dim,
      tokens: '8192',
    }
    setConfirmCtx({
      runtime: 'Local',
      provider: group.name,
      modelId: variant.id,
      dimension: Number(String(metaVals.dim).replace(/[^0-9]/g, '')) || 0,
      maxInputTokens:
        Number(
          String((metaVals as any).tokens || '8192').replace(/[^0-9]/g, '')
        ) || 0,
      similarity: 'cosine',
      downloadSize: variant.download || group.download,
      ramHint: group.ramVram,
    })
    setPendingLocalModelId(variant.id)
    setConfirmOpen(true)
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-40">
      {/* Breadcrumb */}
      <nav className="text-sm md:text-base flex items-center gap-1.5 mb-3">
        <button
          className="text-teal-600 dark:text-teal-400 hover:underline"
          onClick={() => navigate('/chat/rag')}
        >
          RAG
        </button>
        <span className="text-muted-foreground px-1">/</span>
        <button
          className="text-teal-600 dark:text-teal-400 hover:underline"
          onClick={() => navigate(`/chat/rag/${strategyId}`)}
        >
          {strategyName}
        </button>
        <span className="text-muted-foreground px-1">/</span>
        <span className="text-foreground">Change embedding model</span>
      </nav>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg md:text-xl font-medium">
          Change embedding model
        </h2>
        <Button
          variant="outline"
          size="sm"
          onClick={() => navigate(`/chat/rag/${strategyId}`)}
        >
          Back
        </Button>
      </div>

      {/* Current model */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Current model</h3>
          <Badge variant="secondary" size="sm" className="rounded-xl">
            Active
          </Badge>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Input
            readOnly
            className="bg-background max-w-xl"
            value={currentModel}
          />
          <Badge variant="secondary" size="sm" className="rounded-xl">
            1536-d
          </Badge>
        </div>
      </section>

      {/* The rest mirrors the original ChangeEmbeddingModel UI */}
      <section className="rounded-lg border border-border bg-card p-4 md:p-6 flex flex-col gap-4">
        <div className="text-sm text-muted-foreground">
          Select a new embedding model. This mirrors the models flow and uses
          the same styles.
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

        {sourceTab === 'local' && (
          <div className="w-full overflow-hidden rounded-lg border border-border">
            <div className="grid grid-cols-12 items-center bg-secondary text-secondary-foreground text-xs px-3 py-2">
              <div className="col-span-4">Model</div>
              <div className="col-span-2">dim</div>
              <div className="col-span-2">Quality</div>
              <div className="col-span-2">RAM/VRAM</div>
              <div className="col-span-1 text-right">Download</div>
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
                      <span className="truncate font-medium">{group.name}</span>
                    </div>
                    <div className="col-span-2 text-xs text-muted-foreground">
                      {group.dim}
                    </div>
                    <div className="col-span-2 text-xs text-muted-foreground">
                      {group.quality}
                    </div>
                    <div className="col-span-2 text-xs text-muted-foreground">
                      {group.ramVram}
                    </div>
                    <div className="col-span-1 text-xs text-muted-foreground text-right">
                      {group.download}
                    </div>
                    <div className="col-span-1" />
                  </div>
                  {group.variants && isOpen && (
                    <div className="px-3 pb-2">
                      {group.variants.map(v => (
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
                            {group.ramVram}
                          </div>
                          <div className="col-span-1 text-xs text-muted-foreground text-right">
                            {v.download}
                          </div>
                          <div className="col-span-1 flex items-center justify-end pr-2">
                            <Button
                              size="sm"
                              className="h-8 px-3"
                              onClick={() => openConfirmLocal(group, v)}
                              disabled={isApplying !== null}
                            >
                              {isApplying === v.id ? 'Using…' : 'Use'}
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
                  {filteredProviderOptions.map(p => (
                    <DropdownMenuItem
                      key={p}
                      className="w-full justify-start text-left"
                      onClick={() => {
                        setProvider(p)
                        setModel(modelMap[p][0])
                        setHasPickedModel(false)
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
                      onClick={() => {
                        setModel(m)
                        setHasPickedModel(true)
                      }}
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
              {errors.customModel && (
                <div className="text-xs text-destructive">
                  {errors.customModel}
                </div>
              )}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mt-1">
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Vector dimension (d)
                  </Label>
                  <Input
                    value={meta?.dim ?? 'n/a'}
                    readOnly
                    disabled
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Model input limit (tokens)
                  </Label>
                  <Input
                    value={meta?.tokens ?? 'n/a'}
                    readOnly
                    disabled
                    className="h-9"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label className="text-xs text-muted-foreground">
                    Similarity metric
                  </Label>
                  <Input
                    value={hasPickedModel ? 'cosine' : 'n/a'}
                    readOnly
                    disabled
                    className="h-9"
                  />
                </div>
              </div>
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
                    {errors.azureDeployment && (
                      <div className="text-xs text-destructive">
                        {errors.azureDeployment}
                      </div>
                    )}
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
                    {errors.azureResource && (
                      <div className="text-xs text-destructive">
                        {errors.azureResource}
                      </div>
                    )}
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
                    {errors.vertexProjectId && (
                      <div className="text-xs text-destructive">
                        {errors.vertexProjectId}
                      </div>
                    )}
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
                    {errors.vertexLocation && (
                      <div className="text-xs text-destructive">
                        {errors.vertexLocation}
                      </div>
                    )}
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
                    {errors.bedrockRegion && (
                      <div className="text-xs text-destructive">
                        {errors.bedrockRegion}
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            <div className="flex flex-col gap-2">
              <Label className="text-xs text-muted-foreground">API Key</Label>
              <div className="relative">
                <Input
                  type="password"
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
              {errors.apiKey && (
                <div className="text-xs text-destructive">{errors.apiKey}</div>
              )}
              <div className="flex flex-col gap-1">
                <Label className="text-xs text-muted-foreground">
                  Batch size (texts per request)
                </Label>
                <Input
                  type="number"
                  min={1}
                  max={512}
                  value={batchSize}
                  onChange={e => {
                    const v = parseInt(e.target.value || '64', 10)
                    if (Number.isNaN(v)) return
                    const clamped = Math.max(1, Math.min(512, v))
                    setBatchSize(clamped)
                  }}
                  className="h-9"
                />
                <div className="text-xs text-muted-foreground">
                  Controls throughput and cost; provider limits may apply.
                </div>
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
                      setMaxTokens(prev =>
                        prev ? Math.max(prev - 500, 0) : null
                      )
                    }
                  >
                    –
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-8 w-8"
                    onClick={() =>
                      setMaxTokens(prev => (prev ? prev + 500 : 500))
                    }
                  >
                    +
                  </Button>
                </div>
              </div>
            </div>

            <div className="flex justify-end">
              <Button
                onClick={() => {
                  if (
                    apiKey.trim().length === 0 ||
                    !isModelChosen ||
                    !providerRequiredOk
                  )
                    return
                  const chosen = model === 'Custom' ? customModel.trim() : model
                  const metaVals = embeddingMeta[chosen] || {
                    dim: '1536',
                    tokens: '8192',
                  }
                  setConfirmCtx({
                    runtime: 'Cloud',
                    provider,
                    modelId: chosen,
                    dimension: Number(metaVals.dim) || 0,
                    maxInputTokens: Number(metaVals.tokens) || 0,
                    similarity: 'cosine',
                    region:
                      provider === 'Azure OpenAI'
                        ? azureResource || undefined
                        : provider === 'Google'
                          ? vertexLocation || undefined
                          : provider === 'AWS Bedrock'
                            ? bedrockRegion || undefined
                            : undefined,
                  })
                  setConfirmOpen(true)
                }}
                disabled={
                  apiKey.trim().length === 0 ||
                  !isModelChosen ||
                  !providerRequiredOk ||
                  submitState === 'loading'
                }
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
                Apply cloud embedding model
              </Button>
            </div>
          </div>
        )}
      </section>

      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent>
          <DialogTitle>Use this embedding model?</DialogTitle>
          <DialogDescription>
            {confirmCtx && (
              <div className="mt-2 text-sm">
                Are you sure you want to start using
                <span className="mx-1 font-medium text-foreground">
                  {confirmCtx.modelId}
                </span>
                for the
                <span className="mx-1 font-medium text-foreground">
                  {strategyName}
                </span>
                strategy? We’ll download the model if needed and reprocess to
                keep results accurate.
                <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                  <div className="text-muted-foreground">Runtime</div>
                  <div>{confirmCtx.runtime}</div>
                  <div className="text-muted-foreground">
                    Vector dimension (d)
                  </div>
                  <div>{confirmCtx.dimension}</div>
                  <div className="text-muted-foreground">
                    Input limit (tokens)
                  </div>
                  <div>{confirmCtx.maxInputTokens}</div>
                  <div className="text-muted-foreground">Similarity</div>
                  <div>cosine</div>
                  {confirmCtx.region && (
                    <>
                      <div className="text-muted-foreground">Region</div>
                      <div>{confirmCtx.region}</div>
                    </>
                  )}
                </div>
              </div>
            )}
          </DialogDescription>
          <DialogFooter>
            <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
              Cancel
            </Button>
            <Button
              onClick={() => {
                setConfirmOpen(false)
                if (confirmCtx?.runtime === 'Local' && pendingLocalModelId) {
                  applyEmbedding(pendingLocalModelId)
                  setPendingLocalModelId(null)
                } else {
                  handleApplyCloud()
                }
              }}
            >
              {confirmCtx?.runtime === 'Cloud'
                ? 'Use cloud model'
                : 'Download and use'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default ChangeEmbeddingModel
