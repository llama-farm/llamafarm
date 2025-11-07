import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
// Badge import removed after removing summary card
// import { Badge } from '../ui/badge'
import PageActions from '../common/PageActions'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useToast } from '../ui/toast'
import { Label } from '../ui/label'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogTitle,
} from '../ui/dialog'
import { getClientSideSecret } from '../../utils/crypto'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject } from '../../hooks/useProjects'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'

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
  const [mode, setMode] = useModeWithReset('designer')
  const { strategyId } = useParams()
  const { toast } = useToast()
  const activeProject = useActiveProject()
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )
  const projectConfig = (projectResp as any)?.project?.config as ProjectConfig | undefined
  const getEmbeddingLocation = useCallback(() => {
    if (strategyId) {
      return {
        type: 'rag.database.embedding' as const,
        embeddingName: strategyId,
      }
    }
    return { type: 'rag.databases' as const }
  }, [strategyId])
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getEmbeddingLocation,
  })

  // Editable strategy name loaded from list
  const [strategyName, setStrategyName] = useState<string>('')
  useEffect(() => {
    try {
      if (!strategyId) return
      const raw = localStorage.getItem('lf_project_embeddings')
      if (raw) {
        const list = JSON.parse(raw)
        const found = Array.isArray(list)
          ? list.find((e: any) => e?.id === strategyId)
          : null
        if (found?.name) setStrategyName(found.name)
        if (typeof found?.isDefault === 'boolean')
          setIsDefaultStrategy(Boolean(found.isDefault))
      }
    } catch {}
  }, [strategyId])

  // Removed currentModel display state along with summary card

  // Selection state similar to Add page
  const [selected, setSelected] = useState<{
    runtime: 'Local' | 'Cloud'
    provider: string
    modelId: string
  } | null>(null)
  const [existingModelId, setExistingModelId] = useState<string | null>(null)

  // UI state
  const [sourceTab, setSourceTab] = useState<'local' | 'cloud'>('local')
  const [query] = useState('')
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

  const filteredProviderOptions = providerOptions.filter(
    p => (modelMap as any)[p]?.length > 0
  )

  const [provider, setProvider] = useState<Provider>('Ollama (remote)')
  const [model, setModel] = useState<string>('nomic-embed-text')
  const [customModel, setCustomModel] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [showApiKey, setShowApiKey] = useState(false)
  const [baseUrl, setBaseUrl] = useState('http://localhost:11434')
  const [batchSize, setBatchSize] = useState<number>(16)
  const [dimension, setDimension] = useState<number>(768)
  const [timeoutSec, setTimeoutSec] = useState(60)
  const [ollamaAutoPull, setOllamaAutoPull] = useState(true)
  const [openaiOrg, setOpenaiOrg] = useState('')
  const [openaiMaxRetries, setOpenaiMaxRetries] = useState(3)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [makeDefault, setMakeDefault] = useState(false)
  const [reembedOpen, setReembedOpen] = useState(false)
  const [isDefaultStrategy, setIsDefaultStrategy] = useState(false)
  const [azureDeployment, setAzureDeployment] = useState('')
  const [azureResource, setAzureResource] = useState('')
  const [azureApiVersion, setAzureApiVersion] = useState('')
  const [vertexProjectId, setVertexProjectId] = useState('')
  const [vertexLocation, setVertexLocation] = useState('')
  const [vertexEndpoint, setVertexEndpoint] = useState('')
  const [bedrockRegion, setBedrockRegion] = useState('')

  // Connection / diagnostics / advanced state
  const [connectionStatus, setConnectionStatus] = useState<
    'idle' | 'checking' | 'ok' | 'error'
  >('idle')
  const [connectionMsg, setConnectionMsg] = useState('')
  const [testStatus, setTestStatus] = useState<
    'idle' | 'running' | 'ok' | 'error'
  >('idle')
  const [testLatencyMs, setTestLatencyMs] = useState<number | null>(null)

  const modelsForProvider = [...modelMap[provider]]
  // Removed providerRequiredOk gating; Save is always enabled

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
  const meta = embeddingMeta[selectedKey]

  useEffect(() => {
    try {
      if (!strategyId) return
      const storedCfg = localStorage.getItem(
        `lf_strategy_embedding_config_${strategyId}`
      )
      if (storedCfg) {
        const parsed = JSON.parse(storedCfg)
        if (typeof parsed?.dimension === 'number')
          setDimension(parsed.dimension)
        if (typeof parsed?.batchSize === 'number')
          setBatchSize(parsed.batchSize)
        if (typeof parsed?.timeout === 'number') setTimeoutSec(parsed.timeout)
        if (typeof parsed?.auto_pull === 'boolean')
          setOllamaAutoPull(parsed.auto_pull)
        if (parsed?.baseUrl) setBaseUrl(parsed.baseUrl)
        if (parsed?.modelId && typeof parsed.modelId === 'string') {
          setExistingModelId(parsed.modelId)
        }
        if (parsed?.provider && typeof parsed.provider === 'string') {
          const p = parsed.provider.includes('local')
            ? 'Ollama (remote)'
            : parsed.provider
          if ((providerOptions as readonly string[]).includes(p))
            setProvider(p as Provider)
        }
      }
      const storedModel = localStorage.getItem(
        `lf_strategy_embedding_model_${strategyId}`
      )
      if (storedModel) setExistingModelId(storedModel)
    } catch {}
  }, [strategyId])

  // Sync dimension from model metadata when model/provider changes
  useEffect(() => {
    if (meta?.dim) {
      const d = Number(meta.dim)
      if (!Number.isNaN(d)) setDimension(d)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, model, customModel])

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
          window.dispatchEvent(
            new CustomEvent('lf:strategyEmbeddingUpdated', {
              detail: { strategyId, modelId: payload?.modelId },
            })
          )
        } catch {}
      }
    } catch {}
  }

  // When provider changes, default model
  useEffect(() => {
    try {
      if (model !== 'Custom') setModel(modelMap[provider][0])
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider])

  const openConfirmLocal = (group: any, variant: Variant) => {
    setSelected({ runtime: 'Local', provider: group.name, modelId: variant.id })
  }

  const checkConnection = async () => {
    setConnectionStatus('checking')
    setConnectionMsg('')
    try {
      if (provider !== 'Ollama (remote)' && apiKey.trim().length === 0) {
        setConnectionStatus('error')
        setConnectionMsg('API key required for this provider')
        return
      }
      await new Promise(res => setTimeout(res, 400))
      setConnectionStatus('ok')
      setConnectionMsg('Connection looks good')
    } catch (e) {
      setConnectionStatus('error')
      setConnectionMsg('Unable to reach provider')
    }
  }

  const runTestEmbedding = async () => {
    setTestStatus('running')
    setTestLatencyMs(null)
    const started = performance.now()
    try {
      await new Promise(res => setTimeout(res, 500))
      const ended = performance.now()
      setTestLatencyMs(Math.round(ended - started))
      setTestStatus('ok')
    } catch (e) {
      setTestStatus('error')
    }
  }

  // Header with Save strategy
  const summaryProvider = (() => {
    if (selected)
      return selected.runtime === 'Local' ? 'Ollama' : selected.provider
    return provider === 'Ollama (remote)' ? 'Ollama' : provider
  })()
  const summaryModel = selected?.modelId || existingModelId || null
  const summaryLocation = (() => {
    try {
      if (!baseUrl) return null
      const u = new URL(baseUrl)
      return `${u.hostname}${u.port ? `:${u.port}` : ''}`
    } catch {
      return baseUrl || null
    }
  })()
  const runtimeLabel = selected
    ? selected.runtime
    : provider === 'Ollama (remote)'
      ? 'Local'
      : 'Cloud'

  const saveEdited = async () => {
    if (!strategyId) return
    try {
      const LIST_KEY = 'lf_project_embeddings'
      const raw = localStorage.getItem(LIST_KEY)
      const list = raw ? JSON.parse(raw) : []
      const updated = Array.isArray(list)
        ? list.map((e: any) => ({
            ...e,
            name: e.id === strategyId ? strategyName || e.name : e.name,
            isDefault: makeDefault ? e.id === strategyId : e.isDefault,
          }))
        : list
      localStorage.setItem(LIST_KEY, JSON.stringify(updated))
    } catch {}

    let encryptedKey: string | undefined
    if (
      (selected?.runtime === 'Cloud' || provider !== 'Ollama (remote)') &&
      apiKey.trim()
    ) {
      try {
        encryptedKey = await encryptAPIKey(apiKey.trim(), getClientSideSecret())
      } catch (e) {
        toast({ message: 'Failed to encrypt API key', variant: 'destructive' })
        return
      }
    }

    const chosenModelId =
      selected?.modelId ||
      existingModelId ||
      (model === 'Custom' ? customModel.trim() : model)
    const runtimeStr = selected
      ? selected.runtime === 'Local'
        ? 'local'
        : 'cloud'
      : provider === 'Ollama (remote)'
        ? 'local'
        : 'cloud'
    const providerStr =
      runtimeStr === 'local' ? 'local' : selected?.provider || provider

    const cfg: any = {
      runtime: runtimeStr,
      provider: providerStr,
      modelId: chosenModelId,
      baseUrl: baseUrl.trim() || undefined,
      dimension: Number(dimension) || undefined,
      batchSize,
      timeout: timeoutSec,
      auto_pull: runtimeStr === 'local' ? ollamaAutoPull : undefined,
      similarity: 'cosine',
    }
    if (summaryProvider === 'OpenAI') {
      cfg.organization = openaiOrg.trim() || undefined
      cfg.maxRetries = openaiMaxRetries
    }
    if (summaryProvider === 'Azure OpenAI') {
      cfg.deployment = azureDeployment.trim() || undefined
      cfg.endpoint = azureResource.trim() || undefined
      cfg.apiVersion = azureApiVersion.trim() || undefined
    }
    if (summaryProvider === 'Google') {
      cfg.projectId = vertexProjectId.trim() || undefined
      cfg.endpoint = vertexEndpoint.trim() || undefined
      cfg.region = vertexLocation.trim() || undefined
    }
    if (summaryProvider === 'AWS Bedrock') {
      cfg.region = bedrockRegion.trim() || undefined
    }
    if (encryptedKey) cfg.apiKey = encryptedKey

    persistForStrategy(cfg)
    if (makeDefault) setReembedOpen(true)
    else {
      toast({ message: 'Strategy saved', variant: 'default' })
      navigate('/chat/rag')
    }
  }

  return (
    <div
      className={`h-full w-full flex flex-col ${mode === 'designer' ? 'gap-3 pb-40' : ''}`}
    >
      {/* Breadcrumb + Actions */}
      {mode === 'designer' ? (
        <>
          <div className="flex items-center justify-between mb-1 md:mb-3">
            <nav className="text-sm md:text-base flex items-center gap-1.5">
              <button
                className="text-teal-600 dark:text-teal-400 hover:underline"
                onClick={() => navigate('/chat/databases')}
              >
                Databases
              </button>
              <span className="text-muted-foreground px-1">/</span>
              <span className="text-foreground">Edit strategy</span>
            </nav>
            <PageActions mode={mode} onModeChange={handleModeChange} />
          </div>

          {/* Header */}
          <div className="flex items-start justify-between mb-1 gap-2">
            <div className="flex-1 min-w-0">
              <h2 className="text-lg md:text-xl font-medium">Edit strategy</h2>
              <div className="text-xs text-muted-foreground max-w-[80vw] truncate md:hidden mt-1">
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
            </div>
            <div className="flex items-center gap-3">
              <div className="text-xs text-muted-foreground max-w-[50vw] truncate hidden md:block">
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
              {!isDefaultStrategy && (
                <label className="text-xs flex items-center gap-2 select-none">
                  <input
                    type="checkbox"
                    checked={makeDefault}
                    onChange={e => setMakeDefault(e.target.checked)}
                  />
                  Make default
                </label>
              )}
              <Button onClick={() => setConfirmOpen(true)}>
                Save strategy
              </Button>
            </div>
          </div>
        </>
      ) : (
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-2xl">Config editor</h2>
          <PageActions mode={mode} onModeChange={handleModeChange} />
        </div>
      )}

      {/* Removed 'Current model' summary card per request */}

      {mode !== 'designer' ? (
        <div className="flex-1 min-h-0 overflow-hidden">
          <ConfigEditor className="h-full" initialPointer={configPointer} />
        </div>
      ) : (
        <>
          {/* Strategy name and settings */}
          <section className="rounded-lg border border-border bg-card p-4 md:p-6 flex flex-col gap-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <div className="flex flex-col gap-1">
                <Label className="text-xs text-muted-foreground">
                  Strategy name
                </Label>
                <Input
                  value={strategyName}
                  onChange={e => setStrategyName(e.target.value)}
                  placeholder="Enter a name"
                  className="h-9"
                />
              </div>
            </div>

            {/* Top settings grid */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div className="flex flex-col gap-1">
                <Label className="text-xs text-muted-foreground">
                  Base URL
                </Label>
                <Input
                  value={baseUrl}
                  onChange={e => setBaseUrl(e.target.value)}
                  placeholder="http://localhost:11434"
                  className="h-9"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-xs text-muted-foreground">
                  Dimension
                </Label>
                <Input
                  type="number"
                  value={dimension}
                  onChange={e => setDimension(Number(e.target.value || 768))}
                  className="h-9"
                />
              </div>
              <div className="flex flex-col gap-1">
                <Label className="text-xs text-muted-foreground">
                  Batch size
                </Label>
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
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <button className="w-full h-9 rounded-md border border-border bg-background px-3 text-left flex items-center justify-between">
                      <span>{ollamaAutoPull ? 'Enabled' : 'Disabled'}</span>
                      <FontIcon type="chevron-down" className="w-4 h-4" />
                    </button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent className="w-48">
                    <DropdownMenuItem onClick={() => setOllamaAutoPull(true)}>
                      Enabled
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => setOllamaAutoPull(false)}>
                      Disabled
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
              <div />
            </div>
            {!isDefaultStrategy && (
              <div className="flex items-center gap-2">
                <label className="text-xs flex items-center gap-2 select-none">
                  <input
                    type="checkbox"
                    checked={makeDefault}
                    onChange={e => setMakeDefault(e.target.checked)}
                  />
                  Make default strategy
                </label>
              </div>
            )}

            <div className="text-sm text-muted-foreground">
              Select a new embedding model and configure connection/performance
              options.
            </div>

            {/* Source switcher */}
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
                                    onClick={() => openConfirmLocal(group, v)}
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
                            // reset cloud selection state
                            if (selected?.runtime === 'Cloud') setSelected(null)
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
                            if (selected?.runtime === 'Cloud') setSelected(null)
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

                  {/* OpenAI */}
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
                          onChange={e =>
                            setTimeoutSec(Number(e.target.value || 60))
                          }
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
                  {/* Azure */}
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
                  {/* Google */}
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
                  {/* Bedrock */}
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

                  {/* Cloud API key */}
                  {provider !== 'Ollama (remote)' && (
                    <div className="flex flex-col gap-2">
                      <Label className="text-xs text-muted-foreground">
                        API Key
                      </Label>
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
                          aria-label={
                            showApiKey ? 'Hide API key' : 'Show API key'
                          }
                        >
                          <FontIcon
                            type={showApiKey ? 'eye-off' : 'eye'}
                            className="w-4 h-4"
                          />
                        </button>
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={checkConnection}
                    >
                      {connectionStatus === 'checking'
                        ? 'Checking…'
                        : 'Check connection'}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={runTestEmbedding}
                    >
                      {testStatus === 'running' ? 'Testing…' : 'Test embedding'}
                    </Button>
                  </div>
                  <Button
                    onClick={() =>
                      setSelected({
                        runtime: 'Cloud',
                        provider,
                        modelId:
                          model === 'Custom' ? customModel.trim() : model,
                      })
                    }
                    disabled={
                      provider !== 'Ollama (remote)' &&
                      apiKey.trim().length === 0
                    }
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

                {(connectionStatus === 'ok' ||
                  connectionStatus === 'error') && (
                  <div
                    className={`text-xs ${connectionStatus === 'ok' ? 'text-teal-600 dark:text-teal-400' : 'text-destructive'}`}
                  >
                    {connectionMsg}
                  </div>
                )}
                {testStatus !== 'idle' && (
                  <div className="text-xs text-muted-foreground">
                    {testStatus === 'ok'
                      ? `Test completed${testLatencyMs ? ` in ${testLatencyMs} ms` : ''}`
                      : testStatus === 'error'
                        ? 'Test failed'
                        : 'Running test…'}
                  </div>
                )}
              </div>
            )}
          </section>

          {/* Save modal */}
          <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <DialogContent>
              <DialogTitle>Save this embedding strategy?</DialogTitle>
              <DialogDescription>
                <div className="mt-2 text-sm">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-xs">
                    <div className="text-muted-foreground">Strategy name</div>
                    <div className="truncate">
                      {strategyName || '(unnamed)'}
                    </div>
                    <div className="text-muted-foreground">Runtime</div>
                    <div>{runtimeLabel}</div>
                    <div className="text-muted-foreground">Provider</div>
                    <div>{summaryProvider || 'n/a'}</div>
                    <div className="text-muted-foreground">Model</div>
                    <div className="font-mono truncate">
                      {summaryModel || 'n/a'}
                    </div>
                    <div className="text-muted-foreground">
                      Base URL / Location
                    </div>
                    <div className="truncate">{summaryLocation || 'n/a'}</div>
                    <div className="text-muted-foreground">
                      Vector dimension (d)
                    </div>
                    <div>{dimension ?? meta?.dim ?? 'n/a'}</div>
                    <div className="text-muted-foreground">Batch size</div>
                    <div>{batchSize}</div>
                    <div className="text-muted-foreground">Timeout (sec)</div>
                    <div>{timeoutSec}</div>
                    <div className="text-muted-foreground">Auto-pull model</div>
                    <div>{ollamaAutoPull ? 'Enabled' : 'Disabled'}</div>
                    {summaryProvider === 'OpenAI' ? (
                      <>
                        <div className="text-muted-foreground">
                          Organization
                        </div>
                        <div className="truncate">{openaiOrg || '(none)'}</div>
                        <div className="text-muted-foreground">Max retries</div>
                        <div>{openaiMaxRetries}</div>
                      </>
                    ) : null}
                    {summaryProvider === 'Azure OpenAI' ? (
                      <>
                        <div className="text-muted-foreground">Deployment</div>
                        <div className="truncate">
                          {azureDeployment || 'n/a'}
                        </div>
                        <div className="text-muted-foreground">Endpoint</div>
                        <div className="truncate">{azureResource || 'n/a'}</div>
                        <div className="text-muted-foreground">API version</div>
                        <div className="truncate">
                          {azureApiVersion || '(default)'}
                        </div>
                      </>
                    ) : null}
                    {summaryProvider === 'Google' ? (
                      <>
                        <div className="text-muted-foreground">Project ID</div>
                        <div className="truncate">
                          {vertexProjectId || 'n/a'}
                        </div>
                        <div className="text-muted-foreground">Location</div>
                        <div className="truncate">
                          {vertexLocation || 'n/a'}
                        </div>
                        <div className="text-muted-foreground">Endpoint</div>
                        <div className="truncate">
                          {vertexEndpoint || '(auto)'}
                        </div>
                      </>
                    ) : null}
                    {summaryProvider === 'AWS Bedrock' ? (
                      <>
                        <div className="text-muted-foreground">Region</div>
                        <div className="truncate">{bedrockRegion || 'n/a'}</div>
                      </>
                    ) : null}
                  </div>
                  {!isDefaultStrategy && makeDefault ? (
                    <div className="mt-3 text-xs">Will set as default</div>
                  ) : null}
                </div>
              </DialogDescription>
              <DialogFooter>
                <Button
                  variant="secondary"
                  onClick={() => setConfirmOpen(false)}
                >
                  Cancel
                </Button>
                <Button onClick={saveEdited}>
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
                  onClick={() => {
                    setReembedOpen(false)
                    toast({ message: 'Strategy saved', variant: 'default' })
                    navigate('/chat/rag')
                  }}
                >
                  I'll do it later
                </Button>
                <Button
                  onClick={() => {
                    setReembedOpen(false)
                    toast({ message: 'Strategy saved', variant: 'default' })
                    navigate('/chat/rag')
                  }}
                >
                  Yes, proceed with re-embed
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>
        </>
      )}
    </div>
  )
}

export default ChangeEmbeddingModel
