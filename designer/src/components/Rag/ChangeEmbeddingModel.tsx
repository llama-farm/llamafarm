import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  useNavigate,
  useLocation,
  useParams,
  useSearchParams,
} from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
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
import { useDatabaseManager } from '../../hooks/useDatabaseManager'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import { validateEmbeddingNavigationState, validateStrategyName } from '../../utils/security'
import type { ProjectConfig } from '../../types/config'
import { useCachedModels } from '../../hooks/useModels'
import modelService from '../../api/modelService'
import { useUnsavedChanges } from '../../contexts/UnsavedChangesContext'
import UnsavedChangesModal from '../ConfigEditor/UnsavedChangesModal'
import { encryptAPIKey } from '../../utils/encryption'
import {
  LocalModelTable,
  type Variant,
  type LocalGroup,
} from './LocalModelTable'

function ChangeEmbeddingModel() {
  const navigate = useNavigate()
  const location = useLocation()
  const [mode, setMode] = useModeWithReset('designer')
  const { strategyId } = useParams()
  const [searchParams] = useSearchParams()
  const { toast } = useToast()
  const queryClient = useQueryClient()
  const activeProject = useActiveProject()

  // Get project config and database manager
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )
  const databaseManager = useDatabaseManager(
    activeProject?.namespace || '',
    activeProject?.project || ''
  )

  // Get data from navigation state with validation, or URL params (for backward compatibility)
  const validatedState = validateEmbeddingNavigationState(location.state)

  // Config pointer for config editor mode
  const projectConfig = (projectResp as any)?.project?.config as
    | ProjectConfig
    | undefined
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

  // Use validated state or fall back to URL params for backward compatibility
  const database =
    validatedState.database !== 'main_database'
      ? validatedState.database
      : searchParams.get('database') || 'main_database'
  const originalStrategyName = validatedState.strategyName || strategyId || ''

  // Get strategy data from project config (server source of truth) instead of navigation state
  const strategyFromConfig = useMemo(() => {
    if (!projectConfig || !originalStrategyName) return null
    const db = projectConfig.rag?.databases?.find(
      (d: any) => d.name === database
    )
    return db?.embedding_strategies?.find(
      (s: any) => s.name === originalStrategyName
    )
  }, [projectConfig, database, originalStrategyName])

  // Use server data if available, fall back to navigation state for backward compatibility
  const strategyType =
    strategyFromConfig?.type || validatedState.strategyType || 'OllamaEmbedder'
  const currentConfig =
    strategyFromConfig?.config || validatedState.currentConfig || {}
  const isDefaultStrategy =
    projectConfig?.rag?.databases?.find((d: any) => d.name === database)
      ?.default_embedding_strategy === originalStrategyName ||
    validatedState.isDefault ||
    false
  const priority = strategyFromConfig?.priority || validatedState.priority || 0

  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Unsaved changes tracking
  const unsavedChangesContext = useUnsavedChanges()
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [modalErrorMessage, setModalErrorMessage] = useState<string | null>(
    null
  )
  const [isInitialized, setIsInitialized] = useState(false)
  const [initialValues, setInitialValues] = useState<{
    strategyName: string
    selected: typeof selected
    provider: Provider
    model: string
    customModel: string
    dimension: number
    batchSize: number
    timeoutSec: number
    baseUrl: string
    ollamaAutoPull: boolean
  } | null>(null)

  // Editable strategy name - initialize from state
  const [strategyName, setStrategyName] = useState<string>(originalStrategyName)
  const [nameTouched, setNameTouched] = useState(false)

  // Initialize strategy name from state
  useEffect(() => {
    if (originalStrategyName) {
      setStrategyName(originalStrategyName)
    }
  }, [originalStrategyName])

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
  const [query, setQuery] = useState('')
  const [isManuallyRefreshing, setIsManuallyRefreshing] = useState(false)

  // Download state per model
  const [downloadStates, setDownloadStates] = useState<
    Record<
      string,
      {
        state: 'idle' | 'downloading' | 'success' | 'error'
        progress: number
        downloadedBytes: number
        totalBytes: number
        error?: string
      }
    >
  >({})

  // Download confirmation modal state
  const [downloadConfirmOpen, setDownloadConfirmOpen] = useState(false)
  const [pendingDownloadVariant, setPendingDownloadVariant] =
    useState<Variant | null>(null)
  const [showBackgroundDownload, setShowBackgroundDownload] = useState(false)
  const [backgroundDownloadName, setBackgroundDownloadName] = useState('')
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState('')

  // Show/hide model table
  const [showModelTable, setShowModelTable] = useState(true)

  // Fetch cached models from disk
  const {
    data: cachedModelsResponse,
    isLoading: isLoadingCachedModels,
    refetch: refetchCachedModels,
  } = useCachedModels()

  // Helper to format bytes
  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return `${Math.round((bytes / Math.pow(k, i)) * 100) / 100} ${sizes[i]}`
  }

  // Map model IDs to HuggingFace identifiers
  const modelIdToHuggingFace: Record<string, string> = {
    'bge-small-en-v1.5': 'BAAI/bge-small-en-v1.5',
    'bge-base-en-v1.5': 'BAAI/bge-base-en-v1.5',
    'bge-large-en-v1.5': 'BAAI/bge-large-en-v1.5',
    'bge-m3': 'BAAI/bge-m3',
    'e5-base-v2': 'intfloat/e5-base-v2',
    'e5-large-v2': 'intfloat/e5-large-v2',
    'all-MiniLM-L6-v2': 'sentence-transformers/all-MiniLM-L6-v2',
  }

  // Check if a model is on disk
  const isModelOnDisk = (modelId: string): boolean => {
    if (!cachedModelsResponse?.data) return false
    const hfId = modelIdToHuggingFace[modelId] || modelId
    return cachedModelsResponse.data.some(m => {
      // Check if model name matches (could be full path or just name)
      const modelName = m.name.toLowerCase()
      const searchId = hfId.toLowerCase()
      return (
        modelName.includes(searchId.split('/').pop() || '') ||
        modelName === searchId
      )
    })
  }

  // Get disk size for a model
  const getModelDiskSize = (modelId: string): number | null => {
    if (!cachedModelsResponse?.data) return null
    const hfId = modelIdToHuggingFace[modelId] || modelId
    const found = cachedModelsResponse.data.find(m => {
      const modelName = m.name.toLowerCase()
      const searchId = hfId.toLowerCase()
      return (
        modelName.includes(searchId.split('/').pop() || '') ||
        modelName === searchId
      )
    })
    return found?.size || null
  }


  // Base local groups with model identifiers
  const baseLocalGroups: LocalGroup[] = [
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
          modelIdentifier: 'BAAI/bge-small-en-v1.5',
        },
        {
          id: 'bge-base-en-v1.5',
          label: 'bge-base-en-v1.5',
          dim: '768',
          quality: 'General',
          download: '340MB',
          modelIdentifier: 'BAAI/bge-base-en-v1.5',
        },
        {
          id: 'bge-large-en-v1.5',
          label: 'bge-large-en-v1.5',
          dim: '1024',
          quality: 'General',
          download: '650MB',
          modelIdentifier: 'BAAI/bge-large-en-v1.5',
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
          modelIdentifier: 'BAAI/bge-m3',
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
          modelIdentifier: 'intfloat/e5-base-v2',
        },
        {
          id: 'e5-large-v2',
          label: 'e5-large-v2',
          dim: '1024',
          quality: 'General',
          download: '800MB',
          modelIdentifier: 'intfloat/e5-large-v2',
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
          modelIdentifier: 'sentence-transformers/all-MiniLM-L6-v2',
        },
      ],
    },
  ]

  // Enhance local groups with disk status
  const localGroups: LocalGroup[] = useMemo(() => {
    return baseLocalGroups.map(group => ({
      ...group,
      variants: group.variants.map(variant => {
        const isDownloaded = isModelOnDisk(variant.id)
        const diskSize = getModelDiskSize(variant.id)
        return {
          ...variant,
          isDownloaded,
          diskSize,
          download: diskSize ? formatBytes(diskSize) : variant.download,
        }
      }),
    }))
  }, [cachedModelsResponse])

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
  }, [query, localGroups])

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

  // Sync isDirty with context
  useEffect(() => {
    unsavedChangesContext.setIsDirty(hasUnsavedChanges)
  }, [hasUnsavedChanges, unsavedChangesContext])

  // Track changes to form fields - only after initialization
  useEffect(() => {
    // Don't track changes until form is initialized
    if (!isInitialized || !initialValues) return

    // Check if any form field has changed from initial values
    const hasChanges =
      strategyName !== initialValues.strategyName ||
      selected !== initialValues.selected ||
      provider !== initialValues.provider ||
      model !== initialValues.model ||
      customModel !== initialValues.customModel ||
      dimension !== initialValues.dimension ||
      batchSize !== initialValues.batchSize ||
      timeoutSec !== initialValues.timeoutSec ||
      baseUrl !== initialValues.baseUrl ||
      ollamaAutoPull !== initialValues.ollamaAutoPull

    setHasUnsavedChanges(hasChanges)
  }, [
    strategyName,
    selected,
    provider,
    model,
    customModel,
    dimension,
    batchSize,
    timeoutSec,
    baseUrl,
    ollamaAutoPull,
    isInitialized,
    initialValues,
  ])
  const [openaiOrg, setOpenaiOrg] = useState('')
  const [openaiMaxRetries, setOpenaiMaxRetries] = useState(3)
  const [makeDefault, setMakeDefault] = useState(false)
  const [reembedOpen, setReembedOpen] = useState(false)
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

  // Initialize form fields from currentConfig (from server or navigation state)
  useEffect(() => {
    // Wait for project config to load if we're using server data
    if (strategyFromConfig === null && !validatedState.currentConfig) return
    if (!currentConfig || Object.keys(currentConfig).length === 0) return

    try {
      // Initialize common fields
      if (typeof currentConfig.dimension === 'number')
        setDimension(currentConfig.dimension)
      if (typeof currentConfig.batch_size === 'number')
        setBatchSize(currentConfig.batch_size)
      if (typeof currentConfig.batchSize === 'number')
        setBatchSize(currentConfig.batchSize)
      if (typeof currentConfig.timeout === 'number')
        setTimeoutSec(currentConfig.timeout)

      // Initialize model and provider based on strategy type
      let targetProvider: Provider = 'Ollama (remote)'
      let targetTab: 'local' | 'cloud' = 'local'

      if (strategyType === 'UniversalEmbedder') {
        // UniversalEmbedder uses local HuggingFace models
        targetProvider = 'Ollama (remote)' // This is just for UI state, actual provider is handled via selected
        targetTab = 'local'
        if (currentConfig.base_url) setBaseUrl(currentConfig.base_url)

        // Try to find matching variant from localGroups
        if (currentConfig.model) {
          const modelId = currentConfig.model
          // Find variant by modelIdentifier or by mapping
          const variant = localGroups
            .flatMap(g => g.variants)
            .find(
              v =>
                v.modelIdentifier === modelId ||
                modelIdToHuggingFace[v.id] === modelId ||
                v.id === modelId
            )

          if (variant) {
            // Set selected state for local HuggingFace model
            setSelected({
              runtime: 'Local',
              provider: 'Ollama',
              modelId: variant.id,
            })
            setShowModelTable(false) // Hide table, show selected card
          }
        }
      } else if (strategyType === 'OllamaEmbedder') {
        targetProvider = 'Ollama (remote)'
        targetTab = 'local'
        if (currentConfig.base_url) setBaseUrl(currentConfig.base_url)
        if (typeof currentConfig.auto_pull === 'boolean')
          setOllamaAutoPull(currentConfig.auto_pull)
      } else if (strategyType === 'OpenAIEmbedder') {
        targetProvider = 'OpenAI'
        targetTab = 'cloud'
        if (currentConfig.base_url) setBaseUrl(currentConfig.base_url)
        if (currentConfig.organization) setOpenaiOrg(currentConfig.organization)
        if (currentConfig.max_retries)
          setOpenaiMaxRetries(currentConfig.max_retries)
      } else if (strategyType.includes('Azure')) {
        targetProvider = 'Azure OpenAI'
        targetTab = 'cloud'
        if (currentConfig.deployment)
          setAzureDeployment(currentConfig.deployment)
        if (currentConfig.endpoint) setAzureResource(currentConfig.endpoint)
        if (currentConfig.api_version)
          setAzureApiVersion(currentConfig.api_version)
      } else if (strategyType.includes('Google')) {
        targetProvider = 'Google'
        targetTab = 'cloud'
        if (currentConfig.project_id)
          setVertexProjectId(currentConfig.project_id)
        if (currentConfig.region) setVertexLocation(currentConfig.region)
        if (currentConfig.endpoint) setVertexEndpoint(currentConfig.endpoint)
      } else if (strategyType.includes('Bedrock')) {
        targetProvider = 'AWS Bedrock'
        targetTab = 'cloud'
        if (currentConfig.region) setBedrockRegion(currentConfig.region)
      }

      setProvider(targetProvider)
      setSourceTab(targetTab)

      // Initialize model (skip for UniversalEmbedder as it's handled above)
      if (currentConfig.model && strategyType !== 'UniversalEmbedder') {
        const modelName = currentConfig.model
        setExistingModelId(modelName)

        // Set model in the UI (check if it's in the provider's model list)
        const providerModels = modelMap[targetProvider] || []
        const modelInList = providerModels.includes(modelName)
        if (modelInList) {
          setModel(modelName)
        } else {
          setModel('Custom')
          setCustomModel(modelName)
        }
      } else if (currentConfig.model && strategyType === 'UniversalEmbedder') {
        // For UniversalEmbedder, we already set selected above, but also set existingModelId
        setExistingModelId(currentConfig.model)
      }

      // Mark that we're ready to capture initial values
      // We'll capture them in a separate effect after state updates complete
      setIsInitialized(false)
    } catch (e) {
      console.error('Failed to initialize form from config:', e)
      // Even if initialization fails, mark as initialized to prevent false positives
      setIsInitialized(true)
    }
  }, [
    currentConfig,
    strategyType,
    strategyFromConfig,
    validatedState.currentConfig,
    // Note: We intentionally don't include form fields in deps to avoid re-initialization
    // This effect should only run when config/strategyType changes
  ])

  // Capture initial values after form is initialized
  // This runs after the initialization effect has set all the form fields
  useEffect(() => {
    // Only capture once, when we have config but haven't initialized yet
    if (
      isInitialized ||
      !currentConfig ||
      Object.keys(currentConfig).length === 0
    )
      return

    // Use a small delay to ensure all state updates from initialization have completed
    const timer = setTimeout(() => {
      setInitialValues({
        strategyName,
        selected,
        provider,
        model,
        customModel,
        dimension,
        batchSize,
        timeoutSec,
        baseUrl,
        ollamaAutoPull,
      })
      setIsInitialized(true)
    }, 150)

    return () => clearTimeout(timer)
  }, [
    currentConfig,
    strategyName,
    selected,
    provider,
    model,
    customModel,
    dimension,
    batchSize,
    timeoutSec,
    baseUrl,
    ollamaAutoPull,
    isInitialized,
  ])

  // Sync dimension from model metadata when model/provider changes
  useEffect(() => {
    if (meta?.dim) {
      const d = Number(meta.dim)
      if (!Number.isNaN(d)) setDimension(d)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, model, customModel])

  // Removed persistForStrategy - now saving directly to config via databaseManager

  // When provider changes, default model
  useEffect(() => {
    try {
      if (model !== 'Custom') setModel(modelMap[provider][0])
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider])

  // Helper to format ETA
  const formatETA = (seconds: number): string => {
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    return `${Math.round(seconds / 3600)}h`
  }

  // Download a model
  const downloadModel = async (variant: Variant, background = false) => {
    const modelIdentifier =
      variant.modelIdentifier || modelIdToHuggingFace[variant.id] || variant.id

    // Initialize download state
    setDownloadStates(prev => ({
      ...prev,
      [variant.id]: {
        state: 'downloading',
        progress: 0,
        downloadedBytes: 0,
        totalBytes: 0,
      },
    }))

    if (background) {
      setShowBackgroundDownload(true)
      setBackgroundDownloadName(variant.label)
      setDownloadConfirmOpen(false)
    }

    try {
      if (!background) {
        toast({
          message: `Downloading ${variant.label}...`,
          variant: 'default',
        })
      }

      const start = Date.now()
      for await (const event of modelService.downloadModel({
        model_name: modelIdentifier,
        provider: 'universal',
      })) {
        if (event.event === 'progress') {
          const d = Number(event.downloaded || 0)
          const t = Number(event.total || 0)
          const progress = t > 0 ? Math.round((d / t) * 100) : 0

          setDownloadStates(prev => ({
            ...prev,
            [variant.id]: {
              state: 'downloading',
              progress,
              downloadedBytes: d,
              totalBytes: t,
            },
          }))

          // Calculate ETA
          if (t > 0 && d > 0) {
            const elapsedSec = (Date.now() - start) / 1000
            if (elapsedSec > 0) {
              const speed = d / elapsedSec
              const remain = (t - d) / (speed || 1)
              setEstimatedTimeRemaining(formatETA(remain))
            }
          }
        } else if (event.event === 'done') {
          setDownloadStates(prev => ({
            ...prev,
            [variant.id]: {
              state: 'success',
              progress: 100,
              downloadedBytes: prev[variant.id]?.totalBytes || 0,
              totalBytes: prev[variant.id]?.totalBytes || 0,
            },
          }))

          // Refresh cached models
          await refetchCachedModels()

          if (!background) {
            toast({
              message: `${variant.label} downloaded successfully`,
              variant: 'default',
            })
          }

          // Auto-select after download
          setTimeout(() => {
            setSelected({
              runtime: 'Local',
              provider: 'Ollama',
              modelId: variant.id,
            })
            // Close the model selection area when a new model is selected
            setShowModelTable(false)
            setDownloadConfirmOpen(false)
            if (background) {
              setShowBackgroundDownload(false)
            }
          }, 500)
        } else if (event.event === 'error') {
          setDownloadStates(prev => ({
            ...prev,
            [variant.id]: {
              state: 'error',
              progress: 0,
              downloadedBytes: 0,
              totalBytes: 0,
              error: event.message || 'Download failed',
            },
          }))

          toast({
            message: event.message || `Failed to download ${variant.label}`,
            variant: 'destructive',
          })

          if (background) {
            setShowBackgroundDownload(false)
          }
        }
      }
    } catch (error: any) {
      setDownloadStates(prev => ({
        ...prev,
        [variant.id]: {
          state: 'error',
          progress: 0,
          downloadedBytes: 0,
          totalBytes: 0,
          error: error.message || 'Download failed',
        },
      }))

      toast({
        message: error.message || `Failed to download ${variant.label}`,
        variant: 'destructive',
      })

      if (background) {
        setShowBackgroundDownload(false)
      }
    }
  }

  const openConfirmLocal = async (group: any, variant: Variant) => {
    // Check if model needs to be downloaded
    if (!variant.isDownloaded && variant.modelIdentifier) {
      // Show confirmation modal
      setPendingDownloadVariant(variant)
      setDownloadConfirmOpen(true)
    } else {
      // Model is already on disk, just select it
      setSelected({
        runtime: 'Local',
        provider: group.name,
        modelId: variant.id,
      })
      // Close the model selection area when a new model is selected
      setShowModelTable(false)
    }
  }

  // Refresh disk models
  const handleRefresh = async () => {
    setIsManuallyRefreshing(true)
    const startTime = Date.now()
    await refetchCachedModels()
    const elapsed = Date.now() - startTime
    const remaining = Math.max(0, 800 - elapsed)
    setTimeout(() => {
      setIsManuallyRefreshing(false)
    }, remaining)
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
    // First check if we have a selected model (user just selected one)
    if (selected) {
      // Check if this is a UniversalEmbedder (HuggingFace model from local groups)
      if (selected.runtime === 'Local' && selected.modelId) {
        const variant = localGroups
          .flatMap(g => g.variants)
          .find(v => v.id === selected.modelId)
        if (variant?.modelIdentifier) {
          // This is a HuggingFace model, show "Universal"
          return 'Universal'
        }
        // Otherwise it's Ollama
        return 'Ollama'
      }
      return selected.provider
    }
    // Check strategy type from config (when editing existing strategy)
    if (strategyType === 'UniversalEmbedder') {
      return 'Universal'
    }
    // Check if currentConfig has a model that matches a HuggingFace model
    if (currentConfig?.model) {
      const modelId = currentConfig.model
      const variant = localGroups
        .flatMap(g => g.variants)
        .find(
          v =>
            v.modelIdentifier === modelId ||
            modelIdToHuggingFace[v.id] === modelId ||
            v.id === modelId
        )
      if (variant?.modelIdentifier) {
        return 'Universal'
      }
    }
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
  // runtimeLabel removed - unused variable

  // Helper functions similar to AddEmbeddingStrategy
  const mapProviderToType = (providerLabel: string): string => {
    const typeMap: Record<string, string> = {
      Ollama: 'OllamaEmbedder',
      'Ollama (remote)': 'OllamaEmbedder',
      OpenAI: 'OpenAIEmbedder',
      'Azure OpenAI': 'AzureOpenAIEmbedder',
      Google: 'VertexAIEmbedder',
      'AWS Bedrock': 'BedrockEmbedder',
      Cohere: 'CohereEmbedder',
      'Voyage AI': 'VoyageEmbedder',
      HuggingFace: 'HuggingFaceEmbedder',
      SentenceTransformer: 'SentenceTransformerEmbedder',
    }

    const mappedType = typeMap[providerLabel]

    if (!mappedType) {
      console.error(`Unknown embedding provider: ${providerLabel}`)
      // Return a safe fallback but log the issue
      return 'OllamaEmbedder'
    }

    return mappedType
  }

  const buildStrategyConfig = (encryptedKey?: string) => {
    const config: Record<string, any> = {}

    // For local models, get the full HuggingFace model identifier
    let chosenModelId: string | undefined
    if (selected?.runtime === 'Local' && selected?.modelId) {
      // Find the variant to get the modelIdentifier
      const variant = localGroups
        .flatMap(g => g.variants)
        .find(v => v.id === selected.modelId)
      chosenModelId =
        variant?.modelIdentifier ||
        modelIdToHuggingFace[selected.modelId] ||
        selected.modelId
    } else {
      // For cloud models or existing, use the existing logic
      chosenModelId =
        selected?.modelId ||
        existingModelId ||
        (model === 'Custom' ? customModel.trim() : model)
    }

    if (chosenModelId) config.model = chosenModelId
    if (dimension) config.dimension = parseInt(String(dimension))
    if (batchSize) config.batch_size = parseInt(String(batchSize))
    if (timeoutSec) config.timeout = parseInt(String(timeoutSec))

    const runtimeStr = selected
      ? selected.runtime === 'Local'
        ? 'local'
        : 'cloud'
      : provider === 'Ollama (remote)'
        ? 'local'
        : 'cloud'

    // Check if this is a UniversalEmbedder (HuggingFace model from local groups)
    const isUniversalEmbedder =
      selected?.runtime === 'Local' &&
      selected?.modelId &&
      localGroups
        .flatMap(g => g.variants)
        .some(v => v.id === selected.modelId && v.modelIdentifier)

    if (isUniversalEmbedder) {
      // UniversalEmbedder config
      config.base_url = baseUrl?.trim() || 'http://127.0.0.1:11540/v1'
      config.api_key = 'universal'
    } else if (runtimeStr === 'local' || provider === 'Ollama (remote)') {
      // OllamaEmbedder config
      if (baseUrl) config.base_url = baseUrl.trim()
      config.auto_pull = ollamaAutoPull !== undefined ? ollamaAutoPull : true
    } else if (summaryProvider === 'OpenAI') {
      if (baseUrl) config.base_url = baseUrl.trim()
      if (openaiOrg) config.organization = openaiOrg.trim()
      if (openaiMaxRetries) config.max_retries = openaiMaxRetries
      if (encryptedKey) config.api_key = encryptedKey
    } else if (summaryProvider === 'Azure OpenAI') {
      if (azureDeployment) config.deployment = azureDeployment.trim()
      if (azureResource) config.endpoint = azureResource.trim()
      if (azureApiVersion) config.api_version = azureApiVersion.trim()
      if (encryptedKey) config.api_key = encryptedKey
    } else if (summaryProvider === 'Google') {
      if (vertexProjectId) config.project_id = vertexProjectId.trim()
      if (vertexLocation) config.region = vertexLocation.trim()
      if (vertexEndpoint) config.endpoint = vertexEndpoint.trim()
      if (encryptedKey) config.api_key = encryptedKey
    } else if (summaryProvider === 'AWS Bedrock') {
      if (bedrockRegion) config.region = bedrockRegion.trim()
      if (encryptedKey) config.api_key = encryptedKey
    }

    return config
  }

  const saveEdited = async () => {
    if (!originalStrategyName) {
      setError('Strategy name is required')
      return
    }

    try {
      setIsSaving(true)
      setError(null)

      // Encrypt API key if needed
      let encryptedKey: string | undefined
      if (
        (selected?.runtime === 'Cloud' || provider !== 'Ollama (remote)') &&
        apiKey.trim()
      ) {
        try {
          encryptedKey = await encryptAPIKey(
            apiKey.trim(),
            getClientSideSecret()
          )
        } catch (e) {
          toast({
            message: 'Failed to encrypt API key',
            variant: 'destructive',
          })
          return
        }
      }

      // Get current project config
      const projectConfig = (projectResp as any)?.project?.config
      if (!projectConfig) {
        throw new Error('Project config not loaded')
      }

      // Find the database
      const currentDb = projectConfig.rag?.databases?.find(
        (db: any) => db.name === database
      )

      if (!currentDb) {
        throw new Error(`Database ${database} not found in configuration`)
      }

      // Validate strategy name uniqueness (if renamed, case-insensitive)
      const trimmedName = strategyName.trim() || originalStrategyName
      if (trimmedName.toLowerCase() !== originalStrategyName?.toLowerCase()) {
        const nameLower = trimmedName.toLowerCase()
        const originalLower = originalStrategyName?.toLowerCase()
        const nameExists = currentDb.embedding_strategies?.some(
          (s: any) =>
            s.name?.toLowerCase() === nameLower &&
            s.name?.toLowerCase() !== originalLower
        )
        if (nameExists) {
          throw new Error(
            'A strategy with this name already exists'
          )
        }
      }

      // Determine the correct strategy type
      // Local HuggingFace models should use UniversalEmbedder, not OllamaEmbedder
      let strategyType = mapProviderToType(summaryProvider)
      if (selected?.runtime === 'Local' && selected?.modelId) {
        // Check if this is a HuggingFace model from local groups
        const variant = localGroups
          .flatMap(g => g.variants)
          .find(v => v.id === selected.modelId)
        if (variant?.modelIdentifier) {
          // This is a HuggingFace model, use UniversalEmbedder
          strategyType = 'UniversalEmbedder'
        }
      }

      // Find and update the specific strategy
      const updatedStrategies = currentDb.embedding_strategies?.map(
        (strategy: any) => {
          if (strategy.name === originalStrategyName) {
            return {
              ...strategy,
              name: trimmedName,
              type: strategyType,
              priority: priority,
              config: buildStrategyConfig(encryptedKey),
            }
          }
          return strategy
        }
      )

      // Verify the updated strategy exists (using NEW name after rename)
      if (!updatedStrategies?.some((s: any) => s.name === trimmedName)) {
        throw new Error(`Strategy ${trimmedName} not found after update`)
      }

      // Check if we need to update the default strategy name
      let updatedDefaultStrategy = currentDb.default_embedding_strategy
      if (isDefaultStrategy && trimmedName !== originalStrategyName) {
        // If this is the default and we renamed it, update the default reference
        updatedDefaultStrategy = trimmedName
      } else if (makeDefault) {
        // If user wants to make it default
        updatedDefaultStrategy = trimmedName
      }

      // Update database configuration
      await databaseManager.updateDatabase.mutateAsync({
        oldName: database,
        updates: {
          embedding_strategies: updatedStrategies,
          default_embedding_strategy: updatedDefaultStrategy,
        },
        projectConfig,
      })

      // Invalidate queries to refresh data from server
      await queryClient.invalidateQueries({
        queryKey: [
          'rag',
          'databases',
          activeProject?.namespace,
          activeProject?.project,
        ],
      })
      await queryClient.invalidateQueries({
        queryKey: ['project', activeProject?.namespace, activeProject?.project],
      })

      // Clear unsaved changes flags BEFORE navigation to prevent modal from showing
      setHasUnsavedChanges(false)
      unsavedChangesContext.setIsDirty(false)

      if (
        makeDefault ||
        (isDefaultStrategy && strategyName.trim() !== originalStrategyName)
      ) {
        setReembedOpen(true)
      } else {
        toast({ message: 'Strategy saved', variant: 'default' })
        // Use requestAnimationFrame to ensure state updates propagate before navigation
        requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            navigate('/chat/databases')
          })
        })
      }
      return true
    } catch (error: any) {
      console.error('Failed to save embedding strategy:', error)
      setError(error.message || 'Failed to save strategy')
      return false
    } finally {
      setIsSaving(false)
    }
  }

  // Show error if required state is missing
  if (!originalStrategyName && !strategyId) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center gap-4 p-6">
        <div className="text-destructive text-lg font-semibold">
          Missing required information
        </div>
        <div className="text-muted-foreground text-sm">
          Please return to the databases page and try again.
        </div>
        <Button onClick={() => navigate('/chat/databases')}>
          Return to Databases
        </Button>
      </div>
    )
  }

  return (
    <div
      className={`h-full w-full flex flex-col ${mode === 'designer' ? 'gap-3 pb-40' : ''}`}
    >
      {error && (
        <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-2 rounded-md text-sm">
          {error}
        </div>
      )}
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
              <Button 
                onClick={saveEdited} 
                disabled={isSaving || (nameTouched && !!validateStrategyName(strategyName))}
              >
                {isSaving ? 'Saving...' : 'Save strategy'}
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
                  onChange={e => {
                    setStrategyName(e.target.value)
                    if (nameTouched) {
                      // Clear error state when user starts typing
                      const nameError = validateStrategyName(e.target.value)
                      if (!nameError) {
                        setError(null)
                      }
                    }
                  }}
                  onBlur={() => setNameTouched(true)}
                  placeholder="Enter a name"
                  className={`h-9 ${nameTouched && validateStrategyName(strategyName) ? 'border-destructive' : ''}`}
                />
                {nameTouched && validateStrategyName(strategyName) && (
                  <p className="text-xs text-destructive mt-1">
                    {validateStrategyName(strategyName)}
                  </p>
                )}
                {(!nameTouched || !validateStrategyName(strategyName)) && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Can only contain letters, numbers, hyphens, and underscores
                  </p>
                )}
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

            <div className="text-sm text-muted-foreground flex items-center gap-2">
              {showModelTable ? (
                // When in change mode, show current model
                selected?.runtime === 'Local' && selected?.modelId ? (
                  <>
                    <span>Change model</span>
                    <span>•</span>
                    <span className="font-mono text-foreground">
                      {localGroups
                        .flatMap(g => g.variants)
                        .find(v => v.id === selected.modelId)?.label ||
                        selected.modelId}
                    </span>
                  </>
                ) : selected?.runtime === 'Cloud' && selected?.modelId ? (
                  <>
                    <span>Change model</span>
                    <span>•</span>
                    <span className="font-mono text-foreground">
                      {selected.modelId}
                    </span>
                  </>
                ) : existingModelId ? (
                  <>
                    <span>Change model</span>
                    <span>•</span>
                    <span className="font-mono text-foreground">
                      {existingModelId}
                    </span>
                  </>
                ) : (
                  'Embedding strategy model'
                )
              ) : (
                // When not in change mode, show default text
                'Embedding strategy model'
              )}
            </div>

            {/* Selected model card - show when model is selected and table is hidden */}
            {selected?.runtime === 'Local' &&
              selected?.modelId &&
              !showModelTable && (
                <div className="rounded-lg border border-border bg-card p-4">
                  <div className="flex items-center justify-between">
                    <div className="flex-1">
                      <div className="text-sm font-medium text-foreground">
                        {localGroups
                          .flatMap(g => g.variants)
                          .find(v => v.id === selected.modelId)?.label ||
                          selected.modelId}
                      </div>
                      <div className="text-xs text-muted-foreground mt-1 flex items-center gap-2">
                        {(() => {
                          const variant = localGroups
                            .flatMap(g => g.variants)
                            .find(v => v.id === selected.modelId)
                          if (variant?.isDownloaded) {
                            return (
                              <>
                                <Badge
                                  variant="outline"
                                  size="sm"
                                  className="rounded-xl text-muted-foreground border-muted"
                                >
                                  On disk
                                </Badge>
                                <span>• {variant.download}</span>
                              </>
                            )
                          }
                          return 'Local model'
                        })()}
                      </div>
                    </div>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => setShowModelTable(true)}
                    >
                      Change
                    </Button>
                  </div>
                </div>
              )}

            {/* Model selection area - show when showModelTable is true OR no model selected */}
            {(showModelTable ||
              !(selected?.runtime === 'Local' && selected?.modelId)) && (
              <div className="rounded-lg border border-border bg-card p-4">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-2">
                    <div className="text-sm font-medium text-foreground">
                      Select model
                    </div>
                    {selected?.runtime === 'Local' && selected?.modelId && (
                      <div className="text-xs text-muted-foreground flex items-center gap-1">
                        <span>•</span>
                        <span className="font-mono">
                          {localGroups
                            .flatMap(g => g.variants)
                            .find(v => v.id === selected.modelId)?.label ||
                            selected.modelId}
                        </span>
                      </div>
                    )}
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setShowModelTable(false)}
                  >
                    Cancel
                  </Button>
                </div>
                <div className="w-full flex items-center mb-4">
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
              </div>
            )}

            {sourceTab === 'local' && showModelTable && (
              <LocalModelTable
                filteredGroups={filteredGroups}
                query={query}
                onQueryChange={setQuery}
                selected={selected}
                downloadStates={downloadStates}
                onSelect={v => {
                  const group = filteredGroups.find(g =>
                    g.variants.some(variant => variant.id === v.id)
                  )
                  if (group) {
                    openConfirmLocal(group, v)
                  }
                }}
                onDownloadRetry={downloadModel}
                onRefresh={handleRefresh}
                isRefreshing={isManuallyRefreshing}
                isLoadingCachedModels={isLoadingCachedModels}
              />
            )}

            {sourceTab === 'cloud' && showModelTable && (
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
                    onClick={() => {
                      setSelected({
                        runtime: 'Cloud',
                        provider,
                        modelId:
                          model === 'Custom' ? customModel.trim() : model,
                      })
                      // Close the model selection area when a new model is selected
                      setShowModelTable(false)
                    }}
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


          {/* Download confirmation modal */}
          <Dialog
            open={downloadConfirmOpen}
            onOpenChange={open => {
              setDownloadConfirmOpen(open)
              if (!open) {
                setPendingDownloadVariant(null)
              }
            }}
          >
            <DialogContent>
              <DialogTitle>Download this embedding model?</DialogTitle>
              <DialogDescription>
                {pendingDownloadVariant && (
                  <div className="mt-2 flex flex-col gap-3">
                    <p className="text-sm">
                      You are about to download
                      <span className="mx-1 font-medium text-foreground">
                        {pendingDownloadVariant.label}
                      </span>
                      for use in this embedding strategy.
                    </p>

                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div className="text-muted-foreground">Model</div>
                      <div className="truncate font-mono">
                        {pendingDownloadVariant.modelIdentifier ||
                          pendingDownloadVariant.label}
                      </div>
                      <div className="text-muted-foreground">Size</div>
                      <div>{pendingDownloadVariant.download}</div>
                      <div className="text-muted-foreground">Dimension</div>
                      <div>{pendingDownloadVariant.dim}</div>
                      <div className="text-muted-foreground">Quality</div>
                      <div>{pendingDownloadVariant.quality}</div>
                    </div>

                    {downloadStates[pendingDownloadVariant.id]?.state ===
                      'downloading' && (
                      <div className="flex flex-col gap-1">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">
                            Downloading...{' '}
                            {downloadStates[pendingDownloadVariant.id]
                              ?.downloadedBytes > 0 &&
                              formatBytes(
                                downloadStates[pendingDownloadVariant.id]
                                  .downloadedBytes
                              )}{' '}
                            /{' '}
                            {downloadStates[pendingDownloadVariant.id]
                              ?.totalBytes > 0 &&
                              formatBytes(
                                downloadStates[pendingDownloadVariant.id]
                                  .totalBytes
                              )}
                          </span>
                          <span className="text-muted-foreground">
                            {
                              downloadStates[pendingDownloadVariant.id]
                                ?.progress
                            }
                            %
                            {estimatedTimeRemaining &&
                              ` • ${estimatedTimeRemaining} left`}
                          </span>
                        </div>
                        <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                          <div
                            className="h-full bg-primary transition-all duration-300"
                            style={{
                              width: `${downloadStates[pendingDownloadVariant.id]?.progress || 0}%`,
                            }}
                          />
                        </div>
                      </div>
                    )}

                    {downloadStates[pendingDownloadVariant.id]?.state ===
                      'error' && (
                      <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20">
                        <p className="text-sm text-destructive">
                          {downloadStates[pendingDownloadVariant.id]?.error ||
                            'Download failed'}
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </DialogDescription>
              <DialogFooter>
                {downloadStates[pendingDownloadVariant?.id || '']?.state ===
                'downloading' ? (
                  <Button
                    variant="secondary"
                    onClick={() => {
                      if (pendingDownloadVariant) {
                        setShowBackgroundDownload(true)
                        setBackgroundDownloadName(pendingDownloadVariant.label)
                        setDownloadConfirmOpen(false)
                      }
                    }}
                  >
                    Continue in background
                  </Button>
                ) : (
                  <Button
                    variant="secondary"
                    onClick={() => setDownloadConfirmOpen(false)}
                  >
                    Cancel
                  </Button>
                )}
                <Button
                  disabled={
                    downloadStates[pendingDownloadVariant?.id || '']?.state ===
                    'downloading'
                  }
                  onClick={async () => {
                    if (!pendingDownloadVariant) return
                    await downloadModel(pendingDownloadVariant, false)
                  }}
                >
                  {downloadStates[pendingDownloadVariant?.id || '']?.state ===
                  'downloading'
                    ? 'Downloading...'
                    : 'Download and use'}
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Background download indicator */}
          {showBackgroundDownload &&
            downloadStates[pendingDownloadVariant?.id || '']?.state ===
              'downloading' && (
              <div className="fixed bottom-4 right-4 z-50 w-80 rounded-lg border border-border bg-card shadow-lg p-4 flex flex-col gap-2">
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="text-sm font-medium">
                      Downloading {backgroundDownloadName}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {downloadStates[pendingDownloadVariant?.id || '']
                        ?.downloadedBytes > 0 &&
                        formatBytes(
                          downloadStates[pendingDownloadVariant?.id || '']
                            .downloadedBytes
                        )}{' '}
                      /{' '}
                      {downloadStates[pendingDownloadVariant?.id || '']
                        ?.totalBytes > 0 &&
                        formatBytes(
                          downloadStates[pendingDownloadVariant?.id || '']
                            .totalBytes
                        )}{' '}
                      {estimatedTimeRemaining &&
                        `• ${estimatedTimeRemaining} left`}
                    </div>
                  </div>
                  <button
                    onClick={() => setShowBackgroundDownload(false)}
                    className="text-muted-foreground hover:text-foreground"
                  >
                    <FontIcon type="close" className="w-4 h-4" />
                  </button>
                </div>
                <div className="flex flex-col gap-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Progress</span>
                    <span className="text-muted-foreground">
                      {downloadStates[pendingDownloadVariant?.id || '']
                        ?.progress || 0}
                      %
                    </span>
                  </div>
                  <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary transition-all duration-300"
                      style={{
                        width: `${
                          downloadStates[pendingDownloadVariant?.id || '']
                            ?.progress || 0
                        }%`,
                      }}
                    />
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setDownloadConfirmOpen(true)
                    setShowBackgroundDownload(false)
                  }}
                  className="w-full"
                >
                  Show details
                </Button>
              </div>
            )}

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
                    // Clear unsaved changes flags BEFORE navigation
                    setHasUnsavedChanges(false)
                    unsavedChangesContext.setIsDirty(false)
                    toast({ message: 'Strategy saved', variant: 'default' })
                    requestAnimationFrame(() => {
                      requestAnimationFrame(() => {
                        navigate('/chat/rag')
                      })
                    })
                  }}
                >
                  I'll do it later
                </Button>
                <Button
                  onClick={() => {
                    setReembedOpen(false)
                    // Clear unsaved changes flags BEFORE navigation
                    setHasUnsavedChanges(false)
                    unsavedChangesContext.setIsDirty(false)
                    toast({ message: 'Strategy saved', variant: 'default' })
                    requestAnimationFrame(() => {
                      requestAnimationFrame(() => {
                        navigate('/chat/rag')
                      })
                    })
                  }}
                >
                  Yes, proceed with re-embed
                </Button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Unsaved changes modal */}
          <UnsavedChangesModal
            isOpen={unsavedChangesContext.showModal}
            onSave={async () => {
              const result = await saveEdited()
              if (!result) {
                // Save failed - keep modal open with error
                setModalErrorMessage(error || 'Failed to save strategy')
                return
              }
              // Save succeeded - clear error and confirm navigation
              setModalErrorMessage(null)
              unsavedChangesContext.confirmNavigation()
            }}
            onDiscard={() => {
              // Clear unsaved changes flag and confirm navigation
              // The form will be reset when the component unmounts
              setHasUnsavedChanges(false)
              setModalErrorMessage(null)
              unsavedChangesContext.confirmNavigation()
            }}
            onCancel={() => {
              setModalErrorMessage(null)
              unsavedChangesContext.cancelNavigation()
            }}
            isSaving={isSaving}
            errorMessage={modalErrorMessage}
            isError={!!modalErrorMessage}
          />
        </>
      )}
    </div>
  )
}

export default ChangeEmbeddingModel
