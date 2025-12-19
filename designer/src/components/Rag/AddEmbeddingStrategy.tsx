import { useMemo, useState, useEffect } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject } from '../../hooks/useProjects'
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
import { useDatabaseManager } from '../../hooks/useDatabaseManager'
import { getClientSideSecret } from '../../utils/crypto'
import { validateStrategyName } from '../../utils/security'
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

function AddEmbeddingStrategy() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [searchParams] = useSearchParams()
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

  // Get the database from URL query params (defaults to main_database if not provided)
  const database = searchParams.get('database') || 'main_database'

  // Get existing embedding strategies for copy from dropdown
  const existingStrategies = useMemo(() => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig) return []
    const db = projectConfig.rag?.databases?.find(
      (d: any) => d.name === database
    )
    return db?.embedding_strategies || []
  }, [projectResp, database])

  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Name
  const [name, setName] = useState('new-embedding-strategy')
  const [nameTouched, setNameTouched] = useState(false)

  // Copy from existing strategy
  const [copyFrom, setCopyFrom] = useState<string>('')

  // Unsaved changes tracking
  const unsavedChangesContext = useUnsavedChanges()
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [modalErrorMessage, setModalErrorMessage] = useState<string | null>(
    null
  )

  // Editable settings shown at top
  const [baseUrl, setBaseUrl] = useState('http://localhost:11434')
  const [dimension, setDimension] = useState<number>(768)
  const [batchSize, setBatchSize] = useState<number>(16)
  const [timeoutSec, setTimeoutSec] = useState<number>(60)
  const [autoPull, setAutoPull] = useState<boolean>(true)

  // Selection UI state (reusing the edit page's structure)
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
  }, [cachedModelsResponse, query])

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

  // Track changes to form fields (after all state is declared)
  useEffect(() => {
    // Check if any form field has been modified from defaults
    const hasChanges =
      name !== 'new-embedding-strategy' ||
      copyFrom !== '' ||
      selected !== null ||
      baseUrl !== 'http://localhost:11434' ||
      dimension !== 768 ||
      batchSize !== 16 ||
      timeoutSec !== 60 ||
      autoPull !== true ||
      provider !== 'Ollama (remote)' ||
      model !== 'nomic-embed-text' ||
      customModel !== '' ||
      apiKey !== '' ||
      openaiOrg !== '' ||
      openaiMaxRetries !== 3 ||
      azureDeployment !== '' ||
      azureResource !== '' ||
      azureApiVersion !== '' ||
      vertexProjectId !== '' ||
      vertexLocation !== '' ||
      vertexEndpoint !== '' ||
      bedrockRegion !== ''

    setHasUnsavedChanges(hasChanges)
  }, [
    name,
    copyFrom,
    selected,
    baseUrl,
    dimension,
    batchSize,
    timeoutSec,
    autoPull,
    provider,
    model,
    customModel,
    apiKey,
    openaiOrg,
    openaiMaxRetries,
    azureDeployment,
    azureResource,
    azureApiVersion,
    vertexProjectId,
    vertexLocation,
    vertexEndpoint,
    bedrockRegion,
  ])

  // Sync isDirty with context
  useEffect(() => {
    unsavedChangesContext.setIsDirty(hasUnsavedChanges)
  }, [hasUnsavedChanges, unsavedChangesContext])

  // Handle copy from existing strategy
  useEffect(() => {
    if (!copyFrom || !projectResp) return

    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig) return

    const db = projectConfig.rag?.databases?.find(
      (d: any) => d.name === database
    )
    const strategy = db?.embedding_strategies?.find(
      (s: any) => s.name === copyFrom
    )

    if (!strategy?.config) return

    const config = strategy.config
    const strategyType = strategy.type || 'OllamaEmbedder'

    // Populate common fields
    if (typeof config.dimension === 'number') setDimension(config.dimension)
    if (typeof config.batch_size === 'number') setBatchSize(config.batch_size)
    if (typeof config.timeout === 'number') setTimeoutSec(config.timeout)
    if (config.base_url) setBaseUrl(config.base_url)
    if (typeof config.auto_pull === 'boolean') setAutoPull(config.auto_pull)

    // Handle model and provider based on strategy type
    if (strategyType === 'UniversalEmbedder') {
      // UniversalEmbedder - local HuggingFace model
      setSourceTab('local')
      setProvider('Ollama (remote)')

      if (config.model) {
        const modelId = config.model
        // Use a timeout to ensure localGroups is populated
        setTimeout(() => {
          // Find variant by modelIdentifier or mapping
          const variant = localGroups
            .flatMap(g => g.variants)
            .find(
              v =>
                v.modelIdentifier === modelId ||
                modelIdToHuggingFace[v.id] === modelId ||
                v.id === modelId
            )

          if (variant) {
            setSelected({
              runtime: 'Local',
              provider: 'Ollama',
              modelId: variant.id,
            })
            setShowModelTable(false)
          }
        }, 100)
      }
    } else if (strategyType === 'OllamaEmbedder') {
      setSourceTab('local')
      setProvider('Ollama (remote)')
      if (config.model) {
        const modelName = config.model
        const providerModels = modelMap['Ollama (remote)'] || []
        if (providerModels.includes(modelName)) {
          setModel(modelName)
        } else {
          setModel('Custom')
          setCustomModel(modelName)
        }
        setSelected({
          runtime: 'Local',
          provider: 'Ollama',
          modelId: modelName,
        })
      }
    } else if (strategyType === 'OpenAIEmbedder') {
      setSourceTab('cloud')
      setProvider('OpenAI')
      if (config.model) {
        const modelName = config.model
        const providerModels = modelMap['OpenAI'] || []
        if (providerModels.includes(modelName)) {
          setModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'OpenAI',
            modelId: modelName,
          })
        } else {
          setModel('Custom')
          setCustomModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'OpenAI',
            modelId: modelName,
          })
        }
      }
      if (config.organization) setOpenaiOrg(config.organization)
      if (config.max_retries) setOpenaiMaxRetries(config.max_retries)
    } else if (strategyType.includes('Azure')) {
      setSourceTab('cloud')
      setProvider('Azure OpenAI')
      if (config.model) {
        const modelName = config.model
        const providerModels = modelMap['Azure OpenAI'] || []
        if (providerModels.includes(modelName)) {
          setModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'Azure OpenAI',
            modelId: modelName,
          })
        } else {
          setModel('Custom')
          setCustomModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'Azure OpenAI',
            modelId: modelName,
          })
        }
      }
      if (config.deployment) setAzureDeployment(config.deployment)
      if (config.endpoint) setAzureResource(config.endpoint)
      if (config.api_version) setAzureApiVersion(config.api_version)
    } else if (strategyType.includes('Google')) {
      setSourceTab('cloud')
      setProvider('Google')
      if (config.model) {
        const modelName = config.model
        const providerModels = modelMap['Google'] || []
        if (providerModels.includes(modelName)) {
          setModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'Google',
            modelId: modelName,
          })
        } else {
          setModel('Custom')
          setCustomModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'Google',
            modelId: modelName,
          })
        }
      }
      if (config.project_id) setVertexProjectId(config.project_id)
      if (config.region) setVertexLocation(config.region)
      if (config.endpoint) setVertexEndpoint(config.endpoint)
    } else if (strategyType.includes('Bedrock')) {
      setSourceTab('cloud')
      setProvider('AWS Bedrock')
      if (config.model) {
        const modelName = config.model
        const providerModels = modelMap['AWS Bedrock'] || []
        if (providerModels.includes(modelName)) {
          setModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'AWS Bedrock',
            modelId: modelName,
          })
        } else {
          setModel('Custom')
          setCustomModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'AWS Bedrock',
            modelId: modelName,
          })
        }
      }
      if (config.region) setBedrockRegion(config.region)
    } else if (strategyType === 'CohereEmbedder') {
      setSourceTab('cloud')
      setProvider('Cohere')
      if (config.model) {
        const modelName = config.model
        const providerModels = modelMap['Cohere'] || []
        if (providerModels.includes(modelName)) {
          setModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'Cohere',
            modelId: modelName,
          })
        } else {
          setModel('Custom')
          setCustomModel(modelName)
          setSelected({
            runtime: 'Cloud',
            provider: 'Cohere',
            modelId: modelName,
          })
        }
      }
    }
  }, [copyFrom, projectResp, database, localGroups])

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
            const meta = embeddingMeta[variant.id]
            if (meta?.dim)
              setDimension(Number(String(meta.dim).replace(/[^0-9]/g, '')))
            if (!baseUrl.trim()) setBaseUrl('http://localhost:11434')
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

  // Handlers to select models (no modal here)
  const selectLocal = async (v: Variant) => {
    // Check if model needs to be downloaded
    if (!v.isDownloaded && v.modelIdentifier) {
      // Show confirmation modal
      setPendingDownloadVariant(v)
      setDownloadConfirmOpen(true)
    } else {
      // Model is already on disk, just select it
      const meta = embeddingMeta[v.id]
      if (meta?.dim)
        setDimension(Number(String(meta.dim).replace(/[^0-9]/g, '')))
      if (!baseUrl.trim()) setBaseUrl('http://localhost:11434')
      setSelected({ runtime: 'Local', provider: 'Ollama', modelId: v.id })
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

  const selectCloud = () => {
    const key = model === 'Custom' ? customModel.trim() : model
    const meta = key ? embeddingMeta[key] : undefined
    if (meta?.dim) setDimension(Number(String(meta.dim).replace(/[^0-9]/g, '')))
    setSelected({ runtime: 'Cloud', provider, modelId: key || 'Custom' })
  }

  // Derived summary
  const summaryProvider = (() => {
    if (selected?.runtime === 'Local' && selected?.modelId) {
      // Check if this is a UniversalEmbedder (HuggingFace model from local groups)
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
    return selected?.provider || null
  })()
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

  // Helper to map UI provider names to config types
  const mapProviderToType = (providerLabel: string): string => {
    const typeMap: Record<string, string> = {
      Ollama: 'OllamaEmbedder',
      'Ollama (remote)': 'OllamaEmbedder',
      OpenAI: 'OpenAIEmbedder',
      Google: 'OpenAIEmbedder', // Uses OpenAI-compatible endpoint
      'Azure OpenAI': 'OpenAIEmbedder',
      HuggingFace: 'HuggingFaceEmbedder',
      SentenceTransformer: 'SentenceTransformerEmbedder',
    }
    return typeMap[providerLabel] || 'OllamaEmbedder'
  }

  // Helper to build strategy configuration
  const buildStrategyConfig = (
    runtime: 'Local' | 'Cloud',
    providerLabel: string,
    chosenModel: string,
    encryptedApiKey?: string
  ) => {
    const config: Record<string, any> = {}

    // For local models, get the full HuggingFace model identifier
    let modelIdentifier = chosenModel
    if (runtime === 'Local') {
      // Find the variant to get the modelIdentifier
      const variant = localGroups
        .flatMap(g => g.variants)
        .find(v => v.id === chosenModel)
      modelIdentifier =
        variant?.modelIdentifier ||
        modelIdToHuggingFace[chosenModel] ||
        chosenModel
    }

    // Add common fields
    if (modelIdentifier) config.model = modelIdentifier
    if (dimension) config.dimension = parseInt(String(dimension))
    if (batchSize) config.batch_size = parseInt(String(batchSize))
    if (timeoutSec) config.timeout = parseInt(String(timeoutSec))

    // Check if this is a UniversalEmbedder (HuggingFace model from local groups)
    const isUniversalEmbedder =
      runtime === 'Local' &&
      localGroups
        .flatMap(g => g.variants)
        .some(v => v.id === chosenModel && v.modelIdentifier)

    // Add provider-specific fields
    if (isUniversalEmbedder) {
      // UniversalEmbedder config
      config.base_url = baseUrl?.trim() || 'http://127.0.0.1:11540/v1'
      config.api_key = 'universal'
    } else if (runtime === 'Local' || providerLabel === 'Ollama (remote)') {
      // OllamaEmbedder config
      if (baseUrl) config.base_url = baseUrl.trim()
      config.auto_pull = autoPull !== undefined ? autoPull : true
    } else if (runtime === 'Cloud') {
      if (providerLabel === 'OpenAI') {
        if (baseUrl) config.base_url = baseUrl.trim()
        if (openaiOrg) config.organization = openaiOrg.trim()
        if (openaiMaxRetries) config.max_retries = openaiMaxRetries
        if (encryptedApiKey) config.api_key = encryptedApiKey
      } else if (providerLabel === 'Azure OpenAI') {
        if (azureDeployment) config.deployment = azureDeployment.trim()
        if (azureResource) config.endpoint = azureResource.trim()
        if (azureApiVersion) config.api_version = azureApiVersion.trim()
        if (encryptedApiKey) config.api_key = encryptedApiKey
      } else if (providerLabel === 'Google') {
        if (vertexProjectId) config.project_id = vertexProjectId.trim()
        if (vertexLocation) config.region = vertexLocation.trim()
        if (vertexEndpoint) config.endpoint = vertexEndpoint.trim()
        if (encryptedApiKey) config.api_key = encryptedApiKey
      } else if (providerLabel === 'AWS Bedrock') {
        if (bedrockRegion) config.region = bedrockRegion.trim()
        if (encryptedApiKey) config.api_key = encryptedApiKey
      } else if (providerLabel === 'Cohere') {
        if (baseUrl) config.base_url = baseUrl.trim()
        if (encryptedApiKey) config.api_key = encryptedApiKey
      } else if (providerLabel === 'Voyage AI') {
        if (baseUrl) config.base_url = baseUrl.trim()
        if (encryptedApiKey) config.api_key = encryptedApiKey
      }
    }

    return config
  }

  // Real-time duplicate name validation (case-insensitive)
  // Check immediately for default names, or after field is touched
  const duplicateNameError = useMemo(() => {
    if (!projectResp || !name.trim()) return null
    
    // Always check for duplicates (don't wait for touch) so default names are validated
    const projectConfig = (projectResp as any)?.project?.config
    const currentDb = projectConfig?.rag?.databases?.find(
      (db: any) => db.name === database
    )
    if (currentDb) {
      const nameLower = name.trim().toLowerCase()
      const nameExists = currentDb.embedding_strategies?.some(
        (s: any) => s.name?.toLowerCase() === nameLower
      )
      if (nameExists) {
        return 'A strategy with this name already exists'
      }
    }
    return null
  }, [name, projectResp, database])

  // Validation
  const validateStrategy = (): string[] => {
    const errors: string[] = []

    // Validate strategy name with security checks
    const nameError = validateStrategyName(name)
    if (nameError) {
      errors.push(nameError)
    }

    // Validate model selection
    if (!selected) {
      errors.push('Please select a model')
    } else {
      // Validate the selected model is valid
      const selectedModelId = selected.modelId

      // For custom models, validate the custom model name
      if (selectedModelId === 'Custom' || model === 'Custom') {
        if (!customModel || !customModel.trim()) {
          errors.push('Custom model name is required')
        } else if (!/^[a-zA-Z0-9\/_.-]+$/.test(customModel)) {
          errors.push(
            'Custom model name contains invalid characters. Only letters, numbers, slashes, hyphens, dots, and underscores are allowed.'
          )
        }
      } else {
        // Validate non-custom model exists in our list
        const isValidLocal = localGroups.some(
          group =>
            group.name === selectedModelId ||
            group.variants?.some(v => v.id === selectedModelId)
        )

        // For cloud providers, we trust the selected object since it came from our provider data
        const isCloudProvider = selected.runtime === 'Cloud'

        if (!isValidLocal && !isCloudProvider) {
          errors.push(
            'Selected model is not supported. Please choose a valid model.'
          )
        }
      }
    }

    // Validate dimension
    if (dimension && (dimension < 1 || dimension > 8192)) {
      errors.push('Dimension must be between 1 and 8192')
    }

    // Check for duplicate strategy name
    if (duplicateNameError) {
      errors.push(duplicateNameError)
    }

    return errors
  }

  const saveStrategyToConfig = async (
    runtime: 'Local' | 'Cloud',
    providerLabel: string,
    chosenModel: string,
    encryptedApiKey?: string
  ) => {
    try {
      setIsSaving(true)
      setError(null)

      // Validate BEFORE attempting save
      setNameTouched(true) // Mark as touched when attempting save
      const validationErrors = validateStrategy()
      if (validationErrors.length > 0) {
        const errorMessage = validationErrors.join(', ')
        setError(errorMessage)
        // Only show toast for non-duplicate-name errors
        const duplicateNameError = validationErrors.some(e => 
          e.includes('already exists')
        )
        if (!duplicateNameError) {
          toast({
            message: errorMessage,
            variant: 'destructive',
          })
        }
        return false
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

      // Determine the correct strategy type
      // Local HuggingFace models should use UniversalEmbedder, not OllamaEmbedder
      let strategyType = mapProviderToType(providerLabel)
      if (runtime === 'Local' && chosenModel) {
        // Check if this is a HuggingFace model from local groups
        const variant = localGroups
          .flatMap(g => g.variants)
          .find(v => v.id === chosenModel)
        if (variant?.modelIdentifier) {
          // This is a HuggingFace model, use UniversalEmbedder
          strategyType = 'UniversalEmbedder'
        }
      }

      // Build the new strategy
      const newStrategy = {
        name: name.trim(),
        type: strategyType,
        priority: (currentDb.embedding_strategies?.length || 0) * 10,
        config: buildStrategyConfig(
          runtime,
          providerLabel,
          chosenModel,
          encryptedApiKey
        ),
      }

      // Add to existing strategies
      const updatedStrategies = [
        ...(currentDb.embedding_strategies || []),
        newStrategy,
      ]

      // Determine if this should be default
      const shouldBeDefault = makeDefault || updatedStrategies.length === 1

      // Update database configuration
      await databaseManager.updateDatabase.mutateAsync({
        oldName: database,
        updates: {
          embedding_strategies: updatedStrategies,
          default_embedding_strategy: shouldBeDefault
            ? newStrategy.name
            : currentDb.default_embedding_strategy,
        },
        projectConfig,
      })

      return true
    } catch (error: any) {
      console.error('Failed to save embedding strategy:', error)
      setError(error.message || 'Failed to save strategy')
      return false
    } finally {
      setIsSaving(false)
    }
  }

  const finalizeAndRedirect = () => {
    // Clear unsaved changes flags BEFORE navigation to prevent modal from showing
    setHasUnsavedChanges(false)
    unsavedChangesContext.setIsDirty(false)
    toast({ message: 'Strategy saved', variant: 'default' })
    // Use requestAnimationFrame to ensure state updates propagate before navigation
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
    navigate('/chat/databases')
      })
    })
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-40">
      {error && (
        <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-2 rounded-md text-sm">
          {error}
        </div>
      )}
      <div className="flex items-center justify-between mb-1 md:mb-3">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/databases')}
          >
            Databases
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
            disabled={
              !selected || 
              name.trim().length === 0 || 
              !!validateStrategyName(name) || 
              !!duplicateNameError
            }
          >
            Save strategy
          </Button>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-card p-4 md:p-6 flex flex-col gap-4">
        {/* Name and Copy from */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">
              Strategy name
            </Label>
            <Input
              value={name}
              onChange={e => {
                setName(e.target.value)
                // Clear error state when user starts typing if validation passes
                if (nameTouched) {
                  const nameError = validateStrategyName(e.target.value)
                  const dupError = duplicateNameError
                  if (!nameError && !dupError) {
                    setError(null)
                  }
                }
              }}
              onBlur={() => setNameTouched(true)}
              placeholder="Enter a name"
              className={`h-9 ${
                (validateStrategyName(name) || duplicateNameError)
                  ? 'border-destructive'
                  : ''
              }`}
            />
            {validateStrategyName(name) && (
              <p className="text-xs text-destructive mt-1">
                {validateStrategyName(name)}
              </p>
            )}
            {!validateStrategyName(name) && duplicateNameError && (
              <p className="text-xs text-destructive mt-1">
                {duplicateNameError}
              </p>
            )}
            {!validateStrategyName(name) && !duplicateNameError && (
              <p className="text-xs text-muted-foreground mt-1">
                Can only contain letters, numbers, hyphens, and underscores
              </p>
            )}
          </div>
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">Copy from</Label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="w-full h-9 rounded-md border border-border bg-background px-3 text-left flex items-center justify-between text-sm">
                  <span
                    className={
                      copyFrom ? 'text-foreground' : 'text-muted-foreground'
                    }
                  >
                    {copyFrom || 'Select a strategy to copy...'}
                  </span>
                  <FontIcon type="chevron-down" className="w-4 h-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64">
                <DropdownMenuItem
                  onClick={() => {
                    setCopyFrom('')
                    // Reset form to defaults
                    setName('new-embedding-strategy')
                    setBaseUrl('http://localhost:11434')
                    setDimension(768)
                    setBatchSize(16)
                    setTimeoutSec(60)
                    setAutoPull(true)
                    setProvider('Ollama (remote)')
                    setModel('nomic-embed-text')
                    setCustomModel('')
                    setSelected(null)
                    setSourceTab('local')
                    setShowModelTable(true)
                    // Reset cloud fields
                    setApiKey('')
                    setOpenaiOrg('')
                    setOpenaiMaxRetries(3)
                    setAzureDeployment('')
                    setAzureResource('')
                    setAzureApiVersion('')
                    setVertexProjectId('')
                    setVertexLocation('')
                    setVertexEndpoint('')
                    setBedrockRegion('')
                  }}
                >
                  None
                </DropdownMenuItem>
                {existingStrategies.map((strategy: any) => (
                  <DropdownMenuItem
                    key={strategy.name}
                    onClick={() => setCopyFrom(strategy.name)}
                  >
                    {strategy.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
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
            {selected?.runtime === 'Local' && selected?.modelId
              ? 'Change model'
              : 'Select the model you would like to use for this strategy.'}
          </div>
          <div />
        </div>

        {/* Selected model card - always show when model is selected */}
        {selected?.runtime === 'Local' && selected?.modelId && (
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="text-sm font-medium text-foreground">
                  {localGroups
                    .flatMap(g => g.variants)
                    .find(v => v.id === selected.modelId)?.label ||
                    selected.modelId}
                </div>
                <div className="text-xs text-muted-foreground mt-1">
                  {(() => {
                    const variant = localGroups
                      .flatMap(g => g.variants)
                      .find(v => v.id === selected.modelId)
                    if (variant?.isDownloaded) {
                      return `On disk • ${variant.download}`
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

        {/* Model selection area - show when showModelTable is true */}
        {showModelTable && (
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm font-medium text-foreground">
                Select model
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
            onSelect={selectLocal}
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
                onClick={() => {
                  selectCloud()
                  // Close the model selection area when a new model is selected
                  setShowModelTable(false)
                }}
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
        <Dialog 
          open={confirmOpen} 
          onOpenChange={(open) => {
            setConfirmOpen(open)
            if (!open) {
              // Clear error when modal closes
              setError(null)
            }
          }}
        >
          <DialogContent>
            <DialogTitle>Save this embedding strategy?</DialogTitle>
            {error && (
              <div className="bg-destructive/10 border border-destructive text-destructive px-4 py-2 rounded-md text-sm mt-2">
                {error}
              </div>
            )}
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

                  const success = await saveStrategyToConfig(
                    selected.runtime,
                    summaryProvider || 'Provider',
                    selected.modelId,
                    encryptedKey
                  )

                  if (!success) {
                    // Validation failed - keep modal open, error is already set
                    return
                  }

                  // Clear unsaved changes flags BEFORE closing modal/navigating
                  setHasUnsavedChanges(false)
                  unsavedChangesContext.setIsDirty(false)

                  setConfirmOpen(false)
                  if (makeDefault) {
                    setReembedOpen(true)
                  } else {
                    finalizeAndRedirect()
                  }
                }}
                disabled={
                  isSaving || 
                  !!validateStrategyName(name) || 
                  !!duplicateNameError
                }
              >
                {isSaving
                  ? 'Saving...'
                  : selected?.runtime === 'Local' &&
                      selected?.modelId &&
                      !isModelOnDisk(selected.modelId)
                    ? 'Save strategy (model will be downloaded)'
                    : 'Save strategy'}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

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
                          {downloadStates[pendingDownloadVariant.id]?.progress}%
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
                  const meta = embeddingMeta[pendingDownloadVariant.id]
                  if (meta?.dim)
                    setDimension(
                      Number(String(meta.dim).replace(/[^0-9]/g, ''))
                    )
                  if (!baseUrl.trim()) setBaseUrl('http://localhost:11434')
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

        {/* Unsaved changes modal */}
        <UnsavedChangesModal
          isOpen={unsavedChangesContext.showModal}
          onSave={async () => {
            if (!selected) {
              setModalErrorMessage('Please select a model before saving')
              return
            }

            try {
              setIsSaving(true)
              setModalErrorMessage(null)

              // Encrypt API key if needed
              let encryptedKey: string | undefined
              if (selected.runtime === 'Cloud' && apiKey.trim()) {
                const secret = await getClientSideSecret()
                encryptedKey = await encryptAPIKey(apiKey, secret)
              }

              const success = await saveStrategyToConfig(
                selected.runtime,
                summaryProvider || 'Provider',
                selected.modelId,
                encryptedKey
              )

              if (!success) {
                setModalErrorMessage(error || 'Failed to save strategy')
                setIsSaving(false)
                return
              }

              // Save succeeded - clear error and confirm navigation
              setModalErrorMessage(null)
              setHasUnsavedChanges(false)
              setIsSaving(false)
              unsavedChangesContext.confirmNavigation()
            } catch (e: any) {
              setModalErrorMessage(e?.message || 'Failed to save strategy')
              setIsSaving(false)
            }
          }}
          onDiscard={() => {
            // Clear unsaved changes flag and confirm navigation
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
      </div>
    </div>
  )
}

export default AddEmbeddingStrategy
