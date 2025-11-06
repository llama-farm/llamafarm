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
import { parsePromptSets } from '../../utils/promptSets'
import { useCachedModels } from '../../hooks/useModels'
import modelService from '../../api/modelService'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import { PromptSetSelector } from './PromptSetSelector'
import { DeviceModelsSection, type DeviceModel } from './DeviceModelsSection'
import { CustomDownloadDialog } from './CustomDownloadDialog'
import { DeleteDeviceModelDialog } from './DeleteDeviceModelDialog'
import {
  transformCatalogToLocalGroups,
  searchModelGroups,
  getRecommendedByCategory,
  filterGroupsByRuntime,
  filterCloudModels,
  filterLocalModels,
  getVariantProviders,
  type LocalModelVariant,
  type Runtime,
  type ProviderInfo,
} from '../../utils/modelCatalog'
import { ProviderSelectionDialog } from './ProviderSelectionDialog'
import { ModelDownloadDialog } from './ModelDownloadDialog'
import { CloudModelConfigDialog } from './CloudModelConfigDialog'
import { useModelDownload } from '../../hooks/useModelDownload'

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
        <div className="mt-3 md:mt-0 md:justify-self-end w-full md:pl-4 mr-6 md:mr-8">
          <PromptSetSelector
            promptSetNames={promptSetNames}
            selectedPromptSets={selectedPromptSets}
            onTogglePromptSet={onTogglePromptSet}
            onClearPromptSets={onClearPromptSets}
            label="Prompt sets"
          />
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

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  let i = Math.floor(Math.log(bytes) / Math.log(1024))
  if (i >= units.length) i = units.length - 1
  const val = bytes / Math.pow(1024, i)
  return `${val.toFixed(i >= 2 ? 1 : 0)} ${units[i]}`
}

function formatETA(seconds: number): string {
  if (!isFinite(seconds) || seconds <= 0) return ''
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  if (m >= 60) {
    const h = Math.floor(m / 60)
    const rm = m % 60
    return `~${h}h ${rm}m`
  }
  if (m > 0) return `~${m}m ${s}s`
  return `~${s}s`
}

function AddOrChangeModels({
  onAddModel,
  onGoToProject,
  promptSetNames,
  customModelOpen,
  setCustomModelOpen,
  customDownloadState,
  setCustomDownloadState,
  customDownloadProgress,
  setCustomDownloadProgress,
  setShowBackgroundDownload,
  setBackgroundDownloadName,
  projectModels,
  downloadedBytes,
  setDownloadedBytes,
  totalBytes,
  setTotalBytes,
  estimatedTimeRemaining,
  setEstimatedTimeRemaining,
  selectedProviderInfo,
  setSelectedProviderInfo,
  setCloudConfig,
}: {
  onAddModel: (m: InferenceModel, promptSets?: string[]) => void
  onGoToProject: () => void
  promptSetNames: string[]
  customModelOpen: boolean
  setCustomModelOpen: (open: boolean) => void
  customDownloadState: 'idle' | 'downloading' | 'success' | 'error'
  setCustomDownloadState: (
    state: 'idle' | 'downloading' | 'success' | 'error'
  ) => void
  customDownloadProgress: number
  setCustomDownloadProgress: (progress: number) => void
  setShowBackgroundDownload: (show: boolean) => void
  setBackgroundDownloadName: (name: string) => void
  projectModels: InferenceModel[]
  downloadedBytes: number
  setDownloadedBytes: (n: number) => void
  totalBytes: number
  setTotalBytes: (n: number) => void
  estimatedTimeRemaining: string
  setEstimatedTimeRemaining: (s: string) => void
  selectedProviderInfo: ProviderInfo | null
  setSelectedProviderInfo: (info: ProviderInfo | null) => void
  setCloudConfig: (config: { apiKey: string; baseUrl: string } | null) => void
}) {
  const [sourceTab, setSourceTab] = useState<'local' | 'cloud'>('local')
  const [query, setQuery] = useState('')
  const [runtimeFilter, setRuntimeFilter] = useState<Runtime>('all')
  const [expandedGroupId, setExpandedGroupId] = useState<number | null>(null)
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [pendingVariant, setPendingVariant] = useState<LocalModelVariant | null>(
    null
  )
  const [providerDialogOpen, setProviderDialogOpen] = useState(false)
  const [pendingVariantForProvider, setPendingVariantForProvider] =
    useState<LocalModelVariant | null>(null)
  const [cloudConfigDialogOpen, setCloudConfigDialogOpen] = useState(false)
  const [downloadDialogOpen, setDownloadDialogOpen] = useState(false)
  const [submitState, setSubmitState] = useState<
    'idle' | 'loading' | 'success'
  >('idle')

  // Model download hook
  const modelDownload = useModelDownload()
  const [modelName, setModelName] = useState('')
  const [modelDescription, setModelDescription] = useState('')
  const [selectedPromptSets, setSelectedPromptSets] = useState<string[]>([])

  // Device model state
  const [deviceConfirmOpen, setDeviceConfirmOpen] = useState(false)
  const [pendingDeviceModel, setPendingDeviceModel] =
    useState<DeviceModel | null>(null)
  const [deviceSubmitState, setDeviceSubmitState] = useState<
    'idle' | 'loading' | 'success'
  >('idle')
  const [deviceModelName, setDeviceModelName] = useState('')
  const [deviceModelDescription, setDeviceModelDescription] = useState('')
  const [deviceSelectedPromptSets, setDeviceSelectedPromptSets] = useState<
    string[]
  >([])

  // Delete device model state
  const [deleteConfirmModelOpen, setDeleteConfirmModelOpen] = useState(false)
  const [modelToDelete, setModelToDelete] = useState<DeviceModel | null>(null)
  const [deleteState, setDeleteState] = useState<
    'idle' | 'deleting' | 'success' | 'error'
  >('idle')
  const [deleteError, setDeleteError] = useState('')

  // Manual refresh state to ensure visible feedback
  const [isManuallyRefreshing, setIsManuallyRefreshing] = useState(false)

  // Custom model local state (not shared)
  const [customModelInput, setCustomModelInput] = useState('')
  const [customModelName, setCustomModelName] = useState('')
  const [customModelDescription, setCustomModelDescription] = useState('')
  const [customSelectedPromptSets, setCustomSelectedPromptSets] = useState<
    string[]
  >([])
  const [customDownloadError, setCustomDownloadError] = useState('')

  // Load model groups from catalog
  const localGroups = transformCatalogToLocalGroups()

  // Get recommended models
  const recommendedByCategory = getRecommendedByCategory()

  // Filter groups based on source tab (local vs cloud)
  const sourceFilteredGroups =
    sourceTab === 'cloud' ? filterCloudModels(localGroups) : filterLocalModels(localGroups)

  // Filter groups based on runtime and search query (only for local tab)
  const runtimeFilteredGroups =
    sourceTab === 'local'
      ? filterGroupsByRuntime(sourceFilteredGroups, runtimeFilter)
      : sourceFilteredGroups
  const filteredGroups = searchModelGroups(runtimeFilteredGroups, query)

  // Fetch cached models from backend
  const {
    data: cachedModelsResponse,
    isLoading: isLoadingCachedModels,
    refetch: refetchCachedModels,
  } = useCachedModels()

  // Convert cached models to device models format
  const deviceModels: DeviceModel[] =
    cachedModelsResponse?.data.map(cachedModel => ({
      id: cachedModel.id,
      name: cachedModel.name,
      modelIdentifier: cachedModel.name,
      meta: formatBytes(cachedModel.size),
      badges: ['Local', 'Disk'],
    })) || []

  const handleUseDeviceModel = (model: DeviceModel) => {
    setPendingDeviceModel(model)
    setDeviceModelName(model.name)
    setDeviceConfirmOpen(true)
  }

  const handleDeleteDeviceModel = (model: DeviceModel) => {
    setModelToDelete(model)
    setDeleteConfirmModelOpen(true)
  }

  const confirmDeleteDeviceModel = async () => {
    if (!modelToDelete) return

    setDeleteState('deleting')
    setDeleteError('')

    try {
      await modelService.deleteModel(modelToDelete.modelIdentifier)
      setDeleteState('success')
      // Refresh the cached models list
      refetchCachedModels()
      // Close dialog after short delay
      setTimeout(() => {
        setDeleteConfirmModelOpen(false)
        setModelToDelete(null)
        setDeleteState('idle')
      }, 1000)
    } catch (error: any) {
      setDeleteState('error')
      setDeleteError(
        error.response?.data?.detail ||
          error.message ||
          'Failed to delete model'
      )
    }
  }

  // Check if a device model is already in the project
  const isModelInUse = (modelId: string): boolean => {
    return projectModels.some(pm => pm.modelIdentifier === modelId)
  }

  // Handle variant selection - show provider dialog
  const handleVariantSelect = (variant: LocalModelVariant) => {
    setPendingVariantForProvider(variant)
    setProviderDialogOpen(true)
  }

  // Handle provider selection from dialog
  const handleProviderSelect = async (provider: ProviderInfo) => {
    if (!pendingVariantForProvider) return

    setSelectedProviderInfo(provider)
    setProviderDialogOpen(false)

    // Check if this is a cloud provider (OpenAI-compatible API)
    if (provider.runtime === 'openai') {
      // Show cloud configuration dialog to get API key
      setCloudConfigDialogOpen(true)
      return
    }

    // Check if this is a universal provider that needs downloading
    if (provider.runtime === 'universal' && provider.modelId) {
      // Check if model is already in cache
      const isInCache = deviceModels.some(
        m => m.modelIdentifier === provider.modelId
      )

      if (!isInCache) {
        // Model needs to be downloaded - show download dialog
        setDownloadDialogOpen(true)
        const success = await modelDownload.downloadModel(
          provider.modelId,
          'universal'
        )

        if (success) {
          // Download successful, refresh cached models
          await refetchCachedModels()
          // Now show the model configuration dialog
          setPendingVariant(pendingVariantForProvider)
          setConfirmOpen(true)
        }

        setDownloadDialogOpen(false)
        setPendingVariantForProvider(null)
      } else {
        // Model already downloaded, go straight to configuration
        setPendingVariant(pendingVariantForProvider)
        setConfirmOpen(true)
        setPendingVariantForProvider(null)
      }
    } else {
      // Non-universal/non-cloud provider, go straight to configuration
      setPendingVariant(pendingVariantForProvider)
      setConfirmOpen(true)
      setPendingVariantForProvider(null)
    }
  }

  // Handle cloud model configuration (after API key is entered)
  const handleCloudModelConfigure = (config: {
    apiKey: string
    baseUrl: string
  }) => {
    setCloudConfig(config)
    setCloudConfigDialogOpen(false)
    // Now show the model configuration dialog
    setPendingVariant(pendingVariantForProvider)
    setConfirmOpen(true)
    setPendingVariantForProvider(null)
  }

  // Handle custom model download
  const handleCustomModelDownload = async () => {
    setCustomDownloadState('downloading')
    setCustomDownloadProgress(5)
    setCustomDownloadError('')
    setBackgroundDownloadName(customModelName.trim())
    setDownloadedBytes(0)
    setTotalBytes(0)
    setEstimatedTimeRemaining('')
    const start = Date.now()

    const downloadAsync = async () => {
      try {
        for await (const event of modelService.downloadModel({
          model_name: customModelInput.trim(),
          provider: 'universal',
        })) {
          if (event.event === 'progress') {
            const d = Number(event.n || 0)
            const t = Number(event.total || 0)
            setDownloadedBytes(d)
            setTotalBytes(t)
            if (t > 0 && isFinite(d) && d >= 0) {
              const percent = Math.max(
                5,
                Math.min(95, Math.round((d / t) * 90) + 5)
              )
              setCustomDownloadProgress(percent)
              const elapsedSec = (Date.now() - start) / 1000
              if (elapsedSec > 0) {
                const speed = d / elapsedSec
                const remain = (t - d) / (speed || 1)
                setEstimatedTimeRemaining(formatETA(remain))
              }
            }
          } else if (event.event === 'done') {
            setCustomDownloadProgress(100)
            setCustomDownloadState('success')
            setEstimatedTimeRemaining('')
            onAddModel(
              {
                id: `custom-${customModelInput.trim()}`,
                name: customModelName.trim(),
                modelIdentifier: customModelInput.trim(),
                meta:
                  customModelDescription.trim() ||
                  'Downloaded from HuggingFace',
                badges: ['Local', 'HuggingFace'],
                status: 'ready',
              },
              customSelectedPromptSets.length > 0
                ? customSelectedPromptSets
                : undefined
            )
            refetchCachedModels()
            setTimeout(() => {
              setCustomModelOpen(false)
              onGoToProject()
            }, 1000)
            setTimeout(() => {
              setShowBackgroundDownload(false)
              setCustomDownloadState('idle')
            }, 4000)
          } else if (event.event === 'error') {
            setCustomDownloadState('error')
            setCustomDownloadError(
              event.message ||
                'Failed to download model. Please check the model name and try again.'
            )
            setShowBackgroundDownload(false)
          }
        }
      } catch (error: any) {
        setCustomDownloadState('error')
        setCustomDownloadError(
          error.message ||
            'Failed to download model. Please check the model name and try again.'
        )
        setShowBackgroundDownload(false)
      }
    }

    downloadAsync()
  }

  return (
    <>
      {/* Models on device section */}
      <DeviceModelsSection
        models={deviceModels}
        isLoading={isLoadingCachedModels}
        isRefreshing={isManuallyRefreshing}
        onUse={handleUseDeviceModel}
        onDelete={handleDeleteDeviceModel}
        onRefresh={async () => {
          setIsManuallyRefreshing(true)
          const startTime = Date.now()
          await refetchCachedModels()
          const elapsed = Date.now() - startTime
          const remaining = Math.max(0, 800 - elapsed)
          setTimeout(() => {
            setIsManuallyRefreshing(false)
          }, remaining)
        }}
        isModelInUse={isModelInUse}
      />

      {/* Download or use other models section */}
      <div className="flex flex-col gap-4">
        <div>
          <h3 className="font-medium">Download or use other models</h3>
          <div className="h-1" />
          <div className="text-sm text-muted-foreground">
            Add a new model provider or switch which models are enabled for this
            project.
          </div>
        </div>
        <div className="rounded-xl border border-border bg-card p-4 md:p-6 flex flex-col gap-4 mb-12">
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

          {/* Runtime filter and search - only show for local models */}
          {sourceTab === 'local' && (
            <div className="flex flex-col gap-3 w-full">
              <div className="flex items-center gap-2 w-full">
                <div className="flex items-center gap-2">
                  <Label className="text-xs text-muted-foreground whitespace-nowrap">
                    Filter by runtime:
                  </Label>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <button className="h-9 rounded-md border border-border bg-background px-3 text-sm flex items-center gap-2 hover:bg-accent/50">
                        <span className="capitalize">{runtimeFilter}</span>
                        <FontIcon type="chevron-down" className="w-4 h-4" />
                      </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="start">
                      <DropdownMenuItem onClick={() => setRuntimeFilter('all')}>
                        All runtimes
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => setRuntimeFilter('universal')}
                      >
                        Universal (Transformers)
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setRuntimeFilter('ollama')}>
                        Ollama (GGUF)
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => setRuntimeFilter('lemonade')}
                      >
                        Lemonade (All formats)
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={() => setRuntimeFilter('openai')}>
                        OpenAI (API)
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>

                <div className="relative flex-1">
                  <FontIcon
                    type="search"
                    className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2"
                  />
                  <Input
                    placeholder="Search local options"
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    className="pl-9 h-9"
                  />
                </div>
                <Button
                  variant="outline"
                  onClick={() => {
                    setCustomModelOpen(true)
                    setCustomModelInput('')
                    setCustomModelName('')
                    setCustomModelDescription('')
                    setCustomSelectedPromptSets([])
                    setCustomDownloadState('idle')
                    setCustomDownloadError('')
                  }}
                  className="h-9 whitespace-nowrap"
                >
                  Add HuggingFace model
                </Button>
              </div>
            </div>
          )}

          {/* Recommended models section */}
          {sourceTab === 'local' &&
            Object.keys(recommendedByCategory).length > 0 &&
            !query.trim() && (
              <div className="w-full">
                <div className="flex items-center gap-2 mb-3">
                  <FontIcon
                    type="checkmark-filled"
                    className="w-4 h-4 text-primary"
                  />
                  <h3 className="font-medium text-sm">Recommended models</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 mb-4">
                  {Object.entries(recommendedByCategory).map(
                    ([category, models]) => (
                      <div
                        key={category}
                        className="rounded-lg border border-border bg-card/50 p-4"
                      >
                        <div className="text-sm font-medium mb-1">
                          {category}
                        </div>
                        {models[0]?.categoryDescription && (
                          <div className="text-xs text-muted-foreground mb-3">
                            {models[0].categoryDescription}
                          </div>
                        )}
                        <div className="flex flex-col gap-2">
                          {models.slice(0, 2).map(model => (
                            <div
                              key={model.variantId}
                              className="flex items-center justify-between gap-2 text-xs"
                            >
                              <div className="flex-1 min-w-0">
                                <div className="font-medium truncate">
                                  {model.displayName}
                                </div>
                                <div className="text-muted-foreground">
                                  {model.parameters} • {model.downloadSize}
                                </div>
                              </div>
                              <Button
                                size="sm"
                                className="h-7 px-2 text-xs flex-shrink-0"
                                onClick={() => {
                                  // Find the variant in localGroups
                                  const variant = localGroups
                                    .flatMap(g => g.variants)
                                    .find(
                                      v =>
                                        v.label ===
                                        model.variantId.replace(':', ',')
                                    )
                                  if (variant) {
                                    handleVariantSelect(variant)
                                  }
                                }}
                              >
                                Add
                              </Button>
                            </div>
                          ))}
                        </div>
                      </div>
                    )
                  )}
                </div>
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
              {filteredGroups.length === 0 ? (
                <div className="p-6 flex flex-col items-center justify-center text-center">
                  <div className="text-sm text-muted-foreground mb-3">
                    No matching results. Want to download a different local
                    model from Hugging Face?
                  </div>
                  <Button
                    size="sm"
                    onClick={() => {
                      setCustomModelOpen(true)
                      setCustomModelInput('')
                      setCustomModelName('')
                      setCustomModelDescription('')
                      setCustomSelectedPromptSets([])
                      setCustomDownloadState('idle')
                      setCustomDownloadError('')
                    }}
                  >
                    Add HuggingFace model
                  </Button>
                </div>
              ) : (
                filteredGroups.map(group => {
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
                                <span className="ml-2 truncate">
                                  {variant.label}
                                </span>
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
                                  onClick={() => handleVariantSelect(variant)}
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
                })
              )}
            </div>
          )}
          {sourceTab === 'cloud' && (
            <div className="flex flex-col gap-2">
              {/* Cloud models search */}
              <div className="flex items-center gap-2 w-full mb-2">
                <div className="relative flex-1">
                  <FontIcon
                    type="search"
                    className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none"
                  />
                  <Input
                    type="text"
                    placeholder="Search cloud models..."
                    value={query}
                    onChange={e => setQuery(e.target.value)}
                    className="pl-9 bg-background"
                  />
                </div>
              </div>

              {/* Cloud model groups */}
              {filteredGroups.length === 0 ? (
                <div className="text-sm text-muted-foreground p-6 text-center">
                  No cloud models found
                </div>
              ) : (
                filteredGroups.map(group => {
                  const isExpanded = expandedGroupId === group.id

                  return (
                    <div
                      key={group.id}
                      className="border border-border rounded-lg bg-card overflow-hidden"
                    >
                      {/* Group header */}
                      <div
                        className="flex items-center justify-between px-4 py-3 cursor-pointer hover:bg-secondary/50 transition-colors"
                        onClick={() =>
                          setExpandedGroupId(isExpanded ? null : group.id)
                        }
                      >
                        <div className="flex-1">
                          <div className="font-medium text-foreground capitalize">
                            {group.name}
                          </div>
                          <div className="text-xs text-muted-foreground mt-0.5">
                            {group.parameterSummary}
                          </div>
                        </div>
                        <div className="flex items-center gap-3">
                          <div className="text-xs text-muted-foreground">
                            API-based
                          </div>
                          <FontIcon
                            type="chevron-down"
                            className={`w-4 h-4 text-muted-foreground transition-transform ${
                              isExpanded ? 'rotate-180' : ''
                            }`}
                          />
                        </div>
                      </div>

                      {/* Expanded variants */}
                      {isExpanded && (
                        <div className="border-t border-border bg-secondary/20">
                          {group.variants.map(variant => (
                            <div
                              key={variant.id}
                              className="grid grid-cols-[1fr_auto_auto] gap-3 items-center px-4 py-3 hover:bg-secondary/40 transition-colors border-b border-border last:border-b-0"
                            >
                              <div className="col-span-1">
                                <div className="font-medium text-sm text-foreground">
                                  {variant.label}
                                </div>
                              </div>
                              <div className="col-span-1 text-xs text-muted-foreground">
                                {variant.downloadSize}
                              </div>
                              <div className="col-span-1 flex items-center justify-end pr-2">
                                <Button
                                  size="sm"
                                  className="h-8 px-3"
                                  onClick={() => handleVariantSelect(variant)}
                                >
                                  Add
                                </Button>
                              </div>
                            </div>
                          ))}
                          <div className="flex justify-end pr-3 py-2">
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
                })
              )}
            </div>
          )}
        </div>
      </div>

      {/* Custom model download dialog */}
      <CustomDownloadDialog
        open={customModelOpen}
        onOpenChange={open => {
          setCustomModelOpen(open)
          if (!open) {
            if (customDownloadState === 'downloading') {
              setShowBackgroundDownload(true)
            } else {
              setCustomModelInput('')
              setCustomModelName('')
              setCustomModelDescription('')
              setCustomSelectedPromptSets([])
              setCustomDownloadState('idle')
              setCustomDownloadProgress(0)
              setCustomDownloadError('')
            }
          }
        }}
        promptSetNames={promptSetNames}
        customModelInput={customModelInput}
        setCustomModelInput={setCustomModelInput}
        customModelName={customModelName}
        setCustomModelName={setCustomModelName}
        customModelDescription={customModelDescription}
        setCustomModelDescription={setCustomModelDescription}
        customSelectedPromptSets={customSelectedPromptSets}
        setCustomSelectedPromptSets={setCustomSelectedPromptSets}
        customDownloadState={customDownloadState}
        customDownloadProgress={customDownloadProgress}
        customDownloadError={customDownloadError}
        downloadedBytes={downloadedBytes}
        totalBytes={totalBytes}
        estimatedTimeRemaining={estimatedTimeRemaining}
        onDownload={handleCustomModelDownload}
        onMoveToBackground={() => {
          setShowBackgroundDownload(true)
          setCustomModelOpen(false)
        }}
      />

      {/* Device model confirmation dialog */}
      <Dialog
        open={deviceConfirmOpen}
        onOpenChange={open => {
          setDeviceConfirmOpen(open)
          if (!open) {
            setDeviceSubmitState('idle')
            setPendingDeviceModel(null)
            setDeviceModelName('')
            setDeviceModelDescription('')
            setDeviceSelectedPromptSets([])
          }
        }}
      >
        <DialogContent>
          <DialogTitle>Use this model?</DialogTitle>
          <DialogDescription>
            {pendingDeviceModel ? (
              <div className="mt-2 flex flex-col gap-3">
                <p className="text-sm">
                  You are about to add
                  <span className="mx-1 font-medium text-foreground">
                    {pendingDeviceModel.name}
                  </span>
                  to your project.
                </p>

                <div>
                  <label
                    className="text-xs text-muted-foreground"
                    htmlFor="device-model-name"
                  >
                    Name
                  </label>
                  <input
                    id="device-model-name"
                    type="text"
                    placeholder="Enter model name"
                    value={deviceModelName}
                    onChange={e => setDeviceModelName(e.target.value)}
                    className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                  />
                </div>

                <div>
                  <label
                    className="text-xs text-muted-foreground"
                    htmlFor="device-model-description"
                  >
                    Description
                  </label>
                  <textarea
                    id="device-model-description"
                    rows={2}
                    placeholder="Enter model description"
                    value={deviceModelDescription}
                    onChange={e => setDeviceModelDescription(e.target.value)}
                    className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                  />
                </div>

                <PromptSetSelector
                  promptSetNames={promptSetNames}
                  selectedPromptSets={deviceSelectedPromptSets}
                  onTogglePromptSet={(name, checked) => {
                    if (checked) {
                      setDeviceSelectedPromptSets(prev => [...prev, name])
                    } else {
                      setDeviceSelectedPromptSets(prev =>
                        prev.filter(s => s !== name)
                      )
                    }
                  }}
                  onClearPromptSets={() => setDeviceSelectedPromptSets([])}
                  triggerId="device-prompt-sets-trigger"
                  label="Prompt sets"
                />

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-muted-foreground">Provider</div>
                  <div>Ollama</div>
                  <div className="text-muted-foreground">Source</div>
                  <div>Disk</div>
                </div>
              </div>
            ) : null}
          </DialogDescription>
          <DialogFooter>
            <Button
              variant="secondary"
              onClick={() => setDeviceConfirmOpen(false)}
            >
              Cancel
            </Button>
            <Button
              disabled={
                deviceSubmitState === 'loading' || !deviceModelName.trim()
              }
              onClick={() => {
                if (!pendingDeviceModel) return
                onAddModel(
                  {
                    id: `disk-${pendingDeviceModel.id}`,
                    name: deviceModelName.trim(),
                    modelIdentifier: pendingDeviceModel.modelIdentifier,
                    meta: deviceModelDescription.trim() || 'Model from disk',
                    badges: ['Local', 'Disk'],
                    status: 'ready',
                  },
                  deviceSelectedPromptSets.length > 0
                    ? deviceSelectedPromptSets
                    : undefined
                )
                setDeviceSubmitState('loading')
                setTimeout(() => {
                  setDeviceSubmitState('success')
                  setTimeout(() => {
                    setDeviceConfirmOpen(false)
                    onGoToProject()
                    setDeviceSubmitState('idle')
                  }, 600)
                }, 1000)
              }}
            >
              {deviceSubmitState === 'loading' && (
                <span className="mr-2 inline-flex">
                  <Loader
                    size={14}
                    className="border-blue-400 dark:border-blue-100"
                  />
                </span>
              )}
              {deviceSubmitState === 'success' && (
                <span className="mr-2 inline-flex">
                  <FontIcon type="checkmark-filled" className="w-4 h-4" />
                </span>
              )}
              Add to project
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete device model confirmation dialog */}
      <DeleteDeviceModelDialog
        open={deleteConfirmModelOpen}
        onOpenChange={open => {
          setDeleteConfirmModelOpen(open)
          if (!open && deleteState !== 'deleting') {
            setModelToDelete(null)
            setDeleteState('idle')
            setDeleteError('')
          }
        }}
        model={modelToDelete}
        deleteState={deleteState}
        deleteError={deleteError}
        onConfirmDelete={confirmDeleteDeviceModel}
      />

      {/* Model download dialog */}
      <ModelDownloadDialog
        open={downloadDialogOpen}
        onOpenChange={setDownloadDialogOpen}
        modelName={selectedProviderInfo?.modelId || ''}
        progress={modelDownload.progress}
        error={modelDownload.error}
        isDownloading={modelDownload.isDownloading}
        onCancel={() => {
          modelDownload.reset()
          setDownloadDialogOpen(false)
          setPendingVariantForProvider(null)
        }}
        onComplete={() => {
          modelDownload.reset()
        }}
      />

      {/* Cloud model configuration dialog */}
      <CloudModelConfigDialog
        open={cloudConfigDialogOpen}
        onOpenChange={setCloudConfigDialogOpen}
        modelName={selectedProviderInfo?.modelId || ''}
        provider={selectedProviderInfo?.runtime || ''}
        defaultBaseUrl={selectedProviderInfo?.baseUrl || ''}
        onConfigure={handleCloudModelConfigure}
        onCancel={() => {
          setCloudConfigDialogOpen(false)
          setPendingVariantForProvider(null)
        }}
      />

      {/* Provider selection dialog */}
      {pendingVariantForProvider && (
        <ProviderSelectionDialog
          open={providerDialogOpen}
          onOpenChange={setProviderDialogOpen}
          variantName={pendingVariantForProvider.label}
          parameters={pendingVariantForProvider.parameterSize}
          downloadSize={pendingVariantForProvider.downloadSize}
          providers={getVariantProviders(
            pendingVariantForProvider.label.replace(',', ':')
          )}
          onSelectProvider={handleProviderSelect}
        />
      )}

      {/* Download model confirmation dialog */}
      <Dialog
        open={confirmOpen}
        onOpenChange={open => {
          setConfirmOpen(open)
          if (!open) {
            setSubmitState('idle')
            setPendingVariant(null)
            setSelectedProviderInfo(null)
            setModelName('')
            setModelDescription('')
            setSelectedPromptSets([])
          }
        }}
      >
        <DialogContent>
          <DialogTitle>
            {selectedProviderInfo?.runtime === 'openai'
              ? 'Configure and add this model?'
              : 'Download and add this model?'}
          </DialogTitle>
          <DialogDescription>
            {pendingVariant ? (
              <div className="mt-2 flex flex-col gap-3">
                <p className="text-sm">
                  {selectedProviderInfo?.runtime === 'openai'
                    ? 'You are about to configure and add'
                    : 'You are about to download and add'}
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

                <PromptSetSelector
                  promptSetNames={promptSetNames}
                  selectedPromptSets={selectedPromptSets}
                  onTogglePromptSet={(name, checked) => {
                    if (checked) {
                      setSelectedPromptSets(prev => [...prev, name])
                    } else {
                      setSelectedPromptSets(prev =>
                        prev.filter(s => s !== name)
                      )
                    }
                  }}
                  onClearPromptSets={() => setSelectedPromptSets([])}
                  triggerId="prompt-sets-trigger"
                  label="Prompt sets"
                />

                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-muted-foreground">Provider</div>
                  <div className="capitalize">
                    {selectedProviderInfo?.runtime || 'Universal'}
                    {selectedProviderInfo?.format && (
                      <span className="ml-1 text-muted-foreground">
                        ({selectedProviderInfo.format.toUpperCase()})
                      </span>
                    )}
                  </div>
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
                    meta:
                      modelDescription.trim() ||
                      (selectedProviderInfo?.runtime === 'openai'
                        ? 'Cloud API model'
                        : 'Downloading…'),
                    badges:
                      selectedProviderInfo?.runtime === 'openai'
                        ? ['Cloud', 'OpenAI']
                        : ['Local', 'Ollama'],
                    status:
                      selectedProviderInfo?.runtime === 'openai' ? 'ready' : 'downloading',
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
              {selectedProviderInfo?.runtime === 'openai' ? 'Add to project' : 'Download and add'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
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

  // Background download state (shared across component)
  const [showBackgroundDownload, setShowBackgroundDownload] = useState(false)
  const [backgroundDownloadName, setBackgroundDownloadName] = useState('')
  const [customDownloadState, setCustomDownloadState] = useState<
    'idle' | 'downloading' | 'success' | 'error'
  >('idle')
  const [customDownloadProgress, setCustomDownloadProgress] = useState(0)
  const [customModelOpen, setCustomModelOpen] = useState(false)
  const [downloadedBytes, setDownloadedBytes] = useState(0)
  const [totalBytes, setTotalBytes] = useState(0)
  const [estimatedTimeRemaining, setEstimatedTimeRemaining] = useState('')

  // Cloud model configuration state
  const [selectedProviderInfo, setSelectedProviderInfo] =
    useState<ProviderInfo | null>(null)
  const [cloudConfig, setCloudConfig] = useState<{
    apiKey: string
    baseUrl: string
  } | null>(null)

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
        ? provider === 'ollama' ||
          provider === 'lemonade' ||
          provider === 'universal'
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

    // Determine provider, base_url, and api_key from selected provider info or defaults
    const provider = selectedProviderInfo?.runtime || 'ollama'
    const baseUrl =
      cloudConfig?.baseUrl ||
      selectedProviderInfo?.baseUrl ||
      (provider === 'ollama' ? 'http://localhost:11434' : undefined)
    const apiKey = cloudConfig?.apiKey

    const newModel: any = {
      name: m.name,
      description: m.meta === 'Downloading…' ? '' : m.meta,
      provider: provider,
      model: m.modelIdentifier || m.name,
      prompt_format: 'unstructured',
      prompts: promptSets || [],
    }

    // Add base_url if present
    if (baseUrl) {
      newModel.base_url = baseUrl
    }

    // Add api_key if present (for cloud providers)
    if (apiKey) {
      newModel.api_key = apiKey
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

      // Clear cloud config and provider info after successful save
      setCloudConfig(null)
      setSelectedProviderInfo(null)
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
              customModelOpen={customModelOpen}
              setCustomModelOpen={setCustomModelOpen}
              customDownloadState={customDownloadState}
              setCustomDownloadState={setCustomDownloadState}
              customDownloadProgress={customDownloadProgress}
              setCustomDownloadProgress={setCustomDownloadProgress}
              setShowBackgroundDownload={setShowBackgroundDownload}
              setBackgroundDownloadName={setBackgroundDownloadName}
              projectModels={projectModels}
              downloadedBytes={downloadedBytes}
              setDownloadedBytes={setDownloadedBytes}
              totalBytes={totalBytes}
              setTotalBytes={setTotalBytes}
              estimatedTimeRemaining={estimatedTimeRemaining}
              setEstimatedTimeRemaining={setEstimatedTimeRemaining}
              selectedProviderInfo={selectedProviderInfo}
              setSelectedProviderInfo={setSelectedProviderInfo}
              setCloudConfig={setCloudConfig}
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

      {/* Background download indicator */}
      {showBackgroundDownload && customDownloadState === 'downloading' && (
        <div className="fixed bottom-4 right-4 z-50 w-80 rounded-lg border border-border bg-card shadow-lg p-4 flex flex-col gap-2">
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="text-sm font-medium">
                Downloading {backgroundDownloadName}
              </div>
              <div className="text-xs text-muted-foreground">
                {formatBytes(downloadedBytes)} / {formatBytes(totalBytes)}{' '}
                {estimatedTimeRemaining && `• ${estimatedTimeRemaining} left`}
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
                {customDownloadProgress}%
              </span>
            </div>
            <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
              <div
                className="h-full bg-primary transition-all duration-300"
                style={{ width: `${customDownloadProgress}%` }}
              />
            </div>
          </div>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              setCustomModelOpen(true)
              setShowBackgroundDownload(false)
            }}
            className="w-full"
          >
            Show details
          </Button>
        </div>
      )}

      {/* Background download success notification */}
      {showBackgroundDownload && customDownloadState === 'success' && (
        <div className="fixed bottom-4 right-4 z-50 w-80 rounded-lg border border-border bg-card shadow-lg p-4 flex items-start gap-3">
          <div className="flex-shrink-0">
            <FontIcon
              type="checkmark-filled"
              className="w-5 h-5 text-primary"
            />
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium">Download complete</div>
            <div className="text-xs text-muted-foreground">
              {backgroundDownloadName} is ready to use
            </div>
          </div>
          <button
            onClick={() => setShowBackgroundDownload(false)}
            className="text-muted-foreground hover:text-foreground"
          >
            <FontIcon type="close" className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  )
}

export default Models
