import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
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
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject, useUpdateProject } from '../../hooks/useProjects'
import { parsePromptSets } from '../../utils/promptSets'
import { useCachedModels } from '../../hooks/useModels'
import modelService from '../../api/modelService'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import { PromptSetSelector } from './PromptSetSelector'
import { ModelSelector } from './ModelSelector'
import { DeviceModelsSection, type DeviceModel } from './DeviceModelsSection'
import { CustomDownloadDialog } from './CustomDownloadDialog'
import { DeleteDeviceModelDialog } from './DeleteDeviceModelDialog'
import { DiskSpaceWarningDialog } from './DiskSpaceWarningDialog'
import { DiskSpaceErrorDialog } from './DiskSpaceErrorDialog'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'
import { useToast } from '../ui/toast'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip'
import {
  sanitizeModelName,
  formatBytes,
  formatETA,
  validateModelName,
} from '../../utils/modelUtils'
import {
  recommendedQuantizations,
  localGroups,
  type LocalModelGroup,
  type ModelVariant,
} from './modelConstants'
import type { InferenceModel, ModelStatus } from './types'
import { CloudModelsForm } from './CloudModelsForm'
import TrainedModels from './TrainedModels'

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

interface ModelCardProps {
  model: InferenceModel
  onMakeDefault?: () => void
  onDelete?: () => void
  onRename?: (newName: string) => void
  promptSetNames: string[]
  selectedPromptSets: string[]
  onTogglePromptSet: (name: string, checked: boolean | string) => void
  onClearPromptSets: () => void
  availableProjectModels: Array<{ identifier: string; name: string }>
  availableDeviceModels: Array<{ identifier: string; name: string }>
  onModelChange: (newModelIdentifier: string) => void
}

function RenameModelModal({
  open,
  onOpenChange,
  currentName,
  existingNames,
  onRename,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  currentName: string
  existingNames: string[]
  onRename: (newName: string) => void
}) {
  const [editedName, setEditedName] = useState(currentName)
  const [nameError, setNameError] = useState<string>('')

  useEffect(() => {
    if (open) {
      setEditedName(currentName)
      setNameError('')
    }
  }, [open, currentName])

  const handleSave = () => {
    const validation = validateModelName(
      editedName.trim(),
      existingNames,
      currentName
    )
    if (!validation.isValid) {
      setNameError(validation.error || 'Invalid model name')
      return
    }

    if (editedName.trim() !== currentName) {
      onRename(editedName.trim())
    }
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Rename model</DialogTitle>
        </DialogHeader>
        <div className="flex flex-col gap-3 pt-1">
          <div>
            <label className="text-xs text-muted-foreground">Model name</label>
            <input
              className={`w-full mt-1 bg-transparent rounded-lg py-2 px-3 border text-foreground ${
                nameError ? 'border-destructive' : 'border-input'
              }`}
              placeholder="Enter model name"
              value={editedName}
              onChange={e => {
                const sanitized = sanitizeModelName(e.target.value)
                setEditedName(sanitized)
                if (nameError) {
                  setNameError('')
                }
              }}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  handleSave()
                }
              }}
              autoFocus
            />
            {nameError && (
              <p className="text-xs text-destructive mt-1">{nameError}</p>
            )}
            <p className="text-xs text-muted-foreground mt-1">
              Only letters, numbers, underscores (_), and hyphens (-) allowed.
              No spaces.
            </p>
          </div>
        </div>
        <DialogFooter>
          <Button variant="secondary" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button
            onClick={handleSave}
            disabled={!editedName.trim() || editedName.trim() === currentName}
          >
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ModelCard({
  model,
  onMakeDefault,
  onDelete,
  onRename,
  promptSetNames,
  selectedPromptSets,
  onTogglePromptSet,
  onClearPromptSets,
  availableProjectModels,
  availableDeviceModels,
  onModelChange,
  existingModelNames = [],
}: ModelCardProps & { existingModelNames?: string[] }) {
  const [isRenameModalOpen, setIsRenameModalOpen] = useState(false)

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

      <div className="grid grid-cols-1 md:grid-cols-2 md:items-center gap-6 w-full">
        <div className="pr-4">
          <div className="text-sm text-muted-foreground mb-2">
            {model.modelIdentifier || model.name}
          </div>

          <div className="flex items-center gap-2 mb-2">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5 group">
                <div className="text-lg font-medium min-h-[28px] flex items-center">
                  {model.name}
                </div>
                {onRename && (
                  <button
                    onClick={() => setIsRenameModalOpen(true)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-1 hover:bg-accent/30 rounded"
                    aria-label="Rename model"
                  >
                    <FontIcon
                      type="edit"
                      className="w-3.5 h-3.5 text-muted-foreground"
                    />
                  </button>
                )}
              </div>
            </div>
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
        {/* Model selector and Prompt sets multi-select column */}
        <div className="mt-3 md:mt-0 md:justify-self-end w-full md:pl-4 mr-6 md:mr-8 flex flex-col gap-3">
          <ModelSelector
            currentModelIdentifier={model.modelIdentifier || model.name}
            availableProjectModels={availableProjectModels.map(m => ({
              ...m,
              source: 'project' as const,
            }))}
            availableDeviceModels={availableDeviceModels.map(m => ({
              ...m,
              source: 'disk' as const,
            }))}
            onModelChange={onModelChange}
            label="Model"
          />
          <PromptSetSelector
            promptSetNames={promptSetNames}
            selectedPromptSets={selectedPromptSets}
            onTogglePromptSet={onTogglePromptSet}
            onClearPromptSets={onClearPromptSets}
            label="Prompt sets"
          />
        </div>
      </div>

      {onRename && (
        <RenameModelModal
          open={isRenameModalOpen}
          onOpenChange={setIsRenameModalOpen}
          currentName={model.name}
          existingNames={existingModelNames}
          onRename={onRename}
        />
      )}
    </div>
  )
}

function ProjectInferenceModels({
  models,
  onMakeDefault,
  onDelete,
  onRename,
  getSelected,
  promptSetNames,
  onToggle,
  onClear,
  availableProjectModels,
  availableDeviceModels,
  onModelChange,
}: {
  models: InferenceModel[]
  onMakeDefault: (id: string) => void
  onDelete: (id: string) => void
  onRename: (id: string, newName: string) => void
  getSelected: (id: string) => string[]
  promptSetNames: string[]
  onToggle: (id: string, name: string, checked: boolean | string) => void
  onClear: (id: string) => void
  availableProjectModels: Array<{ identifier: string; name: string }>
  availableDeviceModels: Array<{ identifier: string; name: string }>
  onModelChange: (modelId: string, newModelIdentifier: string) => void
}) {
  const existingNames = models.map(m => m.name)

  return (
    <div className="grid grid-cols-1 md:grid-cols-1 gap-2 mb-6 pb-8">
      {models.map((m, index) => (
        <ModelCard
          key={`${m.modelIdentifier}-${index}`}
          model={m}
          onMakeDefault={() => onMakeDefault(m.id)}
          onDelete={() => onDelete(m.id)}
          onRename={newName => onRename(m.id, newName)}
          promptSetNames={promptSetNames}
          selectedPromptSets={getSelected(m.id)}
          onTogglePromptSet={(name, checked) => onToggle(m.id, name, checked)}
          onClearPromptSets={() => onClear(m.id)}
          availableProjectModels={availableProjectModels}
          availableDeviceModels={availableDeviceModels}
          onModelChange={newModelIdentifier =>
            onModelChange(m.id, newModelIdentifier)
          }
          existingModelNames={existingNames}
        />
      ))}
    </div>
  )
}

// Constants and helpers for quantization selection
const QUANTIZATION_FALLBACK_ORDER = [
  'Q4_K_M',
  'Q4_K_S',
  'Q3_K_M',
  'Q3_K_S',
  'Q2_K',
] as const

/**
 * Finds a fallback quantization from available options
 */
function findFallbackQuantization(
  validOptions: Array<{ quantization: string | null }>,
  diskSpaceValidations?: Record<
    string,
    { can_download: boolean; warning: boolean }
  >
): string | null {
  for (const fallbackQuant of QUANTIZATION_FALLBACK_ORDER) {
    const fallbackOption = validOptions.find(
      opt => opt.quantization === fallbackQuant
    )
    if (fallbackOption) {
      // If disk space validations provided, check if it fits
      if (diskSpaceValidations) {
        const validation = diskSpaceValidations[fallbackQuant]
        if (validation && validation.can_download && !validation.warning) {
          return fallbackQuant
        }
      } else {
        return fallbackQuant
      }
    }
  }
  return null
}

/**
 * Determines the recommended quantization for a model
 */
function getRecommendedQuantization(
  baseModelId: string,
  validOptions: Array<{ quantization: string | null }>,
  diskSpaceValidations: Record<
    string,
    { can_download: boolean; warning: boolean }
  >,
  recommendedQuantizations: Record<
    string,
    { quantization: string; description: string }
  >
): { quantization: string | null; description: string | null } {
  const recommendation = recommendedQuantizations[baseModelId]
  if (!recommendation) {
    return { quantization: null, description: null }
  }

  const recommendedOption = validOptions.find(
    opt => opt.quantization === recommendation.quantization
  )
  const recommendedValidation =
    recommendedOption && diskSpaceValidations[recommendation.quantization]

  if (
    recommendedOption &&
    (!recommendedValidation ||
      (recommendedValidation.can_download && !recommendedValidation.warning))
  ) {
    return {
      quantization: recommendation.quantization,
      description: recommendation.description,
    }
  }

  // Recommended option doesn't fit, find next best that fits
  const fallbackQuant = findFallbackQuantization(
    validOptions,
    diskSpaceValidations
  )
  if (fallbackQuant) {
    return {
      quantization: fallbackQuant,
      description: null, // Don't show description for fallback - it would be misleading
    }
  }

  return { quantization: null, description: null }
}

export function AddOrChangeModels({
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
}) {
  const [sourceTab, setSourceTab] = useState<'local' | 'cloud'>('local')
  const [query, setQuery] = useState('')
  const [confirmOpen, setConfirmOpen] = useState(false)
  const [selectedModelGroup, setSelectedModelGroup] =
    useState<LocalModelGroup | null>(null)
  const [pendingVariant, setPendingVariant] = useState<ModelVariant | null>(
    null
  )
  const [submitState, setSubmitState] = useState<
    'idle' | 'loading' | 'success' | 'error'
  >('idle')
  const [modelName, setModelName] = useState('')
  const [modelDescription, setModelDescription] = useState('')
  const [selectedPromptSets, setSelectedPromptSets] = useState<string[]>([])
  // GGUF options state
  const [ggufOptions, setGgufOptions] = useState<
    Array<{
      filename: string
      quantization: string | null
      size_bytes: number
      size_human: string
    }>
  >([])
  const [isLoadingGgufOptions, setIsLoadingGgufOptions] = useState(false)
  const [selectedQuantization, setSelectedQuantization] = useState<
    string | null
  >(null)
  const optionsScrollRef = useRef<HTMLDivElement>(null)
  const [diskSpaceValidations, setDiskSpaceValidations] = useState<
    Record<string, { can_download: boolean; warning: boolean }>
  >({})
  const [downloadProgress, setDownloadProgress] = useState(0)
  // Track quantization counts and size ranges for each model
  const [modelMetadata, setModelMetadata] = useState<
    Record<string, { count: number; minSize: string; maxSize: string } | null>
  >({})
  const [downloadError, setDownloadError] = useState('')
  const [
    showRecommendedBackgroundDownload,
    setShowRecommendedBackgroundDownload,
  ] = useState(false)
  const [modelNameError, setModelNameError] = useState<string>('')
  const [deviceModelNameError, setDeviceModelNameError] = useState<string>('')

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

  // Fetch GGUF options when dialog opens with a selected model group
  useEffect(() => {
    // Create AbortController for this effect run
    const abortController = new AbortController()
    const signal = abortController.signal

    // Capture the current model group ID to check against in callbacks
    const currentModelGroupId = selectedModelGroup?.baseModelId

    if (confirmOpen && currentModelGroupId) {
      setIsLoadingGgufOptions(true)
      setGgufOptions([])
      setSelectedQuantization(selectedModelGroup.defaultQuantization)

      modelService
        .getGGUFOptions(currentModelGroupId, signal)
        .then(data => {
          // Guard: Check if request was aborted or model group changed
          if (
            signal.aborted ||
            selectedModelGroup?.baseModelId !== currentModelGroupId
          ) {
            return
          }

          if (data && data.options && data.options.length > 0) {
            const validOptions = data.options.filter(opt => opt.quantization)
            setGgufOptions(data.options)

            // Determine recommended quantization (will be set after disk space validation)
            const recommendation = recommendedQuantizations[currentModelGroupId]
            let initialQuantization: string | null = null

            if (recommendation) {
              // Check if recommended option exists
              const recommendedOption = validOptions.find(
                opt => opt.quantization === recommendation.quantization
              )
              if (recommendedOption) {
                initialQuantization = recommendation.quantization
              } else {
                // Try fallback order
                initialQuantization = findFallbackQuantization(validOptions)
              }
            }

            // Fallback to defaultQuantization or first option
            if (!initialQuantization) {
              const defaultOption = validOptions.find(
                opt =>
                  opt.quantization === selectedModelGroup.defaultQuantization
              )
              if (defaultOption) {
                initialQuantization = selectedModelGroup.defaultQuantization
              } else {
                initialQuantization = validOptions[0]?.quantization || null
              }
            }

            setSelectedQuantization(initialQuantization)

            // Validate disk space for each option
            validOptions.forEach(option => {
              // Guard: Check if request was aborted or model group changed before each validation
              if (
                signal.aborted ||
                selectedModelGroup?.baseModelId !== currentModelGroupId
              ) {
                return
              }

              const modelIdentifier = `${currentModelGroupId}:${option.quantization}`

              // Validate asynchronously - don't block UI
              modelService
                .validateModelDownload(modelIdentifier, signal)
                .then(
                  (validation: { can_download: boolean; warning: boolean }) => {
                    // Guard: Check if request was aborted or model group changed
                    if (
                      signal.aborted ||
                      selectedModelGroup?.baseModelId !== currentModelGroupId
                    ) {
                      return
                    }

                    setDiskSpaceValidations(prev => {
                      const updated = {
                        ...prev,
                        [option.quantization!]: {
                          can_download: validation.can_download,
                          warning: validation.warning,
                        },
                      }

                      // After validation, check if we should update selection
                      // If recommended doesn't fit, switch to a fallback
                      const recommendation =
                        recommendedQuantizations[currentModelGroupId]
                      if (recommendation) {
                        const recommendedValidation =
                          updated[recommendation.quantization]

                        // If recommended is validated and doesn't fit, find a fallback
                        if (
                          option.quantization === recommendation.quantization &&
                          recommendedValidation &&
                          (!recommendedValidation.can_download ||
                            recommendedValidation.warning)
                        ) {
                          // Find a fallback that fits
                          const fallbackQuant = findFallbackQuantization(
                            validOptions,
                            updated
                          )
                          if (fallbackQuant) {
                            setSelectedQuantization(fallbackQuant)
                          }
                        }
                        // If recommended fits and we haven't selected it yet, select it
                        else if (
                          option.quantization === recommendation.quantization &&
                          recommendedValidation &&
                          recommendedValidation.can_download &&
                          !recommendedValidation.warning
                        ) {
                          setSelectedQuantization(recommendation.quantization)
                        }
                      }

                      return updated
                    })
                  }
                )
                .catch((err: unknown) => {
                  // Ignore abort errors - they're expected when cleaning up
                  if (
                    signal.aborted ||
                    (err as any)?.name === 'AbortError' ||
                    (err as any)?.code === 'ERR_CANCELED'
                  ) {
                    return
                  }

                  // Guard: Check if model group changed
                  if (selectedModelGroup?.baseModelId !== currentModelGroupId) {
                    return
                  }

                  console.error(
                    `Error validating disk space for ${modelIdentifier}:`,
                    err
                  )
                  // On error, assume it's okay (graceful degradation)
                  setDiskSpaceValidations(prev => ({
                    ...prev,
                    [option.quantization!]: {
                      can_download: true,
                      warning: false,
                    },
                  }))
                })
            })
          }
        })
        .catch(err => {
          // Ignore abort errors - they're expected when cleaning up
          if (
            signal.aborted ||
            (err as any)?.name === 'AbortError' ||
            (err as any)?.code === 'ERR_CANCELED'
          ) {
            return
          }

          // Guard: Check if model group changed
          if (selectedModelGroup?.baseModelId !== currentModelGroupId) {
            return
          }

          console.error('Error loading GGUF options:', err)
          // Don't show error, just continue without options
          setGgufOptions([])
        })
        .finally(() => {
          // Guard: Only update loading state if this is still the current request
          if (
            !signal.aborted &&
            selectedModelGroup?.baseModelId === currentModelGroupId
          ) {
            setIsLoadingGgufOptions(false)
            // Scroll to selected option after options load
            setTimeout(() => {
              if (optionsScrollRef.current) {
                const selectedButton = optionsScrollRef.current.querySelector(
                  '[data-selected="true"]'
                ) as HTMLElement
                if (selectedButton) {
                  selectedButton.scrollIntoView({
                    behavior: 'smooth',
                    block: 'center',
                  })
                }
              }
            }, 100)
          }
        })
    } else if (!confirmOpen) {
      // Reset when dialog closes
      setGgufOptions([])
      setSelectedQuantization(null)
      setIsLoadingGgufOptions(false)
      setDiskSpaceValidations({})
    }

    // Cleanup: Cancel all in-flight requests when effect re-runs or unmounts
    return () => {
      abortController.abort()
    }
  }, [confirmOpen, selectedModelGroup])

  // Scroll to selected option when selection changes
  useEffect(() => {
    if (optionsScrollRef.current && selectedQuantization) {
      const selectedButton = optionsScrollRef.current.querySelector(
        '[data-selected="true"]'
      ) as HTMLElement
      if (selectedButton) {
        selectedButton.scrollIntoView({
          behavior: 'smooth',
          block: 'center',
        })
      }
    }
  }, [selectedQuantization])

  // Fetch quantization counts and size ranges for all recommended models
  useEffect(() => {
    const fetchModelMetadata = async () => {
      const metadata: Record<
        string,
        { count: number; minSize: string; maxSize: string }
      > = {}

      await Promise.all(
        localGroups.map(async group => {
          try {
            const data = await modelService.getGGUFOptions(group.baseModelId)
            if (data?.options && data.options.length > 0) {
              const validOptions = data.options.filter(opt => opt.quantization)
              if (validOptions.length > 0) {
                const sizes = validOptions.map(opt => opt.size_bytes)
                const minSize = Math.min(...sizes)
                const maxSize = Math.max(...sizes)
                metadata[group.baseModelId] = {
                  count: validOptions.length,
                  minSize: formatBytes(minSize),
                  maxSize: formatBytes(maxSize),
                }
              }
            }
          } catch (err) {
            // Silently fail - metadata is optional
            console.debug(
              `Could not fetch metadata for ${group.baseModelId}:`,
              err
            )
          }
        })
      )

      setModelMetadata(metadata)
    }

    fetchModelMetadata()
  }, []) // Only run once on mount

  // Custom model local state (not shared)
  const [customModelInput, setCustomModelInput] = useState('')
  const [customModelName, setCustomModelName] = useState('')
  const [customModelDescription, setCustomModelDescription] = useState('')
  const [customSelectedPromptSets, setCustomSelectedPromptSets] = useState<
    string[]
  >([])
  const [customDownloadError, setCustomDownloadError] = useState('')
  const [customModelNameError, setCustomModelNameError] = useState<string>('')

  // Disk space warning/error dialog state
  const [warningDialogOpen, setWarningDialogOpen] = useState(false)
  const [warningDialogMessage, setWarningDialogMessage] = useState('')
  const [warningDialogAvailableBytes, setWarningDialogAvailableBytes] =
    useState(0)
  const [warningDialogRequiredBytes, setWarningDialogRequiredBytes] =
    useState(0)
  const warningDialogResolveRef = useRef<(() => void) | null>(null)
  const warningDialogRejectRef = useRef<(() => void) | null>(null)
  const [errorDialogOpen, setErrorDialogOpen] = useState(false)
  const [errorDialogMessage, setErrorDialogMessage] = useState('')
  const [errorDialogAvailableBytes, setErrorDialogAvailableBytes] = useState(0)
  const [errorDialogRequiredBytes, setErrorDialogRequiredBytes] = useState(0)

  const filteredGroups = localGroups.filter(g =>
    [g.name, ...g.variants.map(v => v.modelIdentifier || v.label)].some(v =>
      v.toLowerCase().includes(query.toLowerCase())
    )
  )

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
      badges: ['Local'],
    })) || []

  const handleUseDeviceModel = (model: DeviceModel) => {
    setPendingDeviceModel(model)
    // Sanitize the model name to remove spaces and special characters
    const sanitized = sanitizeModelName(model.name)
    setDeviceModelName(sanitized)
    setDeviceModelNameError('')
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

  // Handle custom model download
  // Helper function to handle disk space warnings
  // Returns a promise that resolves if user continues, rejects if user cancels
  const handleDiskSpaceWarning = async (
    modelId: string,
    message: string
  ): Promise<void> => {
    return new Promise(async (resolve, reject) => {
      try {
        // Get full validation details for the dialog
        const validation = await modelService.validateModelDownload(modelId)
        setWarningDialogMessage(message)
        setWarningDialogAvailableBytes(validation.available_bytes || 0)
        setWarningDialogRequiredBytes(validation.required_bytes || 0)
        warningDialogResolveRef.current = () => {
          setWarningDialogOpen(false)
          warningDialogResolveRef.current = null
          warningDialogRejectRef.current = null
          resolve()
        }
        warningDialogRejectRef.current = () => {
          setWarningDialogOpen(false)
          warningDialogResolveRef.current = null
          warningDialogRejectRef.current = null
          reject(new Error('User cancelled download'))
        }
        setWarningDialogOpen(true)
      } catch (err) {
        // If validation fails, show dialog with message only
        setWarningDialogMessage(message)
        setWarningDialogAvailableBytes(0)
        setWarningDialogRequiredBytes(0)
        warningDialogResolveRef.current = () => {
          setWarningDialogOpen(false)
          warningDialogResolveRef.current = null
          warningDialogRejectRef.current = null
          resolve()
        }
        warningDialogRejectRef.current = () => {
          setWarningDialogOpen(false)
          warningDialogResolveRef.current = null
          warningDialogRejectRef.current = null
          reject(new Error('User cancelled download'))
        }
        setWarningDialogOpen(true)
      }
    })
  }

  // Handle warning dialog cancel
  const handleWarningDialogCancel = () => {
    if (warningDialogRejectRef.current) {
      warningDialogRejectRef.current()
    }
  }

  // Helper function to check if error is disk space related
  const isDiskSpaceError = (message: string): boolean => {
    const lowerMessage = message.toLowerCase()
    return (
      lowerMessage.includes('insufficient disk space') ||
      lowerMessage.includes('disk space') ||
      lowerMessage.includes('not enough space') ||
      lowerMessage.includes('free up space')
    )
  }

  // Helper function to handle disk space errors
  const handleDiskSpaceError = async (modelId: string, message: string) => {
    try {
      // Get full validation details for the dialog
      const validation = await modelService.validateModelDownload(modelId)
      setErrorDialogMessage(message)
      setErrorDialogAvailableBytes(validation.available_bytes || 0)
      setErrorDialogRequiredBytes(validation.required_bytes || 0)
      setErrorDialogOpen(true)
    } catch (err) {
      // If validation fails, show dialog with message only
      setErrorDialogMessage(message)
      setErrorDialogAvailableBytes(0)
      setErrorDialogRequiredBytes(0)
      setErrorDialogOpen(true)
    }
  }

  const handleCustomModelDownload = async () => {
    // Validate model name
    const existingNames = projectModels.map(m => m.name)
    const validation = validateModelName(customModelName.trim(), existingNames)
    if (!validation.isValid) {
      setCustomModelNameError(validation.error || 'Invalid model name')
      return
    }
    setCustomModelNameError('')

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
          if (event.event === 'warning') {
            // Show warning dialog and wait for user decision
            try {
              await Promise.race([
                handleDiskSpaceWarning(
                  customModelInput.trim(),
                  event.message || 'Low disk space warning'
                ),
                // Timeout after 5 minutes (user should have made a decision)
                new Promise<void>((_, timeoutReject) =>
                  setTimeout(() => timeoutReject(new Error('Timeout')), 300000)
                ),
              ])
              // User chose to continue, proceed with download
            } catch {
              // User cancelled or timeout
              setCustomDownloadState('idle')
              setCustomDownloadProgress(0)
              setShowBackgroundDownload(false)
              return
            }
          } else if (event.event === 'progress') {
            const d = Number(event.downloaded || 0)
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
                badges: ['Local'],
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
            const errorMessage =
              event.message ||
              'Failed to download model. Please check the model name and try again.'
            // Check if it's a disk space error
            if (isDiskSpaceError(errorMessage)) {
              await handleDiskSpaceError(customModelInput.trim(), errorMessage)
              setCustomDownloadState('error')
              setCustomDownloadError(errorMessage)
            } else {
              setCustomDownloadState('error')
              setCustomDownloadError(errorMessage)
            }
            setShowBackgroundDownload(false)
          }
        }
      } catch (error: any) {
        const errorMessage =
          error.message ||
          'Failed to download model. Please check the model name and try again.'
        // Check if it's a disk space error
        if (isDiskSpaceError(errorMessage)) {
          await handleDiskSpaceError(customModelInput.trim(), errorMessage)
          setCustomDownloadState('error')
          setCustomDownloadError(errorMessage)
        } else {
          setCustomDownloadState('error')
          setCustomDownloadError(errorMessage)
        }
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

      {/* Download recommended models section */}
      <div className="flex flex-col gap-4">
        <div>
          <h3 className="font-medium">Download recommended models</h3>
          <div className="h-1" />
          <div className="text-sm text-muted-foreground">
            Download and add recommended GGUF models to your project.
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

          {/* Search - only show for local models */}
          {sourceTab === 'local' && (
            <div className="flex items-center gap-2 w-full">
              <div className="relative flex-1">
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
                  setCustomModelNameError('')
                }}
                className="h-10 whitespace-nowrap"
              >
                Add HuggingFace model
              </Button>
            </div>
          )}

          {/* Table */}
          {sourceTab === 'local' && (
            <div className="w-full overflow-hidden rounded-lg border border-border">
              <div className="grid grid-cols-12 items-center bg-secondary text-secondary-foreground text-xs px-3 py-3">
                <div className="col-span-5">Model</div>
                <div className="col-span-6">Size Range</div>
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
                      setCustomModelNameError('')
                    }}
                  >
                    Add HuggingFace model
                  </Button>
                </div>
              ) : (
                filteredGroups.map(group => {
                  const baseModelName = group.baseModelId.replace(
                    'unsloth/',
                    ''
                  )
                  const metadata = modelMetadata[group.baseModelId]
                  const quantizationCount = metadata?.count ?? null

                  return (
                    <div
                      key={group.id}
                      className="grid grid-cols-12 items-center px-3 py-3 text-sm border-t border-border hover:bg-accent/40"
                    >
                      <div className="col-span-5 truncate">
                        <span className="font-bold text-foreground">
                          {baseModelName}
                        </span>
                        {quantizationCount !== null && (
                          <span className="text-muted-foreground/70 font-normal ml-2.5">
                            ({quantizationCount})
                          </span>
                        )}
                      </div>
                      <div className="col-span-6 text-muted-foreground text-xs">
                        {metadata ? (
                          <span>
                            {metadata.minSize === metadata.maxSize
                              ? metadata.minSize
                              : `${metadata.minSize} - ${metadata.maxSize}`}
                          </span>
                        ) : (
                          <span className="text-muted-foreground/50">—</span>
                        )}
                      </div>
                      <div className="col-span-1 flex items-center justify-end pr-2">
                        <Button
                          size="sm"
                          className="h-8 px-3"
                          onClick={() => {
                            setSelectedModelGroup(group)
                            // Prepopulate name from base model ID
                            const rawName =
                              group.baseModelId
                                .split('/')
                                .pop()
                                ?.replace(/-GGUF.*$/, '') || group.name
                            const sanitized = sanitizeModelName(rawName)
                            setModelName(sanitized)
                            setModelNameError('')
                            setSelectedQuantization(group.defaultQuantization)
                            setConfirmOpen(true)
                          }}
                        >
                          Add
                        </Button>
                      </div>
                    </div>
                  )
                })
              )}
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
              setCustomModelNameError('')
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
        customModelNameError={customModelNameError}
        onClearModelNameError={() => setCustomModelNameError('')}
        downloadedBytes={downloadedBytes}
        totalBytes={totalBytes}
        estimatedTimeRemaining={estimatedTimeRemaining}
        onDownload={handleCustomModelDownload}
        onMoveToBackground={() => {
          setShowBackgroundDownload(true)
          setCustomModelOpen(false)
        }}
      />

      {/* Disk space warning dialog */}
      <DiskSpaceWarningDialog
        open={warningDialogOpen}
        onOpenChange={open => {
          if (!open) {
            // If dialog closes and we still have a reject callback, user cancelled
            if (warningDialogRejectRef.current) {
              warningDialogRejectRef.current()
            }
            setWarningDialogOpen(false)
          }
        }}
        message={warningDialogMessage}
        availableBytes={warningDialogAvailableBytes}
        requiredBytes={warningDialogRequiredBytes}
        onContinue={() => {
          if (warningDialogResolveRef.current) {
            warningDialogResolveRef.current()
          }
        }}
        onCancel={handleWarningDialogCancel}
      />

      {/* Disk space error dialog */}
      <DiskSpaceErrorDialog
        open={errorDialogOpen}
        onOpenChange={setErrorDialogOpen}
        message={errorDialogMessage}
        availableBytes={errorDialogAvailableBytes}
        requiredBytes={errorDialogRequiredBytes}
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
                    onChange={e => {
                      const sanitized = sanitizeModelName(e.target.value)
                      setDeviceModelName(sanitized)
                      // Clear error when user types
                      if (deviceModelNameError) {
                        setDeviceModelNameError('')
                      }
                    }}
                    className={`w-full mt-1 bg-transparent rounded-lg py-2 px-3 border ${
                      deviceModelNameError
                        ? 'border-destructive'
                        : 'border-input'
                    } text-foreground`}
                  />
                  {deviceModelNameError && (
                    <div className="text-xs text-destructive mt-1">
                      {deviceModelNameError}
                    </div>
                  )}
                  <div className="text-xs text-muted-foreground mt-1">
                    Only letters, numbers, underscores (_), and hyphens (-) are
                    allowed. No spaces.
                  </div>
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

                // Validate model name
                const existingNames = projectModels.map(m => m.name)
                const validation = validateModelName(
                  deviceModelName.trim(),
                  existingNames
                )
                if (!validation.isValid) {
                  setDeviceModelNameError(
                    validation.error || 'Invalid model name'
                  )
                  return
                }
                setDeviceModelNameError('')

                onAddModel(
                  {
                    id: `disk-${pendingDeviceModel.id}`,
                    name: deviceModelName.trim(),
                    modelIdentifier: pendingDeviceModel.modelIdentifier,
                    meta: deviceModelDescription.trim() || 'Model from disk',
                    badges: ['Local'],
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

      {/* Download model confirmation dialog */}
      <Dialog
        open={confirmOpen}
        onOpenChange={open => {
          setConfirmOpen(open)
          // If closing while downloading, minimize to background
          if (!open && submitState === 'loading') {
            setShowRecommendedBackgroundDownload(true)
          }
          if (!open && submitState !== 'loading') {
            setSubmitState('idle')
            setPendingVariant(null)
            setModelName('')
            setModelDescription('')
            setSelectedPromptSets([])
            setDownloadProgress(0)
            setDownloadError('')
            setModelNameError('')
            setDownloadedBytes(0)
            setTotalBytes(0)
            setEstimatedTimeRemaining('')
            setSelectedModelGroup(null)
            setGgufOptions([])
            setSelectedQuantization(null)
          }
        }}
      >
        <DialogContent className="max-w-5xl">
          <DialogTitle>
            {selectedModelGroup
              ? selectedModelGroup.baseModelId.split('/').pop() ||
                'Download model'
              : pendingVariant
                ? 'Download and add this model?'
                : 'Download and add this model?'}
          </DialogTitle>
          <DialogDescription>
            {selectedModelGroup || pendingVariant ? (
              <>
                <div className="mt-2 grid grid-cols-2 gap-4">
                  {/* Left Column: Form Fields */}
                  <div className="flex flex-col gap-3">
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
                        onChange={e => {
                          const sanitized = sanitizeModelName(e.target.value)
                          setModelName(sanitized)
                          // Clear error when user types
                          if (modelNameError) {
                            setModelNameError('')
                          }
                        }}
                        className={`w-full mt-1 bg-transparent rounded-lg py-2 px-3 border ${
                          modelNameError ? 'border-destructive' : 'border-input'
                        } text-foreground`}
                      />
                      {modelNameError && (
                        <div className="text-xs text-destructive mt-1">
                          {modelNameError}
                        </div>
                      )}
                      <div className="text-xs text-muted-foreground mt-1">
                        Only letters, numbers, underscores (_), and hyphens (-)
                        are allowed. No spaces.
                      </div>
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
                      <div>Universal</div>
                      <div className="text-muted-foreground">Model</div>
                      <div className="truncate">
                        {selectedModelGroup && selectedQuantization
                          ? `${selectedModelGroup.baseModelId}:${selectedQuantization}`
                          : pendingVariant
                            ? pendingVariant.modelIdentifier
                            : ''}
                      </div>
                    </div>
                  </div>

                  {/* Right Column: GGUF Options */}
                  {selectedModelGroup && (
                    <div>
                      <label
                        className="text-xs text-muted-foreground mb-2 block"
                        htmlFor="quantization-select"
                      >
                        Choose download option
                        {(() => {
                          const validOptions = ggufOptions.filter(
                            opt => opt.quantization
                          )
                          return validOptions.length > 0 ? (
                            <span className="ml-2 text-muted-foreground/70">
                              ({validOptions.length} download options available)
                            </span>
                          ) : null
                        })()}
                      </label>
                      {isLoadingGgufOptions ? (
                        <div className="flex items-center justify-center py-8">
                          <Loader size={20} />
                        </div>
                      ) : (
                        (() => {
                          const validOptions = ggufOptions.filter(
                            opt => opt.quantization
                          )
                          return validOptions.length > 0 ? (
                            <div
                              ref={optionsScrollRef}
                              className="h-full max-h-[400px] overflow-y-auto space-y-2 border border-border rounded-lg p-2"
                            >
                              {(() => {
                                // Determine recommended quantization
                                const {
                                  quantization: recommendedQuantization,
                                  description: recommendationDescription,
                                } = getRecommendedQuantization(
                                  selectedModelGroup.baseModelId,
                                  validOptions,
                                  diskSpaceValidations,
                                  recommendedQuantizations
                                )

                                return validOptions.map((option, index) => {
                                  const isSelected =
                                    option.quantization === selectedQuantization
                                  const isRecommended =
                                    option.quantization ===
                                    recommendedQuantization
                                  return (
                                    <button
                                      key={`${option.quantization}-${index}`}
                                      type="button"
                                      data-selected={isSelected}
                                      onClick={() =>
                                        setSelectedQuantization(
                                          option.quantization
                                        )
                                      }
                                      className={`w-full flex flex-col gap-2 p-3 rounded-lg border transition-colors text-left ${
                                        isSelected
                                          ? 'bg-accent/80 border-primary'
                                          : 'border-border hover:bg-accent/50'
                                      }`}
                                    >
                                      <div className="flex items-center gap-3">
                                        <div className="flex-shrink-0">
                                          {isSelected ? (
                                            <FontIcon
                                              type="checkmark-filled"
                                              className="w-5 h-5 text-primary"
                                            />
                                          ) : (
                                            <div className="w-5 h-5 rounded-full border-2 border-muted-foreground" />
                                          )}
                                        </div>
                                        <div className="flex-1 min-w-0 flex items-center gap-3">
                                          <span className="text-sm font-medium px-3 py-1 rounded-md bg-primary/10 text-primary border border-primary/20">
                                            {option.quantization || 'Unknown'}
                                          </span>
                                          <span className="text-sm text-muted-foreground flex-1 truncate">
                                            {selectedModelGroup.name}
                                          </span>
                                          {isRecommended && (
                                            <span className="text-xs font-medium px-2 py-0.5 rounded-md bg-teal-500/20 dark:bg-teal-500/20 text-teal-600 dark:text-teal-400 border border-teal-500/40 dark:border-teal-500/30">
                                              Recommended
                                            </span>
                                          )}
                                          <div className="flex-shrink-0 flex items-center gap-2">
                                            {diskSpaceValidations[
                                              option.quantization!
                                            ] &&
                                              (!diskSpaceValidations[
                                                option.quantization!
                                              ].can_download ||
                                                diskSpaceValidations[
                                                  option.quantization!
                                                ].warning) && (
                                                <TooltipProvider>
                                                  <Tooltip>
                                                    <TooltipTrigger asChild>
                                                      <span className="cursor-help">
                                                        <FontIcon
                                                          type="alert-triangle"
                                                          className="w-4 h-4 text-amber-500"
                                                        />
                                                      </span>
                                                    </TooltipTrigger>
                                                    <TooltipContent>
                                                      <p className="text-sm">
                                                        {!diskSpaceValidations[
                                                          option.quantization!
                                                        ].can_download
                                                          ? 'Insufficient disk space: This model is too large for your available disk space.'
                                                          : 'Low disk space warning: Your disk space is running low. Consider freeing up space before downloading.'}
                                                      </p>
                                                    </TooltipContent>
                                                  </Tooltip>
                                                </TooltipProvider>
                                              )}
                                            <div className="text-sm font-medium text-foreground">
                                              {option.size_human}
                                            </div>
                                          </div>
                                        </div>
                                      </div>
                                      {isRecommended &&
                                        recommendationDescription && (
                                          <div className="ml-8 text-xs text-muted-foreground">
                                            {recommendationDescription}
                                          </div>
                                        )}
                                    </button>
                                  )
                                })
                              })()}
                            </div>
                          ) : (
                            <div className="text-sm text-muted-foreground py-4">
                              No download options available
                            </div>
                          )
                        })()
                      )}
                    </div>
                  )}
                </div>

                {/* Progress bar */}
                {submitState === 'loading' && (
                  <div className="flex flex-col gap-1 mt-3">
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">
                        Downloading... {formatBytes(downloadedBytes)} /{' '}
                        {formatBytes(totalBytes)}
                      </span>
                      <span className="text-muted-foreground">
                        {downloadProgress}%{' '}
                        {estimatedTimeRemaining &&
                          `• ${estimatedTimeRemaining}`}
                      </span>
                    </div>
                    <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
                      <div
                        className="h-full bg-primary transition-all duration-300"
                        style={{ width: `${downloadProgress}%` }}
                      />
                    </div>
                  </div>
                )}

                {/* Error message */}
                {submitState === 'error' && downloadError && (
                  <div className="p-3 rounded-md bg-destructive/10 border border-destructive/20 mt-3">
                    <p className="text-sm text-destructive">{downloadError}</p>
                  </div>
                )}
              </>
            ) : null}
          </DialogDescription>
          <DialogFooter>
            {submitState === 'loading' ? (
              <Button
                variant="secondary"
                onClick={() => {
                  setShowRecommendedBackgroundDownload(true)
                  setConfirmOpen(false)
                }}
              >
                Continue in background
              </Button>
            ) : (
              <Button variant="secondary" onClick={() => setConfirmOpen(false)}>
                Cancel
              </Button>
            )}
            <Button
              disabled={
                submitState === 'loading' ||
                !modelName.trim() ||
                (selectedModelGroup ? !selectedQuantization : false)
              }
              onClick={async () => {
                // Determine model identifier
                let modelIdentifier: string
                let variantId: number

                if (selectedModelGroup && selectedQuantization) {
                  modelIdentifier = `${selectedModelGroup.baseModelId}:${selectedQuantization}`
                  variantId = selectedModelGroup.id * 1000
                } else if (pendingVariant) {
                  modelIdentifier = pendingVariant.modelIdentifier
                  variantId = pendingVariant.id
                } else {
                  return
                }

                // Validate model name
                const existingNames = projectModels.map(m => m.name)
                const validation = validateModelName(
                  modelName.trim(),
                  existingNames
                )
                if (!validation.isValid) {
                  setModelNameError(validation.error || 'Invalid model name')
                  return
                }
                setModelNameError('')

                setSubmitState('loading')
                setDownloadProgress(5)
                setDownloadError('')
                setDownloadedBytes(0)
                setTotalBytes(0)
                setEstimatedTimeRemaining('')
                const start = Date.now()

                // Show download and add a placeholder card with user-entered data
                onAddModel(
                  {
                    id: `dl-${variantId}`,
                    name: modelName.trim(),
                    modelIdentifier: modelIdentifier,
                    meta: modelDescription.trim() || 'Downloading…',
                    badges: ['Local'],
                    status: 'downloading',
                  },
                  selectedPromptSets.length > 0 ? selectedPromptSets : undefined
                )

                const downloadAsync = async () => {
                  try {
                    for await (const event of modelService.downloadModel({
                      model_name: modelIdentifier,
                      provider: 'universal',
                    })) {
                      if (event.event === 'warning') {
                        // Show warning dialog and wait for user decision
                        try {
                          await Promise.race([
                            handleDiskSpaceWarning(
                              modelIdentifier,
                              event.message || 'Low disk space warning'
                            ),
                            // Timeout after 5 minutes (user should have made a decision)
                            new Promise<void>((_, timeoutReject) =>
                              setTimeout(
                                () => timeoutReject(new Error('Timeout')),
                                300000
                              )
                            ),
                          ])
                          // User chose to continue, proceed with download
                        } catch {
                          // User cancelled or timeout
                          setSubmitState('idle')
                          setDownloadProgress(0)
                          return
                        }
                      } else if (event.event === 'progress') {
                        const d = Number(event.downloaded || 0)
                        const t = Number(event.total || 0)
                        setDownloadedBytes(d)
                        setTotalBytes(t)
                        if (t > 0 && isFinite(d) && d >= 0) {
                          const percent = Math.max(
                            5,
                            Math.min(95, Math.round((d / t) * 90) + 5)
                          )
                          setDownloadProgress(percent)
                          const elapsedSec = (Date.now() - start) / 1000
                          if (elapsedSec > 0) {
                            const speed = d / elapsedSec
                            const remain = (t - d) / (speed || 1)
                            setEstimatedTimeRemaining(formatETA(remain))
                          }
                        }
                      } else if (event.event === 'done') {
                        setDownloadProgress(100)
                        setSubmitState('success')
                        setEstimatedTimeRemaining('')
                        refetchCachedModels()
                        setTimeout(() => {
                          if (!showRecommendedBackgroundDownload) {
                            setConfirmOpen(false)
                            onGoToProject()
                          }
                          setSubmitState('idle')
                          setDownloadProgress(0)
                          setShowRecommendedBackgroundDownload(false)
                        }, 1000)
                      } else if (event.event === 'error') {
                        const errorMessage =
                          event.message ||
                          'Failed to download model. Please check the model name and try again.'
                        // Check if it's a disk space error
                        if (isDiskSpaceError(errorMessage)) {
                          await handleDiskSpaceError(
                            modelIdentifier,
                            errorMessage
                          )
                          setSubmitState('error')
                          setDownloadError(errorMessage)
                        } else {
                          setSubmitState('error')
                          setDownloadError(errorMessage)
                        }
                      }
                    }
                  } catch (error: any) {
                    const errorMessage =
                      error.message ||
                      'Failed to download model. Please check the model name and try again.'
                    // Check if it's a disk space error
                    if (isDiskSpaceError(errorMessage)) {
                      await handleDiskSpaceError(modelIdentifier, errorMessage)
                      setSubmitState('error')
                      setDownloadError(errorMessage)
                    } else {
                      setSubmitState('error')
                      setDownloadError(errorMessage)
                    }
                  }
                }

                downloadAsync()
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
              {(() => {
                if (selectedModelGroup && selectedQuantization) {
                  const validOptions = ggufOptions.filter(
                    opt => opt.quantization
                  )
                  const selectedOption = validOptions.find(
                    opt => opt.quantization === selectedQuantization
                  )
                  if (selectedOption) {
                    return (
                      <>
                        Download and add {selectedOption.size_human}
                        {submitState !== 'loading' &&
                          submitState !== 'success' && (
                            <svg
                              xmlns="http://www.w3.org/2000/svg"
                              viewBox="0 0 32 32"
                              fill="none"
                              className="ml-2"
                              style={{ width: '16px', height: '16px' }}
                            >
                              <path
                                d="M26 24V28H6V24H4V28C4 28.5304 4.21071 29.0391 4.58579 29.4142C4.96086 29.7893 5.46957 30 6 30H26C26.5304 30 27.0391 29.7893 27.4142 29.4142C27.7893 29.0391 28 28.5304 28 28V24H26ZM26 14L24.59 12.59L17 20.17V2H15V20.17L7.41 12.59L6 14L16 24L26 14Z"
                                fill="currentColor"
                              />
                            </svg>
                          )}
                      </>
                    )
                  }
                }
                return (
                  <>
                    Download and add
                    {submitState !== 'loading' && submitState !== 'success' && (
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        viewBox="0 0 32 32"
                        fill="none"
                        className="ml-2"
                        style={{ width: '16px', height: '16px' }}
                      >
                        <path
                          d="M26 24V28H6V24H4V28C4 28.5304 4.21071 29.0391 4.58579 29.4142C4.96086 29.7893 5.46957 30 6 30H26C26.5304 30 27.0391 29.7893 27.4142 29.4142C27.7893 29.0391 28 28.5304 28 28V24H26ZM26 14L24.59 12.59L17 20.17V2H15V20.17L7.41 12.59L6 14L16 24L26 14Z"
                          fill="currentColor"
                        />
                      </svg>
                    )}
                  </>
                )
              })()}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Background download indicator - minimized */}
      {showRecommendedBackgroundDownload && submitState === 'loading' && (
        <div
          role="button"
          tabIndex={0}
          onClick={() => {
            setShowRecommendedBackgroundDownload(false)
            setConfirmOpen(true)
          }}
          onKeyDown={e => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              setShowRecommendedBackgroundDownload(false)
              setConfirmOpen(true)
            }
          }}
          className="fixed bottom-4 right-4 z-[100] w-[320px] rounded-lg border border-border bg-card text-card-foreground shadow-lg p-3 text-left"
          aria-label="Show download progress"
        >
          <div className="flex items-center justify-between mb-2">
            <div className="text-sm font-medium">
              {pendingVariant
                ? `Downloading ${modelName || pendingVariant.label}...`
                : 'Downloading model...'}
            </div>
            <button
              type="button"
              className="h-7 px-2 rounded-md border border-input text-xs hover:bg-accent/30"
              onClick={e => {
                e.stopPropagation()
                setShowRecommendedBackgroundDownload(false)
                setConfirmOpen(true)
              }}
            >
              View
            </button>
          </div>
          <div className="flex items-center gap-3">
            <div className="h-2 w-full rounded-full bg-accent/20">
              <div
                className="h-2 rounded-full bg-primary transition-all"
                style={{ width: `${downloadProgress}%` }}
              />
            </div>
            <div className="text-xs text-muted-foreground whitespace-nowrap">
              {Math.floor(downloadProgress)}%
            </div>
          </div>
          {estimatedTimeRemaining && (
            <div className="mt-2 text-xs text-muted-foreground truncate">
              {estimatedTimeRemaining} remaining
            </div>
          )}
        </div>
      )}

      {/* Background download success notification */}
      {showRecommendedBackgroundDownload && submitState === 'success' && (
        <div className="fixed bottom-4 right-4 z-[100] w-[320px] rounded-lg border border-border bg-card text-card-foreground shadow-lg p-3 flex items-start gap-3">
          <div className="flex-shrink-0">
            <FontIcon
              type="checkmark-filled"
              className="w-5 h-5 text-primary"
            />
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium">Download complete</div>
            <div className="text-xs text-muted-foreground">
              {modelName || pendingVariant?.label || 'Model'} is ready to use
            </div>
          </div>
          <button
            onClick={() => {
              setShowRecommendedBackgroundDownload(false)
              onGoToProject()
            }}
            className="text-muted-foreground hover:text-foreground"
          >
            <FontIcon type="close" className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Background download error notification */}
      {showRecommendedBackgroundDownload && submitState === 'error' && (
        <div className="fixed bottom-4 right-4 z-[100] w-[320px] rounded-lg border border-destructive/20 bg-card text-card-foreground shadow-lg p-3 flex items-start gap-3">
          <div className="flex-shrink-0">
            <FontIcon type="close" className="w-5 h-5 text-destructive" />
          </div>
          <div className="flex-1">
            <div className="text-sm font-medium text-destructive">
              Download failed
            </div>
            <div className="text-xs text-muted-foreground">
              {downloadError || 'Failed to download model'}
            </div>
          </div>
          <button
            onClick={() => {
              setShowRecommendedBackgroundDownload(false)
              setSubmitState('idle')
              setDownloadError('')
            }}
            className="text-muted-foreground hover:text-foreground"
          >
            <FontIcon type="close" className="w-4 h-4" />
          </button>
        </div>
      )}
    </>
  )
}

const Models = () => {
  const navigate = useNavigate()
  const activeProject = useActiveProject()
  const { data: projectResponse } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )
  const updateProject = useUpdateProject()
  const { toast } = useToast()
  const [searchParams] = useSearchParams()
  const initialTab = searchParams.get('tab') === 'training' ? 'training' : 'project'
  const [activeTab, setActiveTab] = useState(initialTab)
  const [mode, setMode] = useModeWithReset('designer')
  const [projectModels, setProjectModels] = useState<InferenceModel[]>([])
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false)
  const [modelToDelete, setModelToDelete] = useState<string | null>(null)

  // Fetch cached models from device
  const { data: cachedModelsResponse } = useCachedModels()

  const projectConfig = (projectResponse as any)?.project?.config as
    | ProjectConfig
    | undefined
  const getModelsLocation = useCallback(
    () => ({ type: 'runtime.models' as const }),
    []
  )
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getModelsLocation,
  })

  // Load models from config
  useEffect(() => {
    if (!projectResponse?.project?.config?.runtime?.models) {
      setProjectModels([])
      return
    }

    const runtimeModels = projectResponse.project.config.runtime.models
    const defaultModelName =
      projectResponse.project.config.runtime.default_model

    // If no explicit default_model is set, use the first model as default
    // Use the same fallback logic as the mapped models: name || model || 'unnamed-model'
    const effectiveDefaultModelName =
      defaultModelName ||
      (runtimeModels.length > 0
        ? runtimeModels[0]?.name || runtimeModels[0]?.model || 'unnamed-model'
        : null)

    const mappedModels: InferenceModel[] = runtimeModels.map((model: any) => {
      const name: string =
        (model && (model.name || model.model)) || 'unnamed-model'
      const provider: string =
        typeof model?.provider === 'string' ? model.provider : ''

      // Determine if model is Local or Cloud
      // Local: ollama, universal (both run locally)
      // Cloud: everything else (openai, anthropic, etc.)
      const isLocal = provider === 'ollama' || provider === 'universal'
      const localityBadge = isLocal ? 'Local' : 'Cloud'

      // Check if model is default: via runtime.default_model, model.default flag, or first model if none set
      const isDefault =
        name === defaultModelName ||
        model?.default === true ||
        (!defaultModelName && name === effectiveDefaultModelName)

      return {
        id: name,
        name,
        modelIdentifier: typeof model?.model === 'string' ? model.model : '',
        meta: (model && model.description) || 'Model from config',
        badges: [localityBadge],
        isDefault,
        status: 'ready' as ModelStatus,
      }
    })

    setProjectModels(mappedModels)
  }, [projectResponse])

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
          prompts:
            updatedMap[id] && updatedMap[id].length > 0
              ? updatedMap[id]
              : ['default'],
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
          prompts: ['default'],
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

  const handleModelChange = async (
    modelId: string,
    newModelIdentifier: string
  ) => {
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !projectResponse?.project?.config
    )
      return

    // Optimistically update local state
    const prevModels = [...projectModels]
    setProjectModels(prev =>
      prev.map(m =>
        m.id === modelId ? { ...m, modelIdentifier: newModelIdentifier } : m
      )
    )

    const currentConfig = projectResponse.project.config
    const runtimeModels = currentConfig.runtime?.models || []

    const updatedModels = runtimeModels.map((model: any) => {
      if (model.name === modelId) {
        return {
          ...model,
          model: newModelIdentifier,
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
      toast({
        message: 'Model updated successfully',
        variant: 'default',
      })
    } catch (error) {
      console.error('Failed to update model identifier:', error)
      toast({
        message: 'Failed to update model. Please try again.',
        variant: 'destructive',
      })
      // Rollback on failure
      setProjectModels(prevModels)
    }
  }

  const handleRename = async (modelId: string, newName: string) => {
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !projectResponse?.project?.config
    )
      return

    // Find the model to rename to get its current name
    const modelToRename = projectModels.find(m => m.id === modelId)
    if (!modelToRename) return

    // Validate input using the same validation as other dialogs
    // Pass the model's current name (not modelId) to validateModelName
    const existingNames = projectModels.map(m => m.name)
    const validation = validateModelName(
      newName.trim(),
      existingNames,
      modelToRename.name
    )
    if (!validation.isValid) {
      toast({
        message: validation.error || 'Invalid model name',
        variant: 'destructive',
      })
      return
    }

    const trimmedName = newName.trim()

    // Optimistically update local state
    const prevModels = [...projectModels]
    const prevModelSetMap = { ...modelSetMap }

    setProjectModels(prev =>
      prev.map(m =>
        m.id === modelId ? { ...m, name: trimmedName, id: trimmedName } : m
      )
    )

    // Update modelSetMap to use new name as key
    if (modelSetMap[modelId]) {
      const newMap = { ...modelSetMap }
      newMap[trimmedName] = newMap[modelId]
      delete newMap[modelId]
      setModelSetMap(newMap)
    }

    const currentConfig = projectResponse.project.config
    const runtimeModels = currentConfig.runtime?.models || []
    const wasDefault = currentConfig.runtime?.default_model === modelId

    const updatedModels = runtimeModels.map((model: any) => {
      // Use the model's current name (not modelId) to find the model in config
      if (model.name === modelToRename.name) {
        return {
          ...model,
          name: trimmedName,
        }
      }
      return model
    })

    const nextConfig = {
      ...currentConfig,
      runtime: {
        ...currentConfig.runtime,
        models: updatedModels,
        // Update default_model if this was the default
        default_model: wasDefault
          ? trimmedName
          : currentConfig.runtime?.default_model,
      },
    }

    try {
      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: nextConfig },
      })
      toast({
        message: 'Model renamed successfully',
        variant: 'default',
      })
    } catch (error) {
      console.error('Failed to rename model:', error)
      toast({
        message: 'Failed to rename model. Please try again.',
        variant: 'destructive',
      })
      // Rollback on failure
      setProjectModels(prevModels)
      setModelSetMap(prevModelSetMap)
    }
  }

  // Prepare available models for the selector
  const availableProjectModels = projectModels.map(m => ({
    identifier: m.modelIdentifier || m.name,
    name: m.name,
  }))

  const availableDeviceModels =
    cachedModelsResponse?.data.map(cachedModel => ({
      identifier: cachedModel.name,
      name: cachedModel.name,
    })) || []

  return (
    <div
      className={`h-full w-full flex flex-col ${mode === 'designer' ? 'gap-3 pb-32' : ''}`}
    >
      <div className="flex items-center justify-between mb-2">
        <h2 className="text-2xl">
          {mode === 'designer' ? 'Models' : 'Config editor'}
        </h2>
        <PageActions mode={mode} onModeChange={handleModeChange} />
      </div>

      {mode !== 'designer' ? (
        <div className="flex-1 min-h-0 overflow-hidden pb-6">
          <ConfigEditor className="h-full" initialPointer={configPointer} />
        </div>
      ) : (
        <>
          <TabBar
            activeTab={activeTab}
            onChange={setActiveTab}
            tabs={[
              { id: 'project', label: 'Inference models' },
              { id: 'training', label: 'Trained models' },
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
                    onClick={() => navigate('/chat/models/add')}
                    className="w-full sm:w-auto"
                  >
                    Add models
                  </Button>
                </div>
              </div>
            ) : (
              <>
                {/* Title and Add button row */}
                <div className="flex items-center justify-between mb-2">
                  <h2 className="text-lg font-medium">Project inference models</h2>
                  <Button onClick={() => navigate('/chat/models/add')}>
                    Add inference models
                  </Button>
                </div>
                <ProjectInferenceModels
                models={projectModels}
                onMakeDefault={makeDefault}
                onDelete={deleteModel}
                onRename={handleRename}
                getSelected={getSelectedFor}
                promptSetNames={promptSetNames}
                onToggle={toggleFor}
                onClear={clearFor}
                availableProjectModels={availableProjectModels}
                availableDeviceModels={availableDeviceModels}
                onModelChange={handleModelChange}
              />
              </>
            ))}
          {activeTab === 'training' && <TrainedModels />}
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
