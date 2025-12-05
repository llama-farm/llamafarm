import { useState, useCallback, useRef, useEffect, useMemo } from 'react'
import { useMutation } from '@tanstack/react-query'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import { AVAILABLE_DEMOS } from '../../config/demos'
import * as YAML from 'yaml'
import {
  saveDatasetTaskId,
  saveDatasetResult,
} from '../../utils/datasetStorage'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../ui/dropdown-menu'
import type { RagStrategy } from '../Rag/strategies'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
  DialogClose,
} from '../ui/dialog'
import { Button } from '../ui/button'
import ImportSampleDatasetModal from './ImportSampleDatasetModal'
import { useImportExampleDataset } from '../../hooks/useExamples'
import PageActions from '../common/PageActions'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { useToast } from '../ui/toast'
import { useLocation, useNavigate } from 'react-router-dom'
import { useActiveProject } from '../../hooks/useActiveProject'
import {
  useListDatasets,
  useCreateDataset,
  useDeleteDataset,
  useAvailableStrategies,
} from '../../hooks/useDatasets'
import { uploadFileToDataset } from '../../api/datasets'
import datasetService from '../../api/datasets'
import projectService from '../../api/projectService'
import { useProject } from '../../hooks/useProjects'
import { useDataProcessingStrategies } from '../../hooks/useDataProcessingStrategies'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'
import { isValidFile } from '../../utils/fileValidation'
import { validateDatasetNameWithDuplicateCheck } from '../../utils/datasetValidation'
import { getDatabaseColor } from '../../utils/databaseColors'
import {
  loadDatasetTaskId,
  clearDatasetTaskId,
} from '../../utils/datasetStorage'
import { useTaskStatus } from '../../hooks/useDatasets'

// Batch size for uploads to prevent overwhelming the backend
const UPLOAD_BATCH_SIZE = 3

// Component for individual dataset card with processing status tracking
const DatasetCard = ({
  dataset,
  activeProject,
  databases,
  onNavigate,
  onDelete,
}: {
  dataset: any
  activeProject: any
  databases: any[]
  onNavigate: (id: string) => void
  onDelete: (id: string, name: string) => void
}) => {
  // Check if this dataset has an active processing task
  const taskId = loadDatasetTaskId(
    activeProject?.namespace || '',
    activeProject?.project || '',
    dataset.name
  )

  // Poll task status if task exists
  const { data: taskStatus } = useTaskStatus(
    activeProject?.namespace || '',
    activeProject?.project || '',
    taskId,
    {
      enabled: !!taskId && !!activeProject,
      refetchInterval: 2000,
    }
  )

  // Clear task ID when processing completes
  useEffect(() => {
    if (
      taskStatus &&
      (taskStatus.state === 'SUCCESS' || taskStatus.state === 'FAILURE')
    ) {
      clearDatasetTaskId(
        activeProject?.namespace || '',
        activeProject?.project || '',
        dataset.name
      )
    }
  }, [taskStatus, activeProject, dataset.name])

  // Calculate processing percentage
  const isProcessing =
    taskStatus &&
    taskStatus.state !== 'SUCCESS' &&
    taskStatus.state !== 'FAILURE'
  const processingPercent = useMemo(() => {
    if (!isProcessing || !taskStatus) return 0

    if (taskStatus.meta?.progress) {
      return taskStatus.meta.progress
    }

    if (taskStatus.meta?.current && taskStatus.meta?.total) {
      return Math.round((taskStatus.meta.current / taskStatus.meta.total) * 100)
    }

    return 0
  }, [isProcessing, taskStatus])

  return (
    <div
      key={dataset.id}
      className={`w-full bg-card rounded-lg border flex flex-col gap-3 p-4 relative hover:bg-accent/20 cursor-pointer transition-colors ${
        isProcessing ? 'border-[#1e3a8a] dark:border-white' : 'border-border'
      }`}
      onClick={() => onNavigate(dataset.id)}
      role="button"
      tabIndex={0}
      onKeyDown={e => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault()
          onNavigate(dataset.id)
        }
      }}
    >
      <div className="absolute right-3 top-3">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
              onClick={e => e.stopPropagation()}
              aria-label="Dataset actions"
            >
              <FontIcon type="overflow" className="w-4 h-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="min-w-[10rem] w-[10rem]">
            <DropdownMenuItem
              onClick={e => {
                e.stopPropagation()
                onNavigate(dataset.id)
              }}
            >
              View
            </DropdownMenuItem>
            <DropdownMenuItem
              className="text-destructive focus:text-destructive"
              onClick={e => {
                e.stopPropagation()
                onDelete(dataset.id, dataset.name)
              }}
            >
              Delete
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      <div className="text-sm font-medium">{dataset.name}</div>
      <div className="flex flex-row gap-2 items-center flex-wrap mt-2">
        {dataset.database && (
          <Badge
            variant="default"
            size="sm"
            className={`rounded-xl ${getDatabaseColor(dataset.database, databases)} cursor-pointer hover:opacity-80 transition-opacity`}
            onClick={e => {
              e.stopPropagation()
              onNavigate(
                `/chat/databases?database=${encodeURIComponent(dataset.database)}`
              )
            }}
          >
            {dataset.database}
          </Badge>
        )}
        <Badge
          variant="default"
          size="sm"
          className="rounded-xl bg-muted text-foreground dark:bg-muted dark:text-foreground"
        >
          {dataset.data_processing_strategy}
        </Badge>
      </div>
      <div className="text-xs text-muted-foreground mt-2">
        {dataset.files.length} {dataset.files.length === 1 ? 'file' : 'files'}
      </div>

      {/* Processing percentage badge */}
      {isProcessing && (
        <div className="absolute bottom-3 right-3">
          <Badge
            variant="secondary"
            size="sm"
            className="rounded-xl bg-primary/10 text-primary border border-primary/20"
          >
            Processing {processingPercent}%...
          </Badge>
        </div>
      )}
    </div>
  )
}

const Data = () => {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [mode, setMode] = useModeWithReset('designer')

  // File drag-and-drop state management
  const [pendingFiles, setPendingFiles] = useState<File[]>([])
  const [isSelectDatasetModalOpen, setIsSelectDatasetModalOpen] =
    useState(false)
  const [shouldUploadAfterCreate, setShouldUploadAfterCreate] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadingFileCount, setUploadingFileCount] = useState(0)
  const activeUploadControllersRef = useRef<AbortController[]>([])
  const [isTransitioningToCreate, setIsTransitioningToCreate] = useState(false)

  const navigate = useNavigate()
  const location = useLocation()
  const { toast } = useToast()

  // Get current active project for API calls
  const activeProject = useActiveProject()

  // Cleanup effect: abort all active upload controllers on unmount
  useEffect(() => {
    return () => {
      // Abort all active upload network requests when component unmounts
      activeUploadControllersRef.current.forEach(controller => {
        try {
          controller.abort()
        } catch (err) {
          // Ignore errors from already aborted controllers
          console.debug('Controller aborted on unmount:', err)
        }
      })
    }
  }, []) // Empty dependency array - only run cleanup on unmount

  // File type mapping for parser creation (centralized to avoid duplication)
  const fileTypeMapping = useMemo(
    () => [
      { type: 'PDF', parser: 'PDFParser_LlamaIndex', extensions: ['*.pdf'] },
      {
        type: 'Docx',
        parser: 'DOCXParser_LlamaIndex',
        extensions: ['*.docx'],
      },
      { type: 'Text', parser: 'TEXTParser_LlamaIndex', extensions: ['*.txt'] },
      { type: 'CSV', parser: 'CSVParser_Pandas', extensions: ['*.csv'] },
      {
        type: 'Markdown',
        parser: 'MARKDOWNParser_LlamaIndex',
        extensions: ['*.md', '*.markdown'],
      },
    ],
    []
  )

  // Load project config to get strategies (source of truth)
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )

  // Use React Query hooks for datasets
  const {
    data: apiDatasets,
    isLoading: isDatasetsLoading,
    error: datasetsError,
    refetch: refetchDatasets,
  } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )
  const createDatasetMutation = useCreateDataset()
  const deleteDatasetMutation = useDeleteDataset()
  const importExampleDataset = useImportExampleDataset()

  // Custom upload mutation with proper AbortSignal handling
  const uploadMutation = useMutation({
    mutationFn: async ({
      namespace,
      project,
      dataset,
      file,
      signal,
    }: {
      namespace: string
      project: string
      dataset: string
      file: File
      signal?: AbortSignal
    }) => {
      // Use the API service which properly handles the signal
      return await uploadFileToDataset(
        namespace,
        project,
        dataset,
        file,
        signal
      )
    },
    onError: error => {
      // Don't show error toast for aborted uploads
      if (
        (error instanceof Error && error.name === 'AbortError') ||
        (error as any)?.code === 'ERR_CANCELED'
      ) {
        return
      }
      // Error toasts for actual failures are handled in handleDatasetSelect
    },
  })

  // Fetch available strategies and databases from API
  const { data: availableOptions } = useAvailableStrategies(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )

  // Hook for managing data processing strategies
  const strategies = useDataProcessingStrategies(
    activeProject?.namespace || '',
    activeProject?.project || ''
  )

  const projectConfig = (projectResp as any)?.project?.config as
    | ProjectConfig
    | undefined
  const getDatasetsLocation = useCallback(
    () => ({ type: 'datasets' as const }),
    []
  )
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getDatasetsLocation,
  })

  // Convert API datasets to UI format - only show real datasets from the API
  const datasets = useMemo(() => {
    // Only return datasets from the API, no localStorage, no placeholders
    return (apiDatasets?.datasets || []).map(dataset => ({
      id: dataset.name,
      name: dataset.name,
      database: dataset.database,
      data_processing_strategy: dataset.data_processing_strategy,
      files: dataset.details?.files_metadata || [],
      // Only show fields that actually come from the API
    }))
  }, [apiDatasets])

  // Extract databases from project config for color assignment
  const databases = useMemo(() => {
    const ragDatabases = (projectResp as any)?.project?.config?.rag?.databases
    return Array.isArray(ragDatabases) ? ragDatabases : []
  }, [projectResp])

  // If navigated with ?dataset= query, auto-redirect to that dataset's detail if it exists
  const hasRedirectedFromQuery = useRef(false)
  useEffect(() => {
    if (hasRedirectedFromQuery.current) return
    const params = new URLSearchParams(location.search)
    const datasetParam = params.get('dataset')
    if (!datasetParam) return
    const found = datasets.find(d => d.id === datasetParam)
    if (found) {
      hasRedirectedFromQuery.current = true
      navigate(`/chat/data/${found.id}`, { replace: true })
    }
  }, [location.search, datasets, navigate])

  // Map of fileKey -> array of dataset ids (transient UI state)
  // const [fileAssignments] = useState<Record<string, string[]>>({})

  // Processing strategies state management ----------------------------------
  const [strategyEditOpen, setStrategyEditOpen] = useState(false)
  const [strategyEditId, setStrategyEditId] = useState<string>('')
  const [strategyEditName, setStrategyEditName] = useState('')
  const [strategyEditDescription, setStrategyEditDescription] = useState('')
  const [strategyCreateOpen, setStrategyCreateOpen] = useState(false)
  const [strategyCreateName, setStrategyCreateName] = useState('')
  const [strategyCreateDescription, setStrategyCreateDescription] = useState('')
  const [strategyCopyFromId, setStrategyCopyFromId] = useState('')
  const [strategyCreateFileTypes, setStrategyCreateFileTypes] = useState<
    Set<string>
  >(new Set())

  // Load strategies from config (source of truth - NO hardcoding)
  const displayStrategies = useMemo((): RagStrategy[] => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig?.rag?.data_processing_strategies) {
      // Return empty array if config not loaded yet
      return []
    }

    const configStrategies = projectConfig.rag.data_processing_strategies || []

    // Convert config strategies to UI format
    return configStrategies.map(
      (strategy: any) =>
        ({
          id: `processing-${strategy.name.replace(/_/g, '-')}`, // Convert snake_case to kebab-case
          name: strategy.name
            .replace(/_/g, ' ')
            .replace(/\b\w/g, (c: string) => c.toUpperCase()),
          description: strategy.description || '',
          isDefault: false, // All strategies are equal - no hardcoded defaults
          datasetsUsing: 0, // Will be calculated from datasetsByStrategyName
          configName: strategy.name, // Store original config name for API calls
        }) as RagStrategy
    )
  }, [projectResp])

  // Build mapping of strategy name -> dataset names
  const datasetsByStrategyName = useMemo(() => {
    const map = new Map<string, string[]>()

    // Use the datasets from API
    for (const d of datasets) {
      const strategyName = (d as any).data_processing_strategy
      const datasetName = d.name
      if (typeof strategyName === 'string' && typeof datasetName === 'string') {
        const arr = map.get(strategyName) || []
        arr.push(datasetName)
        map.set(strategyName, arr)
      }
    }

    return map
  }, [datasets])

  const getParsersCount = (strategyName: string): number => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig?.rag?.data_processing_strategies) return 0

    const strategy = projectConfig.rag.data_processing_strategies.find(
      (s: any) => s.name === strategyName
    )
    return strategy?.parsers?.length || 0
  }

  const getExtractorsCount = (strategyName: string): number => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig?.rag?.data_processing_strategies) return 0

    const strategy = projectConfig.rag.data_processing_strategies.find(
      (s: any) => s.name === strategyName
    )
    return strategy?.extractors?.length || 0
  }

  // rawFiles and fileAssignments are transient UI state - no persistence needed

  // Create dataset dialog state
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [isImportOpen, setIsImportOpen] = useState(false)
  const [newDatasetName, setNewDatasetName] = useState('')
  const [newDatasetDatabase, setNewDatasetDatabase] = useState('')
  const [
    newDatasetDataProcessingStrategy,
    setNewDatasetDataProcessingStrategy,
  ] = useState('')
  const [datasetNameError, setDatasetNameError] = useState<string | null>(null)

  // Set default values when dialog opens and options are available
  useEffect(() => {
    if (isCreateOpen && availableOptions) {
      if (
        !newDatasetDataProcessingStrategy &&
        availableOptions.data_processing_strategies?.[0]
      ) {
        setNewDatasetDataProcessingStrategy(
          availableOptions.data_processing_strategies[0]
        )
      }
      if (!newDatasetDatabase && availableOptions.databases?.[0]) {
        setNewDatasetDatabase(availableOptions.databases[0])
      }
    }
  }, [
    isCreateOpen,
    availableOptions,
    newDatasetDataProcessingStrategy,
    newDatasetDatabase,
  ])

  // Simple edit modal state
  // Edit dataset removed - API doesn't support updating datasets
  const [isConfirmDeleteOpen, setIsConfirmDeleteOpen] = useState(false)
  const [confirmDeleteId, setConfirmDeleteId] = useState<string>('')
  const [confirmDeleteName, setConfirmDeleteName] = useState<string>('')

  const handleCreateDataset = async () => {
    const name = newDatasetName.trim()
    if (!name) return

    // Validate dataset name
    const existingDatasetNames = datasets?.map(d => d.name) || []
    const validation = validateDatasetNameWithDuplicateCheck(
      name,
      existingDatasetNames
    )

    if (!validation.isValid) {
      setDatasetNameError(validation.error || 'Invalid dataset name')
      return
    }

    if (!activeProject?.namespace || !activeProject?.project) {
      toast({
        message: 'No active project selected',
        variant: 'destructive',
      })
      return
    }

    try {
      const response = await createDatasetMutation.mutateAsync({
        namespace: activeProject.namespace,
        project: activeProject.project,
        name,
        data_processing_strategy: newDatasetDataProcessingStrategy || 'default',
        database: newDatasetDatabase || 'default',
      })

      toast({ message: 'Dataset created successfully', variant: 'default' })
      setIsCreateOpen(false)
      setNewDatasetName('')
      setNewDatasetDatabase('')
      setNewDatasetDataProcessingStrategy('')
      setDatasetNameError(null)

      // If we should upload files after creating, use the newly created dataset
      // Note: In this system, dataset name serves as the unique identifier (ID)
      if (shouldUploadAfterCreate && response?.dataset?.name) {
        const datasetId = response.dataset.name
        const datasetName = response.dataset.name
        handleDatasetSelect(datasetId, datasetName)
      }
    } catch (error) {
      console.error('Failed to create dataset:', error)
      toast({
        message: 'Failed to create dataset. Please try again.',
        variant: 'destructive',
      })
    }
  }

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    // Set the drop effect to allow dropping
    e.dataTransfer.dropEffect = 'copy'
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback(
    async (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragging(false)

      const files = Array.from(e.dataTransfer.files)

      if (files.length === 0) {
        return
      }

      // Comprehensive file validation with extension, MIME type, and content verification
      const validationResults = await Promise.all(
        files.map(async file => ({
          file,
          validation: await isValidFile(file),
        }))
      )

      const validFiles = validationResults
        .filter(result => result.validation.valid)
        .map(result => result.file)

      const invalidFiles = validationResults.filter(
        result => !result.validation.valid
      )

      if (validFiles.length === 0) {
        const reasons = invalidFiles
          .map(r => `${r.file.name}: ${r.validation.reason}`)
          .join('; ')
        toast({
          message: `No valid files to upload. ${reasons}`,
          variant: 'destructive',
        })
        return
      }

      if (invalidFiles.length > 0) {
        const reasons = invalidFiles
          .slice(0, 3)
          .map(r => `${r.file.name}: ${r.validation.reason}`)
        const message =
          invalidFiles.length <= 3
            ? reasons.join('; ')
            : `${reasons.join('; ')}... and ${invalidFiles.length - 3} more`

        toast({
          message: `${invalidFiles.length} invalid file(s) were rejected. ${message}`,
          variant: 'default',
        })
      }

      setPendingFiles(validFiles)
      setIsSelectDatasetModalOpen(true)
    },
    [toast]
  )

  // Upload files in batches with cancellation support
  const uploadFilesInBatches = useCallback(
    async (
      files: File[],
      datasetId: string,
      namespace: string,
      project: string,
      batchSize: number = UPLOAD_BATCH_SIZE
    ) => {
      const results = []
      let cancelled = false

      for (let i = 0; i < files.length; i += batchSize) {
        // CHECK FOR CANCELLATION AT START OF EACH BATCH
        if (cancelled) {
          // Add cancelled results for remaining files
          const remainingFiles = files.slice(i)
          remainingFiles.forEach(file => {
            results.push({
              file: file.name,
              success: false,
              error: new Error('Cancelled'),
              cancelled: true,
            })
          })
          break
        }

        const batch = files.slice(i, i + batchSize)
        const batchResults = await Promise.all(
          batch.map(async file => {
            const controller = new AbortController()
            activeUploadControllersRef.current.push(controller)

            try {
              const result = await uploadMutation.mutateAsync({
                namespace,
                project,
                dataset: datasetId,
                file,
                signal: controller.signal,
              })
              return {
                file: file.name,
                success: true,
                result,
                skipped: result.skipped || false,
              }
            } catch (error) {
              // Check if this is an abort/cancellation error
              if (
                (error instanceof Error && error.name === 'AbortError') ||
                (error as any)?.code === 'ERR_CANCELED' ||
                (error as any)?.message?.includes('cancel')
              ) {
                cancelled = true // Set flag to stop processing more batches
                return {
                  file: file.name,
                  success: false,
                  error,
                  cancelled: true,
                }
              }
              return { file: file.name, success: false, error }
            } finally {
              activeUploadControllersRef.current =
                activeUploadControllersRef.current.filter(c => c !== controller)
            }
          })
        )

        results.push(...batchResults)

        // Check if any upload in this batch was cancelled
        if (batchResults.some(r => (r as any).cancelled)) {
          cancelled = true
          // Add cancelled results for remaining files
          const remainingFiles = files.slice(i + batchSize)
          remainingFiles.forEach(file => {
            results.push({
              file: file.name,
              success: false,
              error: new Error('Cancelled'),
              cancelled: true,
            })
          })
          break
        }
      }

      return results
    },
    [uploadMutation]
  )

  // Handle file upload to selected dataset
  const handleDatasetSelect = useCallback(
    async (datasetId: string, datasetName: string) => {
      if (!activeProject || pendingFiles.length === 0) {
        return
      }

      const fileCount = pendingFiles.length // Store count before clearing
      setUploadingFileCount(fileCount)
      setIsUploading(true)
      setIsSelectDatasetModalOpen(false)

      const { namespace, project } = activeProject

      try {
        // Upload all files to the selected dataset in batches
        const results = await uploadFilesInBatches(
          pendingFiles,
          datasetId,
          namespace,
          project
        )

        const cancelled = results.some(r => (r as any).cancelled)
        const failures = results.filter(
          r => !r.success && !(r as any).cancelled
        )
        const successes = results.filter(r => r.success && !(r as any).skipped)
        const skipped = results.filter(r => r.success && (r as any).skipped)

        // If upload was cancelled, don't show success/failure toast (already shown in handleCancelUpload)
        if (cancelled) {
          return
        }

        // Show appropriate toast based on results
        if (failures.length > 0 && successes.length > 0) {
          // Partial success: some uploads succeeded, some failed
          const skippedMsg =
            skipped.length > 0 ? `, skipped ${skipped.length} duplicate(s)` : ''
          toast({
            message: `Uploaded ${successes.length} of ${fileCount} file(s)${skippedMsg}. Failed: ${failures.map(f => f.file).join(', ')}`,
            variant: 'destructive',
          })
        } else if (failures.length > 0 && skipped.length > 0) {
          // All files either failed or were skipped
          toast({
            message: `Upload failed for ${failures.length} file(s), skipped ${skipped.length} duplicate(s). Failed: ${failures.map(f => f.file).join(', ')}`,
            variant: 'destructive',
          })
        } else if (failures.length > 0) {
          // All files failed
          toast({
            message: `Upload failed for all files. Failed: ${failures.map(f => f.file).join(', ')}`,
            variant: 'destructive',
          })
        } else if (skipped.length > 0 && successes.length === 0) {
          // All files were duplicates
          toast({
            message: `All ${skipped.length} file(s) were already in ${datasetName}`,
            variant: 'default',
            icon: 'alert-triangle',
          })
        } else if (skipped.length > 0) {
          // Some successes with some skipped
          toast({
            message: `Uploaded ${successes.length} file(s) to ${datasetName}, skipped ${skipped.length} duplicate(s)`,
            variant: 'default',
          })
        } else {
          // All files succeeded
          toast({
            message: `Successfully uploaded ${fileCount} file(s) to ${datasetName}`,
            variant: 'default',
          })
        }

        // Navigate to the dataset view to see uploaded files (only if some succeeded)
        if (successes.length > 0) {
          // Explicitly refetch to ensure fresh data before navigating
          await refetchDatasets()
          navigate(`/chat/data/${datasetId}`)
        }
      } catch (error) {
        console.error('Upload failed:', error)
        // Check if error is due to cancellation
        if (
          (error as any)?.code === 'ERR_CANCELED' ||
          (error as any)?.message?.includes('cancel')
        ) {
          // Cancellation toast already shown, just return
          return
        }
        toast({
          message:
            error instanceof Error ? error.message : 'Failed to upload files',
          variant: 'destructive',
        })
      } finally {
        setPendingFiles([])
        activeUploadControllersRef.current = []
        setIsUploading(false)
        setShouldUploadAfterCreate(false)
        setUploadingFileCount(0)
        // Reset file input to allow reselecting the same files
        if (fileInputRef.current) {
          fileInputRef.current.value = ''
        }
      }
    },
    [
      activeProject,
      pendingFiles,
      uploadFilesInBatches,
      toast,
      navigate,
      refetchDatasets,
    ]
  )

  // Note: Auto-upload logic now handled directly in handleCreateDataset
  // No need for fragile useEffect watching datasets.length

  // Cancel file upload and reset state
  const handleCancelUpload = useCallback(() => {
    // Abort all active upload network requests
    activeUploadControllersRef.current.forEach(controller => {
      try {
        controller.abort()
      } catch (err) {
        // Ignore errors from already aborted controllers
        console.debug('Controller already aborted:', err)
      }
    })
    activeUploadControllersRef.current = []

    // Clear all state
    setPendingFiles([])
    setIsSelectDatasetModalOpen(false)
    setShouldUploadAfterCreate(false)
    setIsUploading(false)
    setUploadingFileCount(0)

    // Show cancellation toast
    toast({
      message: 'Upload cancelled - all active uploads have been stopped',
      variant: 'default',
    })
  }, [toast]) // Removed activeUploadControllers from dependency array since using ref

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files ? Array.from(e.target.files) : []
    if (files.length === 0) return

    // Apply the same comprehensive validation as drag-and-drop
    const validationResults = await Promise.all(
      files.map(async file => ({
        file,
        validation: await isValidFile(file),
      }))
    )

    const validFiles = validationResults
      .filter(result => result.validation.valid)
      .map(result => result.file)

    const invalidFiles = validationResults.filter(
      result => !result.validation.valid
    )

    if (validFiles.length === 0) {
      const reasons = invalidFiles
        .map(r => `${r.file.name}: ${r.validation.reason}`)
        .join('; ')
      toast({
        message: `No valid files selected. ${reasons}`,
        variant: 'destructive',
      })
      // Reset the input
      e.target.value = ''
      return
    }

    if (invalidFiles.length > 0) {
      const reasons = invalidFiles
        .slice(0, 3)
        .map(r => `${r.file.name}: ${r.validation.reason}`)
      const message =
        invalidFiles.length <= 3
          ? reasons.join('; ')
          : `${reasons.join('; ')}... and ${invalidFiles.length - 3} more`

      toast({
        message: `${invalidFiles.length} invalid file(s) were rejected. ${message}`,
        variant: 'default',
      })
    }

    // Reset the input for next selection
    e.target.value = ''

    // Open dataset selection modal with validated files
    setPendingFiles(validFiles)
    setIsSelectDatasetModalOpen(true)
  }

  // Render modal for selecting destination dataset for dropped files
  const renderSelectDatasetModal = () => (
    <Dialog
      open={isSelectDatasetModalOpen}
      onOpenChange={open => {
        // Don't clear state if we're transitioning to create dialog
        if (!open && isTransitioningToCreate) {
          setIsTransitioningToCreate(false)
          setIsSelectDatasetModalOpen(open)
          return
        }

        setIsSelectDatasetModalOpen(open)
        if (!open) {
          // Only clear if user is actually cancelling (not transitioning)
          setPendingFiles([])
          setShouldUploadAfterCreate(false)
        }
      }}
    >
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>Add Files to Dataset</DialogTitle>
          <DialogDescription>
            {pendingFiles.length === 1
              ? `Where would you like to add "${pendingFiles[0].name}"?`
              : `Where would you like to add ${pendingFiles.length} files?`}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {/* Create new dataset with dropped files */}
          <button
            onClick={() => {
              setIsTransitioningToCreate(true)
              setShouldUploadAfterCreate(true)
              setIsSelectDatasetModalOpen(false)
              setIsCreateOpen(true)
            }}
            className="w-full flex items-center gap-3 p-4 rounded-lg border-2 border-dashed border-primary/50 hover:border-primary hover:bg-primary/5 transition-colors"
          >
            <div className="flex-shrink-0 w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
              <svg
                className="w-5 h-5 text-primary"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
            </div>
            <div className="text-left">
              <div className="font-medium">Create New Dataset</div>
              <div className="text-sm text-muted-foreground">
                Start a new dataset with{' '}
                {pendingFiles.length === 1 ? 'this file' : 'these files'}
              </div>
            </div>
          </button>

          {/* Select from existing datasets */}
          {datasets && datasets.length > 0 && (
            <div className="space-y-2">
              <div className="text-sm font-medium text-muted-foreground px-1">
                Or add to existing dataset:
              </div>
              <div className="max-h-60 overflow-y-auto space-y-2">
                {datasets.map(dataset => (
                  <button
                    key={dataset.id}
                    onClick={() =>
                      handleDatasetSelect(dataset.id, dataset.name)
                    }
                    className="w-full flex items-center gap-3 p-3 rounded-lg border hover:bg-accent hover:border-accent-foreground/20 transition-colors text-left"
                  >
                    <div className="flex-shrink-0 w-8 h-8 rounded bg-accent flex items-center justify-center">
                      <svg
                        className="w-4 h-4"
                        fill="none"
                        viewBox="0 0 24 24"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
                        />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-medium truncate">{dataset.name}</div>
                      <div className="text-sm text-muted-foreground">
                        {dataset.files?.length || 0} files
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={handleCancelUpload}>
            Cancel
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )

  return (
    <div
      className="h-full w-full flex flex-col"
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <div className="w-full flex items-center justify-between mb-4 flex-shrink-0">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Data' : 'Config editor'}
        </h2>
        <PageActions mode={mode} onModeChange={handleModeChange} />
      </div>
      <input
        type="file"
        ref={fileInputRef}
        className="hidden"
        multiple
        onChange={handleFileSelect}
      />
      <div className="w-full flex-1 min-h-0 flex flex-col">
        {mode === 'designer' ? (
          <div className="flex-1 min-h-0 w-full overflow-auto">
            <div className="flex flex-col min-h-full">
              {/* Processing strategies section */}
              <div className="flex items-center justify-between mt-0 mb-3">
                <div>
                  <div className="font-medium">Processing strategies</div>
                  <div className="h-1" />
                  <div className="text-xs text-muted-foreground">
                    Processing strategies are applied to datasets.
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => {
                      setStrategyCreateName('')
                      setStrategyCreateDescription('')
                      setStrategyCopyFromId('')
                      setStrategyCreateOpen(true)
                    }}
                  >
                    Create new processing strategy
                  </Button>
                </div>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-8">
                {displayStrategies.map(s => {
                  const assigned =
                    datasetsByStrategyName.get(s.configName || '') || []
                  return (
                    <div
                      key={s.id}
                      className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${displayStrategies.length === 1 ? 'md:col-span-2' : ''}`}
                      onClick={() => navigate(`/chat/data/strategies/${s.id}`)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={e => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault()
                          navigate(`/chat/data/strategies/${s.id}`)
                        }
                      }}
                    >
                      <div className="absolute top-2 right-2">
                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <button
                              className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                              onClick={e => e.stopPropagation()}
                            >
                              <FontIcon type="overflow" className="w-4 h-4" />
                            </button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent
                            align="end"
                            className="min-w-[12rem] w-[12rem]"
                          >
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation()
                                setStrategyEditId(s.id)
                                setStrategyEditName(s.name)
                                setStrategyEditDescription(s.description)
                                setStrategyEditOpen(true)
                              }}
                            >
                              Rename
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              onClick={e => {
                                e.stopPropagation()
                                setStrategyCreateName(`${s.name} (copy)`)
                                setStrategyCreateDescription(s.description)
                                setStrategyCopyFromId(s.id)
                                setStrategyCreateOpen(true)
                              }}
                            >
                              Duplicate
                            </DropdownMenuItem>
                            <DropdownMenuItem
                              className="text-destructive focus:text-destructive"
                              onClick={e => {
                                e.stopPropagation()

                                // Prevent deleting default strategy
                                if (s.isDefault) {
                                  toast({
                                    message: 'Cannot delete default strategy',
                                    variant: 'destructive',
                                  })
                                  return
                                }

                                const ok = confirm(
                                  'Delete this processing strategy?'
                                )
                                if (!ok) return

                                const projectConfig = (projectResp as any)
                                  ?.project?.config
                                if (!projectConfig || !s.configName) {
                                  toast({
                                    message: 'Unable to delete strategy',
                                    variant: 'destructive',
                                  })
                                  return
                                }

                                strategies.deleteStrategy.mutate(
                                  {
                                    strategyName: s.configName,
                                    projectConfig,
                                  },
                                  {
                                    onSuccess: () => {
                                      toast({
                                        message: 'Strategy deleted',
                                        variant: 'default',
                                      })
                                    },
                                    onError: (error: any) => {
                                      toast({
                                        message:
                                          error.message ||
                                          'Failed to delete strategy',
                                        variant: 'destructive',
                                      })
                                    },
                                  }
                                )
                              }}
                            >
                              Delete
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>

                      <div className="text-sm font-medium">{s.name}</div>
                      <div className="text-xs text-primary text-left w-fit">
                        {s.description ||
                          'Unified processor for PDFs, Word docs, CSVs, Markdown, and text files.'}
                      </div>
                      <div className="flex items-center gap-2 flex-wrap pt-0.5">
                        {assigned.slice(0, 4).map(name => (
                          <Badge
                            key={name}
                            variant="default"
                            size="sm"
                            className="rounded-xl bg-muted text-foreground dark:bg-muted dark:text-foreground"
                          >
                            {name}
                          </Badge>
                        ))}
                        {assigned.length > 4 ? (
                          <Badge
                            variant="secondary"
                            size="sm"
                            className="rounded-xl"
                          >
                            +{assigned.length - 4}
                          </Badge>
                        ) : null}
                        {assigned.length === 0 ? (
                          <Badge
                            variant="secondary"
                            size="sm"
                            className="rounded-xl"
                          >
                            No datasets yet
                          </Badge>
                        ) : null}
                      </div>
                      <div className="flex items-center justify-between mt-1">
                        <div className="text-xs text-muted-foreground">
                          {getParsersCount(s.configName || '')} parsers •{' '}
                          {getExtractorsCount(s.configName || '')} extractors
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          className="px-3 h-7"
                          onClick={e => {
                            e.stopPropagation()
                            navigate(`/chat/data/strategies/${s.id}`)
                          }}
                        >
                          Configure
                        </Button>
                      </div>
                    </div>
                  )
                })}
              </div>
              {/* End processing strategies section */}

              {/* Divider */}
              <div className="border-t border-border mb-6"></div>

              {/* Datasets section */}
              <div className="mb-2 flex flex-row gap-2 justify-between items-end flex-shrink-0">
                <div className="font-medium">Datasets</div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => setIsImportOpen(true)}
                  >
                    Import sample dataset
                  </Button>
                  <Dialog
                    open={isCreateOpen}
                    onOpenChange={open => {
                      // Prevent closing dialog during mutation
                      if (!createDatasetMutation.isPending) {
                        setIsCreateOpen(open)
                        if (
                          !open &&
                          shouldUploadAfterCreate &&
                          pendingFiles.length > 0
                        ) {
                          // User cancelled dataset creation with pending files - go back to select modal
                          setIsSelectDatasetModalOpen(true)
                          // Keep pendingFiles and shouldUploadAfterCreate intact
                        } else if (!open) {
                          // Normal close without pending files, clear everything
                          setNewDatasetName('')
                          setNewDatasetDatabase('')
                          setNewDatasetDataProcessingStrategy('')
                          setDatasetNameError(null)
                          setPendingFiles([])
                          setShouldUploadAfterCreate(false)
                        }
                      }
                    }}
                  >
                    <DialogTrigger asChild>
                      <Button variant="default" size="sm">
                        Create new dataset
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>New dataset</DialogTitle>
                      </DialogHeader>
                      <div className="flex flex-col gap-3">
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-muted-foreground">
                            Name
                          </label>
                          <Input
                            autoFocus
                            value={newDatasetName}
                            onChange={e => {
                              const newName = e.target.value
                              setNewDatasetName(newName)

                              // Validate on change for real-time feedback
                              const existingDatasetNames =
                                datasets?.map(d => d.name) || []
                              const validation =
                                validateDatasetNameWithDuplicateCheck(
                                  newName,
                                  existingDatasetNames
                                )
                              setDatasetNameError(
                                validation.isValid
                                  ? null
                                  : validation.error || 'Invalid dataset name'
                              )
                            }}
                            placeholder="Enter dataset name"
                            className={
                              datasetNameError ? 'border-destructive' : ''
                            }
                          />
                          {datasetNameError && (
                            <p className="text-xs text-destructive mt-1">
                              {datasetNameError}
                            </p>
                          )}
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-muted-foreground">
                            Data Processing Strategy
                          </label>
                          <select
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                            value={newDatasetDataProcessingStrategy}
                            onChange={e =>
                              setNewDatasetDataProcessingStrategy(
                                e.target.value
                              )
                            }
                          >
                            <option value="">Select a strategy...</option>
                            {availableOptions?.data_processing_strategies?.map(
                              strategy => (
                                <option key={strategy} value={strategy}>
                                  {strategy}
                                </option>
                              )
                            )}
                          </select>
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-muted-foreground">
                            Database
                          </label>
                          <select
                            className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
                            value={newDatasetDatabase}
                            onChange={e =>
                              setNewDatasetDatabase(e.target.value)
                            }
                          >
                            <option value="">Select a database...</option>
                            {availableOptions?.databases?.map(database => (
                              <option key={database} value={database}>
                                {database}
                              </option>
                            ))}
                          </select>
                        </div>
                      </div>
                      <DialogFooter>
                        <DialogClose
                          asChild
                          disabled={createDatasetMutation.isPending}
                        >
                          <Button variant="secondary">Cancel</Button>
                        </DialogClose>
                        <Button
                          onClick={handleCreateDataset}
                          disabled={
                            !newDatasetName.trim() ||
                            !newDatasetDataProcessingStrategy.trim() ||
                            !newDatasetDatabase.trim() ||
                            !!datasetNameError ||
                            createDatasetMutation.isPending
                          }
                        >
                          {createDatasetMutation.isPending
                            ? 'Creating...'
                            : 'Create'}
                        </Button>
                      </DialogFooter>
                    </DialogContent>
                  </Dialog>
                </div>
              </div>
              <div className="flex-1 min-h-0 pb-6">
                {isDragging ? (
                  <div
                    className={`w-full h-full flex flex-col items-center justify-center border border-dashed rounded-lg p-4 gap-2 transition-colors border-input`}
                  >
                    <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px] text-primary">
                      <FontIcon
                        type="upload"
                        className="w-10 h-10 text-blue-200 dark:text-white"
                      />
                      <div className="text-xl text-foreground">
                        Drop data here
                      </div>
                    </div>
                    <p className="max-w-[527px] text-sm text-muted-foreground text-center mb-10">
                      You can upload PDFs, explore various list formats, or draw
                      inspiration from other data sources to enhance your
                      project with LlaMaFarm.
                    </p>
                  </div>
                ) : (
                  <div>
                    {isDatasetsLoading ? (
                      <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                        <Loader size={32} className="mr-2" />
                        Loading datasets...
                      </div>
                    ) : datasets.length <= 0 ? (
                      <div className="w-full mb-6 flex items-center justify-center rounded-lg py-4 text-primary text-center bg-primary/10">
                        {datasetsError
                          ? 'Unable to load datasets. Using local storage.'
                          : 'No datasets found. Create one to get started.'}
                      </div>
                    ) : (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-2 mb-6">
                        {datasets.map(ds => (
                          <DatasetCard
                            key={ds.id}
                            dataset={ds}
                            activeProject={activeProject}
                            databases={databases}
                            onNavigate={(id: string) => {
                              // Handle navigation to dataset or databases page
                              if (id.startsWith('/')) {
                                navigate(id)
                              } else {
                                navigate(`/chat/data/${id}`)
                              }
                            }}
                            onDelete={(id: string, name: string) => {
                              setConfirmDeleteId(id)
                              setConfirmDeleteName(name)
                              setIsConfirmDeleteOpen(true)
                            }}
                          />
                        ))}
                      </div>
                    )}

                    {/* Project-level raw files UI removed: files now only exist within datasets. */}
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex-1 min-h-0 overflow-hidden pb-6">
            <ConfigEditor className="h-full" initialPointer={configPointer} />
          </div>
        )}
      </div>

      {/* Edit dataset dialog */}
      <ImportSampleDatasetModal
        open={isImportOpen}
        onOpenChange={setIsImportOpen}
        onImport={async ({ name, sourceProjectId }) => {
          try {
            if (!activeProject?.namespace || !activeProject?.project) {
              toast({
                message: 'No active project selected',
                variant: 'destructive',
              })
              return
            }

            // Check if this is a demo import
            const demo = AVAILABLE_DEMOS.find(d => d.id === sourceProjectId)

            if (demo) {
              // Handle demo import
              toast({
                message: `Importing demo dataset "${name}"...`,
                variant: 'default',
              })
              setIsImportOpen(false)

              // Step 1: Fetch demo config to get processing strategy
              const configResponse = await fetch(demo.configPath)
              if (!configResponse.ok) {
                throw new Error('Failed to fetch demo configuration')
              }
              const configText = await configResponse.text()
              const configData = YAML.parse(configText)

              // Extract processing strategy from demo config
              const demoDataset = configData.datasets?.find(
                (ds: any) => ds.name === demo.datasetName
              )
              const processingStrategyName =
                demoDataset?.data_processing_strategy || 'default'
              const database = demoDataset?.database || 'default'

              // Step 2: Import the processing strategy and database into user's project if they don't exist
              const currentProjectConfig = (projectResp as any)?.project?.config
              if (currentProjectConfig) {
                let needsUpdate = false
                let updatedConfig = { ...currentProjectConfig }

                // Check and add processing strategy if needed
                const existingStrategies =
                  currentProjectConfig.rag?.data_processing_strategies || []
                const strategyExists = existingStrategies.some(
                  (s: any) => s.name === processingStrategyName
                )

                if (!strategyExists) {
                  // Find the strategy definition in demo config
                  const demoStrategy =
                    configData.rag?.data_processing_strategies?.find(
                      (s: any) => s.name === processingStrategyName
                    )

                  if (demoStrategy) {
                    updatedConfig = {
                      ...updatedConfig,
                      rag: {
                        ...updatedConfig.rag,
                        data_processing_strategies: [
                          ...(updatedConfig.rag?.data_processing_strategies ||
                            []),
                          demoStrategy,
                        ],
                      },
                    }
                    needsUpdate = true
                  }
                }

                // Check and add database if needed
                const existingDatabases =
                  currentProjectConfig.rag?.databases || []
                const databaseExists = existingDatabases.some(
                  (db: any) => db.name === database
                )

                if (!databaseExists && database !== 'default') {
                  // Find the database definition in demo config
                  const demoDatabase = configData.rag?.databases?.find(
                    (db: any) => db.name === database
                  )

                  if (demoDatabase) {
                    updatedConfig = {
                      ...updatedConfig,
                      rag: {
                        ...updatedConfig.rag,
                        databases: [
                          ...(updatedConfig.rag?.databases || []),
                          demoDatabase,
                        ],
                      },
                    }
                    needsUpdate = true
                  }
                }

                // Update project config if we added anything
                if (needsUpdate) {
                  // Also set the demo's database as the default RAG database
                  // so test chat queries the right database
                  if (database !== 'default') {
                    updatedConfig = {
                      ...updatedConfig,
                      rag: {
                        ...updatedConfig.rag,
                        default_database: database,
                      },
                    }
                  }

                  await projectService.updateProject(
                    activeProject.namespace,
                    activeProject.project,
                    { config: updatedConfig }
                  )

                  // Refetch project to get updated config
                  await refetchDatasets()
                }
              }

              // Step 3: Create dataset with demo's processing strategy
              await createDatasetMutation.mutateAsync({
                namespace: activeProject.namespace,
                project: activeProject.project,
                name: name,
                data_processing_strategy: processingStrategyName,
                database: database,
              })

              // Step 4: Fetch and upload demo files
              toast({
                message: `Uploading ${demo.files.length} file(s)...`,
                variant: 'default',
              })

              for (const demoFile of demo.files) {
                const fileResponse = await fetch(demoFile.path)
                if (!fileResponse.ok) {
                  throw new Error(`Failed to fetch file: ${demoFile.filename}`)
                }
                const fileBlob = await fileResponse.blob()
                const file = new File([fileBlob], demoFile.filename, {
                  type: demoFile.type,
                })

                await uploadFileToDataset(
                  activeProject.namespace,
                  activeProject.project,
                  name,
                  file
                )
              }

              // Step 4.5: Check and setup Ollama embedding model
              toast({
                message: 'Checking Ollama setup...',
                variant: 'default',
              })

              try {
                // Check if Ollama is running
                const ollamaHealthResponse = await fetch(
                  'http://localhost:11434'
                )
                if (!ollamaHealthResponse.ok) {
                  throw new Error('Ollama not responding')
                }

                // Extract embedding model from demo database config
                const demoDatabase = configData.rag?.databases?.find(
                  (db: any) => db.name === database
                )
                const embeddingStrategy =
                  demoDatabase?.embedding_strategies?.[0]
                const embeddingModel = embeddingStrategy?.config?.model

                if (embeddingModel) {
                  // Check if model exists
                  const tagsResponse = await fetch(
                    'http://localhost:11434/api/tags'
                  )
                  const tagsData = await tagsResponse.json()
                  const modelExists = tagsData.models?.some(
                    (m: any) =>
                      m.name === embeddingModel ||
                      m.name.startsWith(embeddingModel + ':')
                  )

                  if (!modelExists) {
                    toast({
                      message: `Pulling embedding model "${embeddingModel}"... (this may take a minute)`,
                      variant: 'default',
                    })

                    // Pull the model
                    const pullResponse = await fetch(
                      'http://localhost:11434/api/pull',
                      {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ name: embeddingModel }),
                      }
                    )

                    if (!pullResponse.ok) {
                      throw new Error(`Failed to pull model: ${embeddingModel}`)
                    }

                    // Stream the pull progress
                    const reader = pullResponse.body?.getReader()
                    const decoder = new TextDecoder()

                    if (reader) {
                      let lastStatus = ''
                      while (true) {
                        const { done, value } = await reader.read()
                        if (done) break

                        const chunk = decoder.decode(value)
                        const lines = chunk
                          .split('\n')
                          .filter(line => line.trim())

                        for (const line of lines) {
                          try {
                            const json = JSON.parse(line)
                            if (json.status && json.status !== lastStatus) {
                              lastStatus = json.status
                              if (
                                json.status.includes('pulling') ||
                                json.status.includes('downloading')
                              ) {
                                const percent =
                                  json.completed && json.total
                                    ? Math.round(
                                        (json.completed / json.total) * 100
                                      )
                                    : null
                                toast({
                                  message: percent
                                    ? `Pulling ${embeddingModel}: ${percent}%`
                                    : `Pulling ${embeddingModel}...`,
                                  variant: 'default',
                                })
                              }
                            }
                          } catch (e) {
                            // Ignore JSON parse errors
                          }
                        }
                      }
                    }

                    toast({
                      message: `Model "${embeddingModel}" ready!`,
                      variant: 'default',
                    })
                  } else {
                    toast({
                      message: `Embedding model "${embeddingModel}" already available`,
                      variant: 'default',
                    })
                  }
                }
              } catch (ollamaError) {
                console.warn('Ollama check failed:', ollamaError)
                toast({
                  message:
                    'Warning: Could not verify Ollama setup. Embeddings may not work properly.',
                  variant: 'default',
                })
              }

              // Step 5: Process dataset and wait for completion
              toast({
                message: `Processing dataset "${name}"...`,
                variant: 'default',
              })

              const processResult = await datasetService.executeDatasetAction(
                activeProject.namespace,
                activeProject.project,
                name,
                { action_type: 'process' }
              )

              // Poll for completion if we got a task ID
              if (processResult.task_id) {
                // Save task ID to localStorage so dataset page can track it
                saveDatasetTaskId(
                  activeProject.namespace,
                  activeProject.project,
                  name,
                  processResult.task_id
                )

                let completed = false
                let attempts = 0
                const maxAttempts = 60 // 2 minutes max

                while (!completed && attempts < maxAttempts) {
                  await new Promise(resolve => setTimeout(resolve, 2000))
                  attempts++

                  const taskStatus = await datasetService.getTaskStatus(
                    activeProject.namespace,
                    activeProject.project,
                    processResult.task_id
                  )

                  if (taskStatus.state === 'SUCCESS') {
                    completed = true

                    toast({
                      message: `Demo dataset "${name}" imported and processed successfully!`,
                      variant: 'default',
                    })

                    // Save processing result to localStorage so dataset page shows processed status
                    if (taskStatus.result) {
                      saveDatasetResult(
                        activeProject.namespace,
                        activeProject.project,
                        name,
                        taskStatus.result
                      )
                    }
                  } else if (taskStatus.state === 'FAILURE') {
                    throw new Error(
                      taskStatus.error || 'Dataset processing failed'
                    )
                  }
                  // Still processing, continue polling...
                }

                if (!completed) {
                  toast({
                    message: `Dataset "${name}" imported, but processing is taking longer than expected. Check back in a moment.`,
                    variant: 'default',
                  })
                }
              } else {
                toast({
                  message: `Demo dataset "${name}" imported successfully!`,
                  variant: 'default',
                })
              }

              // Refetch datasets and wait a moment for localStorage to persist
              await refetchDatasets()
              await new Promise(resolve => setTimeout(resolve, 100))

              navigate(`/chat/data/${name}`)
            } else {
              // Handle example import (old flow)
              await importExampleDataset.mutateAsync({
                exampleId: sourceProjectId,
                namespace: activeProject.namespace,
                project: activeProject.project,
                dataset: name,
                include_strategies: true,
                process: true,
              })
              toast({
                message: `Dataset "${name}" importing…`,
                variant: 'default',
              })
              setIsImportOpen(false)
              navigate(`/chat/data/${name}`)
            }
          } catch (error: any) {
            console.error('Import failed', error)
            const serverMessage =
              (error?.response?.data?.detail as string) ||
              (error?.message as string) ||
              'Unknown error'
            toast({
              message: `Failed to import dataset: ${serverMessage}`,
              variant: 'destructive',
            })
          }
        }}
      />

      {/* Edit dataset dialog */}
      <Dialog open={isConfirmDeleteOpen} onOpenChange={setIsConfirmDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete dataset</DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            Are you sure you want to delete this{' '}
            {confirmDeleteName ? `"${confirmDeleteName}"` : 'dataset'}?
          </div>
          <div className="mt-4 flex items-center justify-end gap-2">
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button
              variant="destructive"
              onClick={async () => {
                const id = confirmDeleteId
                setIsConfirmDeleteOpen(false)
                if (!id) return
                if (!activeProject?.namespace || !activeProject?.project) {
                  toast({
                    message: 'No active project selected',
                    variant: 'destructive',
                  })
                  return
                }
                try {
                  await deleteDatasetMutation.mutateAsync({
                    namespace: activeProject.namespace,
                    project: activeProject.project,
                    dataset: id,
                  })
                  toast({ message: 'Dataset deleted', variant: 'default' })
                } catch (err) {
                  console.error('Delete failed', err)
                  toast({
                    message: 'Failed to delete dataset',
                    variant: 'destructive',
                  })
                }
              }}
            >
              Delete
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Strategy Modal */}
      <Dialog open={strategyEditOpen} onOpenChange={setStrategyEditOpen}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Edit strategy
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3 pt-1">
            <div>
              <label className="text-xs text-muted-foreground">
                Strategy name
              </label>
              <input
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Enter name"
                value={strategyEditName}
                onChange={e => setStrategyEditName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Add a brief description"
                value={strategyEditDescription}
                onChange={e => setStrategyEditDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="flex items-center justify-between gap-2">
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={() => {
                if (!strategyEditId) return

                const strategy = displayStrategies.find(
                  s => s.id === strategyEditId
                )
                if (!strategy || strategy.isDefault) {
                  toast({
                    message: strategy?.isDefault
                      ? 'Cannot delete default strategy'
                      : 'Strategy not found',
                    variant: 'destructive',
                  })
                  return
                }

                const ok = confirm(
                  'Are you sure you want to delete this strategy?'
                )
                if (!ok) return

                const projectConfig = (projectResp as any)?.project?.config
                if (!projectConfig || !strategy.configName) {
                  toast({
                    message: 'Unable to delete strategy',
                    variant: 'destructive',
                  })
                  return
                }

                strategies.deleteStrategy.mutate(
                  {
                    strategyName: strategy.configName,
                    projectConfig,
                  },
                  {
                    onSuccess: () => {
                      setStrategyEditOpen(false)
                      toast({ message: 'Strategy deleted', variant: 'default' })
                    },
                    onError: (error: any) => {
                      toast({
                        message: error.message || 'Failed to delete strategy',
                        variant: 'destructive',
                      })
                    },
                  }
                )
              }}
              type="button"
            >
              Delete
            </button>
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setStrategyEditOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm ${strategyEditName.trim().length > 0 ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
                onClick={() => {
                  if (!strategyEditId || strategyEditName.trim().length === 0)
                    return

                  const strategy = displayStrategies.find(
                    s => s.id === strategyEditId
                  )
                  if (!strategy || !strategy.configName) {
                    toast({
                      message: 'Strategy not found',
                      variant: 'destructive',
                    })
                    return
                  }

                  const projectConfig = (projectResp as any)?.project?.config
                  if (!projectConfig) {
                    toast({
                      message: 'Unable to update strategy',
                      variant: 'destructive',
                    })
                    return
                  }

                  // Convert UI name back to snake_case for config
                  const newConfigName = strategyEditName
                    .trim()
                    .toLowerCase()
                    .replace(/\s+/g, '_')

                  strategies.updateStrategy.mutate(
                    {
                      strategyName: strategy.configName,
                      updates: {
                        name: newConfigName,
                        description: strategyEditDescription,
                      },
                      projectConfig,
                    },
                    {
                      onSuccess: () => {
                        setStrategyEditOpen(false)
                        toast({
                          message: 'Strategy updated',
                          variant: 'default',
                        })
                      },
                      onError: (error: any) => {
                        toast({
                          message: error.message || 'Failed to update strategy',
                          variant: 'destructive',
                        })
                      },
                    }
                  )
                }}
                disabled={strategyEditName.trim().length === 0}
                type="button"
              >
                Save
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Create Strategy Modal */}
      <Dialog
        open={strategyCreateOpen}
        onOpenChange={open => {
          setStrategyCreateOpen(open)
          if (!open) {
            setStrategyCreateName('')
            setStrategyCreateDescription('')
            setStrategyCopyFromId('')
          }
        }}
      >
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Create new processing strategy
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3 pt-1">
            <div>
              <label className="text-xs text-muted-foreground">
                Strategy name
              </label>
              <input
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Enter name"
                value={strategyCreateName}
                onChange={e => setStrategyCreateName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground mb-2 block">
                What type of files do you plan on uploading?
              </label>
              <div className="flex flex-wrap gap-2">
                {fileTypeMapping.map(({ type }) => {
                  const isSelected = strategyCreateFileTypes.has(type)
                  return (
                    <button
                      key={type}
                      type="button"
                      onClick={() => {
                        const newSet = new Set(strategyCreateFileTypes)
                        if (isSelected) {
                          newSet.delete(type)
                        } else {
                          newSet.add(type)
                        }
                        setStrategyCreateFileTypes(newSet)
                      }}
                      className={`px-3 py-1.5 rounded-md text-sm border transition-colors ${
                        isSelected
                          ? 'bg-primary text-primary-foreground border-primary'
                          : 'border-input hover:bg-accent hover:text-accent-foreground'
                      }`}
                    >
                      {type}
                    </button>
                  )
                })}
              </div>
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Copy from existing (optional)
              </label>
              <select
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                value={strategyCopyFromId}
                onChange={e => {
                  const v = e.target.value
                  setStrategyCopyFromId(v)
                  const found = displayStrategies.find(x => x.id === v)
                  if (found) {
                    setStrategyCreateDescription(found.description || '')
                    if (strategyCreateName.trim().length === 0) {
                      setStrategyCreateName(`${found.name} (copy)`)
                    }
                  }
                }}
              >
                <option value="">None</option>
                {displayStrategies.map(s => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Add a brief description"
                value={strategyCreateDescription}
                onChange={e => setStrategyCreateDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="flex items-center justify-between gap-2">
            <button
              className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
              onClick={() => {
                setStrategyCreateOpen(false)
                setStrategyCreateName('')
                setStrategyCreateDescription('')
                setStrategyCopyFromId('')
                setStrategyCreateFileTypes(new Set())
              }}
              type="button"
            >
              Cancel
            </button>
            <button
              className={`px-3 py-2 rounded-md text-sm ${strategyCreateName.trim().length > 0 && !strategies.isUpdating ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
              onClick={async () => {
                const displayName = strategyCreateName.trim()
                if (displayName.length === 0) return

                // Convert display name to snake_case for config
                const strategyName = displayName
                  .toLowerCase()
                  .replace(/[^a-z0-9]+/g, '_')
                  .replace(/^_+|_+$/g, '')

                if (!projectResp) {
                  toast({
                    message: 'Project config not loaded',
                    variant: 'destructive',
                  })
                  return
                }

                const projectConfig = (projectResp as any)?.project?.config

                // Get parsers from copyFrom strategy, selected file types, or use defaults
                const copyFrom = displayStrategies.find(
                  s => s.id === strategyCopyFromId
                )
                let parsers: any[] = []
                let extractors: any[] = [] // Always start with no extractors

                if (copyFrom && projectConfig) {
                  // Find the source strategy in config
                  const sourceStrategy =
                    projectConfig.rag?.data_processing_strategies?.find(
                      (s: any) =>
                        s.name ===
                        copyFrom.name.toLowerCase().replace(/[^a-z0-9]+/g, '_')
                    )
                  if (sourceStrategy) {
                    parsers = sourceStrategy.parsers || []
                    // Don't copy extractors - let user add them manually
                  }
                }

                // If no copy source but file types selected, create parsers from file types
                if (parsers.length === 0 && strategyCreateFileTypes.size > 0) {
                  parsers = Array.from(strategyCreateFileTypes).map(
                    fileType => {
                      const mapping = fileTypeMapping.find(
                        m => m.type === fileType
                      )
                      return {
                        type: mapping?.parser || 'PDFParser_LlamaIndex',
                        config: {},
                        file_include_patterns: mapping?.extensions || ['*.pdf'],
                        priority: 50,
                      }
                    }
                  )
                }

                // If no copy source and no file types selected, create with a default parser
                if (parsers.length === 0) {
                  parsers = [
                    {
                      type: 'PDFParser_LlamaIndex',
                      config: {},
                      file_include_patterns: ['*.pdf'],
                      priority: 50,
                    },
                  ]
                }

                try {
                  // Build strategy object, only including valid fields
                  const description =
                    strategyCreateDescription.trim() || displayName
                  const strategy: any = {
                    name: strategyName,
                    parsers,
                  }

                  // Only include description if it meets the 10 character minimum (schema requirement)
                  if (description.length >= 10) {
                    strategy.description = description
                  }

                  // Only include extractors if there are any
                  if (extractors.length > 0) {
                    strategy.extractors = extractors
                  }

                  await strategies.createStrategy.mutateAsync({
                    strategy,
                    projectConfig,
                  })

                  setStrategyCreateOpen(false)
                  setStrategyCreateName('')
                  setStrategyCreateDescription('')
                  setStrategyCopyFromId('')
                  setStrategyCreateFileTypes(new Set())
                  toast({
                    message: 'Strategy created successfully',
                    variant: 'default',
                  })
                } catch (error) {
                  console.error('Failed to create strategy:', error)
                  toast({
                    message:
                      error instanceof Error
                        ? error.message
                        : 'Failed to create strategy',
                    variant: 'destructive',
                  })
                }
              }}
              disabled={
                strategyCreateName.trim().length === 0 || strategies.isUpdating
              }
              type="button"
            >
              {strategies.isUpdating ? 'Creating...' : 'Create'}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* File drop dataset selection modal */}
      {renderSelectDatasetModal()}

      {/* Upload progress overlay */}
      {isUploading && (
        <div className="fixed inset-0 bg-background/80 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-card border rounded-lg p-6 shadow-lg flex flex-col items-center gap-4 max-w-sm">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
            <div className="text-center">
              <div className="font-medium">Uploading Files...</div>
              <div className="text-sm text-muted-foreground mt-1">
                {uploadingFileCount}{' '}
                {uploadingFileCount === 1 ? 'file' : 'files'}
              </div>
            </div>
            <Button
              variant="outline"
              onClick={handleCancelUpload}
              className="mt-2"
            >
              Cancel Upload
            </Button>
          </div>
        </div>
      )}
    </div>
  )
}

export default Data
