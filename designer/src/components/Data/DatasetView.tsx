import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { validateDatasetNameWithDuplicateCheck } from '../../utils/datasetValidation'
import { getDatabaseColor } from '../../utils/databaseColors'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip'
import SearchInput from '../ui/search-input'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogClose,
  DialogFooter,
} from '../ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '../ui/collapsible'
import { Textarea } from '../ui/textarea'
import { useToast } from '../ui/toast'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProjectSwitchNavigation } from '../../hooks/useProjectSwitchNavigation'
import { useProject } from '../../hooks/useProjects'
import {
  useUploadFileToDataset,
  useProcessDataset,
  useTaskStatus,
  useListDatasets,
  useDeleteDatasetFile,
  useDeleteDataset,
} from '../../hooks/useDatasets'
import {
  DatasetFile,
  ProcessDatasetResponse,
  FileProcessingDetail,
} from '../../types/datasets'
import PageActions from '../common/PageActions'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'
import {
  saveDatasetTaskId,
  loadDatasetTaskId,
  clearDatasetTaskId,
  saveDatasetResult,
  loadDatasetResult,
  clearDatasetResult,
} from '../../utils/datasetStorage'

type Dataset = {
  id: string
  name: string
  lastRun: string | Date
  embedModel: string
  numChunks: number
  processedPercent: number
  version: string
  description?: string
  files?: string[] | DatasetFile[] // Flexible file format from API
}

function DatasetView() {
  const navigate = useNavigate()
  const { datasetId } = useParams()
  const { toast } = useToast()
  const [mode, setMode] = useModeWithReset('designer')

  // Get current active project for API calls
  const activeProject = useActiveProject()
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )

  // Handle automatic navigation when project changes
  useProjectSwitchNavigation()
  const uploadMutation = useUploadFileToDataset()

  // Fetch datasets from API to get file information
  const { data: datasetsResponse, refetch: refetchDatasets } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    {
      enabled: !!activeProject,
      // Reduce stale time to ensure fresh data after uploads
      staleTime: 1000, // 1 second
    }
  )

  // Force refetch on mount to ensure fresh data after navigation from upload
  useEffect(() => {
    if (activeProject?.namespace && activeProject?.project) {
      refetchDatasets()
    }
  }, []) // Empty deps - only run on mount

  // Extract databases from project config for color assignment
  const databases = useMemo(() => {
    const ragDatabases = (projectResp as any)?.project?.config?.rag?.databases
    return Array.isArray(ragDatabases) ? ragDatabases : []
  }, [projectResp])

  // Task tracking state and hooks
  const [currentTaskId, setCurrentTaskId] = useState<string | null>(null)
  const [processingResult, setProcessingResult] =
    useState<ProcessDatasetResponse | null>(null)
  const [processingFailure, setProcessingFailure] = useState<{
    error: string
    timestamp: Date
    taskId: string
  } | null>(null)
  const [isResultsOpen, setIsResultsOpen] = useState(false)

  // Transform async task result from [bool, {...}] format to normalized structure
  const normalizeTaskResult = (rawResult: any): ProcessDatasetResponse => {
    if (!rawResult || !rawResult.details || !Array.isArray(rawResult.details)) {
      return rawResult
    }

    const normalizedDetails: FileProcessingDetail[] = rawResult.details.map(
      (detail: any) => {
        const [success, info] = detail

        // Determine status - prioritize info.status, then check result.status, then infer from success
        const status =
          info.status ||
          info.result?.status ||
          (success ? 'processed' : 'failed')

        // Use filename from result if main filename looks like a hash (64 char SHA)
        const filename =
          info.result?.filename && info.filename?.length === 64
            ? info.result.filename
            : info.filename || ''

        return {
          hash: info.file_hash,
          filename: filename,
          success: success,
          status: status,
          parser: info.parser,
          extractors: info.extractors,
          chunks: info.chunks,
          chunk_size: info.chunk_size,
          embedder: info.embedder,
          error: info.error,
          reason: info.reason,
          stored_count: info.stored_count ?? info.result?.stored_count,
          skipped_count: info.skipped_count ?? info.result?.skipped_count,
        }
      }
    )

    return {
      ...rawResult,
      processed_files: rawResult.processed_files,
      skipped_files: rawResult.skipped_files,
      failed_files: rawResult.failed_files,
      details: normalizedDetails,
    }
  }

  // Load task ID and previous result from storage on mount
  useEffect(() => {
    if (activeProject?.namespace && activeProject?.project && datasetId) {
      // Load task ID from sessionStorage to resume processing if needed
      if (!currentTaskId) {
        const savedTaskId = loadDatasetTaskId(
          activeProject.namespace,
          activeProject.project,
          datasetId
        )
        if (savedTaskId) {
          setCurrentTaskId(savedTaskId)
        }
      }

      // Load previous result from localStorage and recalculate counts
      const savedResult = loadDatasetResult(
        activeProject.namespace,
        activeProject.project,
        datasetId
      )
      if (savedResult) {
        // Normalize if it's raw async task format
        const normalizedResult =
          savedResult.details &&
          Array.isArray(savedResult.details) &&
          savedResult.details[0] &&
          Array.isArray(savedResult.details[0])
            ? normalizeTaskResult(savedResult)
            : savedResult

        setProcessingResult(normalizedResult)
      }
    }
  }, [
    activeProject?.namespace,
    activeProject?.project,
    datasetId,
    currentTaskId,
  ])
  const processMutation = useProcessDataset()
  const deleteFileMutation = useDeleteDatasetFile()
  const deleteDatasetMutation = useDeleteDataset()
  const { data: taskStatus } = useTaskStatus(
    activeProject?.namespace || '',
    activeProject?.project || '',
    currentTaskId,
    { enabled: !!currentTaskId && !!activeProject }
  )

  // Clear task ID from sessionStorage when processing completes
  useEffect(() => {
    if (
      currentTaskId &&
      taskStatus &&
      activeProject?.namespace &&
      activeProject?.project &&
      datasetId
    ) {
      if (taskStatus.state === 'SUCCESS' || taskStatus.state === 'FAILURE') {
        // Clear from sessionStorage
        clearDatasetTaskId(
          activeProject.namespace,
          activeProject.project,
          datasetId
        )
        // Clear local state
        setCurrentTaskId(null)

        // Try to get results - either from result field or construct from meta
        let resultsToSave = taskStatus.result

        // If no result but we have meta.files (on failure), construct result from that
        if (
          !resultsToSave &&
          taskStatus.state === 'FAILURE' &&
          taskStatus.meta?.files
        ) {
          const files = taskStatus.meta.files
          const processedCount = files.filter(
            (f: any) => f.state === 'processed' || f.state === 'success'
          ).length
          const failedCount = files.filter(
            (f: any) => f.state === 'failed' || f.state === 'failure'
          ).length
          const skippedCount = files.filter(
            (f: any) => f.state === 'skipped'
          ).length

          resultsToSave = {
            processed_files: processedCount,
            failed_files: failedCount,
            skipped_files: skippedCount,
            details: files.map((f: any) => ({
              hash: f.file_hash || f.filename,
              filename: f.filename,
              success: f.state === 'processed' || f.state === 'success',
              status: f.state,
              chunks: f.chunks,
              error: f.error,
              reason: f.error,
            })),
          }
        }

        // Save result for display and persist to localStorage (even on failure - preserve partial results)
        if (resultsToSave) {
          setProcessingResult(resultsToSave)
          saveDatasetResult(
            activeProject.namespace,
            activeProject.project,
            datasetId,
            resultsToSave
          )
          // Expand results section when new results arrive (even on failure if there are partial results)
          setIsResultsOpen(true)
        }
        // Refetch datasets to get updated file list
        refetchDatasets()
      }
    }
  }, [
    taskStatus?.state,
    currentTaskId,
    activeProject?.namespace,
    activeProject?.project,
    datasetId,
    taskStatus,
    refetchDatasets,
  ])

  const [dataset, setDataset] = useState<Dataset | null>(null)
  const datasetName = useMemo(
    () => dataset?.name || datasetId || 'dataset',
    [dataset?.name, datasetId]
  )

  const projectConfig = (projectResp as any)?.project?.config as
    | ProjectConfig
    | undefined
  const getDatasetLocation = useCallback(() => {
    const targetName = dataset?.name || datasetId
    if (targetName) {
      return { type: 'dataset' as const, datasetName: targetName }
    }
    return { type: 'datasets' as const }
  }, [dataset?.name, datasetId])
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getDatasetLocation,
  })

  // Get current dataset from API response
  const currentApiDataset = useMemo(() => {
    if (!datasetsResponse?.datasets || !datasetId) return null
    return datasetsResponse.datasets.find(d => d.name === datasetId)
  }, [datasetsResponse, datasetId])

  // Files from API data only
  const files = useMemo(() => {
    const filesSource: any[] = Array.isArray(
      (currentApiDataset as any)?.details?.files_metadata
    )
      ? (currentApiDataset as any).details.files_metadata
      : (currentApiDataset as any)?.files || []
    if (!filesSource) return []
    return filesSource.map((fileObj: any) => {
      // Handle new API response format with file details
      if (
        typeof fileObj === 'object' &&
        fileObj !== null &&
        ('original_filename' in fileObj || 'original_file_name' in fileObj)
      ) {
        return {
          id: fileObj.hash,
          name: fileObj.original_filename || fileObj.original_file_name,
          size: fileObj.size,
          lastModified: new Date(fileObj.timestamp * 1000).toLocaleString(),
          type: fileObj.mime_type,
          fullHash: fileObj.hash, // Store full hash for operations
        }
      }

      // Fallback for legacy format (file hash strings)
      const fileHash =
        typeof fileObj === 'string' ? fileObj : fileObj?.id || fileObj || ''
      const size =
        typeof fileObj === 'object' &&
        fileObj !== null &&
        'size' in fileObj &&
        fileObj.size !== undefined
          ? fileObj.size
          : 'unknown'
      const lastModified =
        typeof fileObj === 'object' &&
        fileObj !== null &&
        'lastModified' in fileObj &&
        fileObj.lastModified !== undefined
          ? fileObj.lastModified
          : 'unknown'
      return {
        id: fileHash,
        name: `${fileHash.substring(0, 12)}...${fileHash.substring(fileHash.length - 8)}`, // Show first 12 and last 8 chars
        size,
        lastModified,
        type: 'unknown',
        fullHash: fileHash, // Store full hash for operations
      }
    })
  }, [currentApiDataset])

  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [editNameError, setEditNameError] = useState<string | null>(null)
  const [showDeleteConfirmation, setShowDeleteConfirmation] = useState(false)
  const [pendingDeleteFileHash, setPendingDeleteFileHash] = useState<
    string | null
  >(null)
  const [showDeleteFileConfirmation, setShowDeleteFileConfirmation] =
    useState(false)
  const [copyStatus, setCopyStatus] = useState<{
    [id: string]: string | undefined
  }>({})
  const [searchValue, setSearchValue] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const filteredFileInputRef = useRef<HTMLInputElement>(null)
  const directoryInputRef = useRef<HTMLInputElement>(null)

  // Set directory attributes on mount
  useEffect(() => {
    const input = directoryInputRef.current
    if (input) {
      input.setAttribute('webkitdirectory', '')
      input.setAttribute('directory', '')
      input.setAttribute('mozdirectory', '')
    }
  }, [])
  type FileUploadStatus = {
    id: string
    name: string
    status: 'pending' | 'uploading' | 'success' | 'error'
    error?: string
  }
  const [fileUploadStatuses, setFileUploadStatuses] = useState<
    FileUploadStatus[]
  >([])

  // File type filtering - flexible input for any file extension
  const [fileTypeFilter, setFileTypeFilter] = useState<string>('')
  const [showFileTypeFilter, setShowFileTypeFilter] = useState(false)
  const [uploadStats, setUploadStats] = useState<{
    total: number
    filtered: number
    uploading: number
  } | null>(null)

  // Common file type suggestions (non-exhaustive)
  const commonFileTypes = [
    '.pdf',
    '.txt',
    '.md',
    '.csv',
    '.json',
    '.xml',
    '.html',
    '.docx',
    '.xlsx',
    '.pptx',
  ]

  const addFileTypeToFilter = (ext: string) => {
    const current = fileTypeFilter
      .split(',')
      .map(s => s.trim())
      .filter(Boolean)

    if (!current.includes(ext)) {
      const newFilter = [...current, ext].join(', ')
      setFileTypeFilter(newFilter)
    }
  }

  const clearFileTypeFilter = () => {
    setFileTypeFilter('')
  }

  const filterFilesByType = (files: File[]): File[] => {
    if (!fileTypeFilter.trim()) {
      // If no filter specified, accept all files
      return files
    }

    // Parse extensions from comma-separated input
    const extensions = fileTypeFilter
      .split(',')
      .map(ext => ext.trim().toLowerCase())
      .filter(Boolean)
      .map(ext => (ext.startsWith('.') ? ext : `.${ext}`)) // Ensure leading dot

    if (extensions.length === 0) {
      return files
    }

    return files.filter(file => {
      const fileName = file.name.toLowerCase()
      return extensions.some(ext => fileName.endsWith(ext))
    })
  }

  // Drag-and-drop state
  const [isDragging, setIsDragging] = useState(false)
  const [isDropped, setIsDropped] = useState(false)
  const dropZoneRef = useRef<HTMLDivElement>(null)

  // Note: Custom strategies now come from API via project config, not localStorage

  const handleDragEnter = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragOver = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    e.dataTransfer.dropEffect = 'copy'
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    // Only set isDragging to false if we're actually leaving the drop zone
    // and not just entering a child element
    const rect = dropZoneRef.current?.getBoundingClientRect()
    const isLeavingZone =
      rect &&
      (e.clientX <= rect.left ||
        e.clientX >= rect.right ||
        e.clientY <= rect.top ||
        e.clientY >= rect.bottom)
    if (isLeavingZone) {
      setIsDragging(false)
    }
  }, [])

  const handleFilesUpload = useCallback(
    async (list: File[], skipFiltering = false) => {
      if (!datasetId || !activeProject?.namespace || !activeProject?.project) {
        toast({
          message: 'Missing required information for upload',
          variant: 'destructive',
        })
        return
      }
      if (list.length === 0) return

      // Apply file type filtering only if not skipped (e.g., for drag-and-drop)
      const totalFiles = list.length
      const filteredFiles = skipFiltering ? list : filterFilesByType(list)
      const filteredCount = filteredFiles.length

      // Show stats if filtering occurred
      if (!skipFiltering && totalFiles !== filteredCount) {
        setUploadStats({
          total: totalFiles,
          filtered: filteredCount,
          uploading: filteredCount,
        })
        toast({
          message: `Filtered ${filteredCount} of ${totalFiles} files based on file type filter`,
          variant: 'default',
        })
      }

      if (filteredFiles.length === 0) {
        console.error('❌ All files filtered out by file type filter!')
        toast({
          message: 'No files match the selected file type filter',
          variant: 'destructive',
        })
        return
      }

      // Note: Filename stripping for folder uploads is now handled in the API layer
      // (see datasets.ts uploadFileToDataset function)

      // Initialize upload statuses
      const initialStatuses: FileUploadStatus[] = filteredFiles.map(f => ({
        id: `${f.name}:${f.size}:${f.lastModified}`,
        name: f.name.split('/').pop() || f.name, // Display basename only
        status: 'pending',
      }))
      setFileUploadStatuses(initialStatuses)

      const results = await Promise.all(
        filteredFiles.map(async (file, idx) => {
          setFileUploadStatuses(prev =>
            prev.map((s, i) => (i === idx ? { ...s, status: 'uploading' } : s))
          )
          try {
            const response = await uploadMutation.mutateAsync({
              namespace: activeProject.namespace!,
              project: activeProject.project!,
              dataset: datasetId!,
              file,
            })
            setFileUploadStatuses(prev =>
              prev.map((s, i) => (i === idx ? { ...s, status: 'success' } : s))
            )
            return {
              success: true,
              skipped: response.skipped || false,
              fileName: file.name,
            }
          } catch (error: any) {
            console.error(`Failed to upload ${file.name}:`, error)
            toast({
              message: `Failed to upload ${file.name}`,
              variant: 'destructive',
            })
            setFileUploadStatuses(prev =>
              prev.map((s, i) =>
                i === idx
                  ? {
                      ...s,
                      status: 'error',
                      error: error?.message || 'Upload failed',
                    }
                  : s
              )
            )
            return { success: false, skipped: false, fileName: file.name }
          }
        })
      )

      // Count successes, skipped, and failures
      const successCount = results.filter(r => r.success && !r.skipped).length
      const skippedCount = results.filter(r => r.skipped).length
      const failureCount = results.filter(r => !r.success).length

      // Show summary toast
      if (failureCount > 0 && successCount > 0) {
        toast({
          message: `Uploaded ${successCount} file(s)${skippedCount > 0 ? `, skipped ${skippedCount} duplicate(s)` : ''}. ${failureCount} failed.`,
          variant: 'destructive',
        })
      } else if (failureCount > 0) {
        toast({
          message: `Upload failed for all ${failureCount} file(s)`,
          variant: 'destructive',
        })
      } else if (skippedCount > 0 && successCount === 0) {
        toast({
          message: `All ${skippedCount} file(s) were already in the dataset`,
          variant: 'default',
          icon: 'alert-triangle',
        })
      } else if (skippedCount > 0) {
        toast({
          message: `Uploaded ${successCount} file(s), skipped ${skippedCount} duplicate(s)`,
          variant: 'default',
        })
      } else {
        toast({
          message: `Successfully uploaded ${successCount} file(s)`,
          variant: 'default',
        })
      }

      // Refetch datasets to update the file list
      await refetchDatasets()

      // Clear stats after upload completes
      setTimeout(() => setUploadStats(null), 5000)
    },
    [
      datasetId,
      activeProject,
      toast,
      filterFilesByType,
      uploadMutation,
      refetchDatasets,
    ]
  )

  const handleDrop = useCallback(
    async (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDropped(true)
      setIsDragging(false)
      const files = Array.from(e.dataTransfer.files)

      if (files.length === 0) {
        setIsDropped(false)
        return
      }

      // Skip filtering for dropped files - user explicitly selected these files
      await handleFilesUpload(files, true)
      setIsDropped(false)
    },
    [handleFilesUpload]
  )

  // Load dataset metadata from API only - no hardcoded values
  useEffect(() => {
    if (!datasetId) return

    if (currentApiDataset) {
      // Use only real API data
      setDataset({
        id: datasetId,
        name: currentApiDataset.name,
        lastRun: '', // Not available from API
        embedModel: '', // Not available from API
        numChunks: 0, // Not available from API - will be in processing results
        processedPercent: 0, // Not available from API
        version: '', // Not available from API
        description: '',
        files: currentApiDataset.files,
      })
    } else {
      // Minimal fallback
      setDataset({
        id: datasetId,
        name: datasetId,
        lastRun: '',
        embedModel: '',
        numChunks: 0,
        processedPercent: 0,
        version: '',
        description: '',
      })
    }
  }, [datasetId, currentApiDataset])

  // Handle task completion
  useEffect(() => {
    if (!taskStatus) return

    if (taskStatus.state === 'SUCCESS') {
      // Task completed successfully
      setCurrentTaskId(null)
      setProcessingFailure(null) // Clear any previous failures

      // Store the processing result - merge with previous results to maintain history
      if (taskStatus.result) {
        // Normalize the result from async task format
        const normalizedResult = normalizeTaskResult(taskStatus.result)

        setProcessingResult(prevResult => {
          if (!prevResult) {
            // No previous result, just use the new one
            return normalizedResult
          }

          // Merge the new results with the old ones
          const oldDetails = prevResult.details || []
          const newDetails = normalizedResult.details || []

          // Create a map of file hashes to their latest processing details
          const detailsMap = new Map()

          // Add old details first
          oldDetails.forEach((detail: FileProcessingDetail) => {
            detailsMap.set(detail.hash, detail)
          })

          // Override/add with new details
          newDetails.forEach((detail: FileProcessingDetail) => {
            detailsMap.set(detail.hash, detail)
          })

          // Combine into array
          const mergedDetails = Array.from(detailsMap.values())

          // Recalculate counters from the merged details to keep totals accurate
          const processedCount = mergedDetails.filter(
            (d: FileProcessingDetail) =>
              d.status === 'processed' || (d.success && d.status !== 'skipped')
          ).length
          const failedCount = mergedDetails.filter(
            (d: FileProcessingDetail) =>
              d.status === 'failed' || (!d.success && d.status !== 'skipped')
          ).length
          const skippedCount = mergedDetails.filter(
            (d: FileProcessingDetail) => d.status === 'skipped'
          ).length

          return {
            ...normalizedResult,
            processed_files: processedCount,
            failed_files: failedCount,
            skipped_files: skippedCount,
            details: mergedDetails,
          }
        })
      }

      toast({
        message: taskStatus.result?.processed_files
          ? `✅ Processing complete! ${taskStatus.result.processed_files} file(s) processed`
          : 'Dataset processing completed successfully',
        variant: 'default',
      })
    } else if (taskStatus.state === 'FAILURE') {
      // Task failed - but preserve partial results if they exist
      console.error('Task failed:', taskStatus.error, taskStatus.traceback)
      console.log('Full taskStatus on failure:', taskStatus)
      console.log('taskStatus.result:', taskStatus.result)
      setCurrentTaskId(null)

      // Try to extract partial results from either result or meta fields
      let partialResults = taskStatus.result

      // If no result but we have meta.files, construct a result from that
      if (!partialResults && taskStatus.meta?.files) {
        const files = taskStatus.meta.files
        const processedCount = files.filter(
          (f: any) => f.state === 'processed' || f.state === 'success'
        ).length
        const failedCount = files.filter(
          (f: any) => f.state === 'failed' || f.state === 'failure'
        ).length
        const skippedCount = files.filter(
          (f: any) => f.state === 'skipped'
        ).length

        // Build a result object from the meta files
        partialResults = {
          processed_files: processedCount,
          failed_files: failedCount,
          skipped_files: skippedCount,
          details: files.map((f: any) => ({
            hash: f.file_hash || f.filename,
            filename: f.filename,
            success: f.state === 'processed' || f.state === 'success',
            status: f.state,
            chunks: f.chunks,
            error: f.error,
            reason: f.error,
          })),
        }
        console.log('Constructed partial results from meta:', partialResults)
      }

      // Keep partial results to show what succeeded and what failed
      if (partialResults) {
        // Normalize if it's raw async task format (has [bool, {...}] details)
        const normalizedPartialResults =
          partialResults.details &&
          Array.isArray(partialResults.details) &&
          partialResults.details[0] &&
          Array.isArray(partialResults.details[0])
            ? normalizeTaskResult(partialResults)
            : partialResults

        // Merge with previous results to maintain history
        setProcessingResult(prevResult => {
          if (!prevResult) {
            return normalizedPartialResults
          }

          // Merge the new results with the old ones
          const oldDetails = prevResult.details || []
          const newDetails = normalizedPartialResults.details || []

          // Create a map of file hashes to their latest processing details
          const detailsMap = new Map()

          // Add old details first
          oldDetails.forEach((detail: FileProcessingDetail) => {
            detailsMap.set(detail.hash, detail)
          })

          // Override/add with new details
          newDetails.forEach((detail: FileProcessingDetail) => {
            detailsMap.set(detail.hash, detail)
          })

          // Combine into array
          const mergedDetails = Array.from(detailsMap.values())

          return {
            message:
              normalizedPartialResults?.message ||
              prevResult.message ||
              'Processing complete',
            processed_files: normalizedPartialResults.processed_files,
            failed_files: normalizedPartialResults.failed_files,
            skipped_files: normalizedPartialResults.skipped_files,
            strategy:
              normalizedPartialResults?.strategy || prevResult.strategy || null,
            database:
              normalizedPartialResults?.database || prevResult.database || null,
            task_id:
              normalizedPartialResults?.task_id || prevResult.task_id || null,
            details: mergedDetails,
          }
        })
      }

      const errorMessage = taskStatus.error || 'Unknown error occurred'
      const hasPartialResults =
        partialResults &&
        (partialResults.processed_files > 0 ||
          partialResults.skipped_files > 0 ||
          partialResults.failed_files > 0)

      // Only set failure state if there are NO partial results (complete failure)
      // If there are partial results, just show them in the results grid
      if (!hasPartialResults) {
        setProcessingFailure({
          error: errorMessage,
          timestamp: new Date(),
          taskId: currentTaskId || 'unknown',
        })
      } else {
        // Clear any previous failures since we have partial results to show
        setProcessingFailure(null)
      }

      // Auto-open the results section to show the failure or partial results
      setIsResultsOpen(true)

      toast({
        message:
          hasPartialResults && partialResults
            ? `⚠️ Processing completed with ${partialResults.failed_files} error(s). ${partialResults.processed_files} file(s) processed successfully.`
            : `❌ Processing failed: ${errorMessage}`,
        variant: hasPartialResults ? 'default' : 'destructive',
      })
    }
  }, [
    taskStatus?.state,
    taskStatus?.error,
    taskStatus?.result,
    toast,
    currentTaskId,
  ])

  const openEdit = () => {
    setEditName(dataset?.name ?? '')
    setEditDescription(dataset?.description ?? '')
    setEditNameError(null)
    setIsEditOpen(true)
  }

  const handleSaveEdit = () => {
    if (!dataset || !datasetId) return

    // Validate dataset name
    const existingDatasetNames =
      datasetsResponse?.datasets?.map((d: any) => d.name) || []
    const validation = validateDatasetNameWithDuplicateCheck(
      editName,
      existingDatasetNames,
      dataset.name // Allow keeping the same name
    )

    if (!validation.isValid) {
      setEditNameError(validation.error || 'Invalid dataset name')
      return
    }

    // Note: Dataset name/description updates are local-only until backend supports PATCH endpoint
    const updatedDataset = {
      ...dataset,
      name: editName.trim() || dataset.name,
      description: editDescription,
    }
    setDataset(updatedDataset)
    setEditNameError(null)
    setIsEditOpen(false)
  }

  const handleDeleteClick = () => {
    setShowDeleteConfirmation(true)
  }

  const handleConfirmDelete = async () => {
    if (!activeProject?.namespace || !activeProject?.project || !datasetId)
      return

    try {
      await deleteDatasetMutation.mutateAsync({
        namespace: activeProject.namespace,
        project: activeProject.project,
        dataset: datasetId,
      })

      setShowDeleteConfirmation(false)
      setIsEditOpen(false)

      toast({
        message: 'Dataset deleted successfully',
        variant: 'default',
      })

      // Navigate away since the dataset no longer exists
      navigate('/chat/data')
    } catch (error) {
      console.error('Dataset deletion failed:', error)
      toast({
        message: 'Failed to delete dataset. Please try again.',
        variant: 'destructive',
      })
    }
  }

  const handleCancelDelete = () => {
    setShowDeleteConfirmation(false)
  }

  // Helper function to get processing status for a file
  const getFileProcessingStatus = (fileHash: string | undefined): boolean => {
    if (!fileHash || !processingResult?.details) {
      return false // Not processed if no hash or no processing data
    }

    const fileDetail = processingResult.details.find(
      (detail: FileProcessingDetail) =>
        detail.hash === fileHash || (detail as any).file_hash === fileHash
    )

    if (!fileDetail) {
      return false // Not processed if not in results
    }

    // Check if file was successfully processed or skipped
    const isSkipped = fileDetail.status === 'skipped'
    const isProcessed = fileDetail.success === true

    return isProcessed || isSkipped
  }

  const handleDeleteFile = (fileHash: string) => {
    if (!activeProject?.namespace || !activeProject?.project || !datasetId)
      return
    setPendingDeleteFileHash(fileHash)
    setShowDeleteFileConfirmation(true)
  }

  const handleConfirmDeleteFile = async () => {
    if (
      !activeProject?.namespace ||
      !activeProject?.project ||
      !datasetId ||
      !pendingDeleteFileHash
    ) {
      setShowDeleteFileConfirmation(false)
      setPendingDeleteFileHash(null)
      return
    }

    try {
      await deleteFileMutation.mutateAsync({
        namespace: activeProject.namespace,
        project: activeProject.project,
        dataset: datasetId,
        fileHash: pendingDeleteFileHash,
        removeFromDisk: true,
      })

      toast({
        message: 'File deleted successfully',
        variant: 'default',
      })
    } catch (error) {
      console.error('File deletion failed:', error)
      toast({
        message: 'Failed to delete file. Please try again.',
        variant: 'destructive',
      })
    } finally {
      setShowDeleteFileConfirmation(false)
      setPendingDeleteFileHash(null)
    }
  }

  const handleCancelDeleteFile = () => {
    setShowDeleteFileConfirmation(false)
    setPendingDeleteFileHash(null)
  }

  // Helper function to check if a specific file is being deleted
  const isFileDeleting = (fileHash: string) => {
    return (
      deleteFileMutation.isPending &&
      deleteFileMutation.variables?.fileHash === fileHash
    )
  }

  // Processing strategy comes from API dataset response
  const currentStrategy =
    (currentApiDataset as any)?.data_processing_strategy ||
    'universal_processor'

  // Note: Parsers/extractors info now comes from project config API, not localStorage
  // Removed unused derived values

  return (
    <div
      ref={dropZoneRef}
      className={`h-full w-full flex flex-col ${mode === 'designer' ? 'gap-3 pb-40' : ''}`}
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {mode === 'designer' ? (
        <div className="flex items-center justify-between mb-3">
          <nav className="text-sm md:text-base flex items-center gap-1.5">
            <button
              className="text-teal-600 dark:text-teal-400 hover:underline"
              onClick={() => navigate('/chat/data')}
            >
              Data
            </button>
            <span className="text-muted-foreground px-1">\</span>
            <span className="text-foreground">{datasetName}</span>
          </nav>
          <PageActions mode={mode} onModeChange={handleModeChange} />
        </div>
      ) : (
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-2xl">Config editor</h2>
          <PageActions mode={mode} onModeChange={handleModeChange} />
        </div>
      )}

      {mode !== 'designer' ? (
        <div className="flex-1 min-h-0 overflow-hidden">
          <ConfigEditor className="h-full" initialPointer={configPointer} />
        </div>
      ) : (
        <div className="flex-1 min-h-0 flex flex-col gap-3">
          {isDragging || isDropped ? (
            <div className="w-full h-full flex flex-col items-center justify-center border border-dashed rounded-lg p-4 gap-2 transition-colors border-input">
              <div className="flex flex-col items-center justify-center gap-4 text-center my-[56px] text-primary">
                {isDropped ? (
                  <>
                    <Loader />
                    <div className="text-xl text-foreground">
                      Uploading files...
                    </div>
                    <p className="max-w-[527px] text-sm text-muted-foreground text-center">
                      Please wait while your files are being uploaded to the
                      dataset.
                    </p>
                  </>
                ) : (
                  <>
                    <FontIcon
                      type="upload"
                      className="w-10 h-10 text-blue-200 dark:text-white"
                    />
                    <div className="text-xl text-foreground">
                      Drop data here
                    </div>
                    <p className="max-w-[527px] text-sm text-muted-foreground text-center mb-10">
                      You can upload PDFs, CSVs, or other documents directly to
                      this dataset.
                    </p>
                  </>
                )}
              </div>
            </div>
          ) : (
            <>
              {/* Combined Overview Card */}
              <div className="rounded-lg border border-border bg-card p-4">
                {/* Header Section */}
                <div className="flex items-start justify-between gap-4">
                  <div className="flex flex-col gap-1 flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <h2 className="text-xl md:text-2xl font-medium">
                        {datasetName}
                      </h2>
                      <button
                        className="p-1 rounded-md hover:bg-accent text-muted-foreground"
                        onClick={openEdit}
                        aria-label="Edit dataset"
                        title="Edit dataset"
                      >
                        <FontIcon type="edit" className="w-4 h-4" />
                      </button>
                    </div>
                    <p className="text-xs text-muted-foreground max-w-[640px] mb-3">
                      {dataset?.description &&
                      dataset.description.trim().length > 0
                        ? dataset.description
                        : 'Add a short description so teammates know what this dataset is for.'}
                    </p>

                    {/* Configuration - Horizontal Stack */}
                    <div className="flex items-center gap-4 flex-wrap">
                      {/* Database */}
                      {(currentApiDataset as any)?.database && (
                        <div className="flex items-center gap-2">
                          <span className="text-sm text-muted-foreground">
                            Database:
                          </span>
                          <Badge
                            variant="default"
                            size="sm"
                            className={`rounded-xl ${getDatabaseColor((currentApiDataset as any).database, databases)} cursor-pointer hover:opacity-80 transition-opacity`}
                            onClick={() => {
                              // Navigate with database query parameter for reliable tab selection
                              const databaseName = (currentApiDataset as any)
                                .database
                              navigate(
                                `/chat/databases?database=${encodeURIComponent(databaseName)}`
                              )
                            }}
                          >
                            {(currentApiDataset as any).database}
                          </Badge>
                        </div>
                      )}

                      {/* Processing Strategy */}
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          Strategy:
                        </span>
                        <Badge
                          variant="default"
                          size="sm"
                          className="rounded-xl bg-muted text-foreground dark:bg-muted dark:text-foreground cursor-pointer hover:opacity-80 transition-opacity"
                          onClick={() =>
                            navigate(`/chat/data/strategies/${currentStrategy}`)
                          }
                        >
                          {currentStrategy}
                        </Badge>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Processing Results */}
              {(processingResult || processingFailure) && (
                <section className="rounded-lg border border-border bg-card">
                  <Collapsible
                    open={isResultsOpen}
                    onOpenChange={setIsResultsOpen}
                  >
                    <CollapsibleTrigger asChild>
                      <div
                        role="button"
                        tabIndex={0}
                        onKeyDown={e => {
                          if (e.key === 'Enter' || e.key === ' ') {
                            e.preventDefault()
                            setIsResultsOpen(!isResultsOpen)
                          }
                        }}
                        className={`flex items-center justify-between px-4 cursor-pointer hover:opacity-70 transition-opacity ${isResultsOpen ? 'pt-4 pb-3' : 'py-4'}`}
                      >
                        <div className="flex items-center gap-2 text-sm font-medium">
                          <FontIcon
                            type="chevron-down"
                            className={`w-4 h-4 flex-shrink-0 transition-transform ${
                              isResultsOpen ? '' : '-rotate-90'
                            }`}
                          />
                          Last Processing Results
                        </div>
                      </div>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      {/* Show failure state if present and no partial results */}
                      {processingFailure && !processingResult && (
                        <div className="px-4 pb-4">
                          <div className="rounded-md border border-red-600/50 dark:border-red-400/50 bg-red-50/50 dark:bg-red-950/20 p-4">
                            <div className="flex items-start gap-3">
                              <FontIcon
                                type="close"
                                className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5"
                              />
                              <div className="flex-1 min-w-0">
                                <div className="text-sm font-semibold text-red-900 dark:text-red-100 mb-1">
                                  Processing Failed
                                </div>
                                <div className="text-sm text-red-800 dark:text-red-200 mb-2">
                                  {processingFailure.error}
                                </div>
                                <div className="text-xs text-red-700 dark:text-red-300">
                                  Failed at{' '}
                                  {processingFailure.timestamp.toLocaleString()}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}

                      {/* Show results grid if we have partial or complete results */}
                      {processingResult && (
                        <>
                          <div className="grid grid-cols-1 md:grid-cols-3 gap-2 mb-3 px-4">
                            <div className="rounded-md border border-border px-3 py-2">
                              <div className="text-xs text-muted-foreground mb-0.5">
                                Processed Files
                              </div>
                              <div className="text-xl font-semibold text-green-600">
                                {processingResult.processed_files || 0}
                              </div>
                            </div>
                            <div className="rounded-md border border-border px-3 py-2">
                              <div className="text-xs text-muted-foreground mb-0.5">
                                Skipped Files
                              </div>
                              <div className="text-xl font-semibold text-yellow-600">
                                {processingResult.skipped_files || 0}
                              </div>
                            </div>
                            <div className="rounded-md border border-border px-3 py-2">
                              <div className="text-xs text-muted-foreground mb-0.5">
                                Failed Files
                              </div>
                              <div className="text-xl font-semibold text-red-600 dark:text-red-400">
                                {processingResult.failed_files || 0}
                              </div>
                            </div>
                          </div>
                          {processingResult.details &&
                            processingResult.details.length > 0 && (
                              <div className="mt-3 px-4 pb-4">
                                <div className="text-xs font-medium text-muted-foreground mb-2">
                                  File Processing Details
                                </div>
                                <div className="rounded-md border border-border max-h-96 overflow-auto">
                                  {processingResult.details.map(
                                    (
                                      fileResult: FileProcessingDetail,
                                      idx: number
                                    ) => {
                                      const isSkipped =
                                        fileResult.status === 'skipped'
                                      const isFailed = !fileResult.success
                                      const isSuccess =
                                        fileResult.success && !isSkipped

                                      // Get filename or fall back to hash
                                      const displayFilename =
                                        fileResult.filename || fileResult.hash
                                      // Show hash when we have both a real filename AND a different has

                                      // Get chunks information
                                      const totalChunks = fileResult.chunks || 0
                                      const storedChunks =
                                        fileResult.stored_count ?? 0
                                      const skippedChunks =
                                        fileResult.skipped_count ?? 0

                                      return (
                                        <div
                                          key={idx}
                                          className="px-3 py-2.5 border-b last:border-b-0 hover:bg-muted/30 transition-colors flex gap-3"
                                        >
                                          {/* Status icon column - spans full height */}
                                          <div className="flex-shrink-0 w-4 flex items-start pt-0.5">
                                            {isSuccess && (
                                              <FontIcon
                                                type="checkmark-filled"
                                                className="w-4 h-4 text-green-600"
                                              />
                                            )}
                                            {isSkipped && (
                                              <div className="w-4 h-4 rounded-full bg-muted border border-border flex items-center justify-center">
                                                <span className="text-foreground text-[10px] font-bold">
                                                  !
                                                </span>
                                              </div>
                                            )}
                                            {isFailed && (
                                              <FontIcon
                                                type="close"
                                                className="w-4 h-4 text-red-600 dark:text-red-400"
                                              />
                                            )}
                                          </div>

                                          {/* Content column */}
                                          <div className="flex-1 min-w-0 space-y-1.5">
                                            {/* File header with status */}
                                            <div className="flex items-start justify-between gap-3">
                                              {/* Filename */}
                                              <div className="flex flex-col flex-1 min-w-0 gap-1">
                                                <span className="text-sm font-medium truncate">
                                                  {displayFilename}
                                                </span>
                                                <TooltipProvider>
                                                  <Tooltip>
                                                    <TooltipTrigger asChild>
                                                      <span className="text-xs text-muted-foreground text-blue-600 text-mono">
                                                        {fileResult.hash}
                                                      </span>
                                                    </TooltipTrigger>
                                                    <TooltipContent>
                                                      <p className="font-mono text-xs">
                                                        {fileResult.hash}
                                                      </p>
                                                    </TooltipContent>
                                                  </Tooltip>
                                                </TooltipProvider>
                                              </div>

                                              {/* Status badge */}
                                              <Badge
                                                variant={
                                                  isSuccess
                                                    ? 'default'
                                                    : isSkipped
                                                      ? 'secondary'
                                                      : 'outline'
                                                }
                                                size="sm"
                                                className="rounded-xl flex-shrink-0 font-medium"
                                              >
                                                {isSuccess && 'SUCCESS'}
                                                {isSkipped &&
                                                  `SKIPPED${fileResult.reason ? ` (${fileResult.reason})` : ''}`}
                                                {isFailed && 'FAILED'}
                                              </Badge>
                                            </div>

                                            {/* Processing stats - condensed */}
                                            <div className="space-y-1.5 text-xs">
                                              {/* Chunks info with reason inline */}
                                              {totalChunks > 0 && (
                                                <div className="flex items-center justify-between gap-3">
                                                  <div className="flex items-center gap-3 flex-wrap">
                                                    <div className="flex items-center gap-1.5">
                                                      <span className="font-semibold text-foreground">
                                                        {totalChunks}
                                                      </span>
                                                      <span className="text-muted-foreground">
                                                        chunk
                                                        {totalChunks !== 1
                                                          ? 's'
                                                          : ''}{' '}
                                                        created
                                                      </span>
                                                    </div>
                                                    {storedChunks > 0 && (
                                                      <div className="flex items-center gap-1 text-green-600 dark:text-green-500">
                                                        <span className="font-semibold">
                                                          {storedChunks}
                                                        </span>
                                                        <span>stored</span>
                                                      </div>
                                                    )}
                                                    {skippedChunks > 0 && (
                                                      <div className="flex items-center gap-1 text-yellow-600 dark:text-yellow-400">
                                                        <span className="font-semibold">
                                                          {skippedChunks}
                                                        </span>
                                                        <span>skipped</span>
                                                      </div>
                                                    )}
                                                  </div>
                                                  {/* Reason inline - only show for failed files (not skipped, since badge already shows reason) */}
                                                  {fileResult.reason &&
                                                    isFailed && (
                                                      <div className="flex items-center gap-1">
                                                        <span className="font-medium text-red-600 dark:text-red-400">
                                                          Reason:
                                                        </span>
                                                        <span className="text-red-600 dark:text-red-400">
                                                          {fileResult.reason}
                                                        </span>
                                                      </div>
                                                    )}
                                                </div>
                                              )}

                                              {/* Processing details - horizontal single row */}
                                              <div className="flex items-center gap-3 flex-wrap text-muted-foreground">
                                                {/* Parser info */}
                                                {fileResult.parser && (
                                                  <div>
                                                    <span className="font-medium text-foreground">
                                                      Parser:
                                                    </span>{' '}
                                                    <span className="font-mono text-xs">
                                                      {fileResult.parser}
                                                    </span>
                                                  </div>
                                                )}

                                                {/* Embedder */}
                                                {fileResult.embedder && (
                                                  <div>
                                                    <span className="font-medium text-foreground">
                                                      Embedder:
                                                    </span>{' '}
                                                    <span className="font-mono text-xs">
                                                      {fileResult.embedder}
                                                    </span>
                                                  </div>
                                                )}

                                                {/* Extractors - inline */}
                                                {fileResult.extractors &&
                                                  fileResult.extractors.length >
                                                    0 && (
                                                    <div className="flex items-center gap-1.5">
                                                      <span className="font-medium text-foreground">
                                                        Extractors:
                                                      </span>
                                                      <div className="inline-flex flex-wrap gap-1">
                                                        {fileResult.extractors.map(
                                                          (
                                                            ext: string,
                                                            i: number
                                                          ) => (
                                                            <Badge
                                                              key={i}
                                                              variant="outline"
                                                              size="sm"
                                                              className="rounded font-mono text-[10px] px-1.5 py-0"
                                                            >
                                                              {ext}
                                                            </Badge>
                                                          )
                                                        )}
                                                      </div>
                                                    </div>
                                                  )}
                                              </div>

                                              {/* Error message for failures - keep on separate line for visibility */}
                                              {isFailed && fileResult.error && (
                                                <div className="mt-1.5 px-2 py-1.5 bg-red-50 dark:bg-red-950/20 border border-red-200 dark:border-red-800 rounded">
                                                  <span className="font-medium text-red-800 dark:text-red-400">
                                                    Error:
                                                  </span>{' '}
                                                  <span className="text-red-700 dark:text-red-400">
                                                    {fileResult.error}
                                                  </span>
                                                </div>
                                              )}
                                            </div>
                                          </div>
                                        </div>
                                      )
                                    }
                                  )}
                                </div>
                              </div>
                            )}
                        </>
                      )}

                      {/* Show failure banner if we have both partial results and a failure */}
                      {processingFailure && processingResult && (
                        <div className="px-4 pb-4">
                          <div className="rounded-md border border-red-600/50 dark:border-red-400/50 bg-red-50/50 dark:bg-red-950/20 p-3">
                            <div className="flex items-start gap-2">
                              <FontIcon
                                type="alert-triangle"
                                className="w-4 h-4 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5"
                              />
                              <div className="flex-1 min-w-0">
                                <div className="text-xs font-semibold text-red-900 dark:text-red-100 mb-0.5">
                                  Completed with errors
                                </div>
                                <div className="text-xs text-red-800 dark:text-red-200">
                                  {processingFailure.error}
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      )}
                    </CollapsibleContent>
                  </Collapsible>
                </section>
              )}

              <Dialog
                open={isEditOpen}
                onOpenChange={open => {
                  setIsEditOpen(open)
                  if (!open) {
                    // Reset confirmation state when modal closes
                    setShowDeleteConfirmation(false)
                  }
                }}
              >
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>
                      {showDeleteConfirmation
                        ? 'Delete Dataset'
                        : 'Edit dataset'}
                    </DialogTitle>
                  </DialogHeader>

                  {showDeleteConfirmation ? (
                    <div className="space-y-4">
                      <div className="text-center">
                        <h3 className="text-lg font-semibold text-red-600">
                          Confirm Deletion
                        </h3>
                        <p className="mt-2 text-sm text-muted-foreground">
                          Are you sure you want to delete the dataset "
                          {datasetName}
                          "?
                        </p>
                        <p className="mt-1 text-xs text-red-500">
                          This action cannot be undone. All files and data will
                          be permanently deleted.
                        </p>
                      </div>

                      <div className="flex gap-3 justify-center">
                        <Button
                          variant="secondary"
                          onClick={handleCancelDelete}
                          disabled={deleteDatasetMutation.isPending}
                        >
                          Cancel
                        </Button>
                        <Button
                          variant="destructive"
                          onClick={handleConfirmDelete}
                          disabled={deleteDatasetMutation.isPending}
                        >
                          {deleteDatasetMutation.isPending ? (
                            <>
                              <span className="inline-block animate-spin mr-2">
                                ⟳
                              </span>
                              Deleting...
                            </>
                          ) : (
                            'Delete Dataset'
                          )}
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <>
                      <div className="flex flex-col gap-3">
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-muted-foreground">
                            Name
                          </label>
                          <Input
                            autoFocus
                            value={editName}
                            onChange={e => {
                              const newName = e.target.value
                              setEditName(newName)

                              // Validate on change for real-time feedback
                              const existingDatasetNames =
                                datasetsResponse?.datasets?.map(
                                  (d: any) => d.name
                                ) || []
                              const validation =
                                validateDatasetNameWithDuplicateCheck(
                                  newName,
                                  existingDatasetNames,
                                  dataset?.name || null // Allow keeping the same name
                                )
                              setEditNameError(
                                validation.isValid
                                  ? null
                                  : validation.error || 'Invalid dataset name'
                              )
                            }}
                            placeholder="Dataset name"
                            className={
                              editNameError ? 'border-destructive' : ''
                            }
                          />
                          {editNameError && (
                            <p className="text-xs text-destructive mt-1">
                              {editNameError}
                            </p>
                          )}
                        </div>
                        <div className="flex flex-col gap-1">
                          <label className="text-xs text-muted-foreground">
                            Description
                          </label>
                          <Textarea
                            value={editDescription}
                            onChange={e => setEditDescription(e.target.value)}
                            placeholder="Optional description"
                            rows={3}
                          />
                        </div>
                      </div>
                      <div className="mt-4 flex flex-col-reverse gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <Button
                          variant="destructive"
                          onClick={handleDeleteClick}
                        >
                          Delete
                        </Button>
                        <div className="flex items-center gap-2">
                          <DialogClose asChild>
                            <Button variant="secondary">Cancel</Button>
                          </DialogClose>
                          <Button
                            onClick={handleSaveEdit}
                            disabled={!editName.trim() || !!editNameError}
                          >
                            Save
                          </Button>
                        </div>
                      </div>
                    </>
                  )}
                </DialogContent>
              </Dialog>

              {/* File deletion confirmation modal */}
              <Dialog
                open={showDeleteFileConfirmation}
                onOpenChange={open => !open && handleCancelDeleteFile()}
              >
                <DialogContent className="sm:max-w-md">
                  <DialogHeader>
                    <DialogTitle className="text-lg text-foreground">
                      Delete File
                    </DialogTitle>
                  </DialogHeader>

                  <div className="flex flex-col gap-3 pt-1">
                    <p className="text-muted-foreground">
                      Are you sure you want to delete this file?
                    </p>
                    <p className="text-sm text-muted-foreground font-mono bg-muted p-2 rounded">
                      {pendingDeleteFileHash?.substring(0, 20)}...
                    </p>
                  </div>

                  <DialogFooter className="flex flex-row items-center justify-end gap-3 sm:justify-end">
                    <Button
                      variant="outline"
                      onClick={handleCancelDeleteFile}
                      disabled={deleteFileMutation.isPending}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="destructive"
                      onClick={handleConfirmDeleteFile}
                      disabled={deleteFileMutation.isPending}
                    >
                      {deleteFileMutation.isPending
                        ? 'Deleting...'
                        : 'Delete File'}
                    </Button>
                  </DialogFooter>
                </DialogContent>
              </Dialog>

              {/* Live Processing Progress - shown while task is running */}
              {currentTaskId && taskStatus && taskStatus.meta?.files && (
                <section className="rounded-lg border border-border bg-card p-4">
                  <div className="flex items-center justify-between mb-3">
                    <h3 className="text-sm font-medium">Processing Progress</h3>
                    <div className="flex items-center gap-2">
                      <Badge
                        variant="secondary"
                        size="sm"
                        className="rounded-xl w-max flex items-center gap-1.5"
                      >
                        {taskStatus.state === 'PENDING' && (
                          <>
                            <div className="w-3 h-3 rounded-full border-2 border-current border-t-transparent animate-spin" />
                            Queued...
                          </>
                        )}
                        {taskStatus.state !== 'PENDING' &&
                          taskStatus.state !== 'SUCCESS' &&
                          taskStatus.state !== 'FAILURE' && (
                            <>
                              <div className="w-3 h-3 rounded-full border-2 border-current border-t-transparent animate-spin" />
                              {taskStatus.meta?.progress
                                ? `Processing ${taskStatus.meta.progress}%`
                                : taskStatus.meta?.current &&
                                    taskStatus.meta?.total
                                  ? `Processing ${Math.round((taskStatus.meta.current / taskStatus.meta.total) * 100)}%`
                                  : 'Processing...'}
                            </>
                          )}
                      </Badge>
                      <Badge
                        variant="secondary"
                        size="sm"
                        className="rounded-xl"
                      >
                        {taskStatus.meta.current || 0} /{' '}
                        {taskStatus.meta.total || 0} files
                      </Badge>
                    </div>
                  </div>
                  <div className="rounded-md border border-border max-h-80 overflow-auto">
                    {taskStatus.meta.files.map((file: any, idx: number) => (
                      <div
                        key={file.task_id || idx}
                        className="p-3 border-b last:border-b-0 text-xs"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            {/* Status icon */}
                            {file.state === 'pending' && (
                              <div className="w-4 h-4 rounded-full border-2 border-muted-foreground/30 flex-shrink-0" />
                            )}
                            {file.state === 'processing' && (
                              <div className="w-4 h-4 rounded-full border-2 border-blue-500 border-t-transparent animate-spin flex-shrink-0" />
                            )}
                            {file.state === 'success' && (
                              <FontIcon
                                type="checkmark-filled"
                                className="w-4 h-4 text-green-600 flex-shrink-0"
                              />
                            )}
                            {file.state === 'failure' && (
                              <FontIcon
                                type="close"
                                className="w-4 h-4 text-red-600 flex-shrink-0"
                              />
                            )}

                            {/* Filename */}
                            <span className="font-mono text-muted-foreground truncate">
                              {file.filename || `File ${idx + 1}`}
                            </span>
                          </div>

                          {/* Status badge */}
                          <Badge
                            variant={
                              file.state === 'success'
                                ? 'default'
                                : file.state === 'failure'
                                  ? 'outline'
                                  : 'secondary'
                            }
                            size="sm"
                            className="rounded-xl flex-shrink-0"
                          >
                            {file.state === 'pending' && 'Queued'}
                            {file.state === 'processing' && 'Processing'}
                            {file.state === 'success' && 'Complete'}
                            {file.state === 'failure' && 'Failed'}
                          </Badge>
                        </div>

                        {/* Additional info */}
                        {file.chunks && (
                          <div className="text-muted-foreground mt-1 ml-6">
                            {file.chunks} chunks created
                          </div>
                        )}
                        {file.error && (
                          <div className="text-red-600 mt-1 ml-6 text-xs">
                            Error: {file.error}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* Processing strategy and Embedding model sections removed per request */}

              {/* Raw data */}
              <section className="rounded-lg border border-border bg-card p-4 mb-40">
                <div className="flex items-center justify-between mb-3">
                  <h3 className="text-sm font-medium">Raw data</h3>
                  <div className="flex items-center gap-2">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button
                          size="sm"
                          variant="secondary"
                          className="gap-1.5"
                        >
                          Upload
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="12"
                            height="12"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            className="opacity-60"
                          >
                            <polyline points="6 9 12 15 18 9"></polyline>
                          </svg>
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end" className="w-[200px]">
                        <DropdownMenuItem
                          onClick={() => fileInputRef.current?.click()}
                        >
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            width="16"
                            height="16"
                            viewBox="0 0 32 32"
                            fill="none"
                            className="mr-2"
                          >
                            <path
                              d="M25.7 9.3L18.7 2.3C18.5 2.1 18.3 2 18 2H8C6.9 2 6 2.9 6 4V28C6 29.1 6.9 30 8 30H24C25.1 30 26 29.1 26 28V10C26 9.7 25.9 9.5 25.7 9.3ZM18 4.4L23.6 10H18V4.4ZM24 28H8V4H16V10C16 11.1 16.9 12 18 12H24V28Z"
                              fill="currentColor"
                            />
                            <path
                              d="M10 22H22V24H10V22ZM10 16H22V18H10V16Z"
                              fill="currentColor"
                            />
                          </svg>
                          Upload files
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() => directoryInputRef.current?.click()}
                        >
                          <FontIcon type="folder" className="w-4 h-4 mr-2" />
                          Upload folder
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={() =>
                            setShowFileTypeFilter(!showFileTypeFilter)
                          }
                        >
                          <FontIcon type="search" className="w-4 h-4 mr-2" />
                          Filtered file search
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                    <DropdownMenu>
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <DropdownMenuTrigger asChild>
                              <Button
                                size="sm"
                                className="gap-1.5"
                                disabled={
                                  processMutation.isPending ||
                                  !!currentTaskId ||
                                  files.length === 0
                                }
                              >
                                {processMutation.isPending
                                  ? 'Starting...'
                                  : currentTaskId && taskStatus
                                    ? 'Processing...'
                                    : 'Process data'}
                                <svg
                                  xmlns="http://www.w3.org/2000/svg"
                                  width="12"
                                  height="12"
                                  viewBox="0 0 24 24"
                                  fill="none"
                                  stroke="currentColor"
                                  strokeWidth="2"
                                  strokeLinecap="round"
                                  strokeLinejoin="round"
                                  className="opacity-60"
                                >
                                  <polyline points="6 9 12 15 18 9"></polyline>
                                </svg>
                              </Button>
                            </DropdownMenuTrigger>
                          </TooltipTrigger>
                          {files.length === 0 && (
                            <TooltipContent>
                              Upload files before processing
                            </TooltipContent>
                          )}
                        </Tooltip>
                      </TooltipProvider>
                      <DropdownMenuContent align="end" className="w-[200px]">
                        <DropdownMenuItem
                          onClick={async () => {
                            if (
                              !datasetId ||
                              !activeProject?.namespace ||
                              !activeProject?.project
                            )
                              return

                            try {
                              // Don't clear previous results - keep them so already processed files still show as processed
                              // Only clear failures since we're starting a new job
                              setProcessingFailure(null)

                              const result = await processMutation.mutateAsync({
                                namespace: activeProject.namespace,
                                project: activeProject.project,
                                dataset: datasetId,
                              })

                              // The process endpoint returns task_id directly
                              if (result.task_id) {
                                setCurrentTaskId(result.task_id)
                                // Save task ID to sessionStorage so it persists across navigation
                                saveDatasetTaskId(
                                  activeProject.namespace,
                                  activeProject.project,
                                  datasetId,
                                  result.task_id
                                )
                                toast({
                                  message: 'Processing new files...',
                                  variant: 'default',
                                })
                              }
                            } catch (error) {
                              console.error(
                                'Failed to start processing:',
                                error
                              )
                              toast({
                                message:
                                  'Failed to start processing. Please try again.',
                                variant: 'destructive',
                              })
                            }
                          }}
                        >
                          Process new files
                        </DropdownMenuItem>
                        <DropdownMenuItem
                          onClick={async () => {
                            if (
                              !datasetId ||
                              !activeProject?.namespace ||
                              !activeProject?.project
                            )
                              return

                            try {
                              // Clear previous results from state and localStorage
                              setProcessingResult(null)
                              clearDatasetResult(
                                activeProject.namespace,
                                activeProject.project,
                                datasetId
                              )

                              const result = await processMutation.mutateAsync({
                                namespace: activeProject.namespace,
                                project: activeProject.project,
                                dataset: datasetId,
                              })

                              // The process endpoint returns task_id directly
                              if (result.task_id) {
                                setCurrentTaskId(result.task_id)
                                // Save task ID to sessionStorage so it persists across navigation
                                saveDatasetTaskId(
                                  activeProject.namespace,
                                  activeProject.project,
                                  datasetId,
                                  result.task_id
                                )
                                toast({
                                  message: 'Reprocessing all files...',
                                  variant: 'default',
                                })
                              }
                            } catch (error) {
                              console.error(
                                'Failed to start processing:',
                                error
                              )
                              toast({
                                message:
                                  'Failed to start processing. Please try again.',
                                variant: 'destructive',
                              })
                            }
                          }}
                        >
                          Reprocess all files
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                </div>

                {/* File type filter section */}
                {showFileTypeFilter && (
                  <div className="mb-3 p-3 bg-muted/20 rounded-md border border-border/60 relative">
                    <button
                      onClick={() => setShowFileTypeFilter(false)}
                      className="absolute top-2 right-2 p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
                      aria-label="Close filter"
                    >
                      <FontIcon type="close" className="w-4 h-4" />
                    </button>
                    <div className="flex flex-col gap-3">
                      <label className="text-xs font-medium text-muted-foreground">
                        File type filter (comma-separated)
                      </label>
                      <div className="flex items-center gap-2">
                        <SearchInput
                          value={fileTypeFilter}
                          onChange={e => setFileTypeFilter(e.target.value)}
                          onClear={clearFileTypeFilter}
                          placeholder="e.g., .pdf, .txt, .docx (leave empty for all files)"
                          className="text-sm"
                        />
                        <Button
                          onClick={() => filteredFileInputRef.current?.click()}
                          className="whitespace-nowrap"
                        >
                          <FontIcon type="search" className="w-4 h-4 mr-2" />
                          Search files with filters
                        </Button>
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        <span className="text-xs text-muted-foreground">
                          Quick add:
                        </span>
                        {commonFileTypes.map(ext => (
                          <button
                            key={ext}
                            onClick={() => addFileTypeToFilter(ext)}
                            className="text-xs px-2 py-0.5 rounded bg-background border border-border hover:bg-accent"
                          >
                            {ext}
                          </button>
                        ))}
                      </div>
                      {fileTypeFilter && (
                        <p className="text-xs text-muted-foreground">
                          Will only upload files matching:{' '}
                          <span className="font-mono font-medium">
                            {fileTypeFilter}
                          </span>
                        </p>
                      )}
                    </div>
                  </div>
                )}

                {/* Upload stats */}
                {uploadStats && (
                  <div className="mb-3 p-2 bg-blue-50 dark:bg-blue-950/20 rounded border border-blue-200 dark:border-blue-800">
                    <p className="text-xs text-blue-700 dark:text-blue-300">
                      Uploading {uploadStats.filtered} of {uploadStats.total}{' '}
                      files (filtered by file type)
                    </p>
                  </div>
                )}

                <input
                  ref={fileInputRef}
                  type="file"
                  className="hidden"
                  multiple
                  onChange={async e => {
                    if (
                      !datasetId ||
                      !activeProject?.namespace ||
                      !activeProject?.project
                    ) {
                      toast({
                        message: 'Missing required information for upload',
                        variant: 'destructive',
                      })
                      return
                    }

                    const list = e.target.files
                      ? Array.from(e.target.files)
                      : []
                    if (list.length === 0) return

                    // Reset input immediately to allow re-selecting same files
                    const target = e.currentTarget
                    try {
                      // Skip filtering for regular file uploads - user explicitly selected these files
                      await handleFilesUpload(list, true)
                    } catch (error) {
                      console.error('File upload error:', error)
                      toast({
                        message:
                          'Failed to upload files. See console for details.',
                        variant: 'destructive',
                      })
                    } finally {
                      // Reset input value after upload completes
                      if (target) {
                        target.value = ''
                      }
                    }
                  }}
                />
                <input
                  ref={filteredFileInputRef}
                  type="file"
                  className="hidden"
                  multiple
                  accept={
                    fileTypeFilter
                      ? fileTypeFilter
                          .split(',')
                          .map(s => s.trim())
                          .filter(Boolean)
                          .map(ext => (ext.startsWith('.') ? ext : `.${ext}`))
                          .join(',')
                      : undefined
                  }
                  onChange={async e => {
                    if (
                      !datasetId ||
                      !activeProject?.namespace ||
                      !activeProject?.project
                    ) {
                      toast({
                        message: 'Missing required information for upload',
                        variant: 'destructive',
                      })
                      return
                    }

                    const list = e.target.files
                      ? Array.from(e.target.files)
                      : []
                    if (list.length === 0) return

                    // Reset input immediately to allow re-selecting same files
                    const target = e.currentTarget
                    try {
                      await handleFilesUpload(list)
                    } catch (error) {
                      console.error('File upload error:', error)
                      toast({
                        message:
                          'Failed to upload files. See console for details.',
                        variant: 'destructive',
                      })
                    } finally {
                      // Reset input value after upload completes
                      if (target) {
                        target.value = ''
                      }
                    }
                  }}
                />
                <input
                  ref={directoryInputRef}
                  type="file"
                  className="hidden"
                  multiple
                  onChange={async e => {
                    if (
                      !datasetId ||
                      !activeProject?.namespace ||
                      !activeProject?.project
                    ) {
                      toast({
                        message: 'Missing required information for upload',
                        variant: 'destructive',
                      })
                      return
                    }

                    const list = e.target.files
                      ? Array.from(e.target.files)
                      : []

                    if (list.length === 0) {
                      toast({
                        message: 'No files selected',
                        variant: 'destructive',
                      })
                      return
                    }

                    // Store reference to avoid null issues after async operation
                    const target = e.currentTarget
                    try {
                      // Skip filtering for folder uploads - user explicitly selected this folder
                      await handleFilesUpload(list, true)
                    } catch (error) {
                      console.error('Folder upload error:', error)
                      toast({
                        message:
                          'Failed to upload folder. See console for details.',
                        variant: 'destructive',
                      })
                    } finally {
                      // Reset input value after upload completes
                      if (target) {
                        target.value = ''
                      }
                    }
                  }}
                />
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-1/2">
                    <SearchInput
                      placeholder="Search raw files"
                      value={searchValue}
                      onChange={e => setSearchValue(e.target.value)}
                      onClear={() => setSearchValue('')}
                    />
                  </div>
                </div>
                <div className="rounded-md border border-input bg-background p-0 text-xs">
                  {files.length === 0 ? (
                    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
                      <FontIcon
                        type="upload"
                        className="w-10 h-10 text-blue-200 dark:text-white mb-4"
                      />
                      <div className="text-base font-medium text-foreground mb-2">
                        Drag and drop files here
                      </div>
                      <div className="text-sm text-muted-foreground mb-4 max-w-[400px]">
                        Upload PDFs, CSVs, or other documents to add them to
                        this dataset
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        Upload files
                      </Button>
                    </div>
                  ) : (
                    <div>
                      <div className="p-3 border-b border-border/60 bg-muted/20">
                        <div className="text-xs font-medium">
                          {files.length} file{files.length !== 1 ? 's' : ''}
                        </div>
                      </div>
                      {(() => {
                        const filteredFiles = files.filter(f =>
                          f.name
                            .toLowerCase()
                            .includes(searchValue.toLowerCase())
                        )

                        if (filteredFiles.length === 0 && searchValue) {
                          return (
                            <div className="flex flex-col items-center justify-center py-8 px-4 text-center">
                              <FontIcon
                                type="search"
                                className="w-8 h-8 text-muted-foreground/40 mb-3"
                              />
                              <div className="text-sm font-medium text-foreground mb-1">
                                No results found
                              </div>
                              <div className="text-xs text-muted-foreground mb-3">
                                No files match "{searchValue}"
                              </div>
                              <button
                                onClick={() => setSearchValue('')}
                                className="text-xs text-blue-600 hover:text-blue-800 dark:text-white dark:hover:text-blue-200"
                              >
                                Clear search
                              </button>
                            </div>
                          )
                        }

                        return (
                          <ul>
                            {filteredFiles.map(f => (
                              <li
                                key={f.id}
                                className="flex items-center justify-between px-3 py-3 border-b last:border-b-0 border-border/60"
                              >
                                <div className="font-mono text-xs text-muted-foreground truncate max-w-[60%] flex flex-col gap-1">
                                  <span>{f.name}</span>
                                  {f.fullHash && (
                                    <button
                                      onClick={async () => {
                                        try {
                                          await navigator.clipboard.writeText(
                                            f.fullHash!
                                          )
                                          setCopyStatus(prev => ({
                                            ...prev,
                                            [f.id]: 'Copied!',
                                          }))
                                        } catch (err) {
                                          setCopyStatus(prev => ({
                                            ...prev,
                                            [f.id]: 'Failed to copy',
                                          }))
                                        }
                                        setTimeout(() => {
                                          setCopyStatus(prev => ({
                                            ...prev,
                                            [f.id]: undefined,
                                          }))
                                        }, 1500)
                                      }}
                                      className={`text-xs text-left ${
                                        copyStatus?.[f.id] === 'Copied!'
                                          ? 'text-green-600'
                                          : copyStatus?.[f.id] ===
                                              'Failed to copy'
                                            ? 'text-red-600'
                                            : 'text-blue-600 hover:text-blue-800'
                                      }`}
                                      title="Click to copy full hash"
                                    >
                                      {copyStatus?.[f.id] || f.fullHash}
                                    </button>
                                  )}
                                </div>
                                <div className="w-1/2 flex items-center justify-between gap-4">
                                  <div className="text-xs flex items-center gap-1.5">
                                    {getFileProcessingStatus(f.fullHash) ? (
                                      <>
                                        <FontIcon
                                          type="checkmark-outline"
                                          className="w-3.5 h-3.5 text-green-600 dark:text-green-400"
                                        />
                                        <span className="text-green-600 dark:text-green-400 font-medium">
                                          Processed
                                        </span>
                                      </>
                                    ) : (
                                      <span className="text-muted-foreground">
                                        Not Processed
                                      </span>
                                    )}
                                  </div>
                                  <div className="flex items-center gap-6">
                                    {fileUploadStatuses.find(s => s.id === f.id)
                                      ?.status === 'uploading' && (
                                      <div className="flex items-center gap-1 text-muted-foreground">
                                        <FontIcon
                                          type="fade"
                                          className="w-4 h-4"
                                        />
                                        <span className="text-xs">
                                          Processing
                                        </span>
                                      </div>
                                    )}
                                    {fileUploadStatuses.find(s => s.id === f.id)
                                      ?.status === 'success' && (
                                      <FontIcon
                                        type="checkmark-outline"
                                        className="w-4 h-4 text-teal-600 dark:text-teal-400"
                                      />
                                    )}
                                    <button
                                      className="w-4 h-4 grid place-items-center text-muted-foreground hover:text-red-600 disabled:opacity-50"
                                      onClick={() =>
                                        f.fullHash &&
                                        handleDeleteFile(f.fullHash)
                                      }
                                      disabled={
                                        f.fullHash
                                          ? isFileDeleting(f.fullHash)
                                          : true
                                      }
                                      aria-label={`Delete ${f.name} from dataset`}
                                      title="Delete file"
                                    >
                                      {f.fullHash &&
                                      isFileDeleting(f.fullHash) ? (
                                        <span className="text-xs">...</span>
                                      ) : (
                                        <FontIcon
                                          type="trashcan"
                                          className="w-4 h-4"
                                        />
                                      )}
                                    </button>
                                  </div>
                                </div>
                              </li>
                            ))}
                          </ul>
                        )
                      })()}
                    </div>
                  )}
                </div>
              </section>

              {/* File deletion now handled directly via API calls with confirmation dialog */}
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default DatasetView
