/**
 * Hook for orchestrating demo project creation workflow
 * Handles: config fetch, project creation, file upload, dataset processing
 */

import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQueryClient } from '@tanstack/react-query'
import YAML from 'yaml'
import projectService from '../api/projectService'
import datasetService from '../api/datasets'
import modelService from '../api/modelService'
import { type FileBasedDemo } from '../config/demos'
import { projectKeys } from './useProjects'
import { setActiveProject } from '../utils/projectUtils'

export type DemoStep =
  | 'idle'
  | 'fetching_config'
  | 'downloading_model'
  | 'creating_project'
  | 'uploading_files'
  | 'processing_dataset'
  | 'completed'
  | 'error'

export interface ApiCall {
  id: string
  timestamp: number
  method: string
  endpoint: string
  status: 'pending' | 'success' | 'error'
  statusCode?: number
  duration?: number
  description: string
}

export interface ProcessingResult {
  totalFiles: number
  totalChunks: number
  parsers: string[]
  embedder: string | null
}

export interface UseDemoWorkflowReturn {
  // State
  currentStep: DemoStep
  lastValidStep: DemoStep
  progress: number
  error: string | null
  apiCalls: ApiCall[]
  projectName: string | null
  processingResult: ProcessingResult | null

  // Actions
  startDemo: (demo: FileBasedDemo, namespace: string) => Promise<void>
  reset: () => void
  navigateToChat: () => void
}

export function useDemoWorkflow(): UseDemoWorkflowReturn {
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [currentStep, setCurrentStep] = useState<DemoStep>('idle')
  const [lastValidStep, setLastValidStep] = useState<DemoStep>('idle')
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const [apiCalls, setApiCalls] = useState<ApiCall[]>([])
  const [projectName, setProjectName] = useState<string | null>(null)
  const [processingResult, setProcessingResult] =
    useState<ProcessingResult | null>(null)

  // Wrapper to update both currentStep and lastValidStep (except for error state)
  const updateStep = (step: DemoStep) => {
    setCurrentStep(step)
    if (step !== 'error') {
      setLastValidStep(step)
    }
  }

  const addApiCall = useCallback((call: Omit<ApiCall, 'id' | 'timestamp'>) => {
    const newCall: ApiCall = {
      ...call,
      id: Math.random().toString(36).substring(7),
      timestamp: Date.now(),
    }
    setApiCalls(prev => [...prev, newCall])
    return newCall.id
  }, [])

  const updateApiCall = useCallback((id: string, updates: Partial<ApiCall>) => {
    setApiCalls(prev =>
      prev.map(call => (call.id === id ? { ...call, ...updates } : call))
    )
  }, [])

  const reset = useCallback(() => {
    setCurrentStep('idle')
    setLastValidStep('idle')
    setProgress(0)
    setError(null)
    setApiCalls([])
    setProjectName(null)
    setProcessingResult(null)
  }, [])

  const navigateToChat = useCallback(() => {
    if (projectName) {
      navigate('/chat/dashboard')
    }
  }, [navigate, projectName])

  const startDemo = useCallback(
    async (demo: FileBasedDemo, namespace: string) => {
      // Always reset state completely before starting
      reset()

      // Force refresh the projects list to get accurate numbering
      await queryClient.invalidateQueries({
        queryKey: projectKeys.list(namespace),
      })

      try {
        // Step 1: Fetch demo config (10%)
        updateStep('fetching_config')
        setProgress(10)

        const configCallId = addApiCall({
          method: 'GET',
          endpoint: demo.configPath,
          status: 'pending',
          description: 'Fetching demo configuration',
        })

        const configStart = Date.now()
        const configResponse = await fetch(demo.configPath)
        if (!configResponse.ok) {
          throw new Error(
            `Failed to fetch config: ${configResponse.statusText}`
          )
        }
        const configText = await configResponse.text()
        const configData = YAML.parse(configText)

        updateApiCall(configCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - configStart,
        })

        setProgress(20)

        // Step 2: Pre-download embedding model if needed
        // This ensures the Celery task doesn't silently download models with no progress visibility
        const embeddingModel = configData.rag?.databases
          ?.flatMap((db: any) => db.embedding_strategies ?? [])
          ?.find((s: any) => s.config?.model)?.config?.model as string | undefined

        if (embeddingModel) {
          // Check if model is already cached
          let modelAlreadyCached = false
          try {
            const cachedModels = await modelService.listCachedModels('universal')
            modelAlreadyCached = cachedModels.data.some(
              m => m.id === embeddingModel || m.name === embeddingModel
            )
          } catch (error) {
            console.warn('Cache check failed, will attempt download:', error)
            // Backend will short-circuit if model is already cached.
          }

          if (!modelAlreadyCached) {
            updateStep('downloading_model')
            setProgress(25)

            const downloadCallId = addApiCall({
              method: 'POST',
              endpoint: '/v1/models/download',
              status: 'pending',
              description: `Downloading embedding model: ${embeddingModel}`,
            })

            const downloadStart = Date.now()

            try {
              let downloadCompleted = false
              let errorMessage: string | undefined
              let indeterminateProgress = 25

              for await (const event of modelService.downloadModel({
                model_name: embeddingModel,
                provider: 'universal',
              })) {
                if (event.event === 'progress') {
                  if (event.total > 0) {
                    const percent = Math.round((event.downloaded / event.total) * 100)
                    // Scale: 25-50% of overall progress (cap at 50 to prevent overflow)
                    setProgress(Math.min(50, 25 + Math.round(percent * 0.25)))
                  } else {
                    // Unknown total size - slowly increment to show activity
                    indeterminateProgress = Math.min(indeterminateProgress + 0.5, 49)
                    setProgress(Math.round(indeterminateProgress))
                  }
                } else if (event.event === 'done') {
                  downloadCompleted = true
                  setProgress(50)
                } else if (event.event === 'error') {
                  errorMessage = event.message
                  break // Exit loop, let post-loop logic handle error
                }
              }

              // Verify download completed successfully
              if (!downloadCompleted) {
                throw new Error(
                  errorMessage
                    ? `Model download failed: ${errorMessage}`
                    : 'Model download stream ended unexpectedly'
                )
              }

              updateApiCall(downloadCallId, {
                status: 'success',
                statusCode: 200,
                duration: Date.now() - downloadStart,
              })
            } catch (err) {
              updateApiCall(downloadCallId, {
                status: 'error',
                duration: Date.now() - downloadStart,
              })
              throw err
            }
          } else {
            setProgress(50)
          }
        } else {
          // No embedding model specified, skip to next step
          setProgress(50)
        }

        // Step 3: Generate unique project name
        // Fetch fresh project list (we invalidated cache at start)
        await new Promise(resolve => setTimeout(resolve, 500)) // Small delay to ensure cache is cleared
        const existingProjects = await projectService.listProjects(namespace)

        const demoProjects = existingProjects.projects
          .map(p => p.name)
          .filter(name => name.startsWith(`${demo.name}-`))

        const numbers = demoProjects
          .map(name => {
            const match = name.match(/-(\d+)$/)
            return match ? parseInt(match[1]) : 0
          })
          .filter(n => !isNaN(n))

        const nextNumber = numbers.length > 0 ? Math.max(...numbers) + 1 : 1
        const newProjectName = `${demo.name}-${nextNumber}`

        setProjectName(newProjectName)

        // Step 4: Create project (50-55%)
        updateStep('creating_project')
        setProgress(52)

        const createCallId = addApiCall({
          method: 'POST',
          endpoint: `/v1/projects/${namespace}`,
          status: 'pending',
          description: `Creating project: ${newProjectName}`,
        })

        const createStart = Date.now()
        await projectService.createProject(namespace, {
          name: newProjectName,
        })

        updateApiCall(createCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - createStart,
        })

        setProgress(55)

        // Step 5: Update project with demo config (55-60%)
        const updateCallId = addApiCall({
          method: 'PUT',
          endpoint: `/v1/projects/${namespace}/${newProjectName}`,
          status: 'pending',
          description: 'Applying demo configuration',
        })

        const updateStart = Date.now()
        await projectService.updateProject(namespace, newProjectName, {
          config: configData,
        })

        updateApiCall(updateCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - updateStart,
        })

        setProgress(60)

        // Step 6: Upload files (60-80%)
        updateStep('uploading_files')

        const fileCount = demo.files.length
        for (let i = 0; i < fileCount; i++) {
          const demoFile = demo.files[i]
          const fileProgress = 60 + (20 / fileCount) * i
          setProgress(fileProgress)

          const uploadCallId = addApiCall({
            method: 'POST',
            endpoint: `/v1/projects/${namespace}/${newProjectName}/datasets/${demo.datasetName}/data`,
            status: 'pending',
            description: `Uploading ${demoFile.filename} (${i + 1}/${fileCount})`,
          })

          const uploadStart = Date.now()

          // Fetch file from public
          const fileResponse = await fetch(demoFile.path)
          if (!fileResponse.ok) {
            throw new Error(`Failed to fetch file: ${demoFile.filename}`)
          }
          const fileBlob = await fileResponse.blob()
          const file = new File([fileBlob], demoFile.filename, {
            type: demoFile.type,
          })

          // Upload to dataset
          await datasetService.uploadFileToDataset(
            namespace,
            newProjectName,
            demo.datasetName,
            file
          )

          updateApiCall(uploadCallId, {
            status: 'success',
            statusCode: 200,
            duration: Date.now() - uploadStart,
          })
        }

        setProgress(80)

        // Small delay to ensure backend is ready (file metadata written, etc.)
        await new Promise(resolve => setTimeout(resolve, 1000))

        // Step 7: Process dataset (80-100%)
        updateStep('processing_dataset')
        setProgress(80)

        const processCallId = addApiCall({
          method: 'POST',
          endpoint: `/v1/projects/${namespace}/${newProjectName}/datasets/${demo.datasetName}/actions`,
          status: 'pending',
          description: 'Processing dataset (embedding & indexing via actions)',
        })

        const processStart = Date.now()
        const processResult = await datasetService.executeDatasetAction(
          namespace,
          newProjectName,
          demo.datasetName,
          { action_type: 'process' }
        )

        // Poll for completion
        let taskResult: any = null
        if (processResult.task_id) {
          let completed = false
          let attempts = 0
          // Model downloads now happen before processing, so this should complete faster
          // Keeping a generous timeout for large document processing
          const maxAttempts = 150 // 5 minutes max (150 * 2s = 300s)

          while (!completed && attempts < maxAttempts) {
            await new Promise(resolve => setTimeout(resolve, 2000))

            const taskStatus = await datasetService.getTaskStatus(
              namespace,
              newProjectName,
              processResult.task_id
            )

            if (taskStatus.state === 'SUCCESS') {
              completed = true
              taskResult = taskStatus.result
            } else if (taskStatus.state === 'FAILURE') {
              // Provide user-friendly error message
              let errorMsg = 'Dataset processing failed'
              if (taskStatus.error) {
                // Check for common error patterns
                if (
                  taskStatus.error.includes('not found') ||
                  taskStatus.error.includes('deleted')
                ) {
                  errorMsg =
                    'Project was deleted or is unavailable. This may be due to a stale background task. Please try again.'
                } else {
                  // Truncate very long errors for display
                  errorMsg =
                    taskStatus.error.length > 200
                      ? taskStatus.error.substring(0, 200) + '...'
                      : taskStatus.error
                }
              }

              // If we have result details with file-specific errors, include those
              if (
                taskStatus.result?.details &&
                Array.isArray(taskStatus.result.details)
              ) {
                const fileErrors = taskStatus.result.details
                  .filter((d: any) => d.error)
                  .map((d: any) => `${d.filename}: ${d.error}`)
                  .join('; ')
                if (fileErrors) {
                  errorMsg += `. File errors: ${fileErrors.substring(0, 300)}`
                }
              }

              throw new Error(errorMsg)
            }

            // Update progress during processing (80-98%)
            // Model downloads now happen before processing, so this should be faster
            const processingProgress =
              80 + Math.min((attempts / maxAttempts) * 18, 18)
            setProgress(processingProgress)
            attempts++
          }

          if (!completed) {
            throw new Error(
              'Dataset processing timed out - the server may still be processing in the background. Please check the RAG page.'
            )
          }
        }

        updateApiCall(processCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - processStart,
        })

        setProgress(100)

        // Extract processing results from task
        // The task returns: { message, namespace, project, dataset, strategy, files, total_files }
        const totalFiles = taskResult?.total_files || demo.files.length
        const strategy = taskResult?.strategy || 'default'

        setProcessingResult({
          totalFiles,
          totalChunks: 0, // Backend doesn't aggregate this yet
          parsers: [strategy],
          embedder: null,
        })

        // Mark as completed
        updateStep('completed')

        // Set as active project (dispatches lf-active-project event)
        setActiveProject(newProjectName)

        // Mark this project as a demo project (for checklist and onboarding logic)
        let storedDemoProjects: string[] = []
        try {
          storedDemoProjects = JSON.parse(localStorage.getItem('lf_demo_projects') || '[]')
        } catch {
          storedDemoProjects = []
        }
        if (!storedDemoProjects.includes(newProjectName)) {
          storedDemoProjects.push(newProjectName)
          localStorage.setItem('lf_demo_projects', JSON.stringify(storedDemoProjects))
        }

        // Invalidate queries
        queryClient.invalidateQueries({ queryKey: projectKeys.list(namespace) })

        // Navigate to test page immediately - modal will stay open over the chat page
        navigate('/chat/test', { state: { fromDemo: true } })
      } catch (err) {
        setCurrentStep('error')
        setError(err instanceof Error ? err.message : 'Unknown error occurred')
      }
    },
    [navigate, queryClient, addApiCall, updateApiCall, reset]
  )

  return {
    currentStep,
    lastValidStep,
    progress,
    error,
    apiCalls,
    projectName,
    processingResult,
    startDemo,
    reset,
    navigateToChat,
  }
}
