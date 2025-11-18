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
import { DemoConfig } from '../config/demos'
import { projectKeys } from './useProjects'

export type DemoStep =
  | 'idle'
  | 'fetching_config'
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
  startDemo: (demo: DemoConfig, namespace: string) => Promise<void>
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
  const [processingResult, setProcessingResult] = useState<ProcessingResult | null>(null)

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
      timestamp: Date.now()
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
    async (demo: DemoConfig, namespace: string) => {
      reset()

      try {
        // Step 1: Fetch demo config (10%)
        updateStep('fetching_config')
        setProgress(10)

        const configCallId = addApiCall({
          method: 'GET',
          endpoint: demo.configPath,
          status: 'pending',
          description: 'Fetching demo configuration'
        })

        const configStart = Date.now()
        const configResponse = await fetch(demo.configPath)
        if (!configResponse.ok) {
          throw new Error(`Failed to fetch config: ${configResponse.statusText}`)
        }
        const configText = await configResponse.text()
        const configData = YAML.parse(configText)

        updateApiCall(configCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - configStart
        })

        setProgress(20)

        // Step 2: Generate unique project name
        // Find highest demo-{n} number
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

        // Step 3: Create project (30%)
        updateStep('creating_project')
        setProgress(30)

        const createCallId = addApiCall({
          method: 'POST',
          endpoint: `/v1/projects/${namespace}`,
          status: 'pending',
          description: `Creating project: ${newProjectName}`
        })

        const createStart = Date.now()
        await projectService.createProject(namespace, {
          name: newProjectName
        })

        updateApiCall(createCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - createStart
        })

        setProgress(40)

        // Step 4: Update project with demo config (50%)
        const updateCallId = addApiCall({
          method: 'PUT',
          endpoint: `/v1/projects/${namespace}/${newProjectName}`,
          status: 'pending',
          description: 'Applying demo configuration'
        })

        const updateStart = Date.now()
        await projectService.updateProject(namespace, newProjectName, {
          config: configData
        })

        updateApiCall(updateCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - updateStart
        })

        setProgress(55)

        // Step 5: Upload files (60-80%)
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
            description: `Uploading ${demoFile.filename} (${i + 1}/${fileCount})`
          })

          const uploadStart = Date.now()

          // Fetch file from public
          const fileResponse = await fetch(demoFile.path)
          if (!fileResponse.ok) {
            throw new Error(`Failed to fetch file: ${demoFile.filename}`)
          }
          const fileBlob = await fileResponse.blob()
          const file = new File([fileBlob], demoFile.filename, { type: demoFile.type })

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
            duration: Date.now() - uploadStart
          })
        }

        setProgress(80)

        // Step 6: Process dataset (90%)
        updateStep('processing_dataset')
        setProgress(90)

        const processCallId = addApiCall({
          method: 'POST',
          endpoint: `/v1/projects/${namespace}/${newProjectName}/datasets/${demo.datasetName}/process`,
          status: 'pending',
          description: 'Processing dataset (embedding & indexing)'
        })

        const processStart = Date.now()
        const processResult = await datasetService.processDataset(
          namespace,
          newProjectName,
          demo.datasetName
        )

        // Poll for completion
        let taskResult: any = null
        if (processResult.task_id) {
          let completed = false
          let attempts = 0
          const maxAttempts = 60 // 2 minutes max

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
              throw new Error('Dataset processing failed')
            }

            // Update progress during processing (90-98%)
            const processingProgress = 90 + Math.min(attempts / maxAttempts * 8, 8)
            setProgress(processingProgress)
            attempts++
          }

          if (!completed) {
            throw new Error('Dataset processing timed out')
          }
        }

        updateApiCall(processCallId, {
          status: 'success',
          statusCode: 200,
          duration: Date.now() - processStart
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
          embedder: null
        })

        // Mark as completed
        updateStep('completed')

        // Set as active project
        localStorage.setItem('activeProject', newProjectName)

        // Invalidate queries
        queryClient.invalidateQueries({ queryKey: projectKeys.list(namespace) })

        // Navigate to test page immediately - modal will stay open over the chat page
        navigate('/chat/test', { state: { fromDemo: true } })

      } catch (err) {
        console.error('Demo creation failed:', err)
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
    navigateToChat
  }
}
