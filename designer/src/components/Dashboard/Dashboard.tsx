import FontIcon from '../../common/FontIcon'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import PageActions from '../common/PageActions'
import DataCards from './DataCards'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import { useProjectModalContext } from '../../contexts/ProjectModalContext'
import { useOnboardingContext } from '../../contexts/OnboardingContext'
import { useProject } from '../../hooks/useProjects'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useListDatasets, useUploadFileToDataset, useCreateDataset, useAvailableStrategies } from '../../hooks/useDatasets'
import { useTrainAndSaveClassifier, useTrainAndSaveAnomaly } from '../../hooks/useMLModels'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import { useConfigPointer } from '../../hooks/useConfigPointer'
import type { ProjectConfig } from '../../types/config'
import {
  GettingStartedChecklist,
  OnboardingWizard,
  RestartOnboardingBanner,
  CollapsedChecklist,
} from '../Onboarding'
import { useToast } from '../ui/toast'
import { getDemoById, isModelBasedDemo, isFileBasedDemo, type FileBasedDemo } from '../../config/demos'
import { CLASSIFIER_SAMPLE_DATASETS, ANOMALY_SAMPLE_DATASETS } from '../Models/sampleDatasets'
import { parseNumericTrainingData } from '../../types/ml'

const Dashboard = () => {
  const navigate = useNavigate()
  const { toast } = useToast()
  const activeProject = useActiveProject()

  // All state declarations first
  const [mode, setMode] = useModeWithReset('designer')
  const [showValidationDetails, setShowValidationDetails] = useState(false)
  const [projectName, setProjectName] = useState<string>('Dashboard')
  // Datasets list for Data card
  const { data: apiDatasets, isLoading: isDatasetsLoading, refetch: refetchDatasets } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )

  // Active project details for description and project brief fields
  const { data: projectDetail } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )
  const projectConfig = (projectDetail as any)?.project?.config as ProjectConfig | undefined
  const getRootLocation = useCallback(
    () => ({ type: 'root' as const }),
    []
  )
  const { configPointer, handleModeChange } = useConfigPointer({
    mode,
    setMode,
    config: projectConfig,
    getLocation: getRootLocation,
  })

  const datasets = useMemo(() => {
    // Only return datasets from the API, no localStorage fallback
    if (apiDatasets?.datasets && apiDatasets.datasets.length > 0) {
      return apiDatasets.datasets.map(dataset => ({
        id: dataset.name,
        name: dataset.name,
        lastRun: new Date(),
      }))
    }
    return [] as Array<{ id: string; name: string; lastRun: string | Date }>
  }, [apiDatasets])

  // Calculate dashboard stats
  const filesProcessed = useMemo(() => {
    if (apiDatasets?.datasets && apiDatasets.datasets.length > 0) {
      return apiDatasets.datasets.reduce((sum, dataset) => {
        return sum + (dataset.files?.length || 0)
      }, 0)
    }
    return 0
  }, [apiDatasets])

  const databaseCount = useMemo(() => {
    const databases = projectDetail?.project?.config?.rag?.databases
    return Array.isArray(databases) ? databases.length : 0
  }, [projectDetail])

  const modelsCount = useMemo(() => {
    const models = projectDetail?.project?.config?.runtime?.models
    return Array.isArray(models) ? models.length : 0
  }, [projectDetail])

  // Upload mutation for onboarding files
  const uploadMutation = useUploadFileToDataset()
  const createDatasetMutation = useCreateDataset()

  // Get available strategies and databases for dataset creation
  const { data: strategiesData } = useAvailableStrategies(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject?.namespace && !!activeProject?.project }
  )

  // Shared modal hook
  const projectModal = useProjectModalContext()

  // Onboarding state
  const onboarding = useOnboardingContext()
  const onboardingRef = useRef(onboarding)
  onboardingRef.current = onboarding

  // Determine if we should show onboarding components
  const showWizard = onboarding.state.wizardOpen
  const showChecklist =
    onboarding.state.onboardingCompleted && !onboarding.state.checklistDismissed
  // Show collapsed checklist when onboarding completed but checklist was dismissed
  const showCollapsedChecklist =
    onboarding.state.onboardingCompleted && onboarding.state.checklistDismissed
  // Show restart banner when onboarding was never completed (skipped before finishing)
  const showRestartBanner =
    !onboarding.state.wizardOpen &&
    !onboarding.state.onboardingCompleted &&
    !showChecklist

  // Show loading state while we're about to open the wizard (prevents dashboard flash)
  const isWaitingForWizard =
    !onboarding.state.onboardingCompleted &&
    !onboarding.state.wizardOpen &&
    !onboarding.state.checklistDismissed &&
    !onboarding.isDemo &&
    (isDatasetsLoading || filesProcessed === 0)

  // Auto-open wizard on first visit to an empty project (but NOT for demo projects)
  useEffect(() => {
    // Only auto-open if:
    // 1. Onboarding not completed
    // 2. Wizard not already open
    // 3. No datasets loaded (empty project)
    // 4. Datasets finished loading
    // 5. NOT a demo project (demos skip the wizard entirely)
    if (
      !onboarding.state.onboardingCompleted &&
      !onboarding.state.wizardOpen &&
      !onboarding.state.checklistDismissed &&
      filesProcessed === 0 &&
      !isDatasetsLoading &&
      !onboarding.isDemo
    ) {
      // Small delay to prevent flash
      const timer = setTimeout(() => {
        onboarding.openWizard()
      }, 500)
      return () => clearTimeout(timer)
    }
  }, [
    onboarding.state.onboardingCompleted,
    onboarding.state.wizardOpen,
    onboarding.state.checklistDismissed,
    filesProcessed,
    isDatasetsLoading,
    onboarding.openWizard,
    onboarding.isDemo,
  ])

  // Training mutations for auto-training classifier/anomaly models
  const trainClassifierMutation = useTrainAndSaveClassifier()
  const trainAnomalyMutation = useTrainAndSaveAnomaly()
  const trainClassifierMutationRef = useRef(trainClassifierMutation)
  const trainAnomalyMutationRef = useRef(trainAnomalyMutation)
  trainClassifierMutationRef.current = trainClassifierMutation
  trainAnomalyMutationRef.current = trainAnomalyMutation

  // State for pending model training
  const [pendingModelTraining, setPendingModelTraining] = useState<{
    demoId: string
    modelType: 'classifier' | 'anomaly'
    sampleDataId: string
  } | null>(null)

  // State for pending file-based demo import (RAG/doc-qa demos)
  const [pendingFileBasedDemo, setPendingFileBasedDemo] = useState<FileBasedDemo | null>(null)

  // Listen for onboarding sample import event and navigate appropriately
  useEffect(() => {
    const handleImportSample = (event: Event) => {
      const demoId = (event as CustomEvent<{ demoId: string }>).detail?.demoId
      if (demoId) {
        const demo = getDemoById(demoId)
        if (demo && isModelBasedDemo(demo)) {
          // For classifier/anomaly demos, queue auto-training
          setPendingModelTraining({
            demoId,
            modelType: demo.modelType,
            sampleDataId: demo.sampleDataId,
          })
        } else if (demo && isFileBasedDemo(demo)) {
          // For RAG/doc-qa demos, queue import in background (don't navigate away)
          // This lets user see their checklist first
          setPendingFileBasedDemo(demo)
        }
      }
    }

    window.addEventListener('lf-onboarding-import-sample', handleImportSample)
    return () => {
      window.removeEventListener('lf-onboarding-import-sample', handleImportSample)
    }
  }, [navigate])

  // Process pending model training
  useEffect(() => {
    if (!pendingModelTraining) return

    const { modelType, sampleDataId } = pendingModelTraining
    setPendingModelTraining(null)

    // Set training flag to disable the checklist button
    onboardingRef.current.setIsTrainingSampleModel(true)

    if (modelType === 'classifier') {
      // Find the classifier sample dataset
      const dataset = CLASSIFIER_SAMPLE_DATASETS.find(d => d.id === sampleDataId)
      if (!dataset?.data) {
        onboardingRef.current.setIsTrainingSampleModel(false)
        toast({
          message: 'Sample dataset not found.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
        return
      }

      // Generate model name from sample ID
      const modelName = `sample-${sampleDataId}`

      toast({
        message: `Training ${dataset.name} classifier...`,
      })

      // Train the classifier
      trainClassifierMutationRef.current.mutate(
        {
          model: modelName,
          training_data: dataset.data,
          description: `Sample classifier trained from ${dataset.name} dataset`,
        },
        {
          onSuccess: result => {
            toast({
              message: `${dataset.name} classifier trained successfully!`,
              icon: 'checkmark-filled',
            })
            // Store the trained model name in onboarding state
            // This updates the checklist link to point to the trained model
            onboardingRef.current.setTrainedModel(result.fitResult.versioned_name, 'classifier')
            // Don't auto-complete the step - let user click to view it first
          },
          onError: (error: Error) => {
            onboardingRef.current.setIsTrainingSampleModel(false)
            toast({
              message: error.message || 'Failed to train classifier.',
              variant: 'destructive',
              icon: 'alert-triangle',
            })
          },
        }
      )
    } else if (modelType === 'anomaly') {
      // Find the anomaly sample dataset
      const dataset = ANOMALY_SAMPLE_DATASETS.find(d => d.id === sampleDataId)
      if (!dataset?.data) {
        onboardingRef.current.setIsTrainingSampleModel(false)
        toast({
          message: 'Sample dataset not found.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
        return
      }

      // Generate model name from sample ID
      const modelName = `sample-${sampleDataId}`

      toast({
        message: `Training ${dataset.name} detector...`,
      })

      // Parse the data based on type
      let parsedData: number[][] | null = null
      if (dataset.type === 'numeric') {
        parsedData = parseNumericTrainingData(dataset.data)
      } else {
        // For text data, we need to convert to hash encoding
        // This matches how AnomalyModel.tsx handles text data
        const lines = dataset.data.split('\n').map(line => line.trim()).filter(Boolean)
        // Simple hash function for text -> numeric conversion
        parsedData = lines.map(line => {
          const values = line.split(',').map(v => v.trim())
          return values.map(v => {
            // Simple hash: sum of char codes
            let hash = 0
            for (let i = 0; i < v.length; i++) {
              hash = ((hash << 5) - hash) + v.charCodeAt(i)
              hash = hash & hash // Convert to 32-bit integer
            }
            return Math.abs(hash) % 10000 // Normalize to reasonable range
          })
        })
      }

      if (!parsedData) {
        onboardingRef.current.setIsTrainingSampleModel(false)
        toast({
          message: 'Failed to parse training data.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
        return
      }

      // Train the anomaly detector
      trainAnomalyMutationRef.current.mutate(
        {
          model: modelName,
          data: parsedData,
          backend: 'isolation_forest',
          description: `Sample detector trained from ${dataset.name} dataset`,
        },
        {
          onSuccess: result => {
            toast({
              message: `${dataset.name} detector trained successfully!`,
              icon: 'checkmark-filled',
            })
            // Store the trained model name in onboarding state
            // This updates the checklist link to point to the trained model
            onboardingRef.current.setTrainedModel(result.fitResult.versioned_name, 'anomaly')
            // Don't auto-complete the step - let user click to view it first
          },
          onError: (error: Error) => {
            onboardingRef.current.setIsTrainingSampleModel(false)
            toast({
              message: error.message || 'Failed to train detector.',
              variant: 'destructive',
              icon: 'alert-triangle',
            })
          },
        }
      )
    }
  }, [pendingModelTraining, toast])

  // Process pending file-based demo import when we have active project
  useEffect(() => {
    if (!pendingFileBasedDemo) {
      return
    }
    if (!activeProject?.namespace || !activeProject?.project) {
      return
    }

    const demo = pendingFileBasedDemo
    setPendingFileBasedDemo(null)

    const importDemo = async () => {
      try {
        toast({
          message: `Importing "${demo.displayName}"...`,
        })

        // Always use the project's available strategies and databases
        // Demo configs have their own database names that won't exist in the user's project
        const processingStrategy = strategiesData?.data_processing_strategies?.[0] || 'universal_processor'
        const database = strategiesData?.databases?.[0] || 'main_database'

        // Create dataset
        try {
          await createDatasetMutation.mutateAsync({
            namespace: activeProject.namespace,
            project: activeProject.project,
            name: demo.datasetName,
            data_processing_strategy: processingStrategy,
            database: database,
          })
        } catch (error: any) {
          // If dataset already exists, that's fine - continue with upload
          if (!(error?.response?.status === 409 || error?.message?.includes('already exists'))) {
            throw error
          }
        }

        // Upload each demo file
        for (const file of demo.files) {
          const fileResponse = await fetch(file.path)
          if (!fileResponse.ok) {
            throw new Error(`Failed to fetch ${file.filename}`)
          }
          const blob = await fileResponse.blob()
          const fileObj = new File([blob], file.filename, { type: file.type })

          await uploadMutation.mutateAsync({
            namespace: activeProject.namespace,
            project: activeProject.project,
            dataset: demo.datasetName,
            file: fileObj,
          })
        }

        toast({
          message: `"${demo.displayName}" imported successfully!`,
          icon: 'checkmark-filled',
        })

        // Refresh datasets list
        refetchDatasets()
      } catch (error) {
        toast({
          message: `Failed to import demo: ${error}`,
          variant: 'destructive',
          icon: 'alert-triangle',
        })
      }
    }

    importDemo()
  }, [pendingFileBasedDemo, activeProject, toast, createDatasetMutation, uploadMutation, refetchDatasets, strategiesData])

  // State for pending file upload from onboarding
  const [pendingFileUpload, setPendingFileUpload] = useState<{
    files: File[]
    datasetName: string
  } | null>(null)

  // Listen for onboarding file upload event
  useEffect(() => {
    const handleUploadFiles = (event: Event) => {
      const detail = (event as CustomEvent<{ files: File[]; datasetName: string }>).detail
      if (detail?.files && detail.files.length > 0) {
        setPendingFileUpload({
          files: detail.files,
          datasetName: detail.datasetName,
        })
      }
    }

    window.addEventListener('lf-onboarding-upload-files', handleUploadFiles)
    return () => {
      window.removeEventListener('lf-onboarding-upload-files', handleUploadFiles)
    }
  }, [])

  // Process pending file upload when we have active project
  useEffect(() => {
    if (!pendingFileUpload) {
      return
    }
    if (!activeProject?.namespace || !activeProject?.project) {
      return
    }

    const { files, datasetName } = pendingFileUpload
    setPendingFileUpload(null)

    // Create dataset first, then upload files
    const createAndUploadFiles = async () => {
      toast({
        message: `Creating dataset "${datasetName}"...`,
      })

      // First, create the dataset
      // Use the first available strategy and database from the project config
      const strategy = strategiesData?.data_processing_strategies?.[0] || 'universal_processor'
      const database = strategiesData?.databases?.[0] || 'main_database'

      try {
        await createDatasetMutation.mutateAsync({
          namespace: activeProject.namespace,
          project: activeProject.project,
          name: datasetName,
          data_processing_strategy: strategy,
          database: database,
        })
      } catch (error: any) {
        // If dataset already exists, that's fine - continue with upload
        if (!(error?.response?.status === 409 || error?.message?.includes('already exists'))) {
          toast({
            message: `Failed to create dataset: ${error?.message || 'Unknown error'}`,
            variant: 'destructive',
            icon: 'alert-triangle',
          })
          return
        }
      }

      // Now upload files
      toast({
        message: `Uploading ${files.length} file${files.length > 1 ? 's' : ''}...`,
      })

      let successCount = 0
      for (const file of files) {
        try {
          await uploadMutation.mutateAsync({
            namespace: activeProject.namespace,
            project: activeProject.project,
            dataset: datasetName,
            file,
          })
          successCount++
        } catch {
          // Continue with other files even if one fails
        }
      }

      if (successCount === files.length) {
        toast({
          message: `Successfully uploaded ${successCount} file${successCount > 1 ? 's' : ''} to "${datasetName}".`,
          icon: 'checkmark-filled',
        })
      } else if (successCount > 0) {
        toast({
          message: `Uploaded ${successCount} of ${files.length} files. Some uploads failed.`,
          variant: 'destructive',
          icon: 'alert-triangle',
        })
      } else {
        toast({
          message: 'Failed to upload files.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
      }

      // Refresh datasets list
      refetchDatasets()
    }

    createAndUploadFiles()
  }, [pendingFileUpload, activeProject, toast, uploadMutation, createDatasetMutation, refetchDatasets, strategiesData])

  useEffect(() => {
    const refresh = () => {
      try {
        const stored = localStorage.getItem('activeProject')
        if (stored) setProjectName(stored)
      } catch {}
    }
    refresh()
    const handler = (e: Event) => {
      // @ts-ignore custom event detail
      const detailName = (e as CustomEvent<string>).detail
      if (detailName) setProjectName(detailName)
      else refresh()
    }
    window.addEventListener('lf-active-project', handler as EventListener)
    return () =>
      window.removeEventListener('lf-active-project', handler as EventListener)
  }, [])

  // Listen for project deletions and redirect to home if current project was deleted
  useEffect(() => {
    const handleProjectDeleted = (event: Event) => {
      const deletedProjectName = (event as CustomEvent<string>).detail
      if (deletedProjectName === projectName) {
        // Current project was deleted, redirect to home
        navigate('/')
      }
    }
    window.addEventListener(
      'lf-project-deleted',
      handleProjectDeleted as EventListener
    )
    return () =>
      window.removeEventListener(
        'lf-project-deleted',
        handleProjectDeleted as EventListener
      )
  }, [projectName, navigate])

  // Get default model name from config
  const defaultModelName = useMemo(() => {
    const config = projectDetail?.project?.config
    const runtime = (config && (config as Record<string, any>).runtime) || null
    const def = runtime && (runtime as Record<string, any>).default_model
    if (!def || typeof def !== 'string' || def.trim().length === 0) {
      return 'No model configured'
    }
    return def
  }, [projectDetail])

  // Show loading screen while waiting for wizard to open
  if (isWaitingForWizard) {
    return (
      <div className="w-full h-full flex flex-col items-center justify-center">
        <div className="flex flex-col items-center gap-4 animate-in fade-in duration-300">
          <div className="text-4xl">🦙</div>
          <div className="text-lg font-medium text-foreground">Setting up your project...</div>
          <div className="text-sm text-muted-foreground">Just a moment</div>
        </div>
      </div>
    )
  }

  return (
    <>
      <div className="w-full h-full flex flex-col">
        {/* Hide header when wizard is open */}
        {!showWizard && (
          <div className="flex items-center justify-between mb-4 flex-shrink-0">
            <div className="flex items-center gap-2">
              <h2 className="text-2xl break-words ">
                {mode === 'designer' ? projectName : 'Config editor'}
              </h2>
              {mode === 'designer' && (
                <button
                  className="rounded-sm hover:opacity-80"
                  onClick={() => {
                    projectModal.openEditModal(projectName)
                  }}
                >
                  <FontIcon type="edit" className="w-5 h-5 text-primary" />
                </button>
              )}
            </div>
            <PageActions mode={mode} onModeChange={handleModeChange} />
          </div>
        )}

        {/* Validation Error Banner - also hide when wizard is open */}
        {!showWizard && projectDetail?.project?.validation_error &&
          (() => {
            // Parse actual error count from validation messages
            const errorText = projectDetail.project.validation_error
            let errorCount = 1 // Default to 1 if we can't parse

            // Try to extract error count from patterns like "5 validation errors" or "(and 3 more errors)"
            const countMatch = errorText.match(
              /(\d+)\s+(?:validation\s+)?errors?/i
            )
            if (countMatch) {
              errorCount = parseInt(countMatch[1], 10)
            } else {
              // Count semicolon-separated error messages as individual errors
              const parts = errorText
                .split(';')
                .filter(s => s.trim().length > 0)
              if (parts.length > 1) {
                errorCount = parts.length
              }
            }

            return (
              <div className="mb-4 rounded-lg border border-red-600 bg-red-50 dark:bg-red-950/20">
                <button
                  onClick={() =>
                    setShowValidationDetails(!showValidationDetails)
                  }
                  className="w-full px-4 py-3 flex items-center justify-between hover:bg-red-100 dark:hover:bg-red-950/30 rounded-lg transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <FontIcon
                      type="alert-triangle"
                      className="w-5 h-5 text-red-600"
                    />
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-red-100 bg-red-600 rounded-full px-2.5 py-0.5 font-semibold">
                        {errorCount} {errorCount === 1 ? 'error' : 'errors'}
                      </span>
                      <span className="text-sm font-semibold text-red-900 dark:text-red-100">
                        Configuration validation{' '}
                        {errorCount === 1 ? 'issue' : 'issues'} detected
                      </span>
                    </div>
                  </div>
                  <FontIcon
                    type={showValidationDetails ? 'chevron-up' : 'chevron-down'}
                    className="w-5 h-5 text-red-600"
                  />
                </button>
                {showValidationDetails && (
                  <div className="px-4 pb-4 pt-2 border-t border-red-200 dark:border-red-900">
                    <pre className="text-xs text-red-800 dark:text-red-200 whitespace-pre-wrap font-mono bg-red-100 dark:bg-red-950/40 p-3 rounded overflow-x-auto">
                      {projectDetail.project.validation_error}
                    </pre>
                    <div className="mt-3 text-xs text-red-700 dark:text-red-300">
                      <strong>Note:</strong> You can still view and edit this
                      project, but some features may not work correctly until
                      the validation errors are fixed.
                    </div>
                  </div>
                )}
              </div>
            )
          })()}

        {/* Onboarding Wizard - takes over entire dashboard area */}
        {showWizard ? (
          <div className="flex-1 min-h-0 overflow-hidden">
            <OnboardingWizard className="h-full" />
          </div>
        ) : mode !== 'designer' ? (
          <div className="flex-1 min-h-0 overflow-hidden pb-6">
            <ConfigEditor className="h-full" initialPointer={configPointer} />
          </div>
        ) : (
          <>
            {/* Onboarding: Getting Started Checklist - full width top */}
            {showChecklist && (
              <GettingStartedChecklist className="mb-4" />
            )}

            {/* Onboarding: Collapsed checklist when dismissed */}
            {showCollapsedChecklist && (
              <CollapsedChecklist className="mb-4" />
            )}

            {/* Onboarding: Restart banner when skipped/dismissed */}
            {showRestartBanner && (
              <RestartOnboardingBanner className="mb-4" />
            )}

            <DataCards
              filesProcessed={filesProcessed}
              databaseCount={databaseCount}
              modelsCount={modelsCount}
            />
            <div className="w-full grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              {/* Data (1/3) */}
              <div className="flex flex-col min-w-0 overflow-hidden">
                <div className="flex flex-row gap-2 items-center h-[40px] px-2 rounded-tl-lg rounded-tr-lg justify-between bg-card border-b border-border">
                  <div className="flex flex-row gap-2 items-center text-foreground">
                    <FontIcon type="data" className="w-4 h-4" />
                    Data
                  </div>
                  <button
                    className="text-xs text-primary"
                    onClick={() => navigate('/chat/data')}
                  >
                    View and add
                  </button>
                </div>
                <div className="p-4 md:p-6 flex flex-col gap-2 rounded-b-lg bg-card md:min-h-[260px]">
                  {isDatasetsLoading ? (
                    <div className="text-xs text-muted-foreground">
                      Loading…
                    </div>
                  ) : datasets.length === 0 ? (
                    <div className="text-xs text-muted-foreground">
                      No datasets yet
                    </div>
                  ) : (
                    <>
                      {datasets.slice(0, 8).map(d => (
                        <div
                          key={d.id}
                          className="py-1 px-2 rounded-lg flex flex-row gap-2 items-center justify-between bg-secondary cursor-pointer hover:bg-accent/30 min-w-0 overflow-hidden"
                          onClick={() =>
                            navigate(`/chat/data/${encodeURIComponent(d.name)}`)
                          }
                          role="button"
                          aria-label={`Open dataset ${d.name}`}
                        >
                          <div className="text-foreground truncate min-w-0">
                            {d.name}
                          </div>
                          <div className="text-xs text-muted-foreground whitespace-nowrap shrink-0 hidden lg:inline">
                            {(() => {
                              const dt = new Date(d.lastRun)
                              return `Updated ${dt.toLocaleDateString()} ${dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`
                            })()}
                          </div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              </div>

              {/* Models (1/3) */}
              <div className="min-w-0 overflow-hidden">
                <div className="h-[40px] px-2 flex items-center rounded-tl-lg rounded-tr-lg bg-card border-b border-border">
                  <div className="flex flex-row gap-2 items-center text-foreground">
                    <FontIcon type="model" className="w-4 h-4" />
                    Models
                  </div>
                </div>
                <div className="p-4 md:p-6 flex flex-col justify-start md:justify-between rounded-b-lg bg-card md:min-h-[260px]">
                  <div className="flex flex-col gap-3">
                    <div>
                      <label className="text-xs text-muted-foreground flex items-center gap-2">
                        Default inference model
                        <div className="relative group">
                          <FontIcon
                            type="info"
                            className="w-3.5 h-3.5 text-muted-foreground"
                          />
                          <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 mt-2 w-64 rounded-md border border-border bg-popover p-2 text-xs text-popover-foreground opacity-0 group-hover:opacity-100 transition-opacity">
                            Generates AI responses to your questions using the
                            relevant documents as context.
                          </div>
                        </div>
                      </label>
                      <div className="mt-2 rounded-xl border border-primary/50 bg-background px-4 py-2 text-base font-medium text-foreground">
                        {defaultModelName}
                      </div>
                    </div>
                  </div>
                  <div className="pt-4">
                    <button
                      className="w-full text-primary border border-primary rounded-lg py-2 text-base hover:bg-primary/10"
                      onClick={() => navigate('/chat/models')}
                    >
                      Go to models
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
      {/* Modal rendered globally in App */}
    </>
  )
}

export default Dashboard
