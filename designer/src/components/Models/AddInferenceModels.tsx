import { useCallback, useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../ui/button'
import FontIcon from '../../common/FontIcon'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject, useUpdateProject } from '../../hooks/useProjects'
import { parsePromptSets } from '../../utils/promptSets'
import { formatBytes } from '../../utils/modelUtils'
import type { InferenceModel } from './types'
import type { ProjectConfig } from '../../types/config'

// Import AddOrChangeModels from Models.tsx - we'll need to export it
import { AddOrChangeModels } from './Models'

function AddInferenceModels() {
  const navigate = useNavigate()
  const activeProject = useActiveProject()
  const { data: projectResponse } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )
  const updateProject = useUpdateProject()

  // Background download state
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

  // Project models state
  const [projectModels, setProjectModels] = useState<InferenceModel[]>([])

  const projectConfig = (projectResponse as any)?.project?.config as
    | ProjectConfig
    | undefined

  // Load models from config
  useEffect(() => {
    if (!projectResponse?.project?.config?.runtime?.models) {
      setProjectModels([])
      return
    }

    const runtimeModels = projectResponse.project.config.runtime.models
    const defaultModelName =
      projectResponse.project.config.runtime.default_model

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

      const isLocal = provider === 'ollama' || provider === 'universal'
      const localityBadge = isLocal ? 'Local' : 'Cloud'

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
        status: 'ready' as const,
      }
    })

    setProjectModels(mappedModels)
  }, [projectResponse])

  // Get prompt set names from project config
  const promptSetNames =
    projectConfig?.prompts && Array.isArray(projectConfig.prompts)
      ? parsePromptSets(projectConfig.prompts)
          .filter(ps => ps.name && ps.name !== 'default')
          .map(ps => ps.name)
      : []

  const addProjectModel = useCallback(
    async (m: InferenceModel, promptSets?: string[]) => {
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

      try {
        // Add to config
        const currentConfig = projectResponse.project.config
        const runtimeModels = currentConfig.runtime?.models || []

        const modelId = m.modelIdentifier || m.name
        const provider = modelId.includes('/') ? 'universal' : 'ollama'
        const baseUrl =
          provider === 'universal' ? undefined : 'http://localhost:11434'

        const newModel = {
          name: m.name,
          description: m.meta === 'Downloading…' ? '' : m.meta,
          provider,
          model: modelId,
          ...(baseUrl && { base_url: baseUrl }),
          prompt_format: 'unstructured',
          provider_config: {},
          prompts: promptSets && promptSets.length > 0 ? promptSets : ['default'],
        }

        const updatedModels = [...runtimeModels, newModel]

        const nextConfig = {
          ...currentConfig,
          runtime: {
            ...currentConfig.runtime,
            models: updatedModels,
          },
        }

        await updateProject.mutateAsync({
          namespace: activeProject.namespace,
          projectId: activeProject.project,
          request: { config: nextConfig },
        })
      } catch (error) {
        // Rollback optimistic update on error
        setProjectModels(prev => prev.filter(x => x.id !== m.id))
        throw error
      }
    },
    [activeProject, projectResponse, updateProject]
  )

  const handleGoToProject = useCallback(() => {
    navigate('/chat/models')
  }, [navigate])

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-20 px-4 md:px-6 pt-4">
      {/* Breadcrumb + Done button */}
      <div className="flex items-center justify-between mb-1">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/models')}
          >
            Inference models
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">Add inference models</span>
        </nav>
        <Button variant="outline" onClick={() => navigate('/chat/models')}>
          Done
        </Button>
      </div>

      {/* Big title */}
      <h2 className="text-2xl font-medium mb-2">Add inference models</h2>

      {/* Content */}
      <AddOrChangeModels
        onAddModel={addProjectModel}
        onGoToProject={handleGoToProject}
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
      />

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

export default AddInferenceModels
