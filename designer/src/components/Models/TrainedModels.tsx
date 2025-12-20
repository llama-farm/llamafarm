import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import type { TrainedModel, TrainedModelType } from './types'
import {
  useListAnomalyModels,
  useListClassifierModels,
  useDeleteAnomalyModel,
  useDeleteClassifierModel,
} from '../../hooks/useMLModels'
import {
  parseVersionedModelName,
  formatModelTimestamp,
  type AnomalyModelInfo,
  type ClassifierModelInfo,
} from '../../types/ml'

// Convert API model to unified TrainedModel type
function toTrainedModel(
  model: AnomalyModelInfo | ClassifierModelInfo,
  type: TrainedModelType,
  versionCount: number
): TrainedModel {
  const parsed = parseVersionedModelName(model.name)
  return {
    id: parsed.baseName, // Use base name as ID for navigation
    name: parsed.baseName,
    type,
    status: 'ready',
    versionCount,
    lastTrained: model.created || new Date().toISOString(),
    description: model.description,
    // Anomaly-specific
    ...(type === 'anomaly_detection' && {
      backend: (model as AnomalyModelInfo).backend,
    }),
    // Classifier-specific
    ...(type === 'classifier' && {
      labels: (model as ClassifierModelInfo).labels,
    }),
  }
}

// Group models by base name and count versions
function groupModelsByBaseName<T extends { name: string }>(
  models: T[]
): Map<string, T[]> {
  const groups = new Map<string, T[]>()
  for (const model of models) {
    const parsed = parseVersionedModelName(model.name)
    const existing = groups.get(parsed.baseName) || []
    existing.push(model)
    groups.set(parsed.baseName, existing)
  }
  return groups
}

function TrainedModels() {
  const navigate = useNavigate()
  const { toast } = useToast()

  // Fetch models from API
  const {
    data: anomalyData,
    isLoading: isLoadingAnomaly,
    error: anomalyError,
  } = useListAnomalyModels()
  const {
    data: classifierData,
    isLoading: isLoadingClassifier,
    error: classifierError,
  } = useListClassifierModels()

  const deleteAnomalyMutation = useDeleteAnomalyModel()
  const deleteClassifierMutation = useDeleteClassifierModel()

  // Track which model is being deleted
  const [deletingModelId, setDeletingModelId] = useState<string | null>(null)

  // Combine and normalize models
  const trainedModels = useMemo(() => {
    const models: TrainedModel[] = []

    // Process anomaly models
    if (anomalyData?.models) {
      const grouped = groupModelsByBaseName(anomalyData.models)
      for (const [, versions] of grouped) {
        // Get the most recent version for metadata
        const sortedVersions = [...versions].sort((a, b) => {
          const parsedA = parseVersionedModelName(a.name)
          const parsedB = parseVersionedModelName(b.name)
          return (parsedB.timestamp || '').localeCompare(parsedA.timestamp || '')
        })
        const latest = sortedVersions[0]
        models.push(toTrainedModel(latest, 'anomaly_detection', versions.length))
      }
    }

    // Process classifier models
    if (classifierData?.models) {
      const grouped = groupModelsByBaseName(classifierData.models)
      for (const [, versions] of grouped) {
        const sortedVersions = [...versions].sort((a, b) => {
          const parsedA = parseVersionedModelName(a.name)
          const parsedB = parseVersionedModelName(b.name)
          return (parsedB.timestamp || '').localeCompare(parsedA.timestamp || '')
        })
        const latest = sortedVersions[0]
        models.push(toTrainedModel(latest, 'classifier', versions.length))
      }
    }

    // Sort by last trained date (newest first)
    return models.sort(
      (a, b) =>
        new Date(b.lastTrained).getTime() - new Date(a.lastTrained).getTime()
    )
  }, [anomalyData, classifierData])

  const handleDeleteModel = async (model: TrainedModel) => {
    // Confirm deletion
    const confirmMessage = `Delete "${model.name}" and all ${model.versionCount} version${model.versionCount !== 1 ? 's' : ''}? This cannot be undone.`
    if (!window.confirm(confirmMessage)) return

    setDeletingModelId(model.id)
    try {
      // Delete all versions of this model
      if (model.type === 'anomaly_detection' && anomalyData?.models) {
        const versions = anomalyData.models.filter((m: AnomalyModelInfo) => {
          const parsed = parseVersionedModelName(m.name)
          return parsed.baseName === model.id
        })
        await Promise.all(
          versions.map(version => deleteAnomalyMutation.mutateAsync(version.filename))
        )
      } else if (model.type === 'classifier' && classifierData?.models) {
        const versions = classifierData.models.filter((m: ClassifierModelInfo) => {
          const parsed = parseVersionedModelName(m.name)
          return parsed.baseName === model.id
        })
        await Promise.all(
          versions.map(version => deleteClassifierMutation.mutateAsync(version.name))
        )
      }

      toast({
        message: `Successfully deleted ${model.name} and all its versions.`,
        icon: 'checkmark-filled',
      })
    } catch (error) {
      console.error('Failed to delete model:', error)
      toast({
        message: 'Failed to delete some model versions. Please try again.',
        variant: 'destructive',
        icon: 'alert-triangle',
      })
    } finally {
      setDeletingModelId(null)
    }
  }

  const isLoading = isLoadingAnomaly || isLoadingClassifier
  const hasError = anomalyError || classifierError

  return (
    <div className="flex flex-col gap-6">
      {/* Page header */}
      <div>
        <h2 className="text-lg font-medium">Trained models</h2>
        <p className="text-sm text-muted-foreground">
          Create custom models trained on your data. Models you create here are
          available across all your projects.
        </p>
      </div>

      {/* Two action cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Anomaly Detection Card */}
        <div
          className="rounded-lg border border-border bg-card p-5 flex flex-col gap-3 cursor-pointer hover:border-primary/50 transition-colors"
          onClick={() => navigate('/chat/models/train/anomaly/new')}
        >
          <h3 className="font-medium">Anomaly detection models</h3>
          <p className="text-sm text-muted-foreground flex-1">
            Learns what 'normal' looks like in your data, then flags anything
            unusual. No need to define what's wrong—just show it what's right.
          </p>
          <Button
            variant="secondary"
            onClick={e => {
              e.stopPropagation()
              navigate('/chat/models/train/anomaly/new')
            }}
            className="w-fit"
          >
            Create
          </Button>
        </div>

        {/* Classifier Card */}
        <div
          className="rounded-lg border border-border bg-card p-5 flex flex-col gap-3 cursor-pointer hover:border-primary/50 transition-colors"
          onClick={() => navigate('/chat/models/train/classifier/new')}
        >
          <h3 className="font-medium">Classifier models</h3>
          <p className="text-sm text-muted-foreground flex-1">
            Categorizes text into labels you define. Train it with examples of
            each category and it learns to sort new data.
          </p>
          <Button
            variant="secondary"
            onClick={e => {
              e.stopPropagation()
              navigate('/chat/models/train/classifier/new')
            }}
            className="w-fit"
          >
            Create
          </Button>
        </div>
      </div>

      {/* Divider */}
      <hr className="border-border" />

      {/* Your trained models section */}
      <div className="flex flex-col gap-3">
        <h3 className="font-medium">Your trained models</h3>

        {isLoading ? (
          <div className="rounded-lg border border-dashed border-border p-8 text-center">
            <p className="text-sm text-muted-foreground">Loading models...</p>
          </div>
        ) : hasError ? (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4">
            <p className="text-sm text-destructive">
              Failed to load models. Make sure the server is running.
            </p>
          </div>
        ) : trainedModels.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border p-8 text-center">
            <p className="text-sm text-muted-foreground">
              No models yet. Create your first model above to get started.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {trainedModels.map(model => (
              <TrainedModelCard
                key={`${model.type}-${model.id}`}
                model={model}
                onDelete={() => handleDeleteModel(model)}
                isDeleting={deletingModelId === model.id}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

function TrainedModelCard({
  model,
  onDelete,
  isDeleting,
}: {
  model: TrainedModel
  onDelete: () => void
  isDeleting?: boolean
}) {
  const navigate = useNavigate()
  const typeLabel =
    model.type === 'anomaly_detection' ? 'Anomaly detection' : 'Classifier'
  const editPath =
    model.type === 'anomaly_detection'
      ? `/chat/models/train/anomaly/${model.id}`
      : `/chat/models/train/classifier/${model.id}`

  // Use project colors: teal for anomaly detection, purple for classifier
  const typeColorClasses =
    model.type === 'anomaly_detection'
      ? 'bg-teal-100 text-teal-700 dark:bg-teal-500/20 dark:text-teal-300'
      : 'bg-purple-100 text-purple-700 dark:bg-purple-500/20 dark:text-purple-300'

  // Format the last trained date
  const lastTrainedDisplay = (() => {
    try {
      // Try to parse timestamp from model name if available
      const parsed = parseVersionedModelName(model.name)
      if (parsed.timestamp) {
        return formatModelTimestamp(parsed.timestamp)
      }
      return new Date(model.lastTrained).toLocaleDateString()
    } catch {
      return 'Unknown'
    }
  })()

  return (
    <div
      className="rounded-lg border border-border bg-card p-4 flex flex-col gap-2 cursor-pointer hover:border-primary/50 transition-colors"
      onClick={() => navigate(editPath)}
    >
      <div className="flex items-start justify-between">
        <h4 className="font-medium">{model.name}</h4>
        <DropdownMenu>
          <DropdownMenuTrigger
            asChild
            onClick={e => e.stopPropagation()}
          >
            <Button variant="ghost" size="sm" className="h-6 w-6 p-0">
              <FontIcon type="overflow" className="w-4 h-4" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuItem
              onClick={e => {
                e.stopPropagation()
                navigate(editPath)
              }}
            >
              Edit
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={e => {
                e.stopPropagation()
                onDelete()
              }}
              className="text-destructive"
              disabled={isDeleting}
            >
              {isDeleting ? 'Deleting...' : 'Delete'}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
      {model.description && (
        <p className="text-sm text-muted-foreground line-clamp-2">
          {model.description}
        </p>
      )}
      <div className="flex items-center gap-2 mt-1">
        <Badge className={typeColorClasses}>{typeLabel}</Badge>
        <span className="text-xs text-muted-foreground">
          {model.versionCount} version{model.versionCount !== 1 ? 's' : ''}
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        Last trained: {lastTrainedDisplay}
      </p>
    </div>
  )
}

export default TrainedModels
