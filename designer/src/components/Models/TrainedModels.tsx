import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import FontIcon from '../../common/FontIcon'
import type { TrainedModel } from './types'

// Storage key for trained models
const TRAINED_MODELS_KEY = 'llamafarm_trained_models'

// Helper to get trained models from localStorage
export function getTrainedModels(): TrainedModel[] {
  try {
    const stored = localStorage.getItem(TRAINED_MODELS_KEY)
    return stored ? JSON.parse(stored) : []
  } catch {
    return []
  }
}

// Helper to save trained models to localStorage
export function saveTrainedModels(models: TrainedModel[]) {
  localStorage.setItem(TRAINED_MODELS_KEY, JSON.stringify(models))
}

// Helper to add or update a trained model
export function saveTrainedModel(model: TrainedModel) {
  const models = getTrainedModels()
  const existingIndex = models.findIndex(m => m.id === model.id)
  if (existingIndex >= 0) {
    models[existingIndex] = model
  } else {
    models.push(model)
  }
  saveTrainedModels(models)
}

// Helper to delete a trained model
export function deleteTrainedModel(modelId: string) {
  const models = getTrainedModels()
  saveTrainedModels(models.filter(m => m.id !== modelId))
}

function TrainedModels() {
  const navigate = useNavigate()
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([])

  // Load models from localStorage on mount
  useEffect(() => {
    setTrainedModels(getTrainedModels())
  }, [])

  const handleDeleteModel = (modelId: string) => {
    deleteTrainedModel(modelId)
    setTrainedModels(getTrainedModels())
  }

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

        {trainedModels.length === 0 ? (
          <div className="rounded-lg border border-dashed border-border p-8 text-center">
            <p className="text-sm text-muted-foreground">
              No models yet. Create your first model above to get started.
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {trainedModels.map(model => (
              <TrainedModelCard
                key={model.id}
                model={model}
                onDelete={() => handleDeleteModel(model.id)}
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
}: {
  model: TrainedModel
  onDelete: () => void
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
            >
              Delete
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
          v{model.versionCount}
        </span>
      </div>
      <p className="text-xs text-muted-foreground">
        Last trained: {new Date(model.lastTrained).toLocaleDateString()}
      </p>
    </div>
  )
}

export default TrainedModels
