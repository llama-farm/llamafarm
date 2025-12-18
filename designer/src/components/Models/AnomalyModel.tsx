import { useState, useEffect, useCallback } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Select } from '../ui/select'
import { Textarea } from '../ui/textarea'
import { Badge } from '../ui/badge'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import FontIcon from '../../common/FontIcon'
import type { TrainedModel, TrainedModelVersion } from './types'
import { saveTrainedModel, getTrainedModels } from './TrainedModels'

type TrainingState = 'idle' | 'training' | 'success' | 'error'

function AnomalyModel() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const isNewModel = !id || id === 'new'

  // Model data (would come from API for existing models)
  const [model, setModel] = useState<TrainedModel | null>(null)

  // Form state
  const [modelName, setModelName] = useState(
    isNewModel ? 'new-anomaly-model' : ''
  )
  const [description, setDescription] = useState('')
  const [trainingData, setTrainingData] = useState('')

  // Settings state
  const [baseModel, setBaseModel] = useState('auto')
  const [threshold, setThreshold] = useState(0.5)

  // Training state
  const [trainingState, setTrainingState] = useState<TrainingState>('idle')
  const [trainingError, setTrainingError] = useState('')
  const [isTrainingExpanded, setIsTrainingExpanded] = useState(isNewModel)

  // Test state
  const [testInput, setTestInput] = useState('')
  const [testResult, setTestResult] = useState<{
    isAnomaly: boolean
    score: number
  } | null>(null)

  // Versions
  const [versions, setVersions] = useState<TrainedModelVersion[]>([])

  // Load existing model data from localStorage
  useEffect(() => {
    if (!isNewModel && id) {
      const models = getTrainedModels()
      const existingModel = models.find(m => m.id === id)

      if (existingModel) {
        setModel(existingModel)
        setModelName(existingModel.name)
        setDescription(existingModel.description || '')
        setThreshold(existingModel.threshold || 0.5)
        setBaseModel(existingModel.baseModel || 'auto')
        setVersions(existingModel.versions || [])
      }
    }
  }, [id, isNewModel])

  // Auto-save name/description/settings changes (debounced)
  useEffect(() => {
    if (isNewModel || !model) return

    const timer = setTimeout(() => {
      const updatedModel: TrainedModel = {
        ...model,
        name: modelName,
        description: description || undefined,
        threshold,
        baseModel,
      }
      saveTrainedModel(updatedModel)
      setModel(updatedModel)
    }, 500)

    return () => clearTimeout(timer)
  }, [modelName, description, threshold, baseModel, isNewModel, model])

  const hasVersions = versions.length > 0
  const canTest = hasVersions || trainingState === 'success'
  const canTrain = modelName.trim() && trainingData.trim()

  const handleTrain = useCallback(async () => {
    if (!canTrain) return

    setTrainingState('training')
    setTrainingError('')

    try {
      // Mock training - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 2000))

      // Parse training data to count samples
      const samples = trainingData
        .split(/[,\n]/)
        .map(s => s.trim())
        .filter(Boolean)

      // Create new version
      const newVersion: TrainedModelVersion = {
        id: `v${versions.length + 1}`,
        version: versions.length + 1,
        createdAt: new Date().toISOString(),
        trainingSamples: samples.length,
        isActive: true,
        threshold,
        baseModel,
      }

      // Set previous active version to inactive
      const updatedVersions = [
        newVersion,
        ...versions.map(v => ({ ...v, isActive: false })),
      ]
      setVersions(updatedVersions)

      // Generate model ID for new models
      const modelId = isNewModel ? `anomaly-${Date.now()}` : id!

      // Save to localStorage
      const trainedModel: TrainedModel = {
        id: modelId,
        name: modelName,
        type: 'anomaly_detection',
        status: 'ready',
        versionCount: updatedVersions.length,
        lastTrained: new Date().toISOString(),
        description: description || undefined,
        versions: updatedVersions,
        threshold,
        baseModel,
      }
      saveTrainedModel(trainedModel)

      setTrainingState('success')
      setIsTrainingExpanded(false) // Collapse training section after success

      // If new model, redirect to edit page with the new ID
      if (isNewModel) {
        navigate(`/chat/models/train/anomaly/${modelId}`)
      }
    } catch {
      setTrainingState('error')
      setTrainingError('Training failed. Please try again.')
    }
  }, [
    canTrain,
    trainingData,
    versions,
    threshold,
    baseModel,
    isNewModel,
    id,
    modelName,
    description,
    navigate,
  ])

  const handleTest = useCallback(() => {
    if (!testInput.trim()) return

    // Mock test - replace with actual API call
    const mockScore = Math.random()
    setTestResult({
      isAnomaly: mockScore > threshold,
      score: mockScore,
    })
  }, [testInput, threshold])

  const handleSetActiveVersion = useCallback((versionId: string) => {
    setVersions(prev =>
      prev.map(v => ({
        ...v,
        isActive: v.id === versionId,
      }))
    )
    // TODO: Save to API
  }, [])

  const handleDeleteVersion = useCallback((versionId: string) => {
    setVersions(prev => prev.filter(v => v.id !== versionId))
    // TODO: Delete from API
  }, [])

  const pageTitle = isNewModel
    ? 'New anomaly detection model'
    : modelName || 'Anomaly detection model'

  return (
    <div className="h-full w-full flex flex-col gap-4 pb-20">
      {/* Breadcrumb + Done button */}
      <div className="flex items-center justify-between">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/models?tab=training')}
          >
            Trained models
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">{pageTitle}</span>
        </nav>
        <Button
          variant="outline"
          onClick={() => navigate('/chat/models?tab=training')}
        >
          Done
        </Button>
      </div>

      {/* Page title */}
      <h1 className="text-2xl font-medium">{pageTitle}</h1>

      {/* Name and Description row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="model-name" className="text-sm font-medium">
            Model name{' '}
            {isNewModel && <span className="text-destructive">*</span>}
          </Label>
          <Input
            id="model-name"
            placeholder="e.g., fraud-detector"
            value={modelName}
            onChange={e => {
              // Only allow lowercase letters, numbers, and hyphens
              const sanitized = e.target.value
                .toLowerCase()
                .replace(/[^a-z0-9-]/g, '-')
                .replace(/-+/g, '-')
              setModelName(sanitized)
            }}
          />
          <p className="text-xs text-muted-foreground">
            Lowercase letters, numbers, and hyphens only
          </p>
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="description" className="text-sm font-medium">
            Description
          </Label>
          <Input
            id="description"
            placeholder="e.g., Detects unusual transaction patterns"
            value={description}
            onChange={e => setDescription(e.target.value)}
          />
        </div>
      </div>

      {/* Training Data & Settings Card */}
      <div className="rounded-lg border border-border bg-card p-4 flex flex-col gap-4">
        {/* Collapsed view - show when has versions and not expanded */}
        {hasVersions && !isTrainingExpanded ? (
          <div className="flex items-center justify-between">
            <div className="flex flex-col gap-1">
              <h3 className="text-sm font-medium">Training data</h3>
              <p className="text-xs text-muted-foreground">
                Add more training data to improve your model
              </p>
            </div>
            <Button
              variant="secondary"
              onClick={() => setIsTrainingExpanded(true)}
            >
              Retrain
            </Button>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Left: Training Data */}
              <div className="flex flex-col gap-4">
                <div className="flex flex-col gap-1.5">
                  <Label
                    htmlFor="training-data"
                    className="text-sm font-medium"
                  >
                    Training data{' '}
                    {isNewModel && <span className="text-destructive">*</span>}
                  </Label>
                  <p className="text-xs text-muted-foreground">
                    Provide examples of NORMAL data. The model learns this
                    pattern and will flag anything that deviates significantly.
                    Separate entries by new lines or commas.
                  </p>
                  <Textarea
                    id="training-data"
                    placeholder="Paste or type your training data here, separated by new lines or commas"
                    value={trainingData}
                    onChange={e => setTrainingData(e.target.value)}
                    rows={8}
                    className="font-mono text-sm"
                  />
                </div>
              </div>

              {/* Right: Settings */}
              <div className="flex flex-col gap-4">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-medium">Settings</h3>
                  {hasVersions && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setIsTrainingExpanded(false)}
                      className="h-6 w-6 p-0"
                      title="Collapse"
                    >
                      <FontIcon type="chevron-up" className="w-4 h-4" />
                    </Button>
                  )}
                </div>
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-1.5">
                    <Label htmlFor="base-model" className="text-sm">
                      Base model
                    </Label>
                    <Select
                      id="base-model"
                      value={baseModel}
                      onChange={e => setBaseModel(e.target.value)}
                    >
                      <option value="auto">Auto-detect (recommended)</option>
                      <option value="isolation-forest">Isolation Forest</option>
                      <option value="one-class-svm">One-Class SVM</option>
                      <option value="autoencoder">Autoencoder</option>
                    </Select>
                  </div>
                  <div className="flex flex-col gap-1.5">
                    <Label htmlFor="threshold" className="text-sm">
                      Anomaly threshold
                    </Label>
                    <Input
                      id="threshold"
                      type="number"
                      min={0}
                      max={1}
                      step={0.1}
                      value={threshold}
                      onChange={e => setThreshold(parseFloat(e.target.value))}
                    />
                    <p className="text-xs text-muted-foreground">
                      Values above this are flagged as anomalies
                    </p>
                  </div>
                </div>
              </div>
            </div>

            {/* Actions row */}
            <div className="flex items-center gap-3">
              <Button
                onClick={handleTrain}
                disabled={!canTrain || trainingState === 'training'}
              >
                {trainingState === 'training'
                  ? 'Training...'
                  : hasVersions
                    ? `Retrain as v${versions.length + 1}`
                    : 'Train'}
              </Button>
            </div>

            {/* Error message */}
            {trainingState === 'error' && trainingError && (
              <p className="text-sm text-destructive">{trainingError}</p>
            )}
          </>
        )}
      </div>

      {/* Test Panel (full width, below training data and settings) */}
      <div
        className={`rounded-lg border border-border bg-card p-4 flex flex-col gap-4 ${
          !canTest ? 'opacity-50' : ''
        }`}
      >
        {/* Success message - shown at top of test section */}
        {trainingState === 'success' && (
          <div className="flex items-center gap-2 text-primary bg-primary/10 border border-primary/20 rounded-md p-3">
            <FontIcon type="checkmark-filled" className="w-4 h-4" />
            <span className="text-sm font-medium">
              Model trained successfully
            </span>
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <Label className="text-sm font-medium">Test your model</Label>
          <p className="text-xs text-muted-foreground">
            {canTest
              ? 'Enter a value to check if it would be flagged as an anomaly.'
              : 'Train your model first to enable testing.'}
          </p>
        </div>

        <div className="flex gap-2">
          <Input
            placeholder="Enter a value to check"
            value={testInput}
            onChange={e => setTestInput(e.target.value)}
            onKeyDown={e => {
              if (e.key === 'Enter' && canTest) {
                handleTest()
              }
            }}
            disabled={!canTest}
            className="flex-1"
          />
          <Button onClick={handleTest} variant="secondary" disabled={!canTest}>
            Detect
          </Button>
        </div>

        {testResult && (
          <div
            className={`rounded-md p-3 ${
              testResult.isAnomaly
                ? 'bg-destructive/10 border border-destructive/20'
                : 'bg-primary/10 border border-primary/20'
            }`}
          >
            <div className="flex items-center gap-2 mb-1">
              {testResult.isAnomaly ? (
                <>
                  <FontIcon
                    type="alert-triangle"
                    className="w-4 h-4 text-destructive"
                  />
                  <span className="font-medium text-destructive">
                    ANOMALY DETECTED
                  </span>
                </>
              ) : (
                <>
                  <FontIcon
                    type="checkmark-filled"
                    className="w-4 h-4 text-primary"
                  />
                  <span className="font-medium text-primary">Normal</span>
                </>
              )}
            </div>
            <p className="text-sm text-muted-foreground">
              Score: {testResult.score.toFixed(2)} (threshold: {threshold})
            </p>
          </div>
        )}
      </div>

      {/* Model Versions */}
      <div className="flex flex-col gap-3">
        <h3 className="text-sm font-medium">Model versions</h3>
        {hasVersions ? (
          <div className="rounded-lg border border-border overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left px-4 py-2 font-medium">Version</th>
                  <th className="text-left px-4 py-2 font-medium">
                    Date created
                  </th>
                  <th className="text-left px-4 py-2 font-medium">
                    Training samples
                  </th>
                  <th className="text-left px-4 py-2 font-medium">Status</th>
                  <th className="text-right px-4 py-2 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {versions.map(version => (
                  <tr key={version.id} className="bg-card">
                    <td className="px-4 py-3">v{version.version}</td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {new Date(version.createdAt).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-3 text-muted-foreground">
                      {version.trainingSamples} entries
                    </td>
                    <td className="px-4 py-3">
                      {version.isActive ? (
                        <Badge variant="default">Active</Badge>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </td>
                    <td className="px-4 py-3 text-right">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button variant="ghost" size="sm">
                            <FontIcon type="overflow" className="w-4 h-4" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          {!version.isActive && (
                            <DropdownMenuItem
                              onClick={() => handleSetActiveVersion(version.id)}
                            >
                              Set as active
                            </DropdownMenuItem>
                          )}
                          <DropdownMenuItem
                            onClick={() => handleDeleteVersion(version.id)}
                            className="text-destructive"
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-border p-8 text-center">
            <p className="text-sm text-muted-foreground">
              No versions yet. Train your model to create your first version.
            </p>
          </div>
        )}
      </div>

      {/* Bottom spacer */}
      <div className="h-20" />
    </div>
  )
}

export default AnomalyModel
