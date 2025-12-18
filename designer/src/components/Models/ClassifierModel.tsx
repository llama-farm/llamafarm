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

interface ClassLabel {
  id: string
  name: string
  examples: string[]
}

function ClassifierModel() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const isNewModel = !id || id === 'new'

  // Model data (would come from API for existing models)
  const [model, setModel] = useState<TrainedModel | null>(null)

  // Form state
  const [modelName, setModelName] = useState(isNewModel ? 'new-classifier-model' : '')
  const [description, setDescription] = useState('')

  // Class labels state
  const [classLabels, setClassLabels] = useState<ClassLabel[]>([
    { id: '1', name: '', examples: [] },
    { id: '2', name: '', examples: [] },
  ])

  // Settings state
  const [baseModel, setBaseModel] = useState('auto')

  // Training state
  const [trainingState, setTrainingState] = useState<TrainingState>('idle')
  const [trainingError, setTrainingError] = useState('')
  const [isTrainingExpanded, setIsTrainingExpanded] = useState(isNewModel)

  // Test state
  const [testInput, setTestInput] = useState('')
  const [testResult, setTestResult] = useState<{
    label: string
    confidence: number
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
        setBaseModel(existingModel.baseModel || 'auto')
        setVersions(existingModel.versions || [])
        // Note: classLabels are not stored in localStorage yet
        // Users will need to re-enter class labels when editing
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
        baseModel,
      }
      saveTrainedModel(updatedModel)
      setModel(updatedModel)
    }, 500)

    return () => clearTimeout(timer)
  }, [modelName, description, baseModel, isNewModel, model])

  const hasVersions = versions.length > 0
  const canTest = hasVersions || trainingState === 'success'

  // Check if we have at least 2 classes with names and examples
  const validClasses = classLabels.filter(
    c => c.name.trim() && c.examples.length > 0
  )
  const canTrain = modelName.trim() && validClasses.length >= 2

  const handleAddClass = useCallback(() => {
    const newId = String(Date.now())
    setClassLabels(prev => [...prev, { id: newId, name: '', examples: [] }])
  }, [])

  const handleRemoveClass = useCallback(
    (classId: string) => {
      if (classLabels.length <= 2) return // Minimum 2 classes
      setClassLabels(prev => prev.filter(c => c.id !== classId))
    },
    [classLabels]
  )

  const handleClassNameChange = useCallback((classId: string, name: string) => {
    setClassLabels(prev =>
      prev.map(c => (c.id === classId ? { ...c, name } : c))
    )
  }, [])

  const handleTrain = useCallback(async () => {
    if (!canTrain) return

    setTrainingState('training')
    setTrainingError('')

    try {
      // Mock training - replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 2500))

      // Count total training samples
      const totalSamples = classLabels.reduce(
        (sum, c) => sum + c.examples.length,
        0
      )

      // Create new version
      const newVersion: TrainedModelVersion = {
        id: `v${versions.length + 1}`,
        version: versions.length + 1,
        createdAt: new Date().toISOString(),
        trainingSamples: totalSamples,
        isActive: true,
        baseModel,
      }

      // Set previous active version to inactive
      const updatedVersions = [
        newVersion,
        ...versions.map(v => ({ ...v, isActive: false })),
      ]
      setVersions(updatedVersions)

      // Generate model ID for new models
      const modelId = isNewModel ? `classifier-${Date.now()}` : id!

      // Save to localStorage
      const trainedModel: TrainedModel = {
        id: modelId,
        name: modelName,
        type: 'classifier',
        status: 'ready',
        versionCount: updatedVersions.length,
        lastTrained: new Date().toISOString(),
        description: description || undefined,
        versions: updatedVersions,
        baseModel,
      }
      saveTrainedModel(trainedModel)

      setTrainingState('success')
      setIsTrainingExpanded(false) // Collapse training section after success

      // If new model, redirect to edit page with the new ID
      if (isNewModel) {
        navigate(`/chat/models/train/classifier/${modelId}`)
      }
    } catch {
      setTrainingState('error')
      setTrainingError('Training failed. Please try again.')
    }
  }, [canTrain, classLabels, versions, baseModel, isNewModel, id, modelName, description, navigate])

  const handleTest = useCallback(() => {
    if (!testInput.trim()) return

    // Mock test - replace with actual API call
    // Randomly pick a class and generate confidence
    const randomClass =
      validClasses[Math.floor(Math.random() * validClasses.length)]
    const mockConfidence = 0.6 + Math.random() * 0.4 // 0.6 - 1.0

    setTestResult({
      label: randomClass?.name || 'Unknown',
      confidence: mockConfidence,
    })
  }, [testInput, validClasses])

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
    ? 'New classifier model'
    : modelName || 'Classifier model'

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
        <Button variant="outline" onClick={() => navigate('/chat/models?tab=training')}>
          Done
        </Button>
      </div>

      {/* Page title */}
      <h1 className="text-2xl font-medium">{pageTitle}</h1>

      {/* Name and Description row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-2">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="model-name" className="text-sm font-medium">
            Model name {isNewModel && <span className="text-destructive">*</span>}
          </Label>
          <Input
            id="model-name"
            placeholder="e.g., sentiment-classifier"
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
            placeholder="e.g., Classifies customer feedback sentiment"
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
          {/* Left: Class Labels as Pairs */}
          <div className="flex flex-col gap-4">
            <div className="flex flex-col gap-1">
              <Label className="text-sm font-medium">
                Class labels{' '}
                {isNewModel && <span className="text-destructive">*</span>}
              </Label>
              <p className="text-xs text-muted-foreground">
                Define at least 2 classes with example texts for each.
              </p>
            </div>

            {/* Class pairs */}
            <div className="flex flex-col gap-4">
              {classLabels.map((classLabel, index) => (
                <div
                  key={classLabel.id}
                  className="flex flex-col gap-2 p-3 rounded-lg border border-border bg-muted/30"
                >
                  <div className="flex items-center gap-2">
                    <Input
                      value={classLabel.name}
                      onChange={e => handleClassNameChange(classLabel.id, e.target.value)}
                      placeholder={`Class ${index + 1} name`}
                      className="flex-1 font-medium"
                    />
                    {classLabels.length > 2 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleRemoveClass(classLabel.id)}
                        className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                      >
                        <FontIcon type="close" className="w-4 h-4" />
                      </Button>
                    )}
                  </div>
                  <Textarea
                    placeholder="Enter examples for this class (one per line or comma-separated)"
                    value={classLabel.examples.join('\n')}
                    onChange={e => {
                      const examples = e.target.value
                        .split(/[\n,]/)
                        .map(s => s.trim())
                        .filter(Boolean)
                      setClassLabels(prev =>
                        prev.map(c =>
                          c.id === classLabel.id ? { ...c, examples } : c
                        )
                      )
                    }}
                    rows={2}
                    className="font-mono text-sm"
                  />
                  {classLabel.examples.length > 0 && (
                    <p className="text-xs text-muted-foreground">
                      {classLabel.examples.length} example{classLabel.examples.length !== 1 ? 's' : ''}
                    </p>
                  )}
                </div>
              ))}
            </div>

            {/* Add class button */}
            <Button
              variant="outline"
              onClick={handleAddClass}
              className="gap-2 w-full"
            >
              <FontIcon type="add" className="w-4 h-4" />
              Add class
            </Button>
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
                <option value="logistic-regression">Logistic Regression</option>
                <option value="naive-bayes">Naive Bayes</option>
                <option value="svm">Support Vector Machine</option>
                <option value="transformer">Transformer-based</option>
              </Select>
              <p className="text-xs text-muted-foreground">
                The algorithm used for classification
              </p>
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

            {/* Validation message */}
            {!canTrain && modelName.trim() && (
              <p className="text-sm text-muted-foreground">
                Add at least 2 classes with names and examples to train.
              </p>
            )}

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
            <span className="text-sm font-medium">Model trained successfully</span>
          </div>
        )}

        <div className="flex flex-col gap-1.5">
          <Label className="text-sm font-medium">Test your model</Label>
          <p className="text-xs text-muted-foreground">
            {canTest
              ? 'Enter text to see which class it would be assigned to.'
              : 'Train your model first to enable testing.'}
          </p>
        </div>

        <div className="flex gap-2">
          <Input
            placeholder="Enter text to classify"
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
            Classify
          </Button>
        </div>

        {testResult && (
          <div className="rounded-md p-3 bg-primary/10 border border-primary/20">
            <div className="flex items-center gap-2 mb-1">
              <FontIcon type="checkmark-filled" className="w-4 h-4 text-primary" />
              <span className="font-medium text-primary">
                {testResult.label}
              </span>
            </div>
            <p className="text-sm text-muted-foreground">
              Confidence: {(testResult.confidence * 100).toFixed(1)}%
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
                  <th className="text-left px-4 py-2 font-medium">Date created</th>
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

export default ClassifierModel
