import { useState, useEffect, useCallback, useMemo } from 'react'
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
import type { ClassifierTestResult } from './types'
import {
  useListClassifierModels,
  useTrainAndSaveClassifier,
  usePredictClassifier,
  useLoadClassifier,
  useDeleteClassifierModel,
} from '../../hooks/useMLModels'
import {
  parseVersionedModelName,
  formatModelTimestamp,
  generateUniqueModelName,
  type ClassifierModelInfo,
  type ClassifierTrainingData,
} from '../../types/ml'

type TrainingState = 'idle' | 'training' | 'success' | 'error'

// Default SetFit base model
const DEFAULT_BASE_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

interface ClassLabel {
  id: string
  name: string
  examples: string[]
}

interface ModelVersion {
  id: string
  versionNumber: number
  versionedName: string
  createdAt: string
  trainingSamples: number
  isActive: boolean
  labels: string[]
}

function ClassifierModel() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const isNewModel = !id || id === 'new'

  // Form state - modelName will be set after loading existing models
  const [modelName, setModelName] = useState('')
  const [description, setDescription] = useState('')
  const [nameExistsWarning, setNameExistsWarning] = useState(false)

  // Class labels state
  const [classLabels, setClassLabels] = useState<ClassLabel[]>([
    { id: '1', name: '', examples: [] },
    { id: '2', name: '', examples: [] },
  ])

  // Settings state - disabled for now, using default SetFit model
  const [baseModel] = useState(DEFAULT_BASE_MODEL)

  // Training state
  const [trainingState, setTrainingState] = useState<TrainingState>('idle')
  const [trainingError, setTrainingError] = useState('')
  const [isTrainingExpanded, setIsTrainingExpanded] = useState(isNewModel)

  // Test state
  const [testInput, setTestInput] = useState('')
  const [testHistory, setTestHistory] = useState<ClassifierTestResult[]>([])

  // Versions - derived from API models with same base name
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [activeVersionName, setActiveVersionName] = useState<string | null>(null)

  // API hooks
  const { data: modelsData, isLoading: isLoadingModels } = useListClassifierModels()
  const trainAndSaveMutation = useTrainAndSaveClassifier()
  const predictMutation = usePredictClassifier()
  const loadMutation = useLoadClassifier()
  const deleteMutation = useDeleteClassifierModel()

  // Parse the model ID to get base name for filtering versions
  const baseModelName = useMemo(() => {
    if (isNewModel) return null
    if (!id) return null
    const parsed = parseVersionedModelName(id)
    return parsed.baseName
  }, [id, isNewModel])

  // Extract all existing base names from models for uniqueness check
  const existingBaseNames = useMemo(() => {
    const names = new Set<string>()
    if (modelsData?.models) {
      for (const model of modelsData.models) {
        const parsed = parseVersionedModelName(model.name)
        names.add(parsed.baseName)
      }
    }
    return names
  }, [modelsData])

  // Set unique default model name for new models once data is loaded
  useEffect(() => {
    if (isNewModel && !modelName && !isLoadingModels) {
      const uniqueName = generateUniqueModelName('new-classifier-model', existingBaseNames)
      setModelName(uniqueName)
    }
  }, [isNewModel, modelName, isLoadingModels, existingBaseNames])

  // Check if model name already exists (for warning display)
  useEffect(() => {
    if (isNewModel && modelName) {
      setNameExistsWarning(existingBaseNames.has(modelName))
    } else {
      setNameExistsWarning(false)
    }
  }, [isNewModel, modelName, existingBaseNames])

  // Build versions list from API models
  useEffect(() => {
    if (!modelsData?.models || !baseModelName) {
      setVersions([])
      return
    }

    // Filter models that match our base name
    const matchingModels = modelsData.models.filter((m: ClassifierModelInfo) => {
      const parsed = parseVersionedModelName(m.name)
      return parsed.baseName === baseModelName
    })

    // Sort by timestamp (newest first) and build version list
    const sortedModels = [...matchingModels].sort((a, b) => {
      const parsedA = parseVersionedModelName(a.name)
      const parsedB = parseVersionedModelName(b.name)
      return (parsedB.timestamp || '').localeCompare(parsedA.timestamp || '')
    })

    const versionList: ModelVersion[] = sortedModels.map((m, index) => ({
      id: m.name,
      versionNumber: sortedModels.length - index,
      versionedName: m.name,
      createdAt: m.created || new Date().toISOString(),
      trainingSamples: 0,
      isActive: m.name === activeVersionName,
      labels: m.labels || [],
    }))

    setVersions(versionList)

    // Set first model as active if none selected
    if (!activeVersionName && versionList.length > 0) {
      setActiveVersionName(versionList[0].versionedName)
    }
  }, [modelsData, baseModelName, activeVersionName])

  // Load model metadata when editing existing model
  useEffect(() => {
    if (isNewModel || !baseModelName) return
    setModelName(baseModelName)
  }, [isNewModel, baseModelName])

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
      if (classLabels.length <= 2) return
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

    // Use unique name if the current name already exists
    const finalModelName = isNewModel
      ? generateUniqueModelName(modelName, existingBaseNames)
      : modelName

    try {
      // Convert class labels to API training data format
      const trainingData: ClassifierTrainingData[] = []
      for (const classLabel of validClasses) {
        for (const example of classLabel.examples) {
          trainingData.push({
            text: example,
            label: classLabel.name,
          })
        }
      }

      const result = await trainAndSaveMutation.mutateAsync({
        model: finalModelName,
        base_model: baseModel,
        training_data: trainingData,
        overwrite: false,
        description: description.trim() || undefined,
      })

      // Update state with new version
      const newVersionName = result.fitResult.versioned_name
      setActiveVersionName(newVersionName)
      setTrainingState('success')
      setIsTrainingExpanded(false)

      // If new model, redirect to edit page with the base name
      if (isNewModel) {
        navigate(`/chat/models/train/classifier/${finalModelName}`)
      }
    } catch (error) {
      setTrainingState('error')
      setTrainingError(
        error instanceof Error ? error.message : 'Training failed. Please try again.'
      )
    }
  }, [canTrain, validClasses, modelName, baseModel, trainAndSaveMutation, isNewModel, navigate, existingBaseNames])

  const handleTest = useCallback(async () => {
    if (!testInput.trim() || !activeVersionName) return

    try {
      // Ensure model is loaded before predicting
      await loadMutation.mutateAsync({ model: activeVersionName })

      const result = await predictMutation.mutateAsync({
        model: activeVersionName,
        texts: [testInput.trim()],
      })

      if (result.data && result.data.length > 0) {
        const prediction = result.data[0]
        const newResult: ClassifierTestResult = {
          id: String(Date.now()),
          input: testInput.trim(),
          label: prediction.label.trim(), // API may return label with trailing space
          confidence: prediction.score, // API uses 'score' not 'confidence'
          timestamp: new Date().toISOString(),
        }
        setTestHistory(prev => [newResult, ...prev])
      }
    } catch (error) {
      const errorResult: ClassifierTestResult = {
        id: String(Date.now()),
        input: testInput.trim(),
        label: 'Error',
        confidence: 0,
        timestamp: new Date().toISOString(),
      }
      setTestHistory(prev => [
        { ...errorResult, input: `Error: ${error instanceof Error ? error.message : 'Test failed'}` },
        ...prev,
      ])
    }
    setTestInput('')
  }, [testInput, activeVersionName, loadMutation, predictMutation])

  const handleSetActiveVersion = useCallback(
    async (versionName: string) => {
      try {
        await loadMutation.mutateAsync({ model: versionName })
        setActiveVersionName(versionName)
        setVersions(prev =>
          prev.map(v => ({
            ...v,
            isActive: v.versionedName === versionName,
          }))
        )
      } catch (error) {
        console.error('Failed to load model version:', error)
      }
    },
    [loadMutation]
  )

  const handleDeleteVersion = useCallback(
    async (versionName: string) => {
      try {
        await deleteMutation.mutateAsync(versionName)

        if (versionName === activeVersionName) {
          const remaining = versions.filter(v => v.versionedName !== versionName)
          if (remaining.length > 0) {
            setActiveVersionName(remaining[0].versionedName)
          } else {
            setActiveVersionName(null)
          }
        }
      } catch (error) {
        console.error('Failed to delete model version:', error)
      }
    },
    [versions, activeVersionName, deleteMutation]
  )

  const pageTitle = isNewModel
    ? 'New classifier model'
    : modelName || 'Classifier model'

  if (isLoadingModels && !isNewModel) {
    return (
      <div className="flex-1 min-h-0 overflow-auto pb-20">
        <div className="flex items-center justify-center h-64">
          <div className="text-muted-foreground">Loading model...</div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex-1 min-h-0 overflow-auto pb-20">
      <div className="flex flex-col gap-4">
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
                const sanitized = e.target.value
                  .toLowerCase()
                  .replace(/[^a-z0-9-]/g, '-')
                  .replace(/-+/g, '-')
                setModelName(sanitized)
              }}
              disabled={!isNewModel}
              className={nameExistsWarning ? 'border-amber-500' : ''}
            />
            {nameExistsWarning ? (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                A model with this name exists. Will be saved as "{generateUniqueModelName(modelName, existingBaseNames)}".
              </p>
            ) : (
              <p className="text-xs text-muted-foreground">
                Lowercase letters, numbers, and hyphens only
              </p>
            )}
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
                      disabled // Disabled for now - only SetFit supported
                    >
                      <option value={DEFAULT_BASE_MODEL}>
                        SetFit (all-MiniLM-L6-v2)
                      </option>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Uses SetFit few-shot learning for text classification
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

        {/* Test Panel */}
        <div
          className={`rounded-lg border border-border bg-card p-4 flex flex-col gap-4 ${
            !canTest ? 'opacity-50' : ''
          }`}
        >
          {/* Success message */}
          {trainingState === 'success' && (
            <div className="flex items-center gap-2 text-primary bg-primary/10 border border-primary/20 rounded-md p-3">
              <FontIcon type="checkmark-filled" className="w-4 h-4" />
              <span className="text-sm font-medium">Model trained successfully</span>
            </div>
          )}

          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Label className="text-sm font-medium">Test your model</Label>
              {activeVersionName && (
                <Badge variant="secondary" className="text-xs font-normal">
                  {activeVersionName}
                </Badge>
              )}
            </div>
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
              disabled={!canTest || predictMutation.isPending}
              className="flex-1"
            />
            <Button
              onClick={handleTest}
              variant="secondary"
              disabled={!canTest || predictMutation.isPending}
            >
              {predictMutation.isPending ? 'Classifying...' : 'Classify'}
            </Button>
          </div>

          {testHistory.length > 0 && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">Test history</span>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setTestHistory([])}
                  className="text-xs h-5 px-1.5 text-muted-foreground"
                >
                  Clear
                </Button>
              </div>
              <div className="flex flex-col gap-0.5 max-h-[150px] overflow-y-auto">
                {testHistory.map(result => (
                  <div
                    key={result.id}
                    className="flex items-center gap-2 px-2 py-1 rounded text-sm bg-muted/50"
                  >
                    <FontIcon type="checkmark-filled" className="w-3 h-3 text-primary shrink-0" />
                    <span className="font-medium text-primary w-20 shrink-0 truncate" title={result.label}>
                      {result.label}
                    </span>
                    <span className="text-muted-foreground w-10 shrink-0">
                      {(result.confidence * 100).toFixed(0)}%
                    </span>
                    <span className="text-muted-foreground truncate" title={result.input}>
                      {result.input}
                    </span>
                  </div>
                ))}
              </div>
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
                    <th className="text-left px-4 py-2 font-medium">Model name</th>
                    <th className="text-left px-4 py-2 font-medium">Created</th>
                    <th className="text-left px-4 py-2 font-medium">Labels</th>
                    <th className="text-left px-4 py-2 font-medium">Status</th>
                    <th className="text-right px-4 py-2 font-medium">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {versions.map(version => {
                    const parsed = parseVersionedModelName(version.versionedName)
                    return (
                      <tr key={version.id} className="bg-card">
                        <td className="px-4 py-3">v{version.versionNumber}</td>
                        <td className="px-4 py-3 font-mono text-xs">
                          {version.versionedName}
                        </td>
                        <td className="px-4 py-3 text-muted-foreground">
                          {parsed.timestamp
                            ? formatModelTimestamp(parsed.timestamp)
                            : new Date(version.createdAt).toLocaleDateString()}
                        </td>
                        <td className="px-4 py-3 text-muted-foreground">
                          {version.labels.length > 0
                            ? version.labels.join(', ')
                            : '—'}
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
                                  onClick={() =>
                                    handleSetActiveVersion(version.versionedName)
                                  }
                                >
                                  Set as active
                                </DropdownMenuItem>
                              )}
                              <DropdownMenuItem
                                onClick={() =>
                                  handleDeleteVersion(version.versionedName)
                                }
                                className="text-destructive"
                              >
                                Delete
                              </DropdownMenuItem>
                            </DropdownMenuContent>
                          </DropdownMenu>
                        </td>
                      </tr>
                    )
                  })}
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
      </div>
    </div>
  )
}

export default ClassifierModel
