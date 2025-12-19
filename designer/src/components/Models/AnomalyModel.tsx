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
import type { AnomalyTestResult } from './types'
import {
  useListAnomalyModels,
  useTrainAndSaveAnomaly,
  useScoreAnomaly,
  useLoadAnomaly,
  useDeleteAnomalyModel,
} from '../../hooks/useMLModels'
import {
  parseNumericTrainingData,
  validateFeatureConsistency,
  parseVersionedModelName,
  formatModelTimestamp,
  generateUniqueModelName,
  type AnomalyBackend,
  type AnomalyModelInfo,
} from '../../types/ml'

type TrainingState = 'idle' | 'training' | 'success' | 'error'

// Map API backend to display name
const BACKEND_OPTIONS: { value: string; label: string; apiValue: AnomalyBackend }[] = [
  { value: 'isolation_forest', label: 'Isolation Forest (recommended)', apiValue: 'isolation_forest' },
  { value: 'one_class_svm', label: 'One-Class SVM', apiValue: 'one_class_svm' },
  { value: 'local_outlier_factor', label: 'Local Outlier Factor', apiValue: 'local_outlier_factor' },
  { value: 'autoencoder', label: 'Autoencoder', apiValue: 'autoencoder' },
]

interface ModelVersion {
  id: string
  versionNumber: number
  versionedName: string
  createdAt: string
  trainingSamples: number
  isActive: boolean
  backend: AnomalyBackend
}

function AnomalyModel() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const isNewModel = !id || id === 'new'

  // Form state - modelName will be set after loading existing models
  const [modelName, setModelName] = useState('')
  const [description, setDescription] = useState('')
  const [trainingData, setTrainingData] = useState('')
  const [trainingDataError, setTrainingDataError] = useState<string | null>(null)
  const [nameExistsWarning, setNameExistsWarning] = useState(false)

  // Settings state
  const [backend, setBackend] = useState<AnomalyBackend>('isolation_forest')
  const [threshold, setThreshold] = useState(0.5)
  const [contamination, setContamination] = useState(0.1)

  // Training state
  const [trainingState, setTrainingState] = useState<TrainingState>('idle')
  const [trainingError, setTrainingError] = useState('')
  const [isTrainingExpanded, setIsTrainingExpanded] = useState(isNewModel)

  // Test state
  const [testInput, setTestInput] = useState('')
  const [testHistory, setTestHistory] = useState<AnomalyTestResult[]>([])

  // Versions - derived from API models with same base name
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [activeVersionName, setActiveVersionName] = useState<string | null>(null)

  // API hooks
  const { data: modelsData, isLoading: isLoadingModels } = useListAnomalyModels()
  const trainAndSaveMutation = useTrainAndSaveAnomaly()
  const scoreMutation = useScoreAnomaly()
  const loadMutation = useLoadAnomaly()
  const deleteMutation = useDeleteAnomalyModel()

  // Parse the model ID to get base name for filtering versions
  const baseModelName = useMemo(() => {
    if (isNewModel) return null
    if (!id) return null
    // The ID could be a base name or a versioned name
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
      const uniqueName = generateUniqueModelName('new-anomaly-model', existingBaseNames)
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
    const matchingModels = modelsData.models.filter((m: AnomalyModelInfo) => {
      const parsed = parseVersionedModelName(m.name)
      return parsed.baseName === baseModelName
    })

    // Sort by timestamp (newest first) and build version list
    const sortedModels = [...matchingModels].sort((a, b) => {
      const parsedA = parseVersionedModelName(a.name)
      const parsedB = parseVersionedModelName(b.name)
      // Newer timestamps should come first
      return (parsedB.timestamp || '').localeCompare(parsedA.timestamp || '')
    })

    const versionList: ModelVersion[] = sortedModels.map((m, index) => ({
      id: m.name,
      versionNumber: sortedModels.length - index, // v1 is oldest, vN is newest
      versionedName: m.name,
      createdAt: m.created || new Date().toISOString(),
      trainingSamples: 0, // Not stored in API response
      isActive: m.name === activeVersionName,
      backend: m.backend,
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

    // Set the model name from the ID
    setModelName(baseModelName)

    // If we have versions, get backend from the first one
    if (versions.length > 0) {
      setBackend(versions[0].backend)
    }
  }, [isNewModel, baseModelName, versions])

  // Validate training data on change
  useEffect(() => {
    if (!trainingData.trim()) {
      setTrainingDataError(null)
      return
    }

    const parsed = parseNumericTrainingData(trainingData)
    if (!parsed) {
      setTrainingDataError('Invalid format. Enter numeric values separated by commas or newlines.')
      return
    }

    const validation = validateFeatureConsistency(parsed)
    if (!validation.valid) {
      setTrainingDataError(validation.error)
      return
    }

    setTrainingDataError(null)
  }, [trainingData])

  const hasVersions = versions.length > 0
  const canTest = hasVersions || trainingState === 'success'
  const parsedData = useMemo(() => parseNumericTrainingData(trainingData), [trainingData])
  const canTrain = modelName.trim() && parsedData && !trainingDataError

  const handleTrain = useCallback(async () => {
    if (!canTrain || !parsedData) return

    setTrainingState('training')
    setTrainingError('')

    // Use unique name if the current name already exists
    const finalModelName = isNewModel
      ? generateUniqueModelName(modelName, existingBaseNames)
      : modelName

    try {
      const result = await trainAndSaveMutation.mutateAsync({
        model: finalModelName,
        backend,
        data: parsedData,
        contamination,
        overwrite: false, // Always version
        description: description.trim() || undefined,
      })

      // Update state with new version
      const newVersionName = result.fitResult.versioned_name
      setActiveVersionName(newVersionName)
      setTrainingState('success')
      setIsTrainingExpanded(false)

      // If new model, redirect to edit page with the base name
      if (isNewModel) {
        navigate(`/chat/models/train/anomaly/${finalModelName}`)
      }
    } catch (error) {
      setTrainingState('error')
      setTrainingError(
        error instanceof Error ? error.message : 'Training failed. Please try again.'
      )
    }
  }, [
    canTrain,
    parsedData,
    modelName,
    backend,
    contamination,
    trainAndSaveMutation,
    isNewModel,
    navigate,
    existingBaseNames,
  ])

  const handleTest = useCallback(async () => {
    if (!testInput.trim() || !activeVersionName) return

    // Parse test input as numeric data
    const testData = parseNumericTrainingData(testInput)
    if (!testData || testData.length === 0) {
      // Show error in test history
      const errorResult: AnomalyTestResult = {
        id: String(Date.now()),
        input: `Error: ${testInput} (invalid numeric format)`,
        isAnomaly: false,
        score: 0,
        threshold,
        timestamp: new Date().toISOString(),
        status: 'error',
      }
      setTestHistory(prev => [errorResult, ...prev])
      setTestInput('')
      return
    }

    try {
      // Ensure model is loaded before scoring
      await loadMutation.mutateAsync({
        model: activeVersionName,
        backend,
      })

      const result = await scoreMutation.mutateAsync({
        model: activeVersionName,
        backend,
        data: testData,
        threshold,
      })

      // Add results to history
      const newResults: AnomalyTestResult[] = result.results.map((r, idx) => ({
        id: `${Date.now()}-${idx}`,
        input: testData[idx].join(', '),
        isAnomaly: r.is_anomaly,
        score: r.score,
        threshold: result.threshold,
        timestamp: new Date().toISOString(),
        status: 'success',
      }))

      setTestHistory(prev => [...newResults, ...prev])
      setTestInput('')
    } catch (error) {
      const errorResult: AnomalyTestResult = {
        id: String(Date.now()),
        input: `${testInput.trim()} — ${error instanceof Error ? error.message : 'Test failed'}`,
        isAnomaly: false,
        score: 0,
        threshold,
        timestamp: new Date().toISOString(),
        status: 'error',
      }
      setTestHistory(prev => [errorResult, ...prev])
    }
    setTestInput('')
  }, [testInput, activeVersionName, backend, threshold, loadMutation, scoreMutation])

  const handleSetActiveVersion = useCallback(
    async (versionName: string) => {
      try {
        await loadMutation.mutateAsync({
          model: versionName,
          backend,
        })
        setActiveVersionName(versionName)
        // Update versions to reflect new active state
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
    [backend, loadMutation]
  )

  const handleDeleteVersion = useCallback(
    async (versionName: string) => {
      const version = versions.find(v => v.versionedName === versionName)
      if (!version) return

      try {
        // Anomaly models are stored as files with backend suffix
        const filename = `${versionName}_${version.backend}.joblib`
        await deleteMutation.mutateAsync(filename)

        // If deleting active version, set next version as active
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
    ? 'New anomaly detection model'
    : modelName || 'Anomaly detection model'

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
                      Provide NUMERIC examples of NORMAL data. One value per line, or
                      comma-separated values on a single line.
                    </p>
                    <Textarea
                      id="training-data"
                      placeholder="Example (one value per line):
23.5
24.1
23.8
24.0"
                      value={trainingData}
                      onChange={e => setTrainingData(e.target.value)}
                      rows={8}
                      className={`font-mono text-sm ${trainingDataError ? 'border-destructive' : ''}`}
                    />
                    {trainingDataError && (
                      <p className="text-xs text-destructive">{trainingDataError}</p>
                    )}
                    {parsedData && !trainingDataError && (
                      <p className="text-xs text-muted-foreground">
                        {parsedData.length} samples{parsedData[0]?.length > 1 ? ` with ${parsedData[0].length} features each` : ''}
                      </p>
                    )}
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
                      <Label htmlFor="backend" className="text-sm">
                        Algorithm
                      </Label>
                      <Select
                        id="backend"
                        value={backend}
                        onChange={e => setBackend(e.target.value as AnomalyBackend)}
                      >
                        {BACKEND_OPTIONS.map(opt => (
                          <option key={opt.value} value={opt.apiValue}>
                            {opt.label}
                          </option>
                        ))}
                      </Select>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Label htmlFor="contamination" className="text-sm">
                        Contamination
                      </Label>
                      <Input
                        id="contamination"
                        type="number"
                        min={0.01}
                        max={0.5}
                        step={0.01}
                        value={contamination}
                        onChange={e => setContamination(parseFloat(e.target.value))}
                      />
                      <p className="text-xs text-muted-foreground">
                        Expected proportion of anomalies in training data (0.01-0.5)
                      </p>
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Label htmlFor="threshold" className="text-sm">
                        Detection threshold
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
                        Scores above this are flagged as anomalies (0-1)
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
              <span className="text-sm font-medium">
                Model trained successfully
              </span>
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
                ? 'Enter a value to check if it would be flagged as an anomaly.'
                : 'Train your model first to enable testing.'}
            </p>
          </div>

          <div className="flex gap-2">
            <Input
              placeholder="e.g., 25.0"
              value={testInput}
              onChange={e => setTestInput(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' && canTest) {
                  handleTest()
                }
              }}
              disabled={!canTest || scoreMutation.isPending}
              className="flex-1 font-mono"
            />
            <Button
              onClick={handleTest}
              variant="secondary"
              disabled={!canTest || scoreMutation.isPending}
            >
              {scoreMutation.isPending ? 'Testing...' : 'Detect'}
            </Button>
          </div>

          {testHistory.length > 0 && (
            <div className="flex flex-col gap-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-muted-foreground">
                  Test history
                </span>
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
                {testHistory.map(result => {
                  const isError = result.status === 'error'
                  const bgClass = isError
                    ? 'bg-amber-100 dark:bg-amber-900/30'
                    : result.isAnomaly
                      ? 'bg-destructive/10'
                      : 'bg-muted/50'

                  const iconType = isError
                    ? 'info'
                    : result.isAnomaly
                      ? 'alert-triangle'
                      : 'checkmark-filled'

                  const iconClass = isError
                    ? 'w-3 h-3 text-amber-600 dark:text-amber-400 shrink-0'
                    : result.isAnomaly
                      ? 'w-3 h-3 text-destructive shrink-0'
                      : 'w-3 h-3 text-primary shrink-0'

                  const labelClass = isError
                    ? 'text-amber-600 dark:text-amber-400'
                    : result.isAnomaly
                      ? 'text-destructive'
                      : 'text-primary'

                  const label = isError
                    ? 'Error'
                    : result.isAnomaly
                      ? 'Anomaly'
                      : 'Normal'

                  return (
                    <div
                      key={result.id}
                      className={`flex items-center gap-2 px-2 py-1 rounded text-sm ${bgClass}`}
                    >
                      <FontIcon type={iconType} className={iconClass} />
                      <span className={`font-medium w-16 shrink-0 ${labelClass}`}>
                        {label}
                      </span>
                      <span className="text-muted-foreground w-10 shrink-0">
                        {isError ? '—' : result.score.toFixed(2)}
                      </span>
                      <span
                        className="text-muted-foreground truncate font-mono text-xs"
                        title={result.input}
                      >
                        {result.input}
                      </span>
                    </div>
                  )
                })}
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
                    <th className="text-left px-4 py-2 font-medium">
                      Model name
                    </th>
                    <th className="text-left px-4 py-2 font-medium">
                      Created
                    </th>
                    <th className="text-left px-4 py-2 font-medium">Backend</th>
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
                          {version.backend}
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

export default AnomalyModel
