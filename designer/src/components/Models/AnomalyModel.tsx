import { useState, useEffect, useCallback, useMemo, useRef } from 'react'
import { SAMPLE_DATASETS } from './sampleData'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Select } from '../ui/select'
import { Textarea } from '../ui/textarea'
import { Badge } from '../ui/badge'
import { Checkbox } from '../ui/checkbox'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import TrainingLoadingOverlay from './TrainingLoadingOverlay'
import type { AnomalyTestResult } from './types'
import {
  useListAnomalyModels,
  useTrainAndSaveAnomaly,
  useScoreAnomaly,
  useLoadAnomaly,
  useDeleteAnomalyModel,
} from '../../hooks/useMLModels'
import {
  parseVersionedModelName,
  formatModelTimestamp,
  generateUniqueModelName,
  ENCODING_TYPE_OPTIONS,
  type AnomalyBackend,
  type AnomalyModelInfo,
  type FeatureColumn,
  type FeatureEncodingType,
} from '../../types/ml'

type TrainingState = 'idle' | 'training' | 'success' | 'error'
type InputMode = 'text' | 'table'

// Map API backend to display name
const BACKEND_OPTIONS: { value: string; label: string; apiValue: AnomalyBackend }[] = [
  { value: 'isolation_forest', label: 'Isolation Forest', apiValue: 'isolation_forest' },
  { value: 'one_class_svm', label: 'One-Class SVM', apiValue: 'one_class_svm' },
  { value: 'local_outlier_factor', label: 'Local Outlier Factor', apiValue: 'local_outlier_factor' },
  { value: 'autoencoder', label: 'Autoencoder', apiValue: 'autoencoder' },
]

interface ModelVersion {
  id: string
  versionNumber: number
  versionedName: string
  filename: string
  createdAt: string
  trainingSamples: number
  isActive: boolean
  backend: AnomalyBackend
}

interface TableRow {
  id: string
  values: Record<string, string>
}

// Parse pasted table data (TSV or CSV)
function parseTableData(
  input: string
): { columns: FeatureColumn[]; rows: TableRow[] } | null {
  const lines = input.trim().split('\n')
  if (lines.length < 2) return null // Need header + at least 1 row

  // Detect delimiter (tab or comma)
  const delimiter = lines[0].includes('\t') ? '\t' : ','
  const headers = lines[0].split(delimiter).map(h => h.trim())

  if (headers.length === 0 || headers.some(h => !h)) return null

  const rows: TableRow[] = []
  const sampleValues: Record<string, string[]> = {}

  // Initialize sample collection
  headers.forEach(h => {
    sampleValues[h] = []
  })

  // Parse rows
  for (let i = 1; i < lines.length; i++) {
    const values = lines[i].split(delimiter).map(v => v.trim())
    if (values.length !== headers.length) continue // Skip malformed rows

    const row: TableRow = {
      id: String(i),
      values: {},
    }

    headers.forEach((header, idx) => {
      row.values[header] = values[idx]
      sampleValues[header].push(values[idx])
    })

    rows.push(row)
  }

  if (rows.length === 0) return null

  // Infer column types from sample values
  const columns: FeatureColumn[] = headers.map(name => {
    const samples = sampleValues[name]
    const uniqueCount = new Set(samples).size

    // Check if all values are numeric
    const allNumeric = samples.every(v => !isNaN(parseFloat(v)) && v.trim() !== '')
    if (allNumeric) {
      return { name, type: 'numeric' as FeatureEncodingType }
    }

    // Check for binary
    const allBinary = samples.every(v =>
      ['true', 'false', 'yes', 'no', '0', '1', 'on', 'off'].includes(
        v.toLowerCase().trim()
      )
    )
    if (allBinary) {
      return { name, type: 'binary' as FeatureEncodingType }
    }

    // High cardinality -> hash, low cardinality -> label
    if (uniqueCount > 20) {
      return { name, type: 'hash' as FeatureEncodingType }
    }
    return { name, type: 'label' as FeatureEncodingType }
  })

  return { columns, rows }
}

// Parse simple text input (numeric only, backwards compatible)
function parseNumericTrainingData(input: string): number[][] | null {
  try {
    const lines = input
      .split(/[\n]/)
      .map(line => line.trim())
      .filter(Boolean)

    const data: number[][] = []

    for (const line of lines) {
      const values = line
        .split(/[,\s]+/)
        .map(v => v.trim())
        .filter(Boolean)
        .map(v => parseFloat(v))

      if (values.some(isNaN)) {
        return null
      }

      if (values.length > 0) {
        data.push(values)
      }
    }

    return data.length > 0 ? data : null
  } catch {
    return null
  }
}

// Convert table data to text format (comma-separated values per line)
function tableToText(columns: FeatureColumn[], rows: TableRow[]): string {
  if (columns.length === 0 || rows.length === 0) return ''

  const lines = rows.map(row => {
    return columns.map(col => row.values[col.name] || '').join(', ')
  })

  return lines.join('\n')
}

// Convert text to table format - handles tab/comma separated data
function textToTable(
  input: string
): { columns: FeatureColumn[]; rows: TableRow[] } | null {
  const lines = input.trim().split('\n').filter(Boolean)
  if (lines.length === 0) return null

  // Detect delimiter - prefer tab, then comma
  const hasTab = lines[0].includes('\t')
  const delimiter = hasTab ? '\t' : ','

  // Parse all rows
  const parsedRows = lines.map(line =>
    line.split(delimiter).map(v => v.trim())
  )

  // Check consistency - all rows should have same number of columns
  const colCount = parsedRows[0].length
  if (colCount === 0) return null

  // Generate column names (col_1, col_2, etc.)
  const columns: FeatureColumn[] = []
  for (let i = 0; i < colCount; i++) {
    // Infer type from all values in this column
    const colValues = parsedRows.map(row => row[i] || '')
    const allNumeric = colValues.every(
      v => !isNaN(parseFloat(v)) && v.trim() !== ''
    )
    columns.push({
      name: `col_${i + 1}`,
      type: allNumeric ? 'numeric' : 'label',
    })
  }

  // Create rows
  const rows: TableRow[] = parsedRows.map((values, idx) => ({
    id: String(idx + 1),
    values: columns.reduce(
      (acc, col, colIdx) => {
        acc[col.name] = values[colIdx] || ''
        return acc
      },
      {} as Record<string, string>
    ),
  }))

  return { columns, rows }
}

// Format pasted text - convert tabs to commas for cleaner display
function formatPastedText(input: string): string {
  // If it has tabs (from spreadsheet paste), convert to comma-separated
  if (input.includes('\t')) {
    return input
      .split('\n')
      .map(line => line.split('\t').map(v => v.trim()).join(', '))
      .join('\n')
  }
  return input
}


function AnomalyModel() {
  const navigate = useNavigate()
  const { id } = useParams<{ id: string }>()
  const isNewModel = !id || id === 'new'

  // Form state
  const [modelName, setModelName] = useState('')
  const [description, setDescription] = useState('')
  const [nameExistsWarning, setNameExistsWarning] = useState(false)

  // Input mode: text (textarea) or table
  const [inputMode, setInputMode] = useState<InputMode>('text')

  // Text mode state
  const [trainingData, setTrainingData] = useState('')
  const [trainingDataError, setTrainingDataError] = useState<string | null>(null)

  // Table mode state
  const [columns, setColumns] = useState<FeatureColumn[]>([])
  const [tableRows, setTableRows] = useState<TableRow[]>([])

  // CSV import modal state
  const [showCsvModal, setShowCsvModal] = useState(false)
  const [csvFile, setCsvFile] = useState<File | null>(null)
  const [csvFirstRowIsHeader, setCsvFirstRowIsHeader] = useState(true)
  const [isDraggingCsv, setIsDraggingCsv] = useState(false)
  const [isDraggingTrainingArea, setIsDraggingTrainingArea] = useState(false)
  const csvFileInputRef = useRef<HTMLInputElement>(null)
  const trainingAreaRef = useRef<HTMLDivElement>(null)
  const { toast } = useToast()

  // Sample data modal state
  const [showSampleDataModal, setShowSampleDataModal] = useState(false)
  const [selectedSampleDataset, setSelectedSampleDataset] = useState<string | null>(null)
  const [isImportingSampleData, setIsImportingSampleData] = useState(false)

  // Track if user has interacted with training data (for showing low entry warning)
  const [hasBlurredTrainingData, setHasBlurredTrainingData] = useState(false)

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

  // Versions
  const [versions, setVersions] = useState<ModelVersion[]>([])
  const [activeVersionName, setActiveVersionName] = useState<string | null>(null)

  // API hooks
  const { data: modelsData, isLoading: isLoadingModels } = useListAnomalyModels()
  const trainAndSaveMutation = useTrainAndSaveAnomaly()
  const scoreMutation = useScoreAnomaly()
  const loadMutation = useLoadAnomaly()
  const deleteMutation = useDeleteAnomalyModel()

  // Parse the model ID to get base name
  const baseModelName = useMemo(() => {
    if (isNewModel) return null
    if (!id) return null
    const parsed = parseVersionedModelName(id)
    return parsed.baseName
  }, [id, isNewModel])

  // Extract all existing base names
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

  // Set unique default model name for new models
  useEffect(() => {
    if (isNewModel && !modelName && !isLoadingModels) {
      const uniqueName = generateUniqueModelName('new-anomaly-model', existingBaseNames)
      setModelName(uniqueName)
    }
  }, [isNewModel, modelName, isLoadingModels, existingBaseNames])

  // Check if model name already exists
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

    const matchingModels = modelsData.models.filter((m: AnomalyModelInfo) => {
      const parsed = parseVersionedModelName(m.name)
      return parsed.baseName === baseModelName
    })

    const sortedModels = [...matchingModels].sort((a, b) => {
      const parsedA = parseVersionedModelName(a.name)
      const parsedB = parseVersionedModelName(b.name)
      return (parsedB.timestamp || '').localeCompare(parsedA.timestamp || '')
    })

    const versionList: ModelVersion[] = sortedModels.map((m, index) => ({
      id: m.name,
      versionNumber: sortedModels.length - index,
      versionedName: m.name,
      filename: m.filename,
      createdAt: m.created || new Date().toISOString(),
      trainingSamples: 0,
      isActive: m.name === activeVersionName,
      backend: m.backend,
    }))

    setVersions(versionList)

    if (!activeVersionName && versionList.length > 0) {
      setActiveVersionName(versionList[0].versionedName)
    }
  }, [modelsData, baseModelName, activeVersionName])

  // Load model metadata when editing existing model
  useEffect(() => {
    if (isNewModel || !baseModelName) return
    setModelName(baseModelName)
    if (versions.length > 0) {
      setBackend(versions[0].backend)
    }
  }, [isNewModel, baseModelName, versions])

  // Validate training data on change (text mode only)
  useEffect(() => {
    if (inputMode !== 'text') {
      setTrainingDataError(null)
      return
    }

    if (!trainingData.trim()) {
      setTrainingDataError(null)
      return
    }

    // Try to parse as table first (has headers)
    const tableResult = parseTableData(trainingData)
    if (tableResult && tableResult.columns.length > 0) {
      // Valid table data - no error, will be converted on train
      setTrainingDataError(null)
      return
    }

    // Try numeric parsing - if successful, check feature consistency
    const parsed = parseNumericTrainingData(trainingData)
    if (parsed) {
      // Check feature consistency for numeric data
      const featureCount = parsed[0].length
      const inconsistentRow = parsed.findIndex(row => row.length !== featureCount)
      if (inconsistentRow !== -1) {
        setTrainingDataError(
          `Row ${inconsistentRow + 1} has ${parsed[inconsistentRow].length} features, expected ${featureCount}`
        )
        return
      }
    }

    // Text data is allowed - no error for non-numeric input
    setTrainingDataError(null)
  }, [trainingData, inputMode])

  // Handle paste in table mode
  const handleTablePaste = useCallback(
    (e: React.ClipboardEvent<HTMLTextAreaElement>) => {
      const pastedText = e.clipboardData.getData('text')
      const result = parseTableData(pastedText)
      if (result) {
        setColumns(result.columns)
        setTableRows(result.rows)
      }
    },
    []
  )

  // Prepare data for API
  const prepareTrainingData = useCallback((): {
    data: number[][] | Record<string, unknown>[]
    schema?: Record<string, FeatureEncodingType>
  } | null => {
    if (inputMode === 'table') {
      if (columns.length === 0 || tableRows.length === 0) return null

      const schema: Record<string, FeatureEncodingType> = {}
      columns.forEach(col => {
        schema[col.name] = col.type
      })

      const data = tableRows.map(row => {
        const rowData: Record<string, unknown> = {}
        columns.forEach(col => {
          const value = row.values[col.name]
          // Convert numeric strings to numbers
          if (col.type === 'numeric') {
            rowData[col.name] = parseFloat(value) || 0
          } else {
            rowData[col.name] = value
          }
        })
        return rowData
      })

      return { data, schema }
    } else {
      // Text mode - try table parse first, then numeric
      const tableResult = parseTableData(trainingData)
      if (tableResult && tableResult.columns.length > 0) {
        const schema: Record<string, FeatureEncodingType> = {}
        tableResult.columns.forEach(col => {
          schema[col.name] = col.type
        })

        const data = tableResult.rows.map(row => {
          const rowData: Record<string, unknown> = {}
          tableResult.columns.forEach(col => {
            const value = row.values[col.name]
            if (col.type === 'numeric') {
              rowData[col.name] = parseFloat(value) || 0
            } else {
              rowData[col.name] = value
            }
          })
          return rowData
        })

        return { data, schema }
      }

      // Numeric only
      const parsed = parseNumericTrainingData(trainingData)
      if (!parsed) return null
      return { data: parsed }
    }
  }, [inputMode, trainingData, columns, tableRows])

  const hasVersions = versions.length > 0
  const canTest = hasVersions || trainingState === 'success'

  const canTrain = useMemo(() => {
    if (!modelName.trim()) return false
    const prepared = prepareTrainingData()
    return prepared !== null && !trainingDataError
  }, [modelName, prepareTrainingData, trainingDataError])

  const dataStats = useMemo(() => {
    if (inputMode === 'table') {
      return {
        rows: tableRows.length,
        cols: columns.length,
        hasSchema: columns.length > 0,
      }
    }

    const tableResult = parseTableData(trainingData)
    if (tableResult) {
      return {
        rows: tableResult.rows.length,
        cols: tableResult.columns.length,
        hasSchema: true,
      }
    }

    const parsed = parseNumericTrainingData(trainingData)
    if (parsed) {
      return {
        rows: parsed.length,
        cols: parsed[0]?.length || 0,
        hasSchema: false,
      }
    }

    return null
  }, [inputMode, trainingData, columns, tableRows])

  const handleTrain = useCallback(async () => {
    const prepared = prepareTrainingData()
    if (!canTrain || !prepared) return

    setTrainingState('training')
    setTrainingError('')

    const finalModelName = isNewModel
      ? generateUniqueModelName(modelName, existingBaseNames)
      : modelName

    try {
      const result = await trainAndSaveMutation.mutateAsync({
        model: finalModelName,
        backend,
        data: prepared.data,
        schema: prepared.schema,
        contamination,
        overwrite: false,
        description: description.trim() || undefined,
      })

      const newVersionName = result.fitResult.versioned_name
      setActiveVersionName(newVersionName)
      setTrainingState('success')
      setIsTrainingExpanded(false)

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
    prepareTrainingData,
    modelName,
    backend,
    contamination,
    trainAndSaveMutation,
    isNewModel,
    navigate,
    existingBaseNames,
    description,
  ])

  const handleTest = useCallback(async () => {
    if (!testInput.trim() || !activeVersionName) return

    // Parse test input - try table format, then single row with existing schema, then numeric
    const tableResult = parseTableData(testInput)
    let testData: number[][] | Record<string, unknown>[]
    let schema: Record<string, FeatureEncodingType> | undefined

    if (tableResult && tableResult.columns.length > 0) {
      // Full table with headers
      schema = {}
      tableResult.columns.forEach(col => {
        schema![col.name] = col.type
      })
      testData = tableResult.rows.map(row => {
        const rowData: Record<string, unknown> = {}
        tableResult.columns.forEach(col => {
          const value = row.values[col.name]
          if (col.type === 'numeric') {
            rowData[col.name] = parseFloat(value) || 0
          } else {
            rowData[col.name] = value
          }
        })
        return rowData
      })
    } else if (columns.length > 0) {
      // Try to parse as a single row using existing column schema
      const delimiter = testInput.includes('\t') ? '\t' : ','
      const values = testInput.trim().split(delimiter).map(v => v.trim())

      if (values.length === columns.length) {
        // Values match column count - use existing schema
        schema = {}
        columns.forEach(col => {
          schema![col.name] = col.type
        })
        const rowData: Record<string, unknown> = {}
        columns.forEach((col, idx) => {
          const value = values[idx]
          if (col.type === 'numeric') {
            rowData[col.name] = parseFloat(value) || 0
          } else {
            rowData[col.name] = value
          }
        })
        testData = [rowData]
      } else {
        // Fall back to numeric parsing
        const numericData = parseNumericTrainingData(testInput)
        if (!numericData || numericData.length === 0) {
          const errorResult: AnomalyTestResult = {
            id: String(Date.now()),
            input: `Error: Expected ${columns.length} values (${columns.map(c => c.name).join(', ')}), got ${values.length}`,
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
        testData = numericData
      }
    } else {
      const numericData = parseNumericTrainingData(testInput)
      if (!numericData || numericData.length === 0) {
        const errorResult: AnomalyTestResult = {
          id: String(Date.now()),
          input: `Error: ${testInput} (invalid format)`,
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
      testData = numericData
    }

    try {
      await loadMutation.mutateAsync({
        model: activeVersionName,
        backend,
      })

      const result = await scoreMutation.mutateAsync({
        model: activeVersionName,
        backend,
        data: testData,
        schema,
        threshold,
      })

      const newResults: AnomalyTestResult[] = result.results.map((r, idx) => ({
        id: `${Date.now()}-${idx}`,
        input: Array.isArray(testData[idx])
          ? (testData[idx] as number[]).join(', ')
          : JSON.stringify(testData[idx]),
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
  }, [testInput, activeVersionName, backend, threshold, loadMutation, scoreMutation, columns])

  const handleSetActiveVersion = useCallback(
    async (versionName: string) => {
      try {
        await loadMutation.mutateAsync({
          model: versionName,
          backend,
        })
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
    [backend, loadMutation]
  )

  const handleDeleteVersion = useCallback(
    async (versionName: string) => {
      const version = versions.find(v => v.versionedName === versionName)
      if (!version) return

      const versionLabel = `Version ${version.versionNumber}`
      const confirmMessage = `Delete ${versionLabel}? This cannot be undone.`
      if (!window.confirm(confirmMessage)) return

      try {
        // Pass the model name - backend will handle finding the file
        await deleteMutation.mutateAsync(versionName)

        toast({
          message: `Successfully deleted ${versionLabel}.`,
          icon: 'checkmark-filled',
        })

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
        toast({
          message: 'Failed to delete version. Please try again.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
      }
    },
    [versions, activeVersionName, deleteMutation, toast]
  )

  const handleDeleteModel = useCallback(async () => {
    if (!baseModelName || isNewModel) return

    const confirmMessage = `Delete "${baseModelName}" and all ${versions.length} version${versions.length !== 1 ? 's' : ''}? This cannot be undone.`
    if (!window.confirm(confirmMessage)) return

    try {
      // Delete all versions
      await Promise.all(versions.map(v => deleteMutation.mutateAsync(v.versionedName)))

      toast({
        message: `Successfully deleted ${baseModelName} and all its versions.`,
        icon: 'checkmark-filled',
      })

      // Navigate back to models list
      navigate('/chat/models?tab=training')
    } catch (error) {
      console.error('Failed to delete model:', error)
      toast({
        message: 'Failed to delete some model versions. Please try again.',
        variant: 'destructive',
        icon: 'alert-triangle',
      })
    }
  }, [baseModelName, isNewModel, versions, deleteMutation, toast, navigate])

  const handleColumnTypeChange = useCallback(
    (colName: string, newType: FeatureEncodingType) => {
      setColumns(prev =>
        prev.map(col => (col.name === colName ? { ...col, type: newType } : col))
      )
    },
    []
  )

  const handleAddColumn = useCallback(() => {
    const newName = `column_${columns.length + 1}`
    setColumns(prev => [...prev, { name: newName, type: 'numeric' }])
    setTableRows(prev =>
      prev.map(row => ({
        ...row,
        values: { ...row.values, [newName]: '' },
      }))
    )
  }, [columns.length])

  const handleRemoveColumn = useCallback(
    (colIdx: number) => {
      const colName = columns[colIdx]?.name
      if (!colName) return

      setColumns(prev => prev.filter((_, i) => i !== colIdx))
      setTableRows(prev =>
        prev.map(row => {
          const newValues = { ...row.values }
          delete newValues[colName]
          return { ...row, values: newValues }
        })
      )
    },
    [columns]
  )

  const handleAddRow = useCallback(() => {
    const newRow: TableRow = {
      id: String(Date.now()),
      values: {},
    }
    columns.forEach(col => {
      newRow.values[col.name] = ''
    })
    setTableRows(prev => [...prev, newRow])
  }, [columns])

  const handleCellChange = useCallback(
    (rowId: string, colName: string, value: string) => {
      setTableRows(prev =>
        prev.map(row =>
          row.id === rowId
            ? { ...row, values: { ...row.values, [colName]: value } }
            : row
        )
      )
    },
    []
  )

  // Handle paste in table cells - expand into multiple rows AND columns if needed
  const handleCellPaste = useCallback(
    (
      e: React.ClipboardEvent<HTMLInputElement>,
      rowId: string,
      colIdx: number
    ) => {
      const pastedText = e.clipboardData.getData('text')

      // Check if it's multi-line or tab-separated (spreadsheet paste)
      const hasMultipleLines = pastedText.includes('\n')
      const hasTabs = pastedText.includes('\t')

      if (!hasMultipleLines && !hasTabs) {
        // Single value - let default behavior handle it
        return
      }

      e.preventDefault()

      // Parse the pasted data
      const lines = pastedText.trim().split('\n')
      const delimiter = hasTabs ? '\t' : ','

      const parsedRows = lines.map(line =>
        line.split(delimiter).map(v => v.trim())
      )

      // Find the current row index
      const currentRowIdx = tableRows.findIndex(r => r.id === rowId)
      if (currentRowIdx === -1) return

      // Calculate max columns needed
      const maxPastedCols = Math.max(...parsedRows.map(row => row.length))
      const totalColsNeeded = colIdx + maxPastedCols

      // Add new columns if needed
      let updatedColumns = [...columns]
      if (totalColsNeeded > columns.length) {
        const newColsCount = totalColsNeeded - columns.length
        for (let i = 0; i < newColsCount; i++) {
          const newColName = `col_${columns.length + i + 1}`
          updatedColumns.push({ name: newColName, type: 'label' as const })
        }
        setColumns(updatedColumns)
      }

      // Update rows with the updated column list
      setTableRows(prev => {
        const newRows = [...prev]

        // First, add empty values for any new columns to existing rows
        if (updatedColumns.length > columns.length) {
          newRows.forEach(row => {
            updatedColumns.forEach(col => {
              if (!(col.name in row.values)) {
                row.values[col.name] = ''
              }
            })
          })
        }

        parsedRows.forEach((values, pasteRowIdx) => {
          const targetRowIdx = currentRowIdx + pasteRowIdx
          const targetColIdx = colIdx

          if (targetRowIdx < newRows.length) {
            // Update existing row
            const row = newRows[targetRowIdx]
            values.forEach((value, valueColIdx) => {
              const col = updatedColumns[targetColIdx + valueColIdx]
              if (col) {
                row.values[col.name] = value
              }
            })
          } else {
            // Create new row
            const newRow: TableRow = {
              id: String(Date.now() + pasteRowIdx),
              values: {},
            }
            updatedColumns.forEach((col, idx) => {
              const valueIdx = idx - targetColIdx
              if (valueIdx >= 0 && valueIdx < values.length) {
                newRow.values[col.name] = values[valueIdx]
              } else {
                newRow.values[col.name] = ''
              }
            })
            newRows.push(newRow)
          }
        })

        // Remove empty rows (rows where all values are empty)
        return newRows.filter(row =>
          Object.values(row.values).some(v => v.trim() !== '')
        )
      })
    },
    [columns, tableRows]
  )

  // CSV import handlers
  const handleCsvFileSelect = useCallback((file: File) => {
    setCsvFile(file)
  }, [])

  const handleCsvDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDraggingCsv(false)
      const file = e.dataTransfer.files[0]
      if (file && (file.name.endsWith('.csv') || file.type === 'text/csv')) {
        handleCsvFileSelect(file)
      }
    },
    [handleCsvFileSelect]
  )

  const handleCsvImport = useCallback(() => {
    if (!csvFile) return

    const reader = new FileReader()
    reader.onload = e => {
      const text = e.target?.result as string
      if (!text) return

      const lines = text.trim().split('\n').filter(Boolean)
      if (lines.length === 0) return

      // Detect delimiter
      const hasTab = lines[0].includes('\t')
      const delimiter = hasTab ? '\t' : ','

      // Parse all rows
      const allRows = lines.map(line =>
        line.split(delimiter).map(v => v.trim().replace(/^"|"$/g, '')) // Remove surrounding quotes
      )

      // Determine headers and data rows
      let headers: string[]
      let dataRows: string[][]

      if (csvFirstRowIsHeader && allRows.length > 1) {
        headers = allRows[0]
        dataRows = allRows.slice(1)
      } else {
        // Generate column names
        const colCount = allRows[0].length
        headers = Array.from({ length: colCount }, (_, i) => `col_${i + 1}`)
        dataRows = allRows
      }

      // Create columns with type inference
      const newColumns: FeatureColumn[] = headers.map((name, idx) => {
        // Check if all values in this column are numeric
        const colValues = dataRows.map(row => row[idx] || '')
        const allNumeric = colValues.every(
          v => v === '' || (!isNaN(parseFloat(v)) && isFinite(Number(v)))
        )
        return {
          name: name || `col_${idx + 1}`,
          type: allNumeric ? 'numeric' : 'label',
        }
      })

      // Create table rows
      const newRows: TableRow[] = dataRows.map((values, rowIdx) => ({
        id: String(Date.now() + rowIdx),
        values: newColumns.reduce(
          (acc, col, colIdx) => {
            acc[col.name] = values[colIdx] || ''
            return acc
          },
          {} as Record<string, string>
        ),
      }))

      // Update state
      setColumns(newColumns)
      setTableRows(newRows)
      setInputMode('table')

      // Close modal and reset
      setShowCsvModal(false)
      setCsvFile(null)
    }
    reader.readAsText(csvFile)
  }, [csvFile, csvFirstRowIsHeader])

  const handleCsvModalClose = useCallback(() => {
    setShowCsvModal(false)
    setCsvFile(null)
    setIsDraggingCsv(false)
  }, [])

  // Sample data handler - import the selected dataset with loading state
  const handleImportSampleData = useCallback(() => {
    if (!selectedSampleDataset) return

    const dataset = SAMPLE_DATASETS.find(d => d.id === selectedSampleDataset)
    if (!dataset?.data) {
      toast({
        message: 'Sample data not available.',
        variant: 'destructive',
        icon: 'alert-triangle',
      })
      return
    }

    setIsImportingSampleData(true)

    // Simulate a short loading state for better UX
    setTimeout(() => {
      // For multi-column datasets, use table view; for single column, use text view
      if (dataset.columns > 1) {
        const result = textToTable(dataset.data)
        if (result) {
          setColumns(result.columns)
          setTableRows(result.rows)
          setInputMode('table')
        } else {
          setTrainingData(dataset.data)
          setInputMode('text')
        }
      } else {
        setTrainingData(dataset.data)
        setInputMode('text')
      }
      setShowSampleDataModal(false)
      setSelectedSampleDataset(null)
      setIsImportingSampleData(false)
    }, 600)
  }, [selectedSampleDataset, toast])

  // Training area drag handlers
  const handleTrainingAreaDragEnter = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDraggingTrainingArea(true)
    },
    []
  )

  const handleTrainingAreaDragOver = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      e.dataTransfer.dropEffect = 'copy'
    },
    []
  )

  const handleTrainingAreaDragLeave = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      // Only set isDragging to false if we're actually leaving the drop zone
      const rect = trainingAreaRef.current?.getBoundingClientRect()
      const isLeavingZone =
        rect &&
        (e.clientX <= rect.left ||
          e.clientX >= rect.right ||
          e.clientY <= rect.top ||
          e.clientY >= rect.bottom)
      if (isLeavingZone) {
        setIsDraggingTrainingArea(false)
      }
    },
    []
  )

  const handleTrainingAreaDrop = useCallback(
    (e: React.DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDraggingTrainingArea(false)

      const file = e.dataTransfer.files[0]
      if (!file) return

      // Check if it's a CSV file
      const isCsv =
        file.name.toLowerCase().endsWith('.csv') || file.type === 'text/csv'

      if (isCsv) {
        // Valid CSV - open modal with file pre-populated
        setCsvFile(file)
        setShowCsvModal(true)
      } else {
        // Invalid file type - show error toast
        toast({
          message: 'Only CSV files are supported. Please drop a .csv file.',
          variant: 'destructive',
          icon: 'alert-triangle',
        })
      }
    },
    [toast]
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
          <div className="flex items-center gap-2">
            {!isNewModel && versions.length > 0 && (
              <Button
                variant="ghost"
                onClick={handleDeleteModel}
                className="text-sm text-destructive/70 hover:text-destructive hover:bg-destructive/5"
              >
                Delete
              </Button>
            )}
            <Button
              variant="outline"
              onClick={() => navigate('/chat/models?tab=training')}
            >
              Done
            </Button>
          </div>
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
              className={nameExistsWarning ? 'border-amber-500' : ''}
            />
            {nameExistsWarning ? (
              <p className="text-xs text-amber-600 dark:text-amber-400">
                A model with this name exists. Will be saved as "
                {generateUniqueModelName(modelName, existingBaseNames)}".
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
        <div className={`rounded-lg border border-border bg-card p-4 flex flex-col gap-4 relative transition-all duration-300 ${trainingState === 'training' ? 'h-[400px] overflow-hidden' : ''}`}>
          {trainingState === 'training' && <TrainingLoadingOverlay message="Training your anomaly detector..." />}
          {/* Collapsed view */}
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
              {/* Settings row - horizontal across top */}
              <div className="flex flex-wrap items-end gap-4 pb-3 border-b border-border">
                <div className="flex flex-col gap-1">
                  <Label htmlFor="backend" className="text-xs text-muted-foreground">
                    Algorithm
                  </Label>
                  <Select
                    id="backend"
                    value={backend}
                    onChange={e => setBackend(e.target.value as AnomalyBackend)}
                    className="w-48"
                  >
                    {BACKEND_OPTIONS.map(opt => (
                      <option key={opt.value} value={opt.apiValue}>
                        {opt.label}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="contamination" className="text-xs text-muted-foreground">
                    Contamination
                  </Label>
                  <Input
                    id="contamination"
                    type="number"
                    min={0.01}
                    max={0.5}
                    step={0.01}
                    value={contamination}
                    onChange={e => {
                      const val = parseFloat(e.target.value)
                      setContamination(isNaN(val) ? 0.1 : val)
                    }}
                    className="w-24"
                  />
                </div>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="threshold" className="text-xs text-muted-foreground">
                    Threshold
                  </Label>
                  <Input
                    id="threshold"
                    type="number"
                    min={0}
                    max={1}
                    step={0.1}
                    value={threshold}
                    onChange={e => {
                      const val = parseFloat(e.target.value)
                      setThreshold(isNaN(val) ? 0.5 : val)
                    }}
                    className="w-24"
                  />
                </div>
                <div className="flex-1" />
                {hasVersions && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setIsTrainingExpanded(false)}
                    className="h-8"
                  >
                    <FontIcon type="chevron-up" className="w-4 h-4 mr-1" />
                    Collapse
                  </Button>
                )}
              </div>

              {/* Training Data section */}
              <div
                ref={trainingAreaRef}
                className="flex flex-col gap-3 rounded-lg transition-colors relative"
                onDragEnter={handleTrainingAreaDragEnter}
                onDragOver={handleTrainingAreaDragOver}
                onDragLeave={handleTrainingAreaDragLeave}
                onDrop={handleTrainingAreaDrop}
              >
                {/* Drop overlay */}
                {isDraggingTrainingArea && (
                  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center rounded-lg border-2 border-dashed border-primary bg-primary/5 backdrop-blur-[2px]">
                    <div className="flex flex-col items-center gap-3 text-center p-6">
                      <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                        <FontIcon type="upload" className="w-6 h-6 text-primary" />
                      </div>
                      <div className="text-lg font-medium text-foreground">
                        Drop CSV here
                      </div>
                      <p className="text-sm text-muted-foreground max-w-[300px]">
                        Release to import your CSV file as training data
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex items-center justify-between">
                  <div className="flex flex-col gap-0.5">
                    <Label className="text-sm font-medium">
                      Training data{' '}
                      {isNewModel && <span className="text-destructive">*</span>}
                    </Label>
                    <p className="text-xs text-muted-foreground">
                      Provide examples of NORMAL data—the model learns this pattern and flags anything that deviates.
                      Paste from a spreadsheet, drop a CSV file, or type values directly. Each line is one entry.
                      We recommend at least 50 entries for reliable results.
                    </p>
                  </div>
                  <div className="flex items-center gap-2 ml-6 shrink-0">
                    <button
                      onClick={() => setShowSampleDataModal(true)}
                      className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground border border-border rounded-md hover:bg-muted/50 transition-colors"
                    >
                      <FontIcon type="data" className="w-3.5 h-3.5" />
                      Use sample data
                    </button>
                    <button
                      onClick={() => setShowCsvModal(true)}
                      className="flex items-center gap-1.5 px-2.5 py-1 text-xs text-muted-foreground hover:text-foreground border border-border rounded-md hover:bg-muted/50 transition-colors"
                    >
                      <FontIcon type="upload" className="w-3.5 h-3.5" />
                      Import CSV
                    </button>
                    <div className="flex items-center gap-1 bg-muted rounded-md p-0.5">
                      <button
                        onClick={() => {
                          // Switching to text mode - convert table data to text
                          if (inputMode === 'table' && columns.length > 0 && tableRows.length > 0) {
                            const text = tableToText(columns, tableRows)
                            setTrainingData(text)
                          }
                          setInputMode('text')
                        }}
                        className={`px-2.5 py-1 text-xs rounded transition-colors ${
                          inputMode === 'text'
                            ? 'bg-background text-foreground shadow-sm'
                            : 'text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        Text
                      </button>
                      <button
                        onClick={() => {
                          // Switching to table mode - convert text data to table
                          if (inputMode === 'text' && trainingData.trim()) {
                            const result = textToTable(trainingData)
                            if (result) {
                              setColumns(result.columns)
                              setTableRows(result.rows)
                            }
                          }
                          setInputMode('table')
                      }}
                      className={`px-2.5 py-1 text-xs rounded transition-colors ${
                        inputMode === 'table'
                          ? 'bg-background text-foreground shadow-sm'
                          : 'text-muted-foreground hover:text-foreground'
                        }`}
                      >
                        Table
                      </button>
                    </div>
                  </div>
                </div>

                {inputMode === 'text' ? (
                  <div className="flex flex-col gap-1.5">
                    <Textarea
                      id="training-data"
                      placeholder={`Examples:

Arrays (each line is one entry):
1, 2, 3, 4
4, 5, 7, 7
3, 2, 1, 2

Text (one per line):
US
CA
UK
MX`}
                      value={trainingData}
                      onChange={e => setTrainingData(e.target.value)}
                      onPaste={e => {
                        // Format pasted text (convert tabs to commas)
                        const pastedText = e.clipboardData.getData('text')
                        if (pastedText.includes('\t')) {
                          e.preventDefault()
                          const formatted = formatPastedText(pastedText)
                          // Insert at cursor position or replace selection
                          const textarea = e.currentTarget
                          const start = textarea.selectionStart
                          const end = textarea.selectionEnd
                          const newValue =
                            trainingData.substring(0, start) +
                            formatted +
                            trainingData.substring(end)
                          setTrainingData(newValue)
                        }
                      }}
                      className={`font-mono text-sm max-h-[50vh] min-h-[200px] resize-y ${trainingDataError ? 'border-destructive' : ''}`}
                      onBlur={() => setHasBlurredTrainingData(true)}
                    />
                    {trainingDataError ? (
                      <p className="text-xs text-destructive">{trainingDataError}</p>
                    ) : dataStats ? (
                      <p className="text-xs text-muted-foreground">
                        {dataStats.rows} samples
                        {dataStats.cols > 1 ? ` × ${dataStats.cols} features` : ''}
                        {dataStats.hasSchema && ' (with schema)'}
                        {hasBlurredTrainingData && dataStats.rows > 0 && dataStats.rows < 50 && (
                          <span className="text-amber-600 dark:text-amber-400">
                            {' — Please add more entries to increase accuracy'}
                          </span>
                        )}
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground">
                        Each line = one entry. Paste from a spreadsheet or type values directly.
                      </p>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-col gap-3">
                    {/* Paste area for table mode */}
                    {columns.length === 0 && (
                      <div className="border-2 border-dashed border-border rounded-lg p-6 text-center">
                        <Textarea
                          placeholder="Paste table data here (TSV or CSV with headers)..."
                          className="border-0 bg-transparent resize-none text-center focus-visible:ring-0"
                          rows={3}
                          onPaste={handleTablePaste}
                        />
                        <p className="text-xs text-muted-foreground mt-2">
                          Or{' '}
                          <button
                            className="text-primary hover:underline"
                            onClick={handleAddColumn}
                          >
                            add columns manually
                          </button>
                        </p>
                      </div>
                    )}

                    {/* Table editor - spreadsheet style */}
                    {columns.length > 0 && (
                      <div className="border border-border rounded-lg max-h-[50vh] min-h-[200px] overflow-auto">
                        <table className="w-full text-xs border-collapse">
                            {/* Header row with column names - sticky */}
                            <thead className="sticky top-0 z-10">
                              <tr className="bg-muted border-b border-border">
                                <th className="w-8 px-2 py-1.5 text-center text-muted-foreground font-normal border-r border-border">
                                  #
                                </th>
                                {columns.map((col, idx) => (
                                  <th
                                    key={idx}
                                    className={`px-0 py-0 font-medium min-w-[100px] ${idx < columns.length - 1 ? 'border-r border-border' : ''}`}
                                  >
                                    <div className="flex items-center pr-1">
                                      <input
                                        value={col.name}
                                        onChange={e => {
                                          const newName = e.target.value
                                          setColumns(prev =>
                                            prev.map((c, i) =>
                                              i === idx ? { ...c, name: newName } : c
                                            )
                                          )
                                          // Update row values to use new column name
                                          const oldName = col.name
                                          if (oldName !== newName) {
                                            setTableRows(prev =>
                                              prev.map(row => {
                                                const newValues = { ...row.values }
                                                newValues[newName] = newValues[oldName]
                                                delete newValues[oldName]
                                                return { ...row, values: newValues }
                                              })
                                            )
                                          }
                                        }}
                                        className="flex-1 px-2 py-1.5 bg-transparent border-0 outline-none text-xs font-medium focus:bg-background"
                                      />
                                      <select
                                        value={col.type}
                                        onChange={e =>
                                          handleColumnTypeChange(
                                            col.name,
                                            e.target.value as FeatureEncodingType
                                          )
                                        }
                                        className="pl-1 pr-1 py-1 bg-transparent border-0 outline-none text-[10px] text-muted-foreground cursor-pointer hover:text-foreground"
                                        title="Column type"
                                      >
                                        {ENCODING_TYPE_OPTIONS.map(opt => (
                                          <option key={opt.value} value={opt.value}>
                                            {opt.label}
                                          </option>
                                        ))}
                                      </select>
                                      <button
                                        onClick={() => handleRemoveColumn(idx)}
                                        className="ml-1 p-0.5 text-muted-foreground/50 hover:text-destructive"
                                        title="Remove column"
                                      >
                                        <FontIcon type="close" className="w-3 h-3" />
                                      </button>
                                    </div>
                                  </th>
                                ))}
                                <th className="w-8 px-1 border-l border-border">
                                  <button
                                    onClick={handleAddColumn}
                                    className="text-muted-foreground hover:text-primary p-0.5"
                                    title="Add column"
                                  >
                                    <FontIcon type="add" className="w-3.5 h-3.5" />
                                  </button>
                                </th>
                              </tr>
                            </thead>
                            <tbody>
                              {tableRows.map((row, rowIdx) => (
                                <tr
                                  key={row.id}
                                  className="border-b border-border last:border-b-0 hover:bg-muted/30"
                                >
                                  <td className="w-8 px-2 py-0 text-center text-muted-foreground border-r border-border bg-muted/30">
                                    {rowIdx + 1}
                                  </td>
                                  {columns.map((col, colIdx) => (
                                    <td
                                      key={colIdx}
                                      className={`px-0 py-0 ${colIdx < columns.length - 1 ? 'border-r border-border' : ''}`}
                                    >
                                      <input
                                        value={row.values[col.name] || ''}
                                        onChange={e =>
                                          handleCellChange(row.id, col.name, e.target.value)
                                        }
                                        onPaste={e => handleCellPaste(e, row.id, colIdx)}
                                        onBlur={() => setHasBlurredTrainingData(true)}
                                        className="w-full px-2 py-1.5 bg-transparent border-0 outline-none text-xs font-mono focus:bg-primary/5"
                                        placeholder={col.type === 'numeric' ? '0' : '—'}
                                      />
                                    </td>
                                  ))}
                                  <td className="w-8 px-1 border-l border-border">
                                    <button
                                      onClick={() =>
                                        setTableRows(prev =>
                                          prev.filter(r => r.id !== row.id)
                                        )
                                      }
                                      className="text-muted-foreground hover:text-destructive p-0.5"
                                      title="Remove row"
                                    >
                                      <FontIcon type="close" className="w-3 h-3" />
                                    </button>
                                  </td>
                                </tr>
                              ))}
                              {/* Add row button as last row */}
                              <tr className="bg-muted/20">
                                <td
                                  colSpan={columns.length + 2}
                                  className="px-2 py-1"
                                >
                                  <button
                                    onClick={handleAddRow}
                                    className="text-xs text-muted-foreground hover:text-primary flex items-center gap-1"
                                  >
                                    <FontIcon type="add" className="w-3 h-3" />
                                    Add row
                                  </button>
                                </td>
                              </tr>
                            </tbody>
                          </table>
                      </div>
                    )}

                    {columns.length > 0 && (
                      <p className="text-xs text-muted-foreground">
                        {tableRows.length} rows × {columns.length} columns
                        {hasBlurredTrainingData && tableRows.length > 0 && tableRows.length < 50 && (
                          <span className="text-amber-600 dark:text-amber-400">
                            {' — Please add more entries to increase accuracy'}
                          </span>
                        )}
                      </p>
                    )}
                  </div>
                )}

                {/* Clear data button - always visible when there's data */}
                {(trainingData.trim() || columns.length > 0) && (
                  <div className="flex justify-end">
                    <button
                      onClick={() => {
                        setTrainingData('')
                        setColumns([])
                        setTableRows([])
                      }}
                      className="text-xs text-muted-foreground hover:text-destructive"
                    >
                      Clear data
                    </button>
                  </div>
                )}
              </div>

              {/* Actions row */}
              <div className="flex items-center gap-3 pt-2">
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
                ? 'Enter values to check if they would be flagged as anomalies.'
                : 'Train your model first to enable testing.'}
            </p>
          </div>

          <div className="flex gap-2">
            <Input
              placeholder="e.g., 25.0 or paste table row"
              value={testInput}
              onChange={e => setTestInput(e.target.value)}
              onPaste={e => {
                const pastedText = e.clipboardData.getData('text')
                if (pastedText.includes('\t')) {
                  e.preventDefault()
                  setTestInput(formatPastedText(pastedText).split('\n')[0])
                }
              }}
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
                      className={`flex items-start gap-2 px-2 py-1 rounded text-sm ${bgClass}`}
                    >
                      <FontIcon type={iconType} className={iconClass} />
                      <span className={`font-medium w-16 shrink-0 ${labelClass}`}>
                        {label}
                      </span>
                      <span className="text-muted-foreground w-10 shrink-0">
                        {isError ? '—' : result.score.toFixed(2)}
                      </span>
                      <span className="text-muted-foreground break-words font-mono text-xs flex-1 min-w-0">
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

      {/* CSV Import Modal */}
      <Dialog open={showCsvModal} onOpenChange={handleCsvModalClose}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Import from CSV</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-4">
            {/* Drop zone / file display */}
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDraggingCsv
                  ? 'border-primary bg-primary/5'
                  : 'border-border hover:border-muted-foreground/50'
              }`}
              onDragOver={e => {
                e.preventDefault()
                setIsDraggingCsv(true)
              }}
              onDragLeave={() => setIsDraggingCsv(false)}
              onDrop={handleCsvDrop}
            >
              {csvFile ? (
                <div className="flex flex-col items-center gap-2">
                  <FontIcon type="data" className="w-8 h-8 text-primary" />
                  <p className="text-sm font-medium">{csvFile.name}</p>
                  <button
                    onClick={() => setCsvFile(null)}
                    className="text-xs text-muted-foreground hover:text-destructive"
                  >
                    Remove
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3">
                  <FontIcon type="upload" className="w-8 h-8 text-muted-foreground" />
                  <div className="flex flex-col gap-1">
                    <p className="text-sm text-muted-foreground">
                      Drop CSV file here or
                    </p>
                    <button
                      onClick={() => csvFileInputRef.current?.click()}
                      className="text-sm text-primary hover:underline"
                    >
                      browse to upload
                    </button>
                  </div>
                </div>
              )}
              <input
                ref={csvFileInputRef}
                type="file"
                accept=".csv,text/csv"
                className="hidden"
                onChange={e => {
                  const file = e.target.files?.[0]
                  if (file) {
                    handleCsvFileSelect(file)
                  }
                  e.target.value = ''
                }}
              />
            </div>

            {/* First row is header checkbox */}
            <label className="flex items-center gap-2 cursor-pointer">
              <Checkbox
                checked={csvFirstRowIsHeader}
                onCheckedChange={checked => setCsvFirstRowIsHeader(checked === true)}
              />
              <span className="text-sm">First row is a header (excluded)</span>
            </label>
          </div>
          <DialogFooter>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm border border-input hover:bg-accent/30"
              onClick={handleCsvModalClose}
            >
              Cancel
            </button>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-60"
              onClick={handleCsvImport}
              disabled={!csvFile}
            >
              Import
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Sample Data Modal */}
      <Dialog
        open={showSampleDataModal}
        onOpenChange={open => {
          setShowSampleDataModal(open)
          if (!open) {
            setSelectedSampleDataset(null)
            setIsImportingSampleData(false)
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Use sample data</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-2">
            <p className="text-sm text-muted-foreground mb-2">
              Choose a sample dataset to get started quickly.
            </p>
            {SAMPLE_DATASETS.map(dataset => {
              const isSelected = selectedSampleDataset === dataset.id
              return (
                <button
                  key={dataset.id}
                  onClick={() => setSelectedSampleDataset(dataset.id)}
                  disabled={isImportingSampleData}
                  className={`flex items-center gap-3 p-3 rounded-lg border transition-colors text-left group ${
                    isSelected
                      ? 'border-primary bg-primary/5'
                      : 'border-border hover:bg-muted/50 hover:border-muted-foreground/50'
                  } ${isImportingSampleData ? 'opacity-50 cursor-not-allowed' : ''}`}
                >
                  <div
                    className={`w-9 h-9 rounded-lg flex items-center justify-center shrink-0 ${
                      isSelected ? 'bg-primary/10' : 'bg-muted group-hover:bg-muted/80'
                    }`}
                  >
                    <FontIcon
                      type={dataset.type === 'numeric' ? 'numeric' : 'prompt'}
                      className={`w-4 h-4 ${
                        isSelected ? 'text-primary' : 'text-muted-foreground'
                      }`}
                    />
                  </div>
                  <div className="flex flex-col gap-0.5 min-w-0 flex-1">
                    <span className="text-sm font-medium">{dataset.name}</span>
                    <span className="text-xs text-muted-foreground">
                      {dataset.description}
                    </span>
                  </div>
                  {isSelected && (
                    <FontIcon type="checkmark-filled" className="w-4 h-4 text-primary shrink-0" />
                  )}
                </button>
              )
            })}
          </div>
          <DialogFooter>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm border border-input hover:bg-accent/30"
              onClick={() => {
                setShowSampleDataModal(false)
                setSelectedSampleDataset(null)
              }}
              disabled={isImportingSampleData}
            >
              Cancel
            </button>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90 disabled:opacity-60 flex items-center gap-2"
              onClick={handleImportSampleData}
              disabled={!selectedSampleDataset || isImportingSampleData}
            >
              {isImportingSampleData && (
                <span className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin" />
              )}
              {isImportingSampleData ? 'Importing...' : 'Import data'}
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default AnomalyModel
