import { useEffect, useState, useMemo } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Button } from '../ui/button'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject } from '../../hooks/useProjects'
import { useDatabaseManager } from '../../hooks/useDatabaseManager'
import { useToast } from '../ui/toast'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import FontIcon from '../../common/FontIcon'
import {
  getDefaultConfigForRetrieval,
  parseWeightsList,
} from '../../utils/retrievalUtils'
import {
  STRATEGY_TYPES,
  STRATEGY_LABELS,
  STRATEGY_DESCRIPTIONS,
  type StrategyType,
} from '../../utils/strategyCatalog'
import { parseMetadataFilters, validateStrategyName } from '../../utils/security'
import { useUnsavedChanges } from '../../contexts/UnsavedChangesContext'
import UnsavedChangesModal from '../ConfigEditor/UnsavedChangesModal'

function AddRetrievalStrategy() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const { toast } = useToast()
  const activeProject = useActiveProject()
  
  // Get project config and database manager
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )
  const databaseManager = useDatabaseManager(
    activeProject?.namespace || '',
    activeProject?.project || ''
  )

  // Get the database from URL query params (defaults to main_database if not provided)
  const database = searchParams.get('database') || 'main_database'

  // Get existing retrieval strategies for copy from dropdown
  const existingStrategies = useMemo(() => {
    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig) return []
    const db = projectConfig.rag?.databases?.find(
      (d: any) => d.name === database
    )
    return db?.retrieval_strategies || []
  }, [projectResp, database])

  // New retrieval name and default toggle
  const [name, setName] = useState('new-retrieval-strategy')
  const [nameTouched, setNameTouched] = useState(false)
  const [makeDefault, setMakeDefault] = useState(false)
  const [copyFrom, setCopyFrom] = useState<string>('')
  const [isSaving, setIsSaving] = useState(false)
  const [_error, setError] = useState<string | null>(null)

  // Unsaved changes tracking
  const unsavedChangesContext = useUnsavedChanges()
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [modalErrorMessage, setModalErrorMessage] = useState<string | null>(null)

  // Selected type + settings state
  const [selectedType, setSelectedType] = useState<StrategyType | null>(null)

  // Shared UI helpers
  const inputClass =
    'bg-background focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary focus-visible:ring-offset-0'
  const Field = ({
    label,
    children,
  }: {
    label: string
    children: React.ReactNode
  }) => (
    <div className="flex flex-col gap-1 py-1.5">
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {children}
    </div>
  )
  const SelectDropdown = ({
    value,
    onChange,
    options,
    className,
  }: {
    value: string
    onChange: (v: string) => void
    options: string[]
    className?: string
  }) => (
    <div className="relative">
      <button
        className={`h-9 w-full rounded-md border border-input bg-background px-3 text-left flex items-center justify-between ${
          className || ''
        } ${inputClass}`}
        onClick={e => {
          e.preventDefault()
          const idx = options.indexOf(value)
          const next = options[(idx + 1 + options.length) % options.length]
          onChange(next)
        }}
      >
        <span className="truncate">{value}</span>
        <span className="ml-2 text-xs text-muted-foreground">tap to cycle</span>
      </button>
    </div>
  )

  // Strategy-specific state and defaults (match RetrievalMethod)
  const DISTANCE_OPTIONS = ['cosine', 'euclidean', 'manhattan', 'dot']
  const META_FILTER_MODE = ['pre', 'post']
  const MQ_AGGREGATION = ['max', 'mean', 'weighted', 'reciprocal_rank']
  const HYBRID_COMBINATION = ['weighted_average', 'rank_fusion', 'score_fusion']

  const [basicTopK, setBasicTopK] = useState<string>('10')
  const [basicDistance, setBasicDistance] = useState<string>('cosine')
  const [basicScoreThreshold, setBasicScoreThreshold] = useState<string>('')

  const [mfTopK, setMfTopK] = useState<string>('10')
  const [mfFilterMode, setMfFilterMode] = useState<string>('pre')
  const [mfFallbackMultiplier, setMfFallbackMultiplier] = useState<string>('3')
  const [mfFilters, setMfFilters] = useState<
    Array<{ key: string; value: string }>
  >([])

  const [mqNumQueries, setMqNumQueries] = useState<string>('3')
  const [mqTopK, setMqTopK] = useState<string>('10')
  const [mqAggregation, setMqAggregation] = useState<string>('weighted')
  const [mqQueryWeights, setMqQueryWeights] = useState<string>('')

  const [rrInitialK, setRrInitialK] = useState<string>('30')
  const [rrFinalK, setRrFinalK] = useState<string>('10')
  const [rrSimW, setRrSimW] = useState<string>('0.7')
  const [rrRecencyW, setRrRecencyW] = useState<string>('0.1')
  const [rrLengthW, setRrLengthW] = useState<string>('0.1')
  const [rrMetaW, setRrMetaW] = useState<string>('0.1')
  const [rrNormalize, setRrNormalize] = useState<'Enabled' | 'Disabled'>(
    'Enabled'
  )

  type HybridSub = {
    id: string
    type: Exclude<StrategyType, 'HybridUniversalStrategy'>
    weight: string
    config?: Record<string, unknown>
  }
  const [hybStrategies, setHybStrategies] = useState<HybridSub[]>([])
  const [hybCombination, setHybCombination] =
    useState<string>('weighted_average')
  const [hybFinalK, setHybFinalK] = useState<string>('10')

  const getDefaultConfigForType = (
    type: Exclude<StrategyType, 'HybridUniversalStrategy'>
  ): Record<string, unknown> => getDefaultConfigForRetrieval(type)
  const updateHybridSub = (index: number, partial: Partial<HybridSub>) => {
    setHybStrategies(prev => {
      const next = [...prev]
      next[index] = { ...next[index], ...partial }
      return next
    })
  }
  const updateHybridSubConfig = (
    index: number,
    updater: (prev: Record<string, unknown>) => Record<string, unknown>
  ) => {
    setHybStrategies(prev => {
      const next = [...prev]
      const current = (next[index]?.config as Record<string, unknown>) || {}
      next[index] = { ...next[index], config: updater(current) }
      return next
    })
  }

  // Reset defaults when type changes
  useEffect(() => {
    if (!selectedType) return
    setBasicTopK('10')
    setBasicDistance('cosine')
    setBasicScoreThreshold('')
    setMfTopK('10')
    setMfFilterMode('pre')
    setMfFallbackMultiplier('3')
    setMfFilters([])
    setMqNumQueries('3')
    setMqTopK('10')
    setMqAggregation('weighted')
    setMqQueryWeights('')
    setRrInitialK('30')
    setRrFinalK('10')
    setRrSimW('0.7')
    setRrRecencyW('0.1')
    setRrLengthW('0.1')
    setRrMetaW('0.1')
    setRrNormalize('Enabled')
    setHybCombination('weighted_average')
    setHybFinalK('10')
    setHybStrategies([])
  }, [selectedType])

  // Ensure default checked when first retrieval
  useEffect(() => {
    if (projectResp && database) {
      const projectConfig = (projectResp as any)?.project?.config
      const db = projectConfig?.rag?.databases?.find((d: any) => d.name === database)
      const hasStrategies = db?.retrieval_strategies?.length > 0
      setMakeDefault(!hasStrategies)
    }
  }, [projectResp, database])

  // Handle copy from existing strategy
  useEffect(() => {
    if (!copyFrom || !projectResp) return

    const projectConfig = (projectResp as any)?.project?.config
    if (!projectConfig) return

    const db = projectConfig.rag?.databases?.find(
      (d: any) => d.name === database
    )
    const strategy = db?.retrieval_strategies?.find(
      (s: any) => s.name === copyFrom
    )

    if (!strategy?.config || !strategy.type) return

    const config = strategy.config
    const strategyType = strategy.type as StrategyType

    // Set the strategy type
    setSelectedType(strategyType)

    // Populate form fields based on strategy type
    if (strategyType === 'BasicSimilarityStrategy') {
      if (typeof config.top_k === 'number') setBasicTopK(String(config.top_k))
      if (config.distance_metric) setBasicDistance(String(config.distance_metric))
      if (config.score_threshold !== null && config.score_threshold !== undefined) {
        setBasicScoreThreshold(String(config.score_threshold))
      }
    } else if (strategyType === 'MetadataFilteredStrategy') {
      if (typeof config.top_k === 'number') setMfTopK(String(config.top_k))
      if (config.filter_mode) setMfFilterMode(String(config.filter_mode))
      if (typeof config.fallback_multiplier === 'number') {
        setMfFallbackMultiplier(String(config.fallback_multiplier))
      }
      if (Array.isArray(config.filters)) {
        setMfFilters(
          config.filters.map((f: any) => ({
            key: String(f.key || ''),
            value: String(f.value || ''),
          }))
        )
      }
    } else if (strategyType === 'MultiQueryStrategy') {
      if (typeof config.num_queries === 'number') {
        setMqNumQueries(String(config.num_queries))
      }
      if (typeof config.top_k === 'number') setMqTopK(String(config.top_k))
      if (config.aggregation_method) {
        setMqAggregation(String(config.aggregation_method))
      }
      if (Array.isArray(config.query_weights)) {
        setMqQueryWeights(config.query_weights.join(','))
      }
    } else if (strategyType === 'RerankedStrategy') {
      if (typeof config.initial_k === 'number') {
        setRrInitialK(String(config.initial_k))
      }
      if (typeof config.final_k === 'number') {
        setRrFinalK(String(config.final_k))
      }
      if (typeof config.similarity_weight === 'number') {
        setRrSimW(String(config.similarity_weight))
      }
      if (typeof config.recency_weight === 'number') {
        setRrRecencyW(String(config.recency_weight))
      }
      if (typeof config.length_weight === 'number') {
        setRrLengthW(String(config.length_weight))
      }
      if (typeof config.metadata_weight === 'number') {
        setRrMetaW(String(config.metadata_weight))
      }
      if (config.normalize !== undefined) {
        setRrNormalize(config.normalize ? 'Enabled' : 'Disabled')
      }
    } else if (strategyType === 'HybridUniversalStrategy') {
      if (config.combination_method) {
        setHybCombination(String(config.combination_method))
      }
      if (typeof config.final_k === 'number') {
        setHybFinalK(String(config.final_k))
      }
      if (Array.isArray(config.strategies)) {
        setHybStrategies(
          config.strategies.map((s: any, idx: number) => ({
            id: `sub-${Date.now()}-${idx}`,
            type: s.type as Exclude<StrategyType, 'HybridUniversalStrategy'>,
            weight: String(s.weight || '1.0'),
            config: s.config || {},
          }))
        )
      }
    }
  }, [copyFrom, projectResp, database])

  // Track changes to form fields (after all state is declared)
  useEffect(() => {
    // Check if any form field has been modified from defaults
    const hasChanges =
      name !== 'new-retrieval-strategy' ||
      copyFrom !== '' ||
      selectedType !== null ||
      makeDefault !== false ||
      basicTopK !== '10' ||
      basicDistance !== 'cosine' ||
      basicScoreThreshold !== '' ||
      mfTopK !== '10' ||
      mfFilterMode !== 'pre' ||
      mfFallbackMultiplier !== '3' ||
      mfFilters.length > 0 ||
      mqNumQueries !== '3' ||
      mqTopK !== '10' ||
      mqAggregation !== 'weighted' ||
      mqQueryWeights !== '' ||
      rrInitialK !== '30' ||
      rrFinalK !== '10' ||
      rrSimW !== '0.7' ||
      rrRecencyW !== '0.1' ||
      rrLengthW !== '0.1' ||
      rrMetaW !== '0.1' ||
      rrNormalize !== 'Enabled' ||
      hybStrategies.length > 0 ||
      hybCombination !== 'weighted_average' ||
      hybFinalK !== '10'

    setHasUnsavedChanges(hasChanges)
  }, [
    name,
    copyFrom,
    selectedType,
    makeDefault,
    basicTopK,
    basicDistance,
    basicScoreThreshold,
    mfTopK,
    mfFilterMode,
    mfFallbackMultiplier,
    mfFilters,
    mqNumQueries,
    mqTopK,
    mqAggregation,
    mqQueryWeights,
    rrInitialK,
    rrFinalK,
    rrSimW,
    rrRecencyW,
    rrLengthW,
    rrMetaW,
    rrNormalize,
    hybStrategies,
    hybCombination,
    hybFinalK,
  ])

  // Sync isDirty with context
  useEffect(() => {
    unsavedChangesContext.setIsDirty(hasUnsavedChanges)
  }, [hasUnsavedChanges, unsavedChangesContext])

  // Real-time duplicate name validation (case-insensitive)
  // Check immediately for default names, or after field is touched
  const duplicateNameError = useMemo(() => {
    if (!projectResp || !name.trim()) return null
    
    // Always check for duplicates (don't wait for touch) so default names are validated
    const projectConfig = (projectResp as any)?.project?.config
    const currentDb = projectConfig?.rag?.databases?.find(
      (db: any) => db.name === database
    )
    if (currentDb) {
      const nameLower = name.trim().toLowerCase()
      const nameExists = currentDb.retrieval_strategies?.some(
        (s: any) => s.name?.toLowerCase() === nameLower
      )
      if (nameExists) {
        return 'A strategy with this name already exists'
      }
    }
    return null
  }, [name, projectResp, database])

  // Validation function
  const validateStrategy = (): string[] => {
    const errors: string[] = []
    
    if (!selectedType) {
      errors.push('Please select a strategy type')
    }
    
    // Validate strategy name with security checks
    const nameError = validateStrategyName(name)
    if (nameError) {
      errors.push(nameError)
    }
    
    // Check for duplicate strategy name
    if (duplicateNameError) {
      errors.push(duplicateNameError)
    }
    
    // Type-specific validation
    if (selectedType === 'BasicSimilarityStrategy') {
      const topK = Number(basicTopK)
      if (isNaN(topK) || topK < 1 || topK > 1000) {
        errors.push('Top K must be between 1 and 1000')
      }
      if (basicScoreThreshold.trim() && (Number(basicScoreThreshold) < 0 || Number(basicScoreThreshold) > 1)) {
        errors.push('Score threshold must be between 0 and 1')
      }
    } else if (selectedType === 'MetadataFilteredStrategy') {
      const topK = Number(mfTopK)
      if (isNaN(topK) || topK < 1 || topK > 1000) {
        errors.push('Top K must be between 1 and 1000')
      }
    } else if (selectedType === 'MultiQueryStrategy') {
      const numQueries = Number(mqNumQueries)
      if (isNaN(numQueries) || numQueries < 2 || numQueries > 10) {
        errors.push('Number of queries must be between 2 and 10')
      }
    } else if (selectedType === 'RerankedStrategy') {
      const initialK = Number(rrInitialK)
      const finalK = Number(rrFinalK)
      if (isNaN(initialK) || initialK < 1 || initialK > 1000) {
        errors.push('Initial K must be between 1 and 1000')
      }
      if (isNaN(finalK) || finalK < 1 || finalK > initialK) {
        errors.push('Final K must be between 1 and Initial K')
      }
    } else if (selectedType === 'HybridUniversalStrategy' && hybStrategies.length < 2) {
      errors.push('At least 2 sub-strategies are required for hybrid approach')
    }
    
    return errors
  }

  // Save handler - updated to use project config
  const onSave = async (): Promise<void> => {
    try {
      setIsSaving(true)
      setError(null)

      // Validate
      const validationErrors = validateStrategy()
      if (validationErrors.length > 0) {
        const errorMsg = validationErrors.join(', ')
        setError(errorMsg)
        setIsSaving(false)
        throw new Error(errorMsg)
      }

      // Get current project config
      const projectConfig = (projectResp as any)?.project?.config
      if (!projectConfig) {
        throw new Error('Project config not loaded')
      }

      // Find the database
      const currentDb = projectConfig.rag?.databases?.find(
        (db: any) => db.name === database
      )
      
      if (!currentDb) {
        throw new Error(`Database ${database} not found in configuration`)
      }

      // Check if name already exists (case-insensitive)
      const nameLower = name.trim().toLowerCase()
      const existingStrategy = currentDb.retrieval_strategies?.find(
        (s: any) => s.name?.toLowerCase() === nameLower
      )
      if (existingStrategy) {
        throw new Error('A strategy with this name already exists')
      }

      // Build config from current state
      let config: Record<string, unknown> = {}
      switch (selectedType) {
        case 'BasicSimilarityStrategy':
          config = {
            top_k: Number(basicTopK),
            distance_metric: basicDistance,
            score_threshold:
              basicScoreThreshold.trim() === ''
                ? null
                : Number(basicScoreThreshold),
          }
          break
        case 'MetadataFilteredStrategy': {
          // Use secure parsing to sanitize filter keys and values
          const filters = parseMetadataFilters(mfFilters)
          config = {
            top_k: Number(mfTopK),
            filters,
            filter_mode: mfFilterMode,
            fallback_multiplier: Number(mfFallbackMultiplier),
          }
          break
        }
        case 'MultiQueryStrategy': {
          const weights = parseWeightsList(mqQueryWeights, Number(mqNumQueries))
          config = {
            num_queries: Number(mqNumQueries),
            top_k: Number(mqTopK),
            aggregation_method: mqAggregation,
            query_weights: weights,
          }
          break
        }
        case 'RerankedStrategy':
          config = {
            initial_k: Number(rrInitialK),
            final_k: Number(rrFinalK),
            rerank_factors: {
              similarity_weight: Number(rrSimW),
              recency_weight: Number(rrRecencyW),
              length_weight: Number(rrLengthW),
              metadata_weight: Number(rrMetaW),
            },
            normalize_scores: rrNormalize === 'Enabled',
          }
          break
        case 'HybridUniversalStrategy': {
          const strategies = hybStrategies.map(s => ({
            type: s.type,
            weight: Number(s.weight),
            config: s.config || getDefaultConfigForType(s.type),
          }))
          config = {
            strategies,
            combination_method: hybCombination,
            final_k: Number(hybFinalK),
          }
          break
        }
      }

      // Build the new strategy (note: 'default' field is used in config, not 'isDefault')
      const newStrategy = {
        name: name.trim(),
        type: selectedType,
        config,
      }

      // Add to existing strategies
      const updatedStrategies = [
        ...(currentDb.retrieval_strategies || []),
        newStrategy,
      ]

      // Determine default strategy name (no 'default' field on strategy objects)
      const defaultStrategyName = makeDefault || updatedStrategies.length === 1
        ? newStrategy.name
        : currentDb.default_retrieval_strategy

      // Update database configuration
      await databaseManager.updateDatabase.mutateAsync({
        oldName: database,
        updates: {
          retrieval_strategies: updatedStrategies,
          default_retrieval_strategy: defaultStrategyName,
        },
        projectConfig,
      })

      // Clear unsaved changes flags BEFORE navigation to prevent modal from showing
      setHasUnsavedChanges(false)
      unsavedChangesContext.setIsDirty(false)
      
      toast({
        message: 'Strategy saved',
        variant: 'default',
      })
      
      // Use requestAnimationFrame to ensure state updates propagate before navigation
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          navigate('/chat/databases')
        })
      })
    } catch (error: any) {
      console.error('Failed to create retrieval strategy:', error)
      const errorMessage = error.message || 'Failed to create strategy'
      setError(errorMessage)
      setIsSaving(false)
      // Re-throw so callers can catch it
      throw error
    }
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-20">
      {/* Breadcrumb + Actions */}
      <div className="flex items-center justify-between mb-1">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/databases')}
          >
            Databases
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">New retrieval strategy</span>
        </nav>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate('/chat/databases')}
            disabled={isSaving}
          >
            Cancel
          </Button>
          <Button
            onClick={onSave}
            disabled={
              isSaving || 
              !selectedType || 
              name.trim().length === 0 || 
              !!validateStrategyName(name) || 
              !!duplicateNameError
            }
          >
            {isSaving ? 'Saving...' : 'Save strategy'}
          </Button>
        </div>
      </div>

      <section className="rounded-lg border border-border bg-card p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">
              Strategy name
            </Label>
            <Input
              value={name}
              onChange={e => {
                setName(e.target.value)
                // Clear error state when user starts typing if validation passes
                if (nameTouched) {
                  const nameError = validateStrategyName(e.target.value)
                  if (!nameError && !duplicateNameError) {
                    setError(null)
                  }
                }
              }}
              onBlur={() => setNameTouched(true)}
              placeholder="Enter a name"
              className={`h-9 ${
                (validateStrategyName(name) || duplicateNameError)
                  ? 'border-destructive'
                  : ''
              }`}
            />
            {validateStrategyName(name) && (
              <p className="text-xs text-destructive mt-1">
                {validateStrategyName(name)}
              </p>
            )}
            {!validateStrategyName(name) && duplicateNameError && (
              <p className="text-xs text-destructive mt-1">
                {duplicateNameError}
              </p>
            )}
            {!validateStrategyName(name) && !duplicateNameError && (
              <p className="text-xs text-muted-foreground mt-1">
                Can only contain letters, numbers, hyphens, and underscores
              </p>
            )}
          </div>
          <div className="flex flex-col gap-1">
            <Label className="text-xs text-muted-foreground">Copy from</Label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="w-full h-9 rounded-md border border-border bg-background px-3 text-left flex items-center justify-between text-sm">
                  <span
                    className={
                      copyFrom ? 'text-foreground' : 'text-muted-foreground'
                    }
                  >
                    {copyFrom || 'Select a strategy to copy...'}
                  </span>
                  <FontIcon type="chevron-down" className="w-4 h-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64">
                <DropdownMenuItem
                  onClick={() => {
                    setCopyFrom('')
                    // Reset form to defaults
                    setName('new-retrieval-strategy')
                    setSelectedType(null)
                    setMakeDefault(false)
                    setBasicTopK('10')
                    setBasicDistance('cosine')
                    setBasicScoreThreshold('')
                    setMfTopK('10')
                    setMfFilterMode('pre')
                    setMfFallbackMultiplier('3')
                    setMfFilters([])
                    setMqNumQueries('3')
                    setMqTopK('10')
                    setMqAggregation('weighted')
                    setMqQueryWeights('')
                    setRrInitialK('30')
                    setRrFinalK('10')
                    setRrSimW('0.7')
                    setRrRecencyW('0.1')
                    setRrLengthW('0.1')
                    setRrMetaW('0.1')
                    setRrNormalize('Enabled')
                    setHybStrategies([])
                    setHybCombination('weighted_average')
                    setHybFinalK('10')
                  }}
                >
                  None
                </DropdownMenuItem>
                {existingStrategies.map((strategy: any) => (
                  <DropdownMenuItem
                    key={strategy.name}
                    onClick={() => setCopyFrom(strategy.name)}
                  >
                    {strategy.name}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>
        <div className="mt-3">
          <label className="inline-flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={makeDefault}
              onChange={e => setMakeDefault(e.target.checked)}
            />
            <span>Make default</span>
          </label>
        </div>
      </section>

      {/* Type chooser / summary */}
      <section className="rounded-lg border border-border bg-card p-4">
        {!selectedType ? (
          <div className="flex flex-col gap-2">
            <div className="text-sm font-medium">Select retrieval strategy</div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {STRATEGY_TYPES.map(t => (
                <button
                  key={t}
                  className={`text-left rounded-lg border border-border bg-card hover:bg-accent/20 ${
                    t === 'BasicSimilarityStrategy' ? 'pt-2 pb-3 px-3' : 'p-3'
                  }`}
                  onClick={() => setSelectedType(t)}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="text-sm font-medium">
                      {STRATEGY_LABELS[t]}
                    </div>
                    <Button variant="outline" size="sm">
                      Select
                    </Button>
                  </div>
                  <div className="text-xs text-muted-foreground mt-1">
                    {STRATEGY_DESCRIPTIONS[t]}
                  </div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-between">
            <div className="flex-1 min-w-0">
              <div className="text-xs text-muted-foreground mb-1">
                Current strategy
              </div>
              <div className="text-xl md:text-2xl font-medium">
                {STRATEGY_LABELS[selectedType]}
              </div>
            </div>
            <div className="ml-3 shrink-0">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setSelectedType(null)}
              >
                Change
              </Button>
            </div>
          </div>
        )}
      </section>

      {/* Settings per selected type */}
      {selectedType === 'BasicSimilarityStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Basic similarity</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Top K">
              <Input
                type="number"
                value={basicTopK}
                onChange={e => setBasicTopK(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Distance metric">
              <SelectDropdown
                value={basicDistance}
                onChange={setBasicDistance}
                options={DISTANCE_OPTIONS}
              />
            </Field>
            <Field label="Score threshold (0–1, optional)">
              <Input
                type="number"
                step="0.01"
                placeholder="leave blank to disable"
                value={basicScoreThreshold}
                onChange={e => setBasicScoreThreshold(e.target.value)}
                className={inputClass}
              />
            </Field>
          </div>
        </section>
      ) : null}

      {selectedType === 'MetadataFilteredStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Metadata-filtered</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Top K">
              <Input
                type="number"
                value={mfTopK}
                onChange={e => setMfTopK(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Filter mode">
              <SelectDropdown
                value={mfFilterMode}
                onChange={setMfFilterMode}
                options={META_FILTER_MODE}
              />
            </Field>
            <Field label="Fallback multiplier">
              <Input
                type="number"
                value={mfFallbackMultiplier}
                onChange={e => setMfFallbackMultiplier(e.target.value)}
                className={inputClass}
              />
            </Field>
          </div>
          <div className="mt-3">
            <div className="text-xs text-muted-foreground mb-1">
              Metadata filters
            </div>
            <div className="flex flex-col gap-2">
              {mfFilters.map((f, idx) => (
                <div
                  key={idx}
                  className="grid grid-cols-1 md:grid-cols-3 gap-2"
                >
                  <Input
                    placeholder="key"
                    value={f.key}
                    onChange={e => {
                      const next = [...mfFilters]
                      next[idx] = { ...next[idx], key: e.target.value }
                      setMfFilters(next)
                    }}
                    className={inputClass}
                  />
                  <div className="md:col-span-2 flex items-center gap-2">
                    <Input
                      placeholder="value (supports numbers, true/false, or a,b,c)"
                      value={f.value}
                      onChange={e => {
                        const next = [...mfFilters]
                        next[idx] = { ...next[idx], value: e.target.value }
                        setMfFilters(next)
                      }}
                      className={inputClass}
                    />
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        setMfFilters(mfFilters.filter((_, i) => i !== idx))
                      }
                    >
                      Remove
                    </Button>
                  </div>
                </div>
              ))}
              <div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setMfFilters([...mfFilters, { key: '', value: '' }])
                  }
                >
                  Add filter
                </Button>
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {selectedType === 'MultiQueryStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Multi-query</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Number of queries">
              <Input
                type="number"
                value={mqNumQueries}
                onChange={e => setMqNumQueries(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Top K per query">
              <Input
                type="number"
                value={mqTopK}
                onChange={e => setMqTopK(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Aggregation method">
              <SelectDropdown
                value={mqAggregation}
                onChange={setMqAggregation}
                options={MQ_AGGREGATION}
              />
            </Field>
            <Field label={`Query weights (${mqNumQueries} values, optional)`}>
              <Input
                placeholder="e.g. 0.6, 0.3, 0.1"
                value={mqQueryWeights}
                onChange={e => setMqQueryWeights(e.target.value)}
                className={inputClass}
              />
            </Field>
          </div>
        </section>
      ) : null}

      {selectedType === 'RerankedStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Reranked</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Initial candidates (K)">
              <Input
                type="number"
                value={rrInitialK}
                onChange={e => setRrInitialK(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Final results (K)">
              <Input
                type="number"
                value={rrFinalK}
                onChange={e => setRrFinalK(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Normalize scores">
              <SelectDropdown
                value={rrNormalize}
                onChange={v => setRrNormalize(v as 'Enabled' | 'Disabled')}
                options={['Enabled', 'Disabled']}
              />
            </Field>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 mt-2">
            <Field label="Similarity weight (0–1)">
              <Input
                type="number"
                step="0.01"
                value={rrSimW}
                onChange={e => setRrSimW(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Recency weight (0–1)">
              <Input
                type="number"
                step="0.01"
                value={rrRecencyW}
                onChange={e => setRrRecencyW(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Length weight (0–1)">
              <Input
                type="number"
                step="0.01"
                value={rrLengthW}
                onChange={e => setRrLengthW(e.target.value)}
                className={inputClass}
              />
            </Field>
            <Field label="Metadata weight (0–1)">
              <Input
                type="number"
                step="0.01"
                value={rrMetaW}
                onChange={e => setRrMetaW(e.target.value)}
                className={inputClass}
              />
            </Field>
          </div>
        </section>
      ) : null}

      {selectedType === 'HybridUniversalStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Hybrid universal</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Combination method">
              <SelectDropdown
                value={hybCombination}
                onChange={setHybCombination}
                options={HYBRID_COMBINATION}
              />
            </Field>
            <Field label="Final K">
              <Input
                type="number"
                value={hybFinalK}
                onChange={e => setHybFinalK(e.target.value)}
                className={inputClass}
              />
            </Field>
          </div>
          <div className="mt-3">
            <div className="text-xs text-muted-foreground mb-1">
              Sub-strategies
            </div>
            <div className="flex flex-col gap-2">
              {hybStrategies.map((s, idx) => (
                <div key={s.id} className="rounded-md border border-border p-3">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-2 items-center">
                    <SelectDropdown
                      value={STRATEGY_LABELS[s.type]}
                      onChange={label => {
                        const entry = (
                          Object.entries(STRATEGY_LABELS) as Array<
                            [StrategyType, string]
                          >
                        ).find(([, v]) => v === label)
                        if (entry) {
                          const newType = entry[0] as Exclude<
                            StrategyType,
                            'HybridUniversalStrategy'
                          >
                          updateHybridSub(idx, {
                            type: newType,
                            config: getDefaultConfigForType(newType),
                          })
                        }
                      }}
                      options={STRATEGY_TYPES.filter(
                        t => t !== 'HybridUniversalStrategy'
                      ).map(t => STRATEGY_LABELS[t])}
                    />
                    <Input
                      type="number"
                      step="0.01"
                      value={s.weight}
                      onChange={e =>
                        updateHybridSub(idx, { weight: e.target.value })
                      }
                      className={inputClass}
                    />
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setHybStrategies(prev => {
                            const next = [...prev]
                            next.splice(idx, 1)
                            return next
                          })
                        }}
                      >
                        Remove
                      </Button>
                    </div>
                  </div>
                  {/* Inline editors for sub-strategies */}
                  <div className="mt-3">
                    {s.type === 'BasicSimilarityStrategy' ? (
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <Field label="Top K">
                          <Input
                            type="number"
                            value={String(
                              ((s.config as any)?.top_k ?? 10) as number
                            )}
                            onChange={e =>
                              updateHybridSubConfig(idx, prev => ({
                                ...prev,
                                top_k: Number(e.target.value),
                              }))
                            }
                            className={inputClass}
                          />
                        </Field>
                        <Field label="Distance metric">
                          <SelectDropdown
                            value={String(
                              (s.config as any)?.distance_metric ?? 'cosine'
                            )}
                            onChange={v =>
                              updateHybridSubConfig(idx, prev => ({
                                ...prev,
                                distance_metric: v,
                              }))
                            }
                            options={DISTANCE_OPTIONS}
                          />
                        </Field>
                        <Field label="Score threshold (0–1, optional)">
                          <Input
                            type="number"
                            step="0.01"
                            placeholder="leave blank to disable"
                            value={((): string => {
                              const thr = (s.config as any)?.score_threshold
                              return thr === null || thr === undefined
                                ? ''
                                : String(thr)
                            })()}
                            onChange={e =>
                              updateHybridSubConfig(idx, prev => ({
                                ...prev,
                                score_threshold:
                                  e.target.value.trim() === ''
                                    ? null
                                    : Number(e.target.value),
                              }))
                            }
                            className={inputClass}
                          />
                        </Field>
                      </div>
                    ) : null}

                    {s.type === 'MetadataFilteredStrategy' ? (
                      <div className="flex flex-col gap-3">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <Field label="Top K">
                            <Input
                              type="number"
                              value={String(
                                ((s.config as any)?.top_k ?? 10) as number
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  top_k: Number(e.target.value),
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                          <Field label="Filter mode">
                            <SelectDropdown
                              value={String(
                                (s.config as any)?.filter_mode ?? 'pre'
                              )}
                              onChange={v =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  filter_mode: v,
                                }))
                              }
                              options={META_FILTER_MODE}
                            />
                          </Field>
                          <Field label="Fallback multiplier">
                            <Input
                              type="number"
                              value={String(
                                (s.config as any)?.fallback_multiplier ?? 3
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  fallback_multiplier: Number(e.target.value),
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                        </div>
                      </div>
                    ) : null}

                    {s.type === 'MultiQueryStrategy' ? (
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                        <Field label="Number of queries">
                          <Input
                            type="number"
                            value={String(
                              ((s.config as any)?.num_queries ?? 3) as number
                            )}
                            onChange={e =>
                              updateHybridSubConfig(idx, prev => ({
                                ...prev,
                                num_queries: Number(e.target.value),
                              }))
                            }
                            className={inputClass}
                          />
                        </Field>
                        <Field label="Top K per query">
                          <Input
                            type="number"
                            value={String(
                              ((s.config as any)?.top_k ?? 10) as number
                            )}
                            onChange={e =>
                              updateHybridSubConfig(idx, prev => ({
                                ...prev,
                                top_k: Number(e.target.value),
                              }))
                            }
                            className={inputClass}
                          />
                        </Field>
                        <Field label="Aggregation method">
                          <SelectDropdown
                            value={String(
                              (s.config as any)?.aggregation_method ??
                                'weighted'
                            )}
                            onChange={v =>
                              updateHybridSubConfig(idx, prev => ({
                                ...prev,
                                aggregation_method: v,
                              }))
                            }
                            options={MQ_AGGREGATION}
                          />
                        </Field>
                      </div>
                    ) : null}
                  </div>
                </div>
              ))}
              <div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    setHybStrategies([
                      ...hybStrategies,
                      {
                        id: `sub-${Date.now()}`,
                        type: 'BasicSimilarityStrategy',
                        weight: '1.0',
                        config: getDefaultConfigForType(
                          'BasicSimilarityStrategy'
                        ),
                      },
                    ])
                  }
                >
                  Add sub-strategy
                </Button>
              </div>
            </div>
          </div>
        </section>
      ) : null}

      {/* Unsaved changes modal */}
      <UnsavedChangesModal
        isOpen={unsavedChangesContext.showModal}
        onSave={async () => {
          if (!selectedType) {
            setModalErrorMessage('Please select a strategy type before saving')
            return
          }

          try {
            setModalErrorMessage(null)
            setError(null)

            // Confirm navigation before calling onSave (which will navigate on success)
            unsavedChangesContext.confirmNavigation()
            
            // Call onSave - it handles setIsSaving and navigation internally
            // onSave will throw on error, so we can catch it here
            await onSave()
            
            // If we get here, save succeeded - navigation will happen in onSave
            setHasUnsavedChanges(false)
          } catch (e: any) {
            const errorMessage = e?.message || 'Failed to save strategy'
            setModalErrorMessage(errorMessage)
            // Cancel navigation since save failed
            unsavedChangesContext.cancelNavigation()
          }
        }}
        onDiscard={() => {
          // Clear unsaved changes flag and confirm navigation
          setHasUnsavedChanges(false)
          setModalErrorMessage(null)
          unsavedChangesContext.confirmNavigation()
        }}
        onCancel={() => {
          setModalErrorMessage(null)
          unsavedChangesContext.cancelNavigation()
        }}
        isSaving={isSaving}
        errorMessage={modalErrorMessage}
        isError={!!modalErrorMessage}
      />
    </div>
  )
}

export default AddRetrievalStrategy
