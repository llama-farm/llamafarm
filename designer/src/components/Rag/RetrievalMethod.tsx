import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import PageActions from '../common/PageActions'
import { Mode } from '../ModeToggle'
import FontIcon from '../../common/FontIcon'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../ui/dropdown-menu'

function RetrievalMethod() {
  const navigate = useNavigate()
  const { strategyId } = useParams()
  const [mode, setMode] = useState<Mode>('designer')

  const strategyName = useMemo(() => {
    if (!strategyId) return 'Strategy'
    return strategyId
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
  }, [strategyId])

  const storageKey = useMemo(
    () => (strategyId ? `lf_strategy_retrieval_${strategyId}` : ''),
    [strategyId]
  )

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
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={`h-9 w-full rounded-md border border-input bg-background px-3 text-left ${
            className || ''
          } ${inputClass}`}
        >
          {value}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-56" align="start">
        {options.map(opt => (
          <DropdownMenuItem key={opt} onClick={() => onChange(opt)}>
            {opt}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )

  const Collapsible = ({
    title,
    open,
    onToggle,
    children,
  }: {
    title: string
    open: boolean
    onToggle: () => void
    children: React.ReactNode
  }) => (
    <section className="rounded-lg border border-border bg-card p-3 transition-colors">
      <div
        className="flex items-center justify-between mb-2 cursor-pointer select-none"
        onClick={onToggle}
        role="button"
        tabIndex={0}
        onKeyDown={e => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            onToggle()
          }
        }}
        aria-expanded={open}
      >
        <div className="text-sm font-medium">{title}</div>
        <FontIcon
          type="chevron-down"
          className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`}
        />
      </div>
      {open && <div className="mt-1">{children}</div>}
    </section>
  )

  // Basic settings state
  const [searchType, setSearchType] = useState('Hybrid')
  const [resultsCount, setResultsCount] = useState<string>('8')
  const [reranking, setReranking] = useState<'Enabled' | 'Disabled'>('Enabled')

  // Accordions state
  const [searchOpen, setSearchOpen] = useState(false)
  const [rerankOpen, setRerankOpen] = useState(false)
  const [filtersOpen, setFiltersOpen] = useState(false)
  const [perfOpen, setPerfOpen] = useState(false)

  // Search Configuration
  const [semanticWeight, setSemanticWeight] = useState<string>('70')
  const [keywordWeight, setKeywordWeight] = useState<string>('30')
  const [bm25Params, setBm25Params] = useState('k1=1.2, b=0.75')
  const [similarityThreshold, setSimilarityThreshold] = useState<string>('0.65')
  const [maxResultsBeforeRerank, setMaxResultsBeforeRerank] =
    useState<string>('20')
  const [queryExpansion, setQueryExpansion] = useState<'Enabled' | 'Disabled'>(
    'Disabled'
  )

  // Re-ranking
  const [rerankingModel, setRerankingModel] = useState('ms-marco-v2')
  const [rerankThreshold, setRerankThreshold] = useState<string>('0.5')
  const [finalResultCount, setFinalResultCount] = useState<string>('8')
  const [crossEncoderBatchSize, setCrossEncoderBatchSize] =
    useState<string>('32')
  const [gpuAcceleration, setGpuAcceleration] = useState('Auto-detect')

  // Filtering & Boosting
  const [metadataFilters, setMetadataFilters] = useState<string[]>([])
  const [dateRange, setDateRange] = useState<string>('All dates')
  const [boostRecent, setBoostRecent] = useState<'Enabled' | 'Disabled'>(
    'Disabled'
  )
  const [sourcePriority, setSourcePriority] = useState<'Enabled' | 'Disabled'>(
    'Disabled'
  )
  const [sectionTypeBoosting, setSectionTypeBoosting] =
    useState('Headers: 1.2x')
  const [diversityFactor, setDiversityFactor] = useState<string>('0.3')

  // Performance
  const [cacheResults, setCacheResults] = useState<'Enabled' | 'Disabled'>(
    'Enabled'
  )
  const [cacheTtl, setCacheTtl] = useState('1 hour')
  const [asyncProcessing, setAsyncProcessing] = useState<
    'Enabled' | 'Disabled'
  >('Enabled')
  const [maxConcurrentQueries, setMaxConcurrentQueries] = useState<string>('10')
  const [timeoutPerQuery, setTimeoutPerQuery] = useState<string>('30')

  // Validation
  const [errors, setErrors] = useState<Record<string, string>>({})
  const validate = () => {
    const next: Record<string, string> = {}
    const nResults = Number(resultsCount)
    const nSem = Number(semanticWeight)
    const nKey = Number(keywordWeight)
    const nSim = Number(similarityThreshold)
    const nMax = Number(maxResultsBeforeRerank)
    const nFinal = Number(finalResultCount)
    const nBatch = Number(crossEncoderBatchSize)
    const nMaxConc = Number(maxConcurrentQueries)
    const nTimeout = Number(timeoutPerQuery)
    if (!Number.isFinite(nResults) || nResults < 1 || nResults > 50)
      next.resultsCount = 'Enter 1–50'
    if (!Number.isFinite(nSem) || nSem < 0 || nSem > 100)
      next.semanticWeight = 'Enter 0–100'
    if (!Number.isFinite(nKey) || nKey < 0 || nKey > 100)
      next.keywordWeight = 'Enter 0–100'
    if (!Number.isFinite(nSim) || nSim < 0 || nSim > 1)
      next.similarityThreshold = 'Enter 0–1'
    if (!Number.isFinite(nMax) || nMax < 5 || nMax > 100)
      next.maxResultsBeforeRerank = 'Enter 5–100'
    if (!Number.isFinite(nFinal) || nFinal < 1 || nFinal > 20)
      next.finalResultCount = 'Enter 1–20'
    if (!Number.isFinite(nBatch) || nBatch < 1 || nBatch > 128)
      next.crossEncoderBatchSize = 'Enter 1–128'
    if (!Number.isFinite(nMaxConc) || nMaxConc < 1 || nMaxConc > 100)
      next.maxConcurrentQueries = 'Enter 1–100'
    if (!Number.isFinite(nTimeout) || nTimeout < 5 || nTimeout > 300)
      next.timeoutPerQuery = 'Enter 5–300 seconds'
    setErrors(next)
    return Object.keys(next).length === 0
  }

  // Load persisted values
  useEffect(() => {
    try {
      if (!storageKey) return
      const raw = localStorage.getItem(storageKey)
      if (!raw) return
      const cfg = JSON.parse(raw)
      setSearchType(cfg.searchType ?? searchType)
      setResultsCount(String(cfg.resultsCount ?? resultsCount))
      setReranking(cfg.reranking ?? reranking)

      setSemanticWeight(String(cfg.semanticWeight ?? semanticWeight))
      setKeywordWeight(String(cfg.keywordWeight ?? keywordWeight))
      setBm25Params(cfg.bm25Params ?? bm25Params)
      setSimilarityThreshold(
        String(cfg.similarityThreshold ?? similarityThreshold)
      )
      setMaxResultsBeforeRerank(
        String(cfg.maxResultsBeforeRerank ?? maxResultsBeforeRerank)
      )
      setQueryExpansion(cfg.queryExpansion ?? queryExpansion)

      setRerankingModel(cfg.rerankingModel ?? rerankingModel)
      setRerankThreshold(String(cfg.rerankThreshold ?? rerankThreshold))
      setFinalResultCount(String(cfg.finalResultCount ?? finalResultCount))
      setCrossEncoderBatchSize(
        String(cfg.crossEncoderBatchSize ?? crossEncoderBatchSize)
      )
      setGpuAcceleration(cfg.gpuAcceleration ?? gpuAcceleration)

      setMetadataFilters(cfg.metadataFilters ?? metadataFilters)
      setDateRange(cfg.dateRange ?? dateRange)
      setBoostRecent(cfg.boostRecent ?? boostRecent)
      setSourcePriority(cfg.sourcePriority ?? sourcePriority)
      setSectionTypeBoosting(cfg.sectionTypeBoosting ?? sectionTypeBoosting)
      setDiversityFactor(String(cfg.diversityFactor ?? diversityFactor))

      setCacheResults(cfg.cacheResults ?? cacheResults)
      setCacheTtl(cfg.cacheTtl ?? cacheTtl)
      setAsyncProcessing(cfg.asyncProcessing ?? asyncProcessing)
      setMaxConcurrentQueries(
        String(cfg.maxConcurrentQueries ?? maxConcurrentQueries)
      )
      setTimeoutPerQuery(String(cfg.timeoutPerQuery ?? timeoutPerQuery))
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey])

  const [saveState, setSaveState] = useState<'idle' | 'loading' | 'success'>(
    'idle'
  )

  const handleSave = () => {
    if (!validate()) return
    setSaveState('loading')
    setTimeout(() => {
      try {
        const payload = {
          searchType,
          resultsCount: Number(resultsCount),
          reranking,
          semanticWeight: Number(semanticWeight),
          keywordWeight: Number(keywordWeight),
          bm25Params,
          similarityThreshold: Number(similarityThreshold),
          maxResultsBeforeRerank: Number(maxResultsBeforeRerank),
          queryExpansion,
          rerankingModel,
          rerankThreshold: Number(rerankThreshold),
          finalResultCount: Number(finalResultCount),
          crossEncoderBatchSize: Number(crossEncoderBatchSize),
          gpuAcceleration,
          metadataFilters,
          dateRange,
          boostRecent,
          sourcePriority,
          sectionTypeBoosting,
          diversityFactor: Number(diversityFactor),
          cacheResults,
          cacheTtl,
          asyncProcessing,
          maxConcurrentQueries: Number(maxConcurrentQueries),
          timeoutPerQuery: Number(timeoutPerQuery),
        }
        if (storageKey)
          localStorage.setItem(storageKey, JSON.stringify(payload))
        try {
          if (typeof window !== 'undefined') {
            window.dispatchEvent(
              new CustomEvent('lf:strategyRetrievalUpdated', {
                detail: { strategyId, payload },
              })
            )
          }
        } catch {}
      } catch {}
      setSaveState('success')
      setTimeout(() => setSaveState('idle'), 800)
    }, 600)
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-20">
      {/* Breadcrumb + Actions */}
      <div className="flex items-center justify-between mb-1">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/rag')}
          >
            RAG
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate(`/chat/rag/${strategyId}`)}
          >
            {strategyName}
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">Retrieval method</span>
        </nav>
        <PageActions mode={mode} onModeChange={setMode} />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg md:text-xl font-medium">Retrieval method</h2>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/chat/rag/${strategyId}`)}
          >
            Back
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={saveState === 'loading'}
          >
            {saveState === 'success' ? (
              <span className="mr-2 inline-flex">
                <FontIcon type="checkmark-filled" className="w-4 h-4" />
              </span>
            ) : null}
            Save
          </Button>
        </div>
      </div>

      {/* Basic settings */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="text-sm font-medium mb-3">Basic settings</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Field label="Search Type">
            <SelectDropdown
              value={searchType}
              onChange={setSearchType}
              options={[
                'Hybrid',
                'Semantic only',
                'Keyword only',
                'Neural search',
              ]}
            />
          </Field>
          <Field label="Results Count">
            <Input
              type="number"
              value={resultsCount}
              onChange={e => setResultsCount(e.target.value)}
              className={`${inputClass} ${errors.resultsCount ? 'border-destructive' : ''}`}
            />
            {errors.resultsCount && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.resultsCount}
              </div>
            )}
          </Field>
          <Field label="Re-ranking">
            <SelectDropdown
              value={reranking}
              onChange={v => setReranking(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
        </div>
      </section>

      {/* Search Configuration */}
      <Collapsible
        title="Search Configuration"
        open={searchOpen}
        onToggle={() => setSearchOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Semantic Weight (%)">
            <div className="grid grid-cols-3 gap-2 items-center">
              <input
                type="range"
                min={0}
                max={100}
                value={Number(semanticWeight)}
                onChange={e => setSemanticWeight(e.target.value)}
                className="col-span-2 w-full"
              />
              <Input
                type="number"
                value={semanticWeight}
                onChange={e => setSemanticWeight(e.target.value)}
                className={`${inputClass} ${errors.semanticWeight ? 'border-destructive' : ''}`}
              />
            </div>
            {errors.semanticWeight && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.semanticWeight}
              </div>
            )}
          </Field>
          <Field label="Keyword Weight (%)">
            <div className="grid grid-cols-3 gap-2 items-center">
              <input
                type="range"
                min={0}
                max={100}
                value={Number(keywordWeight)}
                onChange={e => setKeywordWeight(e.target.value)}
                className="col-span-2 w-full"
              />
              <Input
                type="number"
                value={keywordWeight}
                onChange={e => setKeywordWeight(e.target.value)}
                className={`${inputClass} ${errors.keywordWeight ? 'border-destructive' : ''}`}
              />
            </div>
            {errors.keywordWeight && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.keywordWeight}
              </div>
            )}
          </Field>
          <Field label="BM25 Parameters">
            <Input
              value={bm25Params}
              onChange={e => setBm25Params(e.target.value)}
              placeholder="k1=value, b=value"
              className={inputClass}
            />
          </Field>
          <Field label="Similarity Threshold (0–1)">
            <div className="grid grid-cols-3 gap-2 items-center">
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={Number(similarityThreshold)}
                onChange={e => setSimilarityThreshold(e.target.value)}
                className="col-span-2 w-full"
              />
              <Input
                type="number"
                step="0.01"
                value={similarityThreshold}
                onChange={e => setSimilarityThreshold(e.target.value)}
                className={`${inputClass} ${errors.similarityThreshold ? 'border-destructive' : ''}`}
              />
            </div>
            {errors.similarityThreshold && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.similarityThreshold}
              </div>
            )}
          </Field>
          <Field label="Max Results Before Re-rank">
            <Input
              type="number"
              value={maxResultsBeforeRerank}
              onChange={e => setMaxResultsBeforeRerank(e.target.value)}
              className={`${inputClass} ${errors.maxResultsBeforeRerank ? 'border-destructive' : ''}`}
            />
            {errors.maxResultsBeforeRerank && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.maxResultsBeforeRerank}
              </div>
            )}
          </Field>
          <Field label="Query Expansion">
            <SelectDropdown
              value={queryExpansion}
              onChange={v => setQueryExpansion(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
        </div>
      </Collapsible>

      {/* Re-ranking */}
      <Collapsible
        title="Re-ranking"
        open={rerankOpen}
        onToggle={() => setRerankOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Re-ranking Model">
            <SelectDropdown
              value={rerankingModel}
              onChange={setRerankingModel}
              options={[
                'ms-marco-v2',
                'cross-encoder-base',
                'custom-model',
                'disabled',
              ]}
            />
          </Field>
          <Field label="Re-ranking Threshold (0–1)">
            <div className="grid grid-cols-3 gap-2 items-center">
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={Number(rerankThreshold)}
                onChange={e => setRerankThreshold(e.target.value)}
                className="col-span-2 w-full"
              />
              <Input
                type="number"
                step="0.01"
                value={rerankThreshold}
                onChange={e => setRerankThreshold(e.target.value)}
                className={inputClass}
              />
            </div>
          </Field>
          <Field label="Final Result Count">
            <Input
              type="number"
              value={finalResultCount}
              onChange={e => setFinalResultCount(e.target.value)}
              className={`${inputClass} ${errors.finalResultCount ? 'border-destructive' : ''}`}
            />
            {errors.finalResultCount && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.finalResultCount}
              </div>
            )}
          </Field>
          <Field label="Cross-encoder Batch Size">
            <Input
              type="number"
              value={crossEncoderBatchSize}
              onChange={e => setCrossEncoderBatchSize(e.target.value)}
              className={`${inputClass} ${errors.crossEncoderBatchSize ? 'border-destructive' : ''}`}
            />
            {errors.crossEncoderBatchSize && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.crossEncoderBatchSize}
              </div>
            )}
          </Field>
          <Field label="Use GPU Acceleration">
            <SelectDropdown
              value={gpuAcceleration}
              onChange={setGpuAcceleration}
              options={['Auto-detect', 'Force GPU', 'Force CPU', 'Disabled']}
            />
          </Field>
        </div>
      </Collapsible>

      {/* Filtering & Boosting */}
      <Collapsible
        title="Filtering & Boosting"
        open={filtersOpen}
        onToggle={() => setFiltersOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Metadata Filters">
            <Input
              value={metadataFilters.join(', ')}
              onChange={e =>
                setMetadataFilters(
                  e.target.value
                    .split(',')
                    .map(v => v.trim())
                    .filter(Boolean)
                )
              }
              placeholder="department, document_type, date_range, author, version"
              className={inputClass}
            />
          </Field>
          <Field label="Date Range Filter">
            <SelectDropdown
              value={dateRange}
              onChange={setDateRange}
              options={[
                'All dates',
                'Past week',
                'Past month',
                'Past year',
                'Custom',
              ]}
            />
          </Field>
          <Field label="Boost Recent Content">
            <SelectDropdown
              value={boostRecent}
              onChange={v => setBoostRecent(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Source Priority Boosting">
            <SelectDropdown
              value={sourcePriority}
              onChange={v => setSourcePriority(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Section Type Boosting">
            <Input
              value={sectionTypeBoosting}
              onChange={e => setSectionTypeBoosting(e.target.value)}
              placeholder="section_type: multiplier"
              className={inputClass}
            />
          </Field>
          <Field label="Diversity Factor (0–1)">
            <div className="grid grid-cols-3 gap-2 items-center">
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={Number(diversityFactor)}
                onChange={e => setDiversityFactor(e.target.value)}
                className="col-span-2 w-full"
              />
              <Input
                type="number"
                step="0.1"
                value={diversityFactor}
                onChange={e => setDiversityFactor(e.target.value)}
                className={inputClass}
              />
            </div>
          </Field>
        </div>
      </Collapsible>

      {/* Performance */}
      <Collapsible
        title="Performance"
        open={perfOpen}
        onToggle={() => setPerfOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Cache Results">
            <SelectDropdown
              value={cacheResults}
              onChange={v => setCacheResults(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Cache TTL">
            <SelectDropdown
              value={cacheTtl}
              onChange={setCacheTtl}
              options={['1 hour', '4 hours', '12 hours', '24 hours', '1 week']}
            />
          </Field>
          <Field label="Async Processing">
            <SelectDropdown
              value={asyncProcessing}
              onChange={v => setAsyncProcessing(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Max Concurrent Queries">
            <Input
              type="number"
              value={maxConcurrentQueries}
              onChange={e => setMaxConcurrentQueries(e.target.value)}
              className={`${inputClass} ${errors.maxConcurrentQueries ? 'border-destructive' : ''}`}
            />
            {errors.maxConcurrentQueries && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.maxConcurrentQueries}
              </div>
            )}
          </Field>
          <Field label="Timeout per Query (seconds)">
            <Input
              type="number"
              value={timeoutPerQuery}
              onChange={e => setTimeoutPerQuery(e.target.value)}
              className={`${inputClass} ${errors.timeoutPerQuery ? 'border-destructive' : ''}`}
            />
            {errors.timeoutPerQuery && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.timeoutPerQuery}
              </div>
            )}
          </Field>
        </div>
      </Collapsible>
    </div>
  )
}

export default RetrievalMethod
