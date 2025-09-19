import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import PageActions from '../common/PageActions'
import { Mode } from '../ModeToggle'
import FontIcon from '../../common/FontIcon'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Badge } from '../ui/badge'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../ui/dropdown-menu'
import {
  getDefaultConfigForRetrieval,
  parseWeightsList,
} from '../../utils/retrievalUtils'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'

function RetrievalMethod() {
  const navigate = useNavigate()
  const { strategyId } = useParams()
  const [mode, setMode] = useState<Mode>('designer')
  const [, setDefaultTick] = useState(0)

  // removed unused strategyName

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
          className={`h-9 w-full rounded-md border border-input bg-background px-3 text-left flex items-center justify-between ${
            className || ''
          } ${inputClass}`}
        >
          <span className="truncate">{value}</span>
          <FontIcon type="chevron-down" className="w-4 h-4 opacity-70 ml-2" />
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

  // removed unused Collapsible

  // Strategy options from schema
  const STRATEGY_TYPES = [
    'BasicSimilarityStrategy',
    'MetadataFilteredStrategy',
    'MultiQueryStrategy',
    'RerankedStrategy',
    'HybridUniversalStrategy',
  ] as const

  type StrategyType = (typeof STRATEGY_TYPES)[number]

  const STRATEGY_LABELS: Record<StrategyType, string> = {
    BasicSimilarityStrategy: 'Basic similarity',
    MetadataFilteredStrategy: 'Metadata-filtered',
    MultiQueryStrategy: 'Multi-query',
    RerankedStrategy: 'Reranked',
    HybridUniversalStrategy: 'Hybrid universal',
  }

  const STRATEGY_SLUG: Record<StrategyType, string> = {
    BasicSimilarityStrategy: 'basic-search',
    MetadataFilteredStrategy: 'metadata-filtered',
    MultiQueryStrategy: 'multi-query',
    RerankedStrategy: 'reranked',
    HybridUniversalStrategy: 'hybrid-universal',
  }

  const STRATEGY_DESCRIPTIONS: Record<StrategyType, string> = {
    BasicSimilarityStrategy:
      'Simple, fast vector search. Returns the top matches by similarity (you set how many and the distance metric). Optionally filter out weak hits with a score threshold.',
    MetadataFilteredStrategy:
      'Search with filters like source, type, date, or tags. Choose whether filters apply before or after retrieval, and automatically widen results when post-filtering removes too much.',
    MultiQueryStrategy:
      'Ask the question several ways at once. We create multiple query variations and merge their results so you catch relevant content even when phrased differently.',
    RerankedStrategy:
      'Pull a larger candidate set first, then sort by quality. Tune weights for similarity, recency, length, and metadata; optionally normalize scores for fair comparisons.',
    HybridUniversalStrategy:
      'Blend multiple strategies into one result set. Combine with weighted average, rank fusion, or score fusion, then keep the best K.',
  }

  const DISTANCE_OPTIONS = ['cosine', 'euclidean', 'manhattan', 'dot']
  const META_FILTER_MODE = ['pre', 'post']
  const MQ_AGGREGATION = ['max', 'mean', 'weighted', 'reciprocal_rank']
  const HYBRID_COMBINATION = ['weighted_average', 'rank_fusion', 'score_fusion']

  const [selectedStrategy, setSelectedStrategy] = useState<StrategyType>(
    'BasicSimilarityStrategy'
  )
  const [isChangeOpen, setIsChangeOpen] = useState(false)
  const [pendingStrategy, setPendingStrategy] = useState<StrategyType | null>(
    null
  )

  // BasicSimilarityStrategy config
  const [basicTopK, setBasicTopK] = useState<string>('10')
  const [basicDistance, setBasicDistance] = useState<string>('cosine')
  const [basicScoreThreshold, setBasicScoreThreshold] = useState<string>('')

  // MetadataFilteredStrategy config
  const [mfTopK, setMfTopK] = useState<string>('10')
  const [mfFilterMode, setMfFilterMode] = useState<string>('pre')
  const [mfFallbackMultiplier, setMfFallbackMultiplier] = useState<string>('3')
  const [mfFilters, setMfFilters] = useState<
    Array<{ key: string; value: string }>
  >([])

  // MultiQueryStrategy config
  const [mqNumQueries, setMqNumQueries] = useState<string>('3')
  const [mqTopK, setMqTopK] = useState<string>('10')
  const [mqAggregation, setMqAggregation] = useState<string>('weighted')
  const [mqQueryWeights, setMqQueryWeights] = useState<string>('')

  // RerankedStrategy config
  const [rrInitialK, setRrInitialK] = useState<string>('30')
  const [rrFinalK, setRrFinalK] = useState<string>('10')
  const [rrSimW, setRrSimW] = useState<string>('0.7')
  const [rrRecencyW, setRrRecencyW] = useState<string>('0.1')
  const [rrLengthW, setRrLengthW] = useState<string>('0.1')
  const [rrMetaW, setRrMetaW] = useState<string>('0.1')
  const [rrNormalize, setRrNormalize] = useState<'Enabled' | 'Disabled'>(
    'Enabled'
  )

  // HybridUniversalStrategy config
  type HybridSub = {
    id: string
    type: Exclude<StrategyType, 'HybridUniversalStrategy'>
    weight: string
    // Minimal nested config per sub-strategy
    config?: Record<string, unknown>
  }
  const [hybStrategies, setHybStrategies] = useState<HybridSub[]>([])
  const [hybCombination, setHybCombination] =
    useState<string>('weighted_average')
  const [hybFinalK, setHybFinalK] = useState<string>('10')

  // Retrieval list helpers for default handling
  const RET_LIST_KEY = 'lf_project_retrievals'
  const getRetrievals = (): Array<{
    id: string
    name: string
    isDefault: boolean
    enabled: boolean
  }> => {
    try {
      const raw = localStorage.getItem(RET_LIST_KEY)
      const arr = raw ? JSON.parse(raw) : []
      if (!Array.isArray(arr)) return []
      return arr.filter(
        (e: any) => e && typeof e.id === 'string' && typeof e.name === 'string'
      )
    } catch {
      return []
    }
  }
  const saveRetrievals = (
    list: Array<{
      id: string
      name: string
      isDefault: boolean
      enabled: boolean
    }>
  ) => {
    try {
      localStorage.setItem(RET_LIST_KEY, JSON.stringify(list))
    } catch {}
  }
  const setDefaultRetrieval = (id: string) => {
    const list = getRetrievals()
    const next = list.map(r => ({ ...r, isDefault: r.id === id }))
    saveRetrievals(next)
    setDefaultTick(t => t + 1)
  }

  // Helpers for Hybrid sub-strategy config defaults and updates
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

  // Validation
  const [errors] = useState<Record<string, string>>({})

  // Load persisted values
  useEffect(() => {
    try {
      if (!storageKey) return
      const raw = localStorage.getItem(storageKey)
      if (!raw) return
      const saved = JSON.parse(raw)
      if (saved && typeof saved === 'object' && saved.type && saved.config) {
        const t = saved.type as StrategyType
        if (STRATEGY_TYPES.includes(t)) setSelectedStrategy(t)
        const cfg = saved.config || {}
        if (t === 'BasicSimilarityStrategy') {
          setBasicTopK(String(cfg.top_k ?? '10'))
          setBasicDistance(String(cfg.distance_metric ?? 'cosine'))
          setBasicScoreThreshold(
            cfg.score_threshold === null || cfg.score_threshold === undefined
              ? ''
              : String(cfg.score_threshold)
          )
        } else if (t === 'MetadataFilteredStrategy') {
          setMfTopK(String(cfg.top_k ?? '10'))
          setMfFilterMode(String(cfg.filter_mode ?? 'pre'))
          setMfFallbackMultiplier(String(cfg.fallback_multiplier ?? '3'))
          const fObj = (cfg.filters as Record<string, unknown>) || {}
          const list: Array<{ key: string; value: string }> = []
          for (const k of Object.keys(fObj)) {
            const v = (fObj as any)[k]
            if (Array.isArray(v)) list.push({ key: k, value: v.join(', ') })
            else list.push({ key: k, value: String(v) })
          }
          setMfFilters(list)
        } else if (t === 'MultiQueryStrategy') {
          setMqNumQueries(String(cfg.num_queries ?? '3'))
          setMqTopK(String(cfg.top_k ?? '10'))
          setMqAggregation(String(cfg.aggregation_method ?? 'weighted'))
          const qw = cfg.query_weights as number[] | null | undefined
          setMqQueryWeights(
            Array.isArray(qw) ? qw.map(n => String(n)).join(', ') : ''
          )
        } else if (t === 'RerankedStrategy') {
          setRrInitialK(String(cfg.initial_k ?? '30'))
          setRrFinalK(String(cfg.final_k ?? '10'))
          const rf = (cfg.rerank_factors as Record<string, unknown>) || {}
          setRrSimW(String(rf.similarity_weight ?? '0.7'))
          setRrRecencyW(String(rf.recency_weight ?? '0.1'))
          setRrLengthW(String(rf.length_weight ?? '0.1'))
          setRrMetaW(String(rf.metadata_weight ?? '0.1'))
          setRrNormalize(
            (cfg.normalize_scores ?? true) ? 'Enabled' : 'Disabled'
          )
        } else if (t === 'HybridUniversalStrategy') {
          setHybCombination(
            String(cfg.combination_method ?? 'weighted_average')
          )
          setHybFinalK(String(cfg.final_k ?? '10'))
          const subs = (cfg.strategies as any[]) || []
          const mapped: HybridSub[] = subs
            .map((s: any, i: number) => {
              const t = (s.type as StrategyType) || 'BasicSimilarityStrategy'
              const subType: Exclude<StrategyType, 'HybridUniversalStrategy'> =
                t === 'HybridUniversalStrategy'
                  ? 'BasicSimilarityStrategy'
                  : (t as Exclude<StrategyType, 'HybridUniversalStrategy'>)
              return {
                id: `${s.type || 'sub'}-${i}`,
                type: subType,
                weight: String(s.weight ?? '1.0'),
                config: s.config || {},
              }
            })
            .filter(Boolean)
          setHybStrategies(mapped)
        }
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey])

  // Auto-save on change
  useEffect(() => {
    try {
      if (!storageKey) return
      let config: Record<string, unknown> = {}
      if (selectedStrategy === 'BasicSimilarityStrategy') {
        config = {
          top_k: Number(basicTopK),
          distance_metric: basicDistance,
          score_threshold:
            basicScoreThreshold.trim() === ''
              ? null
              : Number(basicScoreThreshold),
        }
      } else if (selectedStrategy === 'MetadataFilteredStrategy') {
        const filters: Record<string, unknown> = {}
        for (const { key, value } of mfFilters) {
          if (!key.trim()) continue
          const raw = value.trim()
          if (raw.includes(','))
            filters[key] = raw.split(',').map(v => v.trim())
          else if (raw === 'true' || raw === 'false')
            filters[key] = raw === 'true'
          else if (!Number.isNaN(Number(raw))) filters[key] = Number(raw)
          else filters[key] = raw
        }
        config = {
          top_k: Number(mfTopK),
          filters,
          filter_mode: mfFilterMode,
          fallback_multiplier: Number(mfFallbackMultiplier),
        }
      } else if (selectedStrategy === 'MultiQueryStrategy') {
        const weights = parseWeightsList(mqQueryWeights, Number(mqNumQueries))
        config = {
          num_queries: Number(mqNumQueries),
          top_k: Number(mqTopK),
          aggregation_method: mqAggregation,
          query_weights: weights,
        }
      } else if (selectedStrategy === 'RerankedStrategy') {
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
      } else if (selectedStrategy === 'HybridUniversalStrategy') {
        const strategies = hybStrategies.map(s => ({
          type: s.type,
          weight: Number(s.weight),
          config: s.config || {},
        }))
        config = {
          strategies,
          combination_method: hybCombination,
          final_k: Number(hybFinalK),
        }
      }
      const payload = { type: selectedStrategy, config }
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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    storageKey,
    strategyId,
    selectedStrategy,
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
    hybCombination,
    hybFinalK,
    hybStrategies,
  ])

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
          <span className="text-foreground">Edit retrieval strategy</span>
        </nav>
        <PageActions mode={mode} onModeChange={setMode} />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg md:text-xl font-medium">
          Edit retrieval strategy
        </h2>
        <div className="flex items-center gap-2" />
      </div>
      {/* Strategy summary card */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between">
          <div className="flex-1 min-w-0">
            <div className="text-xs text-muted-foreground mb-1">
              Current strategy
            </div>
            <div className="text-xl md:text-2xl font-medium">
              {STRATEGY_SLUG[selectedStrategy]}
            </div>
            <div className="text-xs text-muted-foreground mt-1">
              {STRATEGY_DESCRIPTIONS[selectedStrategy]}
            </div>
          </div>
          <div className="ml-3 shrink-0 flex items-center gap-3">
            {strategyId
              ? (() => {
                  const list = getRetrievals()
                  const found = list.find(r => r.id === strategyId)
                  const isDefault = Boolean(found?.isDefault)
                  const onlyOne = list.length <= 1
                  return (
                    <label className="inline-flex items-center gap-2 text-xs">
                      <input
                        type="checkbox"
                        checked={isDefault}
                        disabled={isDefault || onlyOne}
                        onChange={e => {
                          if (e.target.checked && strategyId) {
                            setDefaultRetrieval(strategyId)
                          }
                        }}
                      />
                      <span>Make default</span>
                    </label>
                  )
                })()
              : null}
            <Button
              variant="outline"
              size="sm"
              onClick={() => setIsChangeOpen(true)}
            >
              Change
            </Button>
          </div>
        </div>
      </section>

      {/* Strategy-specific settings */}
      {selectedStrategy === 'BasicSimilarityStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Basic similarity</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Top K">
              <Input
                type="number"
                value={basicTopK}
                onChange={e => setBasicTopK(e.target.value)}
                className={`${inputClass} ${errors.basicTopK ? 'border-destructive' : ''}`}
              />
              {errors.basicTopK && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.basicTopK}
                </div>
              )}
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
                className={`${inputClass} ${errors.basicScoreThreshold ? 'border-destructive' : ''}`}
              />
              {errors.basicScoreThreshold && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.basicScoreThreshold}
                </div>
              )}
            </Field>
          </div>
        </section>
      ) : null}

      {selectedStrategy === 'MetadataFilteredStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Metadata-filtered</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Top K">
              <Input
                type="number"
                value={mfTopK}
                onChange={e => setMfTopK(e.target.value)}
                className={`${inputClass} ${errors.mfTopK ? 'border-destructive' : ''}`}
              />
              {errors.mfTopK && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.mfTopK}
                </div>
              )}
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
                className={`${inputClass} ${errors.mfFallbackMultiplier ? 'border-destructive' : ''}`}
              />
              {errors.mfFallbackMultiplier && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.mfFallbackMultiplier}
                </div>
              )}
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

      {selectedStrategy === 'MultiQueryStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Multi-query</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Number of queries">
              <Input
                type="number"
                value={mqNumQueries}
                onChange={e => setMqNumQueries(e.target.value)}
                className={`${inputClass} ${errors.mqNumQueries ? 'border-destructive' : ''}`}
              />
              {errors.mqNumQueries && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.mqNumQueries}
                </div>
              )}
            </Field>
            <Field label="Top K per query">
              <Input
                type="number"
                value={mqTopK}
                onChange={e => setMqTopK(e.target.value)}
                className={`${inputClass} ${errors.mqTopK ? 'border-destructive' : ''}`}
              />
              {errors.mqTopK && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.mqTopK}
                </div>
              )}
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
                className={`${inputClass} ${errors.mqQueryWeights ? 'border-destructive' : ''}`}
              />
              {errors.mqQueryWeights && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.mqQueryWeights}
                </div>
              )}
            </Field>
          </div>
        </section>
      ) : null}

      {selectedStrategy === 'RerankedStrategy' ? (
        <section className="rounded-lg border border-border bg-card p-4">
          <div className="text-sm font-medium mb-3">Reranked</div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <Field label="Initial candidates (K)">
              <Input
                type="number"
                value={rrInitialK}
                onChange={e => setRrInitialK(e.target.value)}
                className={`${inputClass} ${errors.rrInitialK ? 'border-destructive' : ''}`}
              />
              {errors.rrInitialK && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.rrInitialK}
                </div>
              )}
            </Field>
            <Field label="Final results (K)">
              <Input
                type="number"
                value={rrFinalK}
                onChange={e => setRrFinalK(e.target.value)}
                className={`${inputClass} ${errors.rrFinalK ? 'border-destructive' : ''}`}
              />
              {errors.rrFinalK && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.rrFinalK}
                </div>
              )}
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
                className={`${inputClass} ${errors.rrSimW ? 'border-destructive' : ''}`}
              />
              {errors.rrSimW && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.rrSimW}
                </div>
              )}
            </Field>
            <Field label="Recency weight (0–1)">
              <Input
                type="number"
                step="0.01"
                value={rrRecencyW}
                onChange={e => setRrRecencyW(e.target.value)}
                className={`${inputClass} ${errors.rrRecencyW ? 'border-destructive' : ''}`}
              />
              {errors.rrRecencyW && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.rrRecencyW}
                </div>
              )}
            </Field>
            <Field label="Length weight (0–1)">
              <Input
                type="number"
                step="0.01"
                value={rrLengthW}
                onChange={e => setRrLengthW(e.target.value)}
                className={`${inputClass} ${errors.rrLengthW ? 'border-destructive' : ''}`}
              />
              {errors.rrLengthW && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.rrLengthW}
                </div>
              )}
            </Field>
            <Field label="Metadata weight (0–1)">
              <Input
                type="number"
                step="0.01"
                value={rrMetaW}
                onChange={e => setRrMetaW(e.target.value)}
                className={`${inputClass} ${errors.rrMetaW ? 'border-destructive' : ''}`}
              />
              {errors.rrMetaW && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.rrMetaW}
                </div>
              )}
            </Field>
          </div>
        </section>
      ) : null}

      {selectedStrategy === 'HybridUniversalStrategy' ? (
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
                className={`${inputClass} ${errors.hybFinalK ? 'border-destructive' : ''}`}
              />
              {errors.hybFinalK && (
                <div className="text-xs text-destructive mt-0.5">
                  {errors.hybFinalK}
                </div>
              )}
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
                      onChange={e => {
                        updateHybridSub(idx, { weight: e.target.value })
                      }}
                      className={`${inputClass} ${errors[`hybWeight:${s.id}`] ? 'border-destructive' : ''}`}
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
                  {/* Sub-strategy specific settings */}
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
                        <div>
                          <div className="text-xs text-muted-foreground mb-1">
                            Metadata filters
                          </div>
                          <div className="flex flex-col gap-2">
                            {Object.entries(
                              ((s.config as any)?.filters as Record<
                                string,
                                unknown
                              >) || {}
                            ).map(([key, value], fIdx) => (
                              <div
                                key={`${key}-${fIdx}`}
                                className="grid grid-cols-1 md:grid-cols-3 gap-2"
                              >
                                <Input
                                  placeholder="key"
                                  value={key}
                                  onChange={e => {
                                    const newKey = e.target.value
                                    updateHybridSubConfig(idx, prev => {
                                      const current =
                                        (prev.filters as Record<
                                          string,
                                          unknown
                                        >) || {}
                                      const nextFilters: Record<
                                        string,
                                        unknown
                                      > = {}
                                      Object.entries(current).forEach(
                                        ([k, v], i) => {
                                          if (i === fIdx)
                                            nextFilters[newKey] = v
                                          else nextFilters[k] = v
                                        }
                                      )
                                      return { ...prev, filters: nextFilters }
                                    })
                                  }}
                                  className={inputClass}
                                />
                                <div className="md:col-span-2 flex items-center gap-2">
                                  <Input
                                    placeholder="value (supports numbers, true/false, or a,b,c)"
                                    value={
                                      Array.isArray(value)
                                        ? (value as unknown[])
                                            .map(v => String(v))
                                            .join(', ')
                                        : String(value)
                                    }
                                    onChange={e => {
                                      const raw = e.target.value.trim()
                                      const parsed: unknown = raw.includes(',')
                                        ? raw
                                            .split(',')
                                            .map(v => v.trim())
                                            .filter(Boolean)
                                        : raw === 'true' || raw === 'false'
                                          ? raw === 'true'
                                          : Number.isNaN(Number(raw))
                                            ? raw
                                            : Number(raw)
                                      updateHybridSubConfig(idx, prev => {
                                        const current =
                                          (prev.filters as Record<
                                            string,
                                            unknown
                                          >) || {}
                                        const nextFilters: Record<
                                          string,
                                          unknown
                                        > = {}
                                        Object.entries(current).forEach(
                                          ([k, v], i) => {
                                            nextFilters[k] =
                                              i === fIdx ? parsed : v
                                          }
                                        )
                                        return { ...prev, filters: nextFilters }
                                      })
                                    }}
                                    className={inputClass}
                                  />
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() =>
                                      updateHybridSubConfig(idx, prev => {
                                        const current =
                                          (prev.filters as Record<
                                            string,
                                            unknown
                                          >) || {}
                                        const nextFilters: Record<
                                          string,
                                          unknown
                                        > = {}
                                        Object.entries(current).forEach(
                                          ([k, v], i) => {
                                            if (i !== fIdx) nextFilters[k] = v
                                          }
                                        )
                                        return { ...prev, filters: nextFilters }
                                      })
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
                                  updateHybridSubConfig(idx, prev => {
                                    const current =
                                      (prev.filters as Record<
                                        string,
                                        unknown
                                      >) || {}
                                    const nextFilters: Record<string, unknown> =
                                      {
                                        ...current,
                                      }
                                    const newKeyBase = 'key'
                                    let newKey = newKeyBase
                                    let i = 1
                                    while (nextFilters.hasOwnProperty(newKey)) {
                                      newKey = `${newKeyBase}_${i++}`
                                    }
                                    nextFilters[newKey] = ''
                                    return { ...prev, filters: nextFilters }
                                  })
                                }
                              >
                                Add filter
                              </Button>
                            </div>
                          </div>
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
                        <Field
                          label={`Query weights (${String(
                            (s.config as any)?.num_queries ?? 3
                          )} values, optional)`}
                        >
                          <Input
                            placeholder="e.g. 0.6, 0.3, 0.1"
                            value={((): string => {
                              const qw = (s.config as any)?.query_weights
                              return Array.isArray(qw)
                                ? qw.map((n: any) => String(n)).join(', ')
                                : ''
                            })()}
                            onChange={e => {
                              const parts = e.target.value
                                .split(',')
                                .map(s => s.trim())
                                .filter(Boolean)
                              const weights = parts.length
                                ? parts.map(p => Number(p))
                                : null
                              updateHybridSubConfig(idx, prev => ({
                                ...prev,
                                query_weights: weights,
                              }))
                            }}
                            className={inputClass}
                          />
                        </Field>
                      </div>
                    ) : null}

                    {s.type === 'RerankedStrategy' ? (
                      <div className="flex flex-col gap-3">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <Field label="Initial candidates (K)">
                            <Input
                              type="number"
                              value={String(
                                ((s.config as any)?.initial_k ?? 30) as number
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  initial_k: Number(e.target.value),
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                          <Field label="Final results (K)">
                            <Input
                              type="number"
                              value={String(
                                ((s.config as any)?.final_k ?? 10) as number
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  final_k: Number(e.target.value),
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                          <Field label="Normalize scores">
                            <SelectDropdown
                              value={
                                ((s.config as any)?.normalize_scores ?? true)
                                  ? 'Enabled'
                                  : 'Disabled'
                              }
                              onChange={v =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  normalize_scores: v === 'Enabled',
                                }))
                              }
                              options={['Enabled', 'Disabled']}
                            />
                          </Field>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
                          <Field label="Similarity weight (0–1)">
                            <Input
                              type="number"
                              step="0.01"
                              value={String(
                                ((s.config as any)?.rerank_factors
                                  ?.similarity_weight ?? 0.7) as number
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  rerank_factors: {
                                    ...(prev as any).rerank_factors,
                                    similarity_weight: Number(e.target.value),
                                  },
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                          <Field label="Recency weight (0–1)">
                            <Input
                              type="number"
                              step="0.01"
                              value={String(
                                ((s.config as any)?.rerank_factors
                                  ?.recency_weight ?? 0.1) as number
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  rerank_factors: {
                                    ...(prev as any).rerank_factors,
                                    recency_weight: Number(e.target.value),
                                  },
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                          <Field label="Length weight (0–1)">
                            <Input
                              type="number"
                              step="0.01"
                              value={String(
                                ((s.config as any)?.rerank_factors
                                  ?.length_weight ?? 0.1) as number
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  rerank_factors: {
                                    ...(prev as any).rerank_factors,
                                    length_weight: Number(e.target.value),
                                  },
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                          <Field label="Metadata weight (0–1)">
                            <Input
                              type="number"
                              step="0.01"
                              value={String(
                                ((s.config as any)?.rerank_factors
                                  ?.metadata_weight ?? 0.1) as number
                              )}
                              onChange={e =>
                                updateHybridSubConfig(idx, prev => ({
                                  ...prev,
                                  rerank_factors: {
                                    ...(prev as any).rerank_factors,
                                    metadata_weight: Number(e.target.value),
                                  },
                                }))
                              }
                              className={inputClass}
                            />
                          </Field>
                        </div>
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

      {/* Change Strategy Modal */}
      <Dialog
        open={isChangeOpen}
        onOpenChange={open => {
          setIsChangeOpen(open)
          if (open) setPendingStrategy(selectedStrategy)
        }}
      >
        <DialogContent className="sm:max-w-3xl p-0">
          <div className="flex flex-col max-h-[80vh]">
            <DialogHeader className="bg-background p-4 border-b">
              <DialogTitle className="text-lg text-foreground">
                Select retrieval strategy
              </DialogTitle>
            </DialogHeader>
            <div className="flex-1 min-h-0 overflow-y-auto p-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {STRATEGY_TYPES.map(t => {
                  const isCurrent = t === selectedStrategy
                  return (
                    <button
                      key={t}
                      className={`text-left rounded-lg border border-border bg-card transition-colors ${
                        t === 'BasicSimilarityStrategy'
                          ? 'pt-2 pb-3 px-3'
                          : 'p-3'
                      } ${
                        pendingStrategy === t ? 'ring-2 ring-teal-500' : ''
                      } ${isCurrent ? 'opacity-70 cursor-not-allowed' : 'hover:bg-accent/20'}`}
                      onClick={() => {
                        if (!isCurrent) setPendingStrategy(t)
                      }}
                      aria-disabled={isCurrent}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2">
                          <div className="text-sm font-medium">
                            {STRATEGY_LABELS[t]}
                          </div>
                          {isCurrent ? (
                            <Badge
                              variant="secondary"
                              size="sm"
                              className="rounded-xl"
                            >
                              Current
                            </Badge>
                          ) : null}
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          disabled={isCurrent}
                          onClick={e => {
                            e.stopPropagation()
                            if (!isCurrent) setPendingStrategy(t)
                          }}
                        >
                          {isCurrent ? 'Current' : 'Select'}
                        </Button>
                      </div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {STRATEGY_DESCRIPTIONS[t]}
                      </div>
                    </button>
                  )
                })}
              </div>
            </div>
            <DialogFooter className="bg-background p-4 border-t flex items-center gap-2">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsChangeOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
                disabled={
                  !pendingStrategy || pendingStrategy === selectedStrategy
                }
                onClick={() => {
                  if (pendingStrategy) setSelectedStrategy(pendingStrategy)
                  setIsChangeOpen(false)
                }}
                type="button"
              >
                Save changes
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default RetrievalMethod
