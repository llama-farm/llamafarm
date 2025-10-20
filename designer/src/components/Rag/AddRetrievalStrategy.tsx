import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { Button } from '../ui/button'
import { useActiveProject } from '../../hooks/useActiveProject'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
// removed unused imports
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

function AddRetrievalStrategy() {
  const navigate = useNavigate()
  const [searchParams] = useSearchParams()
  const activeProject = useActiveProject()
  const projectKey = useMemo(() => {
    const ns = activeProject?.namespace || 'global'
    const proj = activeProject?.project || 'global'
    return `${ns}__${proj}`
  }, [activeProject?.namespace, activeProject?.project])

  // Get the database from URL query params (defaults to main_database if not provided)
  const database = searchParams.get('database') || 'main_database'

  // New retrieval name and default toggle
  const [name, setName] = useState('New retrieval strategy')
  const [makeDefault, setMakeDefault] = useState(false)

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
    try {
      const raw = localStorage.getItem(
        `lf_ui_${projectKey}_db_${database}_retrievals`
      )
      const list = raw ? JSON.parse(raw) : []
      if (!Array.isArray(list) || list.length === 0) setMakeDefault(true)
    } catch {}
  }, [projectKey, database])

  // Save handler
  const onSave = () => {
    if (!selectedType) return
    // generate id
    const slug = name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
    const baseId = `ret-${slug || Date.now()}`
    const raw = localStorage.getItem(
      `lf_ui_${projectKey}_db_${database}_retrievals`
    )
    const list = raw ? JSON.parse(raw) : []
    const exists =
      Array.isArray(list) && list.some((e: any) => e?.id === baseId)
    const id = exists ? `${baseId}-${Date.now()}` : baseId

    // Add list entry
    const entry = {
      id,
      name: name.trim() || id,
      isDefault: Array.isArray(list) && list.length === 0 ? true : false,
      enabled: true,
    }
    const nextList = Array.isArray(list) ? [...list, entry] : [entry]
    // If makeDefault checked, enforce uniqueness
    const finalList = makeDefault
      ? nextList.map((r: any) => ({ ...r, isDefault: r.id === id }))
      : nextList
    localStorage.setItem(
      `lf_ui_${projectKey}_db_${database}_retrievals`,
      JSON.stringify(finalList)
    )

    // Build config from current state like the edit page
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
    const payload = { type: selectedType, config }
    localStorage.setItem(
      `lf_ui_${projectKey}_db_${database}_retrieval_${id}`,
      JSON.stringify(payload)
    )

    navigate('/chat/databases')
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
          >
            Cancel
          </Button>
          <Button
            onClick={onSave}
            disabled={!selectedType || name.trim().length === 0}
          >
            Save strategy
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
              onChange={e => setName(e.target.value)}
              placeholder="Enter a name"
              className="h-9"
            />
          </div>
          <div className="flex items-end">
            <label className="inline-flex items-center gap-2 text-sm">
              <input
                type="checkbox"
                checked={makeDefault}
                onChange={e => setMakeDefault(e.target.checked)}
              />
              <span>Make default</span>
            </label>
          </div>
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
    </div>
  )
}

export default AddRetrievalStrategy
