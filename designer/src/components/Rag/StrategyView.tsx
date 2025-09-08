import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'

function StrategyView() {
  const navigate = useNavigate()
  const { strategyId } = useParams()

  const strategyName = useMemo(() => {
    if (!strategyId) return 'Strategy'
    return strategyId
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
  }, [strategyId])

  const usedBy = ['aircraft-maintenance-guides', 'another dataset']

  const [currentModel, setCurrentModel] = useState<string>(
    'text-embedding-3-large'
  )
  const [savedModel, setSavedModel] = useState<string>('text-embedding-3-large')
  const [saveState, setSaveState] = useState<'idle' | 'loading' | 'success'>(
    'idle'
  )

  // Load current and saved models, and compute change state
  useEffect(() => {
    try {
      if (!strategyId) return
      const storedCfg = localStorage.getItem(
        `lf_strategy_embedding_config_${strategyId}`
      )
      let nextCurrent = currentModel
      if (storedCfg) {
        const parsed = JSON.parse(storedCfg)
        if (parsed?.modelId) nextCurrent = parsed.modelId
      }
      const storedModel = localStorage.getItem(
        `lf_strategy_embedding_model_${strategyId}`
      )
      if (storedModel) nextCurrent = storedModel
      setCurrentModel(nextCurrent)

      const storedSaved = localStorage.getItem(
        `lf_strategy_saved_embedding_model_${strategyId}`
      )
      if (storedSaved) {
        setSavedModel(storedSaved)
      } else {
        // initialize baseline saved == current
        localStorage.setItem(
          `lf_strategy_saved_embedding_model_${strategyId}`,
          nextCurrent
        )
        setSavedModel(nextCurrent)
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [strategyId])

  // Listen for embedding changes from the ChangeEmbeddingModel page
  useEffect(() => {
    const handler = (e: Event) => {
      try {
        // @ts-ignore detail typing
        const { strategyId: sid, modelId } = (e as CustomEvent).detail || {}
        if (sid && strategyId && sid === strategyId && modelId) {
          setCurrentModel(modelId)
        }
      } catch {}
    }
    window.addEventListener(
      'lf:strategyEmbeddingUpdated',
      handler as EventListener
    )
    return () =>
      window.removeEventListener(
        'lf:strategyEmbeddingUpdated',
        handler as EventListener
      )
  }, [strategyId])

  const docTypeFromExtraction = useMemo(() => {
    try {
      if (!strategyId) return null
      const raw = localStorage.getItem(`lf_strategy_extraction_${strategyId}`)
      if (!raw) return null
      const cfg = JSON.parse(raw)
      return cfg?.documentType || null
    } catch {
      return null
    }
  }, [strategyId])

  useEffect(() => {
    const handler = (e: Event) => {
      try {
        // @ts-ignore custom event
        const { strategyId: sid } = (e as CustomEvent).detail || {}
        if (sid && strategyId && sid === strategyId) {
          // force state update by reading localStorage
          setCurrentModel(prev => prev) // no-op to trigger rerender alongside below
        }
      } catch {}
    }
    window.addEventListener(
      'lf:strategyExtractionUpdated',
      handler as EventListener
    )
    return () =>
      window.removeEventListener(
        'lf:strategyExtractionUpdated',
        handler as EventListener
      )
  }, [strategyId])

  const hasChanges = useMemo(
    () => currentModel !== savedModel,
    [currentModel, savedModel]
  )

  const handleSave = () => {
    if (!strategyId || !hasChanges || saveState === 'loading') return
    setSaveState('loading')
    setTimeout(() => {
      try {
        localStorage.setItem(
          `lf_strategy_saved_embedding_model_${strategyId}`,
          currentModel
        )
      } catch {}
      setSavedModel(currentModel)
      setSaveState('success')
      setTimeout(() => setSaveState('idle'), 800)
    }, 1000)
  }

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-20">
      {/* Breadcrumb */}
      <nav className="text-sm md:text-base flex items-center gap-1.5 mb-1">
        <button
          className="text-teal-600 dark:text-teal-400 hover:underline"
          onClick={() => navigate('/chat/rag')}
        >
          RAG
        </button>
        <span className="text-muted-foreground px-1">/</span>
        <span className="text-foreground">{strategyName}</span>
      </nav>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg md:text-xl font-medium">{strategyName}</h2>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" className="gap-2">
            <FontIcon type="code" className="w-4 h-4" />
            View config
          </Button>
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!hasChanges || saveState === 'loading'}
          >
            {saveState === 'loading' && (
              <span className="mr-2 inline-flex">
                <Loader
                  size={14}
                  className="border-blue-400 dark:border-blue-100"
                />
              </span>
            )}
            {saveState === 'success' && (
              <span className="mr-2 inline-flex">
                <FontIcon type="checkmark-filled" className="w-4 h-4" />
              </span>
            )}
            Save
          </Button>
        </div>
      </div>
      <div className="text-sm text-muted-foreground">
        Optimized for standard PDF documents with clean text content. Perfect
        for manuals, reports, and documentation.
      </div>

      {/* Used by */}
      <div className="flex items-center gap-2 flex-wrap">
        <div className="text-xs text-muted-foreground">Used by</div>
        {usedBy.map(u => (
          <Badge key={u} variant="secondary" size="sm" className="rounded-xl">
            {u}
          </Badge>
        ))}
      </div>

      {/* Extraction settings */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Extraction settings</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/chat/rag/${strategyId}/extraction`)}
          >
            Configure
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Document type</div>
            <Input
              value={docTypeFromExtraction ?? 'PDFs ?'}
              readOnly
              className="bg-background"
            />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">OCR Fallback</div>
            <Input value="On" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Table detection</div>
            <Input value="On" readOnly className="bg-background" />
          </div>
        </div>
      </section>

      {/* Parsing strategy */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Parsing Strategy</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/chat/rag/${strategyId}/parsing`)}
          >
            Configure
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Parsing</div>
            <Input value="PDF-aware" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Chunk size</div>
            <Input value="800" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Overlap</div>
            <Input value="100" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Deduplication</div>
            <Input value="On" readOnly className="bg-background" />
          </div>
        </div>
      </section>

      {/* Retrieval method */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Retrieval Method</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/chat/rag/${strategyId}/retrieval`)}
          >
            Configure
          </Button>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Search type</div>
            <Input value="Hybrid" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Results count</div>
            <Input value="8" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Reranking</div>
            <Input value="On" readOnly className="bg-background" />
          </div>
        </div>
      </section>

      {/* Embedding model card (same layout as DatasetView) */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Embedding model</h3>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={() =>
                navigate(`/chat/rag/${strategyId}/change-embedding`)
              }
            >
              Change
            </Button>
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="default" size="sm" className="rounded-xl">
            {currentModel}
          </Badge>
          <Badge variant="secondary" size="sm" className="rounded-xl">
            Active
          </Badge>
        </div>
      </section>
    </div>
  )
}

export default StrategyView
