import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import FontIcon from '../../common/FontIcon'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../ui/dropdown-menu'

function ParsingStrategy() {
  const navigate = useNavigate()
  const { strategyId } = useParams()

  const strategyName = useMemo(() => {
    if (!strategyId) return 'Strategy'
    return strategyId
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
  }, [strategyId])

  const storageKey = useMemo(
    () => (strategyId ? `lf_strategy_parsing_${strategyId}` : ''),
    [strategyId]
  )

  // UI helpers
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
  const [parsingMethod, setParsingMethod] = useState('PDF-aware')
  const [chunkSize, setChunkSize] = useState<string>('800')
  const [overlap, setOverlap] = useState<string>('100')
  const [deduplication, setDeduplication] = useState<'Enabled' | 'Disabled'>(
    'Enabled'
  )

  // Accordions state
  const [chunkingOpen, setChunkingOpen] = useState(false)
  const [metadataOpen, setMetadataOpen] = useState(false)
  const [contentOpen, setContentOpen] = useState(false)
  const [dedupOpen, setDedupOpen] = useState(false)

  // Chunking Strategy
  const [splittingMethod, setSplittingMethod] = useState('Semantic')
  const [minChunkSize, setMinChunkSize] = useState<string>('100')
  const [maxChunkSize, setMaxChunkSize] = useState<string>('1500')
  const [overlapStrategy, setOverlapStrategy] = useState('Token-based')
  const [boundaryRespect, setBoundaryRespect] = useState('Sentence boundaries')
  const [sentenceSplitting, setSentenceSplitting] = useState<
    'Enabled' | 'Disabled'
  >('Enabled')
  const [paragraphPreservation, setParagraphPreservation] =
    useState('Maintain breaks')
  const [sectionAwareness, setSectionAwareness] = useState('Headers as breaks')

  // Metadata Handling
  const [preservePageNumbers, setPreservePageNumbers] = useState<
    'Enabled' | 'Disabled'
  >('Enabled')
  const [preserveHeaders, setPreserveHeaders] = useState('H1-H6 tags')
  const [extractDocumentTitle, setExtractDocumentTitle] =
    useState('From first page')
  const [sectionNumbering, setSectionNumbering] = useState('Auto-detect')
  const [tableContext, setTableContext] = useState('Include table ID')
  const [figureReferences, setFigureReferences] = useState('Link to images')
  const [customMetadataFields, setCustomMetadataFields] = useState(
    'department, version'
  )

  // Content Processing
  const [languageDetection, setLanguageDetection] = useState('Auto-detect')
  const [textCleaning, setTextCleaning] = useState('Remove artifacts')
  const [normalizeWhitespace, setNormalizeWhitespace] = useState<
    'Enabled' | 'Disabled'
  >('Enabled')
  const [removeHeadersFooters, setRemoveHeadersFooters] =
    useState('Auto-detect')
  const [handleSpecialCharacters, setHandleSpecialCharacters] =
    useState('Preserve Unicode')
  const [acronymExpansion, setAcronymExpansion] = useState<
    'Enabled' | 'Disabled'
  >('Disabled')

  // Deduplication
  const [similarityThreshold, setSimilarityThreshold] = useState<string>('85')
  const [comparisonMethod, setComparisonMethod] = useState('Fuzzy matching')
  const [mergeStrategy, setMergeStrategy] = useState('Keep longest')

  // Validation state
  const [errors, setErrors] = useState<Record<string, string>>({})

  const validate = () => {
    const next: Record<string, string> = {}
    const nChunk = Number(chunkSize)
    const nOverlap = Number(overlap)
    const nMin = Number(minChunkSize)
    const nMax = Number(maxChunkSize)
    const nSim = Number(similarityThreshold)
    if (!Number.isFinite(nChunk) || nChunk <= 0)
      next.chunkSize = 'Enter a positive number'
    if (!Number.isFinite(nOverlap) || nOverlap < 0)
      next.overlap = 'Enter a non-negative number'
    if (!Number.isFinite(nMin) || nMin < 0)
      next.minChunkSize = 'Enter a non-negative number'
    if (!Number.isFinite(nMax) || nMax <= 0 || nMax < nMin)
      next.maxChunkSize = 'Must be >= Min chunk size'
    if (!Number.isFinite(nSim) || nSim < 0 || nSim > 100)
      next.similarityThreshold = 'Enter 0–100'
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
      setParsingMethod(cfg.parsingMethod ?? parsingMethod)
      setChunkSize(String(cfg.chunkSize ?? chunkSize))
      setOverlap(String(cfg.overlap ?? overlap))
      setDeduplication(cfg.deduplication ?? deduplication)

      setSplittingMethod(cfg.splittingMethod ?? splittingMethod)
      setMinChunkSize(String(cfg.minChunkSize ?? minChunkSize))
      setMaxChunkSize(String(cfg.maxChunkSize ?? maxChunkSize))
      setOverlapStrategy(cfg.overlapStrategy ?? overlapStrategy)
      setBoundaryRespect(cfg.boundaryRespect ?? boundaryRespect)
      setSentenceSplitting(cfg.sentenceSplitting ?? sentenceSplitting)
      setParagraphPreservation(
        cfg.paragraphPreservation ?? paragraphPreservation
      )
      setSectionAwareness(cfg.sectionAwareness ?? sectionAwareness)

      setPreservePageNumbers(cfg.preservePageNumbers ?? preservePageNumbers)
      setPreserveHeaders(cfg.preserveHeaders ?? preserveHeaders)
      setExtractDocumentTitle(cfg.extractDocumentTitle ?? extractDocumentTitle)
      setSectionNumbering(cfg.sectionNumbering ?? sectionNumbering)
      setTableContext(cfg.tableContext ?? tableContext)
      setFigureReferences(cfg.figureReferences ?? figureReferences)
      setCustomMetadataFields(cfg.customMetadataFields ?? customMetadataFields)

      setLanguageDetection(cfg.languageDetection ?? languageDetection)
      setTextCleaning(cfg.textCleaning ?? textCleaning)
      setNormalizeWhitespace(cfg.normalizeWhitespace ?? normalizeWhitespace)
      setRemoveHeadersFooters(cfg.removeHeadersFooters ?? removeHeadersFooters)
      setHandleSpecialCharacters(
        cfg.handleSpecialCharacters ?? handleSpecialCharacters
      )
      setAcronymExpansion(cfg.acronymExpansion ?? acronymExpansion)

      setSimilarityThreshold(
        String(cfg.similarityThreshold ?? similarityThreshold)
      )
      setComparisonMethod(cfg.comparisonMethod ?? comparisonMethod)
      setMergeStrategy(cfg.mergeStrategy ?? mergeStrategy)
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
          parsingMethod,
          chunkSize: Number(chunkSize),
          overlap: Number(overlap),
          deduplication,
          splittingMethod,
          minChunkSize: Number(minChunkSize),
          maxChunkSize: Number(maxChunkSize),
          overlapStrategy,
          boundaryRespect,
          sentenceSplitting,
          paragraphPreservation,
          sectionAwareness,
          preservePageNumbers,
          preserveHeaders,
          extractDocumentTitle,
          sectionNumbering,
          tableContext,
          figureReferences,
          customMetadataFields,
          languageDetection,
          textCleaning,
          normalizeWhitespace,
          removeHeadersFooters,
          handleSpecialCharacters,
          acronymExpansion,
          similarityThreshold: Number(similarityThreshold),
          comparisonMethod,
          mergeStrategy,
        }
        if (storageKey)
          localStorage.setItem(storageKey, JSON.stringify(payload))
        try {
          if (typeof window !== 'undefined') {
            window.dispatchEvent(
              new CustomEvent('lf:strategyParsingUpdated', {
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
      {/* Breadcrumb */}
      <nav className="text-sm md:text-base flex items-center gap-1.5 mb-1">
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
        <span className="text-foreground">Parsing strategy</span>
      </nav>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg md:text-xl font-medium">Parsing strategy</h2>
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
          <Field label="Parsing Method">
            <SelectDropdown
              value={parsingMethod}
              onChange={setParsingMethod}
              options={[
                'PDF-aware',
                'Semantic',
                'Fixed-size',
                'Sentence-based',
                'Custom',
              ]}
            />
          </Field>
          <Field label="Chunk Size (tokens)">
            <Input
              type="number"
              value={chunkSize}
              onChange={e => setChunkSize(e.target.value)}
              className={`${inputClass} ${errors.chunkSize ? 'border-destructive' : ''}`}
            />
            {errors.chunkSize && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.chunkSize}
              </div>
            )}
          </Field>
          <Field label="Overlap (tokens)">
            <Input
              type="number"
              value={overlap}
              onChange={e => setOverlap(e.target.value)}
              className={`${inputClass} ${errors.overlap ? 'border-destructive' : ''}`}
            />
            {errors.overlap && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.overlap}
              </div>
            )}
          </Field>
          <Field label="Deduplication">
            <SelectDropdown
              value={deduplication}
              onChange={v => setDeduplication(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
        </div>
      </section>

      {/* Chunking Strategy */}
      <Collapsible
        title="Chunking Strategy"
        open={chunkingOpen}
        onToggle={() => setChunkingOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Splitting Method">
            <SelectDropdown
              value={splittingMethod}
              onChange={setSplittingMethod}
              options={[
                'Semantic',
                'Fixed-size',
                'Sentence-based',
                'Paragraph-based',
                'Custom regex',
              ]}
            />
          </Field>
          <Field label="Min Chunk Size (tokens)">
            <Input
              type="number"
              value={minChunkSize}
              onChange={e => setMinChunkSize(e.target.value)}
              className={`${inputClass} ${errors.minChunkSize ? 'border-destructive' : ''}`}
            />
            {errors.minChunkSize && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.minChunkSize}
              </div>
            )}
          </Field>
          <Field label="Max Chunk Size (tokens)">
            <Input
              type="number"
              value={maxChunkSize}
              onChange={e => setMaxChunkSize(e.target.value)}
              className={`${inputClass} ${errors.maxChunkSize ? 'border-destructive' : ''}`}
            />
            {errors.maxChunkSize && (
              <div className="text-xs text-destructive mt-0.5">
                {errors.maxChunkSize}
              </div>
            )}
          </Field>
          <Field label="Overlap Strategy">
            <SelectDropdown
              value={overlapStrategy}
              onChange={setOverlapStrategy}
              options={[
                'Token-based',
                'Character-based',
                'Sentence-based',
                'Percentage-based',
              ]}
            />
          </Field>
          <Field label="Boundary Respect">
            <SelectDropdown
              value={boundaryRespect}
              onChange={setBoundaryRespect}
              options={[
                'Sentence boundaries',
                'Paragraph boundaries',
                'Section boundaries',
                'None',
              ]}
            />
          </Field>
          <Field label="Sentence Splitting">
            <SelectDropdown
              value={sentenceSplitting}
              onChange={v => setSentenceSplitting(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Paragraph Preservation">
            <SelectDropdown
              value={paragraphPreservation}
              onChange={setParagraphPreservation}
              options={[
                'Maintain breaks',
                'Merge paragraphs',
                'Smart detection',
              ]}
            />
          </Field>
          <Field label="Section Awareness">
            <SelectDropdown
              value={sectionAwareness}
              onChange={setSectionAwareness}
              options={[
                'Headers as breaks',
                'No section detection',
                'Custom markers',
              ]}
            />
          </Field>
        </div>
      </Collapsible>

      {/* Metadata Handling */}
      <Collapsible
        title="Metadata Handling"
        open={metadataOpen}
        onToggle={() => setMetadataOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Preserve Page Numbers">
            <SelectDropdown
              value={preservePageNumbers}
              onChange={v =>
                setPreservePageNumbers(v as 'Enabled' | 'Disabled')
              }
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Preserve Headers">
            <SelectDropdown
              value={preserveHeaders}
              onChange={setPreserveHeaders}
              options={['H1-H6 tags', 'All headers', 'None', 'Custom levels']}
            />
          </Field>
          <Field label="Extract Document Title">
            <SelectDropdown
              value={extractDocumentTitle}
              onChange={setExtractDocumentTitle}
              options={[
                'From first page',
                'From metadata',
                'From filename',
                'Auto-detect',
              ]}
            />
          </Field>
          <Field label="Section Numbering">
            <SelectDropdown
              value={sectionNumbering}
              onChange={setSectionNumbering}
              options={['Auto-detect', 'Manual pattern', 'Disabled']}
            />
          </Field>
          <Field label="Table Context">
            <SelectDropdown
              value={tableContext}
              onChange={setTableContext}
              options={[
                'Include table ID',
                'Full table context',
                'Table caption only',
                'None',
              ]}
            />
          </Field>
          <Field label="Figure References">
            <SelectDropdown
              value={figureReferences}
              onChange={setFigureReferences}
              options={[
                'Link to images',
                'Caption only',
                'Reference text only',
                'None',
              ]}
            />
          </Field>
          <Field label="Custom Metadata Fields">
            <Input
              value={customMetadataFields}
              onChange={e => setCustomMetadataFields(e.target.value)}
              placeholder="department, version"
              className={inputClass}
            />
          </Field>
        </div>
      </Collapsible>

      {/* Content Processing */}
      <Collapsible
        title="Content Processing"
        open={contentOpen}
        onToggle={() => setContentOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Language Detection">
            <SelectDropdown
              value={languageDetection}
              onChange={setLanguageDetection}
              options={[
                'Auto-detect',
                'English only',
                'Multi-language',
                'Custom list',
              ]}
            />
          </Field>
          <Field label="Text Cleaning">
            <SelectDropdown
              value={textCleaning}
              onChange={setTextCleaning}
              options={[
                'Remove artifacts',
                'Basic cleanup',
                'Aggressive cleanup',
                'None',
              ]}
            />
          </Field>
          <Field label="Normalize Whitespace">
            <SelectDropdown
              value={normalizeWhitespace}
              onChange={v =>
                setNormalizeWhitespace(v as 'Enabled' | 'Disabled')
              }
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Remove Headers/Footers">
            <SelectDropdown
              value={removeHeadersFooters}
              onChange={setRemoveHeadersFooters}
              options={['Auto-detect', 'Manual patterns', 'Disabled']}
            />
          </Field>
          <Field label="Handle Special Characters">
            <SelectDropdown
              value={handleSpecialCharacters}
              onChange={setHandleSpecialCharacters}
              options={[
                'Preserve Unicode',
                'ASCII only',
                'Remove special chars',
                'Custom handling',
              ]}
            />
          </Field>
          <Field label="Acronym Expansion">
            <SelectDropdown
              value={acronymExpansion}
              onChange={v => setAcronymExpansion(v as 'Enabled' | 'Disabled')}
              options={['Enabled', 'Disabled']}
            />
          </Field>
        </div>
      </Collapsible>

      {/* Deduplication */}
      <Collapsible
        title="Deduplication"
        open={dedupOpen}
        onToggle={() => setDedupOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Similarity Threshold (%)">
            <div className="grid grid-cols-3 gap-2 items-center">
              <input
                type="range"
                min={0}
                max={100}
                value={Number(similarityThreshold)}
                onChange={e => setSimilarityThreshold(e.target.value)}
                className="col-span-2 w-full"
              />
              <Input
                type="number"
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
          <Field label="Comparison Method">
            <SelectDropdown
              value={comparisonMethod}
              onChange={setComparisonMethod}
              options={[
                'Fuzzy matching',
                'Exact matching',
                'Semantic similarity',
                'Hash-based',
              ]}
            />
          </Field>
          <Field label="Merge Strategy">
            <SelectDropdown
              value={mergeStrategy}
              onChange={setMergeStrategy}
              options={[
                'Keep longest',
                'Keep first',
                'Keep most recent',
                'Manual review',
              ]}
            />
          </Field>
        </div>
      </Collapsible>
    </div>
  )
}

export default ParsingStrategy
