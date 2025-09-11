import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { Button } from '../ui/button'
import PageActions from '../common/PageActions'
import { Mode } from '../ModeToggle'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from '../ui/dropdown-menu'
import { decryptJson, encryptJson } from '@/utils/crypto'

function ExtractionSettings() {
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
    () => `lf_strategy_extraction_${strategyId}`,
    [strategyId]
  )

  const [saveState, setSaveState] = useState<'idle' | 'loading' | 'success'>(
    'idle'
  )

  // Shared focus style for inputs (match edit modal behavior)
  const inputClass =
    'bg-background focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-primary focus-visible:ring-offset-0'
  // Remove event propagation blockers so inputs behave normally
  const inputEvents = {}

  // Collapsible states (collapsed by default)
  const [pdfOpen, setPdfOpen] = useState(false)
  const [ocrOpen, setOcrOpen] = useState(false)
  const [tableOpen, setTableOpen] = useState(false)
  const [imageOpen, setImageOpen] = useState(false)
  const [perfOpen, setPerfOpen] = useState(false)

  // Basic settings
  const [documentType, setDocumentType] = useState('PDFs ?')
  const [ocrFallback, setOcrFallback] = useState('Auto (when needed)')
  const [ocrThreshold, setOcrThreshold] = useState('75%')
  const [tableDetection, setTableDetection] = useState('Enabled')
  const [imageExtraction, setImageExtraction] = useState('Skip > 2MB')

  // PDF engine options
  const [primaryEngine, setPrimaryEngine] = useState('PyMuPDF')
  const [fallbackEngine, setFallbackEngine] = useState('pdfplumber')
  const [textExtractionMode, setTextExtractionMode] = useState('full text')
  const [layoutPreservation, setLayoutPreservation] = useState(
    'maintain formatting'
  )
  const [passwordHandling, setPasswordHandling] = useState('skip protected')
  const [metadataExtraction, setMetadataExtraction] = useState('include all')
  const [fontInfo, setFontInfo] = useState('preserve')
  const [colorInfo, setColorInfo] = useState('discard')

  // OCR configuration
  const [ocrEngine, setOcrEngine] = useState('Tesseract 5.x')
  const [languages, setLanguages] = useState('eng, spa, fra')
  const [dpi, setDpi] = useState('300')
  const [preprocessing, setPreprocessing] = useState('auto-deskew')
  const [charWhitelist, setCharWhitelist] = useState('default')
  const [pageSegMode, setPageSegMode] = useState('auto')
  const [noiseReduction, setNoiseReduction] = useState('enabled')
  const [ocrEngineMode, setOcrEngineMode] = useState('accurate')

  // Table & Structure Detection
  const [tableLib, setTableLib] = useState('Camelot')
  const [minTableSize, setMinTableSize] = useState('3x3')
  const [colSeparator, setColSeparator] = useState('auto')
  const [headerDetection, setHeaderDetection] = useState('first row')
  const [mergeSplitTables, setMergeSplitTables] = useState('Enabled')
  const [cellMerging, setCellMerging] = useState('handle spans')
  const [borderDetection, setBorderDetection] = useState('lattice and stream')
  const [exportFormat, setExportFormat] = useState('preserve format')

  // Image & Media Handling
  const [imageExtractionMode, setImageExtractionMode] = useState('Extract all')
  const [maxImageSize, setMaxImageSize] = useState('5MB')
  const [figureCaptions, setFigureCaptions] = useState('Extract nearby text')
  const [imageOCR, setImageOCR] = useState('only when needed')

  // Performance & Error Handling
  const [parallelProcessing, setParallelProcessing] = useState('4 threads')
  const [memoryLimit, setMemoryLimit] = useState('2GB')
  const [timeoutPerPage, setTimeoutPerPage] = useState('60 seconds')
  const [cacheExtractedText, setCacheExtractedText] = useState('enabled')
  const [batchSize, setBatchSize] = useState('10 documents')
  const [retryFailedPages, setRetryFailedPages] = useState('3 attempts')
  const [tempCleanup, setTempCleanup] = useState('auto-cleanup')

  // Load persisted values (decrypt if needed)
  useEffect(() => {
    ;(async () => {
      try {
        if (!storageKey) return
        const raw = localStorage.getItem(storageKey)
        if (!raw) return
        let cfg: any = null
        try {
          // Try decrypt first
          const secret = strategyId || 'default-project-secret'
          const dec = await decryptJson(raw, secret)
          if (dec && typeof dec === 'object') cfg = dec
        } catch {}
        if (!cfg) {
          try {
            cfg = JSON.parse(raw)
          } catch {
            cfg = null
          }
        }
        if (!cfg) return
        setDocumentType(cfg.documentType ?? documentType)
        setOcrFallback(cfg.ocrFallback ?? ocrFallback)
        setOcrThreshold(cfg.ocrThreshold ?? ocrThreshold)
        setTableDetection(cfg.tableDetection ?? tableDetection)
        setImageExtraction(cfg.imageExtraction ?? imageExtraction)

        setPrimaryEngine(cfg.primaryEngine ?? primaryEngine)
        setFallbackEngine(cfg.fallbackEngine ?? fallbackEngine)
        setTextExtractionMode(cfg.textExtractionMode ?? textExtractionMode)
        setLayoutPreservation(cfg.layoutPreservation ?? layoutPreservation)
        setPasswordHandling(cfg.passwordHandling ?? passwordHandling)
        setMetadataExtraction(cfg.metadataExtraction ?? metadataExtraction)
        setFontInfo(cfg.fontInfo ?? fontInfo)
        setColorInfo(cfg.colorInfo ?? colorInfo)

        setOcrEngine(cfg.ocrEngine ?? ocrEngine)
        setLanguages(cfg.languages ?? languages)
        setDpi(cfg.dpi ?? dpi)
        setPreprocessing(cfg.preprocessing ?? preprocessing)
        setCharWhitelist(cfg.charWhitelist ?? charWhitelist)
        setPageSegMode(cfg.pageSegMode ?? pageSegMode)
        setNoiseReduction(cfg.noiseReduction ?? noiseReduction)
        setOcrEngineMode(cfg.ocrEngineMode ?? ocrEngineMode)

        setTableLib(cfg.tableLib ?? tableLib)
        setMinTableSize(cfg.minTableSize ?? minTableSize)
        setColSeparator(cfg.colSeparator ?? colSeparator)
        setHeaderDetection(cfg.headerDetection ?? headerDetection)
        setMergeSplitTables(cfg.mergeSplitTables ?? mergeSplitTables)
        setCellMerging(cfg.cellMerging ?? cellMerging)
        setBorderDetection(cfg.borderDetection ?? borderDetection)
        setExportFormat(cfg.exportFormat ?? exportFormat)

        setImageExtractionMode(cfg.imageExtractionMode ?? imageExtractionMode)
        setMaxImageSize(cfg.maxImageSize ?? maxImageSize)
        setFigureCaptions(cfg.figureCaptions ?? figureCaptions)
        setImageOCR(cfg.imageOCR ?? imageOCR)

        setParallelProcessing(cfg.parallelProcessing ?? parallelProcessing)
        setMemoryLimit(cfg.memoryLimit ?? memoryLimit)
        setTimeoutPerPage(cfg.timeoutPerPage ?? timeoutPerPage)
        setCacheExtractedText(cfg.cacheExtractedText ?? cacheExtractedText)
        setBatchSize(cfg.batchSize ?? batchSize)
        setRetryFailedPages(cfg.retryFailedPages ?? retryFailedPages)
        setTempCleanup(cfg.tempCleanup ?? tempCleanup)
      } catch {}
    })()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKey])

  const handleSave = () => {
    setSaveState('loading')
    setTimeout(() => {
      try {
        const payload = {
          documentType,
          ocrFallback,
          ocrThreshold,
          tableDetection,
          imageExtraction,
          primaryEngine,
          fallbackEngine,
          textExtractionMode,
          layoutPreservation,
          passwordHandling,
          metadataExtraction,
          fontInfo,
          colorInfo,
          ocrEngine,
          languages,
          dpi,
          preprocessing,
          charWhitelist,
          pageSegMode,
          noiseReduction,
          ocrEngineMode,
          tableLib,
          minTableSize,
          colSeparator,
          headerDetection,
          mergeSplitTables,
          cellMerging,
          borderDetection,
          exportFormat,
          imageExtractionMode,
          maxImageSize,
          figureCaptions,
          imageOCR,
          parallelProcessing,
          memoryLimit,
          timeoutPerPage,
          cacheExtractedText,
          batchSize,
          retryFailedPages,
          tempCleanup,
        }
        if (storageKey) {
          const secret = strategyId || 'default-project-secret'
          // Encrypt before storing to address CodeQL clear-text storage finding
          encryptJson(payload, secret).then(ciphertext => {
            try {
              localStorage.setItem(storageKey, ciphertext)
            } catch {}
          })
        }
        try {
          if (typeof window !== 'undefined') {
            window.dispatchEvent(
              new CustomEvent('lf:strategyExtractionUpdated', {
                detail: { strategyId, payload },
              })
            )
          }
        } catch {}
      } catch {}
      setSaveState('success')
      setTimeout(() => setSaveState('idle'), 800)
    }, 1000)
  }

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
        aria-label={open ? 'Collapse section' : 'Expand section'}
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

  const ToggleDropdown = ({
    value,
    onChange,
    options,
  }: {
    value: string
    onChange: (v: string) => void
    options: string[]
  }) => (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          className={`h-9 w-full rounded-md border border-input bg-background px-3 text-left ${inputClass}`}
        >
          {value}
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent className="w-40" align="start">
        {options.map(opt => (
          <DropdownMenuItem key={opt} onClick={() => onChange(opt)}>
            {opt}
          </DropdownMenuItem>
        ))}
      </DropdownMenuContent>
    </DropdownMenu>
  )

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
          <span className="text-foreground">Extraction settings</span>
        </nav>
        <PageActions mode={mode} onModeChange={setMode} />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <h2 className="text-lg md:text-xl font-medium">Extraction settings</h2>
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

      {/* Basic settings */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="text-sm font-medium mb-3">Basic settings</div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1 py-1.5">
            <Label className="text-xs text-muted-foreground">
              Document type
            </Label>
            <Input
              value={documentType}
              onChange={e => setDocumentType(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </div>
          <div className="flex flex-col gap-1 py-1.5">
            <Label className="text-xs text-muted-foreground">
              OCR Fallback
            </Label>
            <Input
              value={ocrFallback}
              onChange={e => setOcrFallback(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </div>
          <div className="flex flex-col gap-1 py-1.5">
            <Label className="text-xs text-muted-foreground">
              OCR Confidence threshold
            </Label>
            <Input
              value={ocrThreshold}
              onChange={e => setOcrThreshold(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </div>
          <div className="flex flex-col gap-1 py-1.5">
            <Label className="text-xs text-muted-foreground">
              Table detection
            </Label>
            <ToggleDropdown
              value={tableDetection}
              onChange={setTableDetection}
              options={['Enabled', 'Disabled']}
            />
          </div>
          <div className="flex flex-col gap-1 py-1.5">
            <Label className="text-xs text-muted-foreground">
              Image extraction
            </Label>
            <Input
              value={imageExtraction}
              onChange={e => setImageExtraction(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </div>
        </div>
      </section>

      {/* Advanced sections as collapsible cards */}
      <Collapsible
        title="PDF engine options"
        open={pdfOpen}
        onToggle={() => setPdfOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Primary engine">
            <Input
              value={primaryEngine}
              onChange={e => setPrimaryEngine(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Fallback engine">
            <Input
              value={fallbackEngine}
              onChange={e => setFallbackEngine(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Text extraction mode">
            <Input
              value={textExtractionMode}
              onChange={e => setTextExtractionMode(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Layout preservation">
            <Input
              value={layoutPreservation}
              onChange={e => setLayoutPreservation(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Password handling">
            <Input
              value={passwordHandling}
              onChange={e => setPasswordHandling(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Metadata extraction">
            <Input
              value={metadataExtraction}
              onChange={e => setMetadataExtraction(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Font information">
            <Input
              value={fontInfo}
              onChange={e => setFontInfo(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Color information">
            <Input
              value={colorInfo}
              onChange={e => setColorInfo(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
        </div>
      </Collapsible>

      <Collapsible
        title="OCR configuration"
        open={ocrOpen}
        onToggle={() => setOcrOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="OCR engine">
            <Input
              value={ocrEngine}
              onChange={e => setOcrEngine(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Languages">
            <Input
              value={languages}
              onChange={e => setLanguages(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="DPI for OCR">
            <Input
              value={dpi}
              onChange={e => setDpi(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Preprocessing">
            <Input
              value={preprocessing}
              onChange={e => setPreprocessing(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Character whitelist">
            <Input
              value={charWhitelist}
              onChange={e => setCharWhitelist(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Page segmentation mode">
            <Input
              value={pageSegMode}
              onChange={e => setPageSegMode(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Noise reduction">
            <ToggleDropdown
              value={noiseReduction}
              onChange={setNoiseReduction}
              options={['enabled', 'disabled']}
            />
          </Field>
          <Field label="OCR engine mode">
            <Input
              value={ocrEngineMode}
              onChange={e => setOcrEngineMode(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
        </div>
      </Collapsible>

      <Collapsible
        title="Table & Structure Detection"
        open={tableOpen}
        onToggle={() => setTableOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Table detection library">
            <Input
              value={tableLib}
              onChange={e => setTableLib(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Min table size">
            <Input
              value={minTableSize}
              onChange={e => setMinTableSize(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Column separator detection">
            <Input
              value={colSeparator}
              onChange={e => setColSeparator(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Header detection">
            <Input
              value={headerDetection}
              onChange={e => setHeaderDetection(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Merge split tables">
            <ToggleDropdown
              value={mergeSplitTables}
              onChange={setMergeSplitTables}
              options={['Enabled', 'Disabled']}
            />
          </Field>
          <Field label="Cell merging">
            <Input
              value={cellMerging}
              onChange={e => setCellMerging(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Border detection">
            <Input
              value={borderDetection}
              onChange={e => setBorderDetection(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Export format">
            <Input
              value={exportFormat}
              onChange={e => setExportFormat(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
        </div>
      </Collapsible>

      <Collapsible
        title="Image & Media Handling"
        open={imageOpen}
        onToggle={() => setImageOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Image extraction">
            <Input
              value={imageExtractionMode}
              onChange={e => setImageExtractionMode(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Max image size">
            <Input
              value={maxImageSize}
              onChange={e => setMaxImageSize(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Figure captions">
            <Input
              value={figureCaptions}
              onChange={e => setFigureCaptions(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Image OCR">
            <Input
              value={imageOCR}
              onChange={e => setImageOCR(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
        </div>
      </Collapsible>

      <Collapsible
        title="Performance & Error Handling"
        open={perfOpen}
        onToggle={() => setPerfOpen(o => !o)}
      >
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <Field label="Parallel processing">
            <Input
              value={parallelProcessing}
              onChange={e => setParallelProcessing(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Memory limit per process">
            <Input
              value={memoryLimit}
              onChange={e => setMemoryLimit(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Timeout per page">
            <Input
              value={timeoutPerPage}
              onChange={e => setTimeoutPerPage(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Cache extracted text">
            <ToggleDropdown
              value={cacheExtractedText}
              onChange={setCacheExtractedText}
              options={['enabled', 'disabled']}
            />
          </Field>
          <Field label="Batch size">
            <Input
              value={batchSize}
              onChange={e => setBatchSize(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Retry failed pages">
            <Input
              value={retryFailedPages}
              onChange={e => setRetryFailedPages(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
          <Field label="Temp file cleanup">
            <Input
              value={tempCleanup}
              onChange={e => setTempCleanup(e.target.value)}
              className={inputClass}
              {...inputEvents}
            />
          </Field>
        </div>
      </Collapsible>
    </div>
  )
}

export default ExtractionSettings
