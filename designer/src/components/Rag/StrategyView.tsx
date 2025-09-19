import { useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { defaultStrategies } from './strategies'
import ParserSettingsForm from './ParserSettingsForm'
import {
  PARSER_SCHEMAS,
  ORDERED_PARSER_TYPES,
  getDefaultConfigForParser,
} from './parserSchemas'
import { useToast } from '../ui/toast'
import PageActions from '../common/PageActions'
import { Mode } from '../ModeToggle'
import Tabs from '../Tabs'
import ExtractorSettingsForm from './ExtractorSettingsForm'
import {
  EXTRACTOR_SCHEMAS,
  ORDERED_EXTRACTOR_TYPES,
  getDefaultConfigForExtractor,
} from './extractorSchemas'
import { ChevronDown, Plus, Settings, Trash2 } from 'lucide-react'
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

function StrategyView() {
  const navigate = useNavigate()
  const { strategyId } = useParams()
  const { toast } = useToast()
  const [mode, setMode] = useState<Mode>('designer')

  const [strategyMetaTick, setStrategyMetaTick] = useState(0)

  const strategyName = useMemo(() => {
    if (!strategyId) return 'Strategy'
    try {
      const override = localStorage.getItem(
        `lf_strategy_name_override_${strategyId}`
      )
      if (override && override.trim().length > 0) return override
    } catch {}
    const found = defaultStrategies.find(s => s.id === strategyId)
    if (found) return found.name
    // Fallback to title-casing the id
    return strategyId
      .replace(/[-_]/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
  }, [strategyId, strategyMetaTick])

  const strategyDescription = useMemo(() => {
    if (!strategyId) return ''
    try {
      const override = localStorage.getItem(
        `lf_strategy_description_${strategyId}`
      )
      if (override !== null) return override
    } catch {}
    const found = defaultStrategies.find(s => s.id === strategyId)
    return found?.description || ''
  }, [strategyId, strategyMetaTick])

  const usedBy = ['aircraft-maintenance-guides', 'another dataset']

  // Removed embedding model save flow; processing edits persist immediately

  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')

  // (Removed) embedding save flow and listeners

  // Tabbed Parsers/Extractors data -------------------------------------------
  type ParserRow = {
    id: string
    name: string
    priority: number
    include: string
    exclude: string
    summary: string
    config?: Record<string, unknown>
  }
  type ExtractorRow = {
    id: string
    name: string
    priority: number
    applyTo: string
    summary: string
    config?: Record<string, unknown>
  }

  // Map legacy high-number priorities to new low-number scale
  const migratePriority = (value: unknown): number => {
    const n = Number(value)
    if (!Number.isFinite(n)) return 1
    if (n >= 100) return 1
    if (n >= 90) return 2
    if (n >= 80) return 3
    if (n >= 50) return 4
    if (n < 1) return 1
    return n
  }

  const defaultParsers: ParserRow[] = [
    {
      id: 'pdf-llamaindex',
      name: 'PDFParser_LlamaIndex',
      priority: 1,
      include: '*.pdf, *.PDF',
      exclude: '*_draft.pdf, *.tmp.pdf',
      summary:
        'Semantic chunking, 1000 chars, 200 overlap, extract metadata & tables',
      config: getDefaultConfigForParser('PDFParser_LlamaIndex'),
    },
    {
      id: 'pdf-pypdf2',
      name: 'PDFParser_PyPDF2',
      priority: 4,
      include: '*.pdf, *.PDF',
      exclude: '*_draft.pdf, *.tmp.pdf',
      summary: 'Paragraph chunking, 1000 chars, 150 overlap, extract metadata',
      config: getDefaultConfigForParser('PDFParser_PyPDF2'),
    },
    {
      id: 'docx-llamaindex',
      name: 'DocxParser_LlamaIndex',
      priority: 1,
      include: '*.docx, *.DOCX, *.doc, *.DOC',
      exclude: '~$*, *.tmp',
      summary: '1000 chars, 150 overlap, extract tables & metadata',
      config: getDefaultConfigForParser('DocxParser_LlamaIndex'),
    },
    {
      id: 'md-python',
      name: 'MarkdownParser_Python',
      priority: 1,
      include: '*.md, *.markdown, *.mdown, *.mkd, README*',
      exclude: '*.tmp.md, _draft*.md',
      summary: 'Section-based, extract code & links',
      config: getDefaultConfigForParser('MarkdownParser_Python'),
    },
    {
      id: 'csv-pandas',
      name: 'CSVParser_Pandas',
      priority: 1,
      include: '*.csv, *.CSV, *.tsv, *.TSV, *.dat',
      exclude: '*_backup.csv, *.tmp.csv',
      summary: 'Row-based, 500 chars, UTF-8',
      config: getDefaultConfigForParser('CSVParser_Pandas'),
    },
    {
      id: 'excel-pandas',
      name: 'ExcelParser_Pandas',
      priority: 1,
      include: '*.xlsx, *.XLSX, *.xls, *.XLS',
      exclude: '~$*, *.tmp.xlsx',
      summary: 'Process all sheets, 500 chars, extract metadata',
      config: getDefaultConfigForParser('ExcelParser_Pandas'),
    },
    {
      id: 'text-python',
      name: 'TextParser_Python',
      priority: 4,
      include: '*.txt, *.json, *.xml, *.yaml, *.py, *.js, LICENSE*, etc.',
      exclude: '*.pyc, *.pyo, *.class',
      summary: 'Sentence-based, 1200 chars, 200 overlap',
      config: getDefaultConfigForParser('TextParser_Python'),
    },
  ]

  const defaultExtractors: ExtractorRow[] = [
    {
      id: 'content-stats',
      name: 'ContentStatisticsExtractor',
      priority: 1,
      applyTo: 'All files (*)',
      summary: 'Include readability, vocabulary & structure analysis',
    },
    {
      id: 'entity',
      name: 'EntityExtractor',
      priority: 2,
      applyTo: 'All files (*)',
      summary:
        'Extract: PERSON, ORG, GPE, DATE, PRODUCT, MONEY, PERCENT | Min length: 2',
    },
    {
      id: 'keyword',
      name: 'KeywordExtractor',
      priority: 3,
      applyTo: 'All files (*)',
      summary: 'YAKE algorithm, 10 max keywords, 3 min keyword length',
    },
    {
      id: 'table',
      name: 'TableExtractor',
      priority: 1,
      applyTo: '*.pdf, *.PDF only',
      summary: 'Dict format output, extract headers, merge cells',
    },
    {
      id: 'datetime',
      name: 'DateTimeExtractor',
      priority: 1,
      applyTo: '*.csv, *.xlsx, *.xls, *.tsv',
      summary: 'Formats: ISO8601, US, EU | Extract relative dates & times',
    },
    {
      id: 'pattern',
      name: 'PatternExtractor',
      priority: 1,
      applyTo: '*.py, *.js, *.java, *.cpp, *.c, *.h',
      summary:
        'Email, URL, IP, version + custom function/class definition patterns',
    },
    {
      id: 'heading',
      name: 'HeadingExtractor',
      priority: 1,
      applyTo: '*.md, *.markdown, README*',
      summary: 'Max level 6, include hierarchy & outline extraction',
    },
    {
      id: 'link',
      name: 'LinkExtractor',
      priority: 2,
      applyTo: '*.md, *.markdown, *.html, *.htm',
      summary: 'Extract URLs, emails, and domains',
    },
  ]

  const [activeTab, setActiveTab] = useState<'parsers' | 'extractors'>(
    'parsers'
  )
  const [openRows, setOpenRows] = useState<Set<string>>(new Set())

  const [parserRows, setParserRows] = useState<ParserRow[]>(defaultParsers)
  const [extractorRows, setExtractorRows] =
    useState<ExtractorRow[]>(defaultExtractors)

  const storageKeys = useMemo(() => {
    if (!strategyId) return { parsers: '', extractors: '' }
    return {
      parsers: `lf_strategy_parsers_${strategyId}`,
      extractors: `lf_strategy_extractors_${strategyId}`,
    }
  }, [strategyId])

  const loadPersisted = () => {
    try {
      if (!storageKeys.parsers || !storageKeys.extractors) return
      const pRaw = localStorage.getItem(storageKeys.parsers)
      const eRaw = localStorage.getItem(storageKeys.extractors)
      if (pRaw) {
        try {
          const arr = JSON.parse(pRaw)
          if (Array.isArray(arr)) {
            const migrated = arr.map((p: ParserRow) => {
              if (!p || typeof p !== 'object') return p
              const hasSchema =
                typeof p.name === 'string' && PARSER_SCHEMAS[p.name]
              const withConfig =
                hasSchema && !p.config
                  ? { ...p, config: getDefaultConfigForParser(p.name) }
                  : p
              return {
                ...withConfig,
                priority: migratePriority(withConfig.priority),
              }
            })
            setParserRows(migrated)
            try {
              localStorage.setItem(
                storageKeys.parsers,
                JSON.stringify(migrated)
              )
            } catch {}
          }
        } catch {}
      } else {
        setParserRows(defaultParsers)
      }
      if (eRaw) {
        try {
          const arr = JSON.parse(eRaw)
          if (Array.isArray(arr)) {
            const migrated = arr.map((e: ExtractorRow) => {
              if (!e || typeof e !== 'object') return e
              return { ...e, priority: migratePriority(e.priority) }
            })
            setExtractorRows(migrated)
            try {
              localStorage.setItem(
                storageKeys.extractors,
                JSON.stringify(migrated)
              )
            } catch {}
          }
        } catch {}
      } else {
        setExtractorRows(defaultExtractors)
      }
    } catch {}
  }

  useEffect(() => {
    loadPersisted()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [storageKeys.parsers, storageKeys.extractors])

  const saveParsers = (rows: ParserRow[]) => {
    try {
      if (storageKeys.parsers)
        localStorage.setItem(storageKeys.parsers, JSON.stringify(rows))
    } catch {}
  }
  const saveExtractors = (rows: ExtractorRow[]) => {
    try {
      if (storageKeys.extractors)
        localStorage.setItem(storageKeys.extractors, JSON.stringify(rows))
    } catch {}
  }
  const toggleRow = (id: string) => {
    setOpenRows(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const extractExtensions = (patternText: string): string[] => {
    if (!patternText || patternText.trim().length === 0) return []
    const lowered = patternText.toLowerCase()
    // If explicitly says all files
    if (/(all\s*files|\*\))/i.test(patternText)) return ['*']
    const tokens = lowered.split(/[,\s]+/).filter(Boolean)
    const exts = new Set<string>()
    for (const t of tokens) {
      // match .ext (allow patterns like *.pdf or .PDF)
      const m = t.match(/\.([a-z0-9]{1,10})$/i)
      if (m && m[1]) exts.add(m[1])
      // handle README*, LICENSE* as pseudo-types
      if (/readme\*/i.test(t)) exts.add('readme')
      if (/license\*/i.test(t)) exts.add('license')
    }
    return Array.from(exts)
  }

  const summarizeFileTypes = (patternText: string): string => {
    const exts = extractExtensions(patternText)
    if (exts.includes('*')) return 'All (*)'
    if (exts.length === 0) return ''
    const max = 3
    const shown = exts.slice(0, max).join(', ')
    if (exts.length > max) return `${shown} +${exts.length - max}`
    return shown
  }

  const getPriorityVariant = (
    p: number
  ): 'default' | 'secondary' | 'outline' => {
    if (p <= 1) return 'default'
    if (p <= 4) return 'secondary'
    return 'outline'
  }

  // Add Parser modal ----------------------------------------------------------
  const [isAddParserOpen, setIsAddParserOpen] = useState(false)
  const [newParserType, setNewParserType] = useState<string>('')
  const [newParserConfig, setNewParserConfig] = useState<
    Record<string, unknown>
  >({})
  const [newParserPriority, setNewParserPriority] = useState<string>('1')

  const slugify = (str: string) =>
    str
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')

  const handleCreateParser = () => {
    const name = newParserType.trim()
    if (!name) return
    const prio = Number(newParserPriority)
    if (!Number.isFinite(prio)) return
    const idBase = slugify(name) || 'parser'
    const id = `${idBase}-${Date.now()}`
    const next: ParserRow = {
      id,
      name,
      priority: prio,
      include: '',
      exclude: '',
      summary: '',
      config: newParserConfig,
    }
    const rows = [...parserRows, next]
    setParserRows(rows)
    saveParsers(rows)
    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('lf:processingUpdated', {
            detail: { strategyId, type: 'parser:add', item: next },
          })
        )
      }
    } catch {}
    setOpenRows(prev => new Set(prev).add(id))
    setIsAddParserOpen(false)
    setNewParserType('')
    setNewParserConfig({})
    setNewParserPriority('1')
  }

  // Edit/Delete Parser modals -------------------------------------------------
  const [isEditParserOpen, setIsEditParserOpen] = useState(false)
  const [editParserId, setEditParserId] = useState<string>('')
  const [editParserConfig, setEditParserConfig] = useState<
    Record<string, unknown>
  >({})
  const [editParserPriority, setEditParserPriority] = useState<string>('1')

  const openEditParser = (id: string) => {
    const found = parserRows.find(p => p.id === id)
    if (!found) return
    setEditParserId(found.id)
    setEditParserConfig(found.config || getDefaultConfigForParser(found.name))
    setEditParserPriority(String(found.priority))
    setIsEditParserOpen(true)
  }

  const handleUpdateParser = () => {
    if (!editParserId) return
    const prio = Number(editParserPriority)
    if (!Number.isFinite(prio)) return
    const next = parserRows.map(p =>
      p.id === editParserId
        ? { ...p, config: editParserConfig, priority: prio }
        : p
    )
    setParserRows(next)
    saveParsers(next)
    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('lf:processingUpdated', {
            detail: { strategyId, type: 'parser:update', id: editParserId },
          })
        )
      }
    } catch {}
    setIsEditParserOpen(false)
  }

  const [isDeleteParserOpen, setIsDeleteParserOpen] = useState(false)
  const [deleteParserId, setDeleteParserId] = useState<string>('')
  const openDeleteParser = (id: string) => {
    setDeleteParserId(id)
    setIsDeleteParserOpen(true)
  }
  const handleDeleteParser = () => {
    if (!deleteParserId) return
    const next = parserRows.filter(p => p.id !== deleteParserId)
    setParserRows(next)
    saveParsers(next)
    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('lf:processingUpdated', {
            detail: { strategyId, type: 'parser:delete', id: deleteParserId },
          })
        )
      }
    } catch {}
    setIsDeleteParserOpen(false)
    setDeleteParserId('')
  }

  // Add/Edit/Delete Extractor modals -----------------------------------------
  const [isAddExtractorOpen, setIsAddExtractorOpen] = useState(false)
  const [newExtractorType, setNewExtractorType] = useState<string>('')
  const [newExtractorPriority, setNewExtractorPriority] = useState<string>('1')
  const [newExtractorConfig, setNewExtractorConfig] = useState<
    Record<string, unknown>
  >({})

  const handleCreateExtractor = () => {
    const name = newExtractorType.trim()
    const prio = Number(newExtractorPriority)
    if (!name || !Number.isFinite(prio)) return
    const idBase = slugify(name) || 'extractor'
    const id = `${idBase}-${Date.now()}`
    const next: ExtractorRow = {
      id,
      name,
      priority: prio,
      applyTo: 'All files (*)',
      summary: '',
      config: newExtractorConfig,
    }
    const rows = [...extractorRows, next]
    setExtractorRows(rows)
    saveExtractors(rows)
    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('lf:processingUpdated', {
            detail: { strategyId, type: 'extractor:add', item: next },
          })
        )
      }
    } catch {}
    setOpenRows(prev => new Set(prev).add(id))
    setIsAddExtractorOpen(false)
    setNewExtractorType('')
    setNewExtractorPriority('1')
    setNewExtractorConfig({})
  }

  const [isEditExtractorOpen, setIsEditExtractorOpen] = useState(false)
  const [editExtractorId, setEditExtractorId] = useState<string>('')
  const [editExtractorPriority, setEditExtractorPriority] =
    useState<string>('1')
  const [editExtractorConfig, setEditExtractorConfig] = useState<
    Record<string, unknown>
  >({})

  const openEditExtractor = (id: string) => {
    const found = extractorRows.find(e => e.id === id)
    if (!found) return
    setEditExtractorId(found.id)
    setEditExtractorPriority(String(found.priority))
    setEditExtractorConfig(
      found.config || getDefaultConfigForExtractor(found.name)
    )
    setIsEditExtractorOpen(true)
  }
  const handleUpdateExtractor = () => {
    const prio = Number(editExtractorPriority)
    if (!editExtractorId || !Number.isFinite(prio)) return
    const next = extractorRows.map(e =>
      e.id === editExtractorId
        ? {
            ...e,
            priority: prio,
            config: editExtractorConfig,
          }
        : e
    )
    setExtractorRows(next)
    saveExtractors(next)
    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('lf:processingUpdated', {
            detail: {
              strategyId,
              type: 'extractor:update',
              id: editExtractorId,
            },
          })
        )
      }
    } catch {}
    setIsEditExtractorOpen(false)
  }

  const [isDeleteExtractorOpen, setIsDeleteExtractorOpen] = useState(false)
  const [deleteExtractorId, setDeleteExtractorId] = useState<string>('')
  const openDeleteExtractor = (id: string) => {
    setDeleteExtractorId(id)
    setIsDeleteExtractorOpen(true)
  }
  const handleDeleteExtractor = () => {
    if (!deleteExtractorId) return
    const next = extractorRows.filter(e => e.id !== deleteExtractorId)
    setExtractorRows(next)
    saveExtractors(next)
    try {
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('lf:processingUpdated', {
            detail: {
              strategyId,
              type: 'extractor:delete',
              id: deleteExtractorId,
            },
          })
        )
      }
    } catch {}
    setIsDeleteExtractorOpen(false)
    setDeleteExtractorId('')
  }

  // Reset to defaults (for universal strategy) --------------------------------
  const isUniversal = strategyId === 'processing-universal'
  const handleResetDefaults = () => {
    if (!isUniversal) return
    const ok = confirm('Reset parsers and extractors to defaults?')
    if (!ok) return
    try {
      setParserRows(defaultParsers)
      setExtractorRows(defaultExtractors)
      if (storageKeys.parsers)
        localStorage.setItem(
          storageKeys.parsers,
          JSON.stringify(defaultParsers)
        )
      if (storageKeys.extractors)
        localStorage.setItem(
          storageKeys.extractors,
          JSON.stringify(defaultExtractors)
        )
      if (typeof window !== 'undefined') {
        window.dispatchEvent(
          new CustomEvent('lf:processingUpdated', {
            detail: { strategyId, type: 'reset:defaults' },
          })
        )
      }
    } catch {}
  }

  useEffect(() => {
    const handler = (e: Event) => {
      try {
        // @ts-ignore custom event
        const { strategyId: sid } = (e as CustomEvent).detail || {}
        if (sid && strategyId && sid === strategyId) {
          setStrategyMetaTick(t => t + 1)
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

  // (Removed) save button logic – edits persist immediately

  return (
    <div className="w-full flex flex-col gap-3 pb-20">
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
          <span className="text-foreground">Processing strategies</span>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">{strategyName}</span>
        </nav>
        <PageActions mode={mode} onModeChange={setMode} />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-1">
        <div className="flex items-center gap-2">
          <h2 className="text-lg md:text-xl font-medium">{strategyName}</h2>
          <button
            className="p-1 rounded-md hover:bg-accent text-muted-foreground"
            onClick={() => {
              setEditName(strategyName)
              setEditDescription(strategyDescription)
              setIsEditOpen(true)
            }}
            aria-label="Edit strategy"
            title="Edit strategy"
          >
            <FontIcon type="edit" className="w-4 h-4" />
          </button>
        </div>
      </div>
      {strategyDescription && (
        <div className="text-sm text-muted-foreground">
          {strategyDescription}
        </div>
      )}

      {/* Used by + Actions */}
      <div className="flex items-center justify-between gap-2 flex-wrap mb-3">
        <div className="flex items-center gap-2 flex-wrap">
          <div className="text-xs text-muted-foreground">Used by</div>
          {usedBy.map(u => (
            <Badge key={u} variant="secondary" size="sm" className="rounded-xl">
              {u}
            </Badge>
          ))}
        </div>
        <div className="flex items-center gap-2 ml-auto">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAddParserOpen(true)}
          >
            <Plus className="w-4 h-4" /> Add Parser
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAddExtractorOpen(true)}
          >
            <Plus className="w-4 h-4" /> Add Extractor
          </Button>
          {isUniversal ? (
            <Button variant="outline" size="sm" onClick={handleResetDefaults}>
              Reset to defaults
            </Button>
          ) : null}
        </div>
      </div>

      {/* Processing editors */}
      {/* Tabs header outside card */}
      <Tabs
        activeTab={activeTab}
        setActiveTab={t => setActiveTab(t as 'parsers' | 'extractors')}
        tabs={[
          { id: 'parsers', label: `Parsers (${parserRows.length})` },
          { id: 'extractors', label: `Extractors (${extractorRows.length})` },
        ]}
      />
      <section className="rounded-lg border border-border bg-card p-4">
        {activeTab === 'parsers' ? (
          <div className="flex flex-col gap-2">
            {parserRows.map(row => {
              const open = openRows.has(row.id)
              return (
                <div
                  key={row.id}
                  className="rounded-lg border border-border bg-card p-3 hover:bg-accent/20 transition-colors"
                >
                  <div
                    className="w-full flex items-center gap-2 text-left cursor-pointer"
                    onClick={() => toggleRow(row.id)}
                    aria-expanded={open}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        toggleRow(row.id)
                      }
                    }}
                  >
                    <ChevronDown
                      className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`}
                    />
                    <div className="flex-1 text-sm font-medium">{row.name}</div>
                    <div className="hidden md:block text-xs text-muted-foreground mr-4">
                      {summarizeFileTypes(row.include)}
                    </div>
                    <Badge
                      variant={getPriorityVariant(row.priority)}
                      size="sm"
                      className="rounded-xl mr-2"
                    >
                      Priority: {row.priority}
                    </Badge>
                    <div className="flex items-center gap-1 ml-auto">
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="Configure parser"
                        onClick={e => {
                          e.stopPropagation()
                          openEditParser(row.id)
                        }}
                      >
                        <Settings className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="Remove parser"
                        onClick={e => {
                          e.stopPropagation()
                          openDeleteParser(row.id)
                        }}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  {open ? (
                    <div className="mt-2 rounded-md border border-border bg-accent/10 p-2 text-sm">
                      <div className="text-muted-foreground">
                        <span className="font-medium text-foreground">
                          Include:
                        </span>{' '}
                        {row.include}{' '}
                        <span className="ml-2 font-medium text-foreground">
                          Exclude:
                        </span>{' '}
                        {row.exclude}
                      </div>
                      <div className="mt-1 text-muted-foreground">
                        {row.summary}
                      </div>
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>
        ) : (
          <div className="flex flex-col gap-2">
            {extractorRows.map(row => {
              const open = openRows.has(row.id)
              return (
                <div
                  key={row.id}
                  className="rounded-lg border border-border bg-card p-3 hover:bg-accent/20 transition-colors"
                >
                  <div
                    className="w-full flex items-center gap-2 text-left cursor-pointer"
                    onClick={() => toggleRow(row.id)}
                    aria-expanded={open}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        toggleRow(row.id)
                      }
                    }}
                  >
                    <ChevronDown
                      className={`w-4 h-4 transition-transform ${open ? 'rotate-180' : ''}`}
                    />
                    <div className="flex-1 text-sm font-medium">{row.name}</div>
                    <div className="hidden md:block text-xs text-muted-foreground mr-4">
                      {row.applyTo || 'All (*)'}
                    </div>
                    <Badge
                      variant={getPriorityVariant(row.priority)}
                      size="sm"
                      className="rounded-xl mr-2"
                    >
                      Priority: {row.priority}
                    </Badge>
                    <div className="flex items-center gap-1 ml-auto">
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="Configure extractor"
                        onClick={e => {
                          e.stopPropagation()
                          openEditExtractor(row.id)
                        }}
                      >
                        <Settings className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="Remove extractor"
                        onClick={e => {
                          e.stopPropagation()
                          openDeleteExtractor(row.id)
                        }}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                  {open ? (
                    <div className="mt-2 rounded-md border border-border bg-accent/10 p-2 text-sm">
                      <div className="text-muted-foreground">
                        <span className="font-medium text-foreground">
                          Apply to:
                        </span>{' '}
                        {row.applyTo}
                      </div>
                      <div className="mt-1 text-muted-foreground">
                        {row.summary}
                      </div>
                    </div>
                  ) : null}
                </div>
              )
            })}
          </div>
        )}
      </section>

      {/* Add Parser Modal */}
      <Dialog open={isAddParserOpen} onOpenChange={setIsAddParserOpen}>
        <DialogContent className="sm:max-w-xl p-0">
          <div className="flex flex-col max-h-[80vh]">
            <DialogHeader className="bg-background p-4 border-b">
              <DialogTitle className="text-lg text-foreground">
                Add parser
              </DialogTitle>
            </DialogHeader>
            <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">
                  Parser type
                </Label>
                <div className="mt-1">
                  {(() => {
                    const existing = new Set(parserRows.map(p => p.name))
                    const available = ORDERED_PARSER_TYPES.filter(
                      t => !existing.has(t) && PARSER_SCHEMAS[t]
                    )
                    if (!newParserType && available.length > 0) {
                      const first = available[0]
                      setTimeout(() => {
                        setNewParserType(first)
                        setNewParserConfig(getDefaultConfigForParser(first))
                      }, 0)
                    }
                    return (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="outline"
                            className="w-full justify-between"
                          >
                            {newParserType ||
                              available[0] ||
                              'No parsers available'}
                            <ChevronDown className="w-4 h-4 ml-2 opacity-70" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent>
                          {available.length === 0 ? (
                            <DropdownMenuItem disabled>
                              No parsers available
                            </DropdownMenuItem>
                          ) : (
                            available.map(t => (
                              <DropdownMenuItem
                                key={t}
                                onClick={() => {
                                  setNewParserType(t)
                                  setNewParserConfig(
                                    getDefaultConfigForParser(t)
                                  )
                                }}
                              >
                                {t}
                              </DropdownMenuItem>
                            ))
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )
                  })()}
                </div>
              </div>
              {newParserType && PARSER_SCHEMAS[newParserType] ? (
                <>
                  <div className="text-xs text-muted-foreground">
                    {PARSER_SCHEMAS[newParserType].description}
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">
                      Priority
                    </Label>
                    <Input
                      type="number"
                      className="mt-1 bg-background w-40"
                      value={newParserPriority}
                      onChange={e => setNewParserPriority(e.target.value)}
                    />
                  </div>
                  <div className="rounded-lg border border-border bg-accent/10 p-3">
                    <div className="text-sm font-medium mb-2">
                      {PARSER_SCHEMAS[newParserType].title}
                    </div>
                    <ParserSettingsForm
                      schema={PARSER_SCHEMAS[newParserType]}
                      value={newParserConfig}
                      onChange={setNewParserConfig}
                    />
                  </div>
                </>
              ) : null}
            </div>
            <DialogFooter className="bg-background p-4 border-t flex items-center gap-2">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsAddParserOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm ${
                  newParserType.trim().length > 0
                    ? 'bg-primary text-primary-foreground hover:opacity-90'
                    : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'
                }`}
                onClick={handleCreateParser}
                disabled={newParserType.trim().length === 0}
                type="button"
              >
                Add parser
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Parser Modal (config only) */}
      <Dialog open={isEditParserOpen} onOpenChange={setIsEditParserOpen}>
        <DialogContent
          className="sm:max-w-xl p-0"
          onOpenAutoFocus={e => e.preventDefault()}
        >
          <div className="flex flex-col max-h-[80vh]">
            <DialogHeader className="bg-background p-4 border-b">
              <DialogTitle className="text-lg text-foreground">
                Edit parser settings
              </DialogTitle>
            </DialogHeader>
            <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-3">
              {(() => {
                const found = parserRows.find(p => p.id === editParserId)
                if (!found) return null
                const schema = PARSER_SCHEMAS[found.name]
                if (!schema) {
                  return (
                    <div className="text-sm text-muted-foreground">
                      No schema found for this parser type.
                    </div>
                  )
                }
                return (
                  <>
                    <div className="text-xs text-muted-foreground">
                      {schema.description}
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">
                        Priority
                      </Label>
                      <Input
                        type="number"
                        className="mt-1 bg-background w-40"
                        value={editParserPriority}
                        onChange={e => setEditParserPriority(e.target.value)}
                      />
                    </div>
                    <div className="rounded-lg border border-border bg-accent/10 p-3">
                      <div className="text-sm font-medium mb-2">
                        {schema.title}
                      </div>
                      <ParserSettingsForm
                        schema={schema}
                        value={editParserConfig}
                        onChange={setEditParserConfig}
                      />
                    </div>
                  </>
                )
              })()}
            </div>
            <DialogFooter className="bg-background p-4 border-t flex items-center gap-2">
              <div className="mr-auto">
                <button
                  className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
                  onClick={() => {
                    if (!editParserId) return
                    openDeleteParser(editParserId)
                  }}
                  type="button"
                >
                  Remove
                </button>
              </div>
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsEditParserOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90`}
                onClick={handleUpdateParser}
                type="button"
              >
                Save changes
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Parser Modal */}
      <Dialog open={isDeleteParserOpen} onOpenChange={setIsDeleteParserOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Delete parser
            </DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            Are you sure you want to delete this parser? This action cannot be
            undone.
          </div>
          <DialogFooter className="flex items-center gap-2">
            <button
              className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
              onClick={() => setIsDeleteParserOpen(false)}
              type="button"
            >
              Cancel
            </button>
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={handleDeleteParser}
              type="button"
            >
              Delete
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Add Extractor Modal (schema-driven) */}
      <Dialog open={isAddExtractorOpen} onOpenChange={setIsAddExtractorOpen}>
        <DialogContent className="sm:max-w-xl p-0">
          <div className="flex flex-col max-h-[80vh]">
            <DialogHeader className="bg-background p-4 border-b">
              <DialogTitle className="text-lg text-foreground">
                Add extractor
              </DialogTitle>
            </DialogHeader>
            <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-3">
              <div>
                <Label className="text-xs text-muted-foreground">
                  Extractor type
                </Label>
                <div className="mt-1">
                  {(() => {
                    const existing = new Set(extractorRows.map(e => e.name))
                    const available = ORDERED_EXTRACTOR_TYPES.filter(
                      t => !existing.has(t) && EXTRACTOR_SCHEMAS[t]
                    )
                    if (!newExtractorType && available.length > 0) {
                      const first = available[0]
                      setTimeout(() => {
                        setNewExtractorType(first)
                        setNewExtractorConfig(
                          getDefaultConfigForExtractor(first)
                        )
                      }, 0)
                    }
                    return (
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <Button
                            variant="outline"
                            className="w-full justify-between"
                          >
                            {newExtractorType ||
                              available[0] ||
                              'No extractors available'}
                            <ChevronDown className="w-4 h-4 ml-2 opacity-70" />
                          </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent>
                          {available.length === 0 ? (
                            <DropdownMenuItem disabled>
                              No extractors available
                            </DropdownMenuItem>
                          ) : (
                            available.map(t => (
                              <DropdownMenuItem
                                key={t}
                                onClick={() => {
                                  setNewExtractorType(t)
                                  setNewExtractorConfig(
                                    getDefaultConfigForExtractor(t)
                                  )
                                }}
                              >
                                {t}
                              </DropdownMenuItem>
                            ))
                          )}
                        </DropdownMenuContent>
                      </DropdownMenu>
                    )
                  })()}
                </div>
              </div>
              {newExtractorType && EXTRACTOR_SCHEMAS[newExtractorType] ? (
                <>
                  <div className="text-xs text-muted-foreground">
                    {EXTRACTOR_SCHEMAS[newExtractorType].description}
                  </div>
                  <div>
                    <Label className="text-xs text-muted-foreground">
                      Priority
                    </Label>
                    <Input
                      type="number"
                      className="mt-1 bg-background w-40"
                      value={newExtractorPriority}
                      onChange={e => setNewExtractorPriority(e.target.value)}
                    />
                  </div>
                  <div className="rounded-lg border border-border bg-accent/10 p-3">
                    <div className="text-sm font-medium mb-2">
                      {EXTRACTOR_SCHEMAS[newExtractorType].title}
                    </div>
                    <ExtractorSettingsForm
                      schema={EXTRACTOR_SCHEMAS[newExtractorType]}
                      value={newExtractorConfig}
                      onChange={setNewExtractorConfig}
                    />
                  </div>
                </>
              ) : null}
            </div>
            <DialogFooter className="bg-background p-4 border-t flex items-center gap-2">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsAddExtractorOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm ${newExtractorType.trim().length > 0 ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
                onClick={handleCreateExtractor}
                disabled={newExtractorType.trim().length === 0}
                type="button"
              >
                Add extractor
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* Edit Extractor Modal (schema-driven) */}
      <Dialog open={isEditExtractorOpen} onOpenChange={setIsEditExtractorOpen}>
        <DialogContent
          className="sm:max-w-xl p-0"
          onOpenAutoFocus={e => e.preventDefault()}
        >
          <div className="flex flex-col max-h-[80vh]">
            <DialogHeader className="bg-background p-4 border-b">
              <DialogTitle className="text-lg text-foreground">
                Edit extractor
              </DialogTitle>
            </DialogHeader>
            <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-3">
              {(() => {
                const found = extractorRows.find(e => e.id === editExtractorId)
                if (!found) return null
                const schema = EXTRACTOR_SCHEMAS[found.name]
                if (!schema) {
                  return (
                    <div className="text-sm text-muted-foreground">
                      No schema found for this extractor type.
                    </div>
                  )
                }
                return (
                  <>
                    <div className="text-xs text-muted-foreground">
                      {schema.description}
                    </div>
                    <div>
                      <Label className="text-xs text-muted-foreground">
                        Priority
                      </Label>
                      <Input
                        type="number"
                        className="mt-1 bg-background w-40"
                        value={editExtractorPriority}
                        onChange={e => setEditExtractorPriority(e.target.value)}
                      />
                    </div>
                    <div className="rounded-lg border border-border bg-accent/10 p-3">
                      <div className="text-sm font-medium mb-2">
                        {schema.title}
                      </div>
                      <ExtractorSettingsForm
                        schema={schema}
                        value={editExtractorConfig}
                        onChange={setEditExtractorConfig}
                      />
                    </div>
                  </>
                )
              })()}
            </div>
            <DialogFooter className="bg-background p-4 border-t flex items-center gap-2">
              <div className="mr-auto">
                <button
                  className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
                  onClick={() => {
                    if (!editExtractorId) return
                    setDeleteExtractorId(editExtractorId)
                    setIsDeleteExtractorOpen(true)
                  }}
                  type="button"
                >
                  Remove
                </button>
              </div>
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsEditExtractorOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90`}
                onClick={handleUpdateExtractor}
                type="button"
              >
                Save changes
              </button>
            </DialogFooter>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Extractor Modal */}
      <Dialog
        open={isDeleteExtractorOpen}
        onOpenChange={setIsDeleteExtractorOpen}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Delete extractor
            </DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            Are you sure you want to delete this extractor? This action cannot
            be undone.
          </div>
          <DialogFooter className="flex items-center gap-2">
            <button
              className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
              onClick={() => setIsDeleteExtractorOpen(false)}
              type="button"
            >
              Cancel
            </button>
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={handleDeleteExtractor}
              type="button"
            >
              Delete
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
      {/* Edit Strategy Modal */}
      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Edit strategy
            </DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3 pt-1">
            <div>
              <label className="text-xs text-muted-foreground">
                Strategy name
              </label>
              <input
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Enter name"
                value={editName}
                onChange={e => setEditName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Add a brief description"
                value={editDescription}
                onChange={e => setEditDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="flex items-center gap-2">
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={() => {
                if (!strategyId) return
                const ok = confirm(
                  'Are you sure you want to delete this strategy?'
                )
                if (ok) {
                  try {
                    localStorage.removeItem(
                      `lf_strategy_name_override_${strategyId}`
                    )
                    localStorage.removeItem(
                      `lf_strategy_description_${strategyId}`
                    )
                    // Mark strategy as deleted so it disappears from lists
                    const raw = localStorage.getItem('lf_strategy_deleted')
                    const arr = raw ? (JSON.parse(raw) as string[]) : []
                    const set = new Set(arr)
                    set.add(strategyId)
                    localStorage.setItem(
                      'lf_strategy_deleted',
                      JSON.stringify(Array.from(set))
                    )
                  } catch {}
                  setIsEditOpen(false)
                  setStrategyMetaTick(t => t + 1)
                  navigate('/chat/rag')
                  toast({ message: 'Strategy deleted', variant: 'default' })
                }
              }}
              type="button"
            >
              Delete
            </button>
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsEditOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className={`px-3 py-2 rounded-md text-sm ${editName.trim().length > 0 ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
                onClick={() => {
                  if (!strategyId || editName.trim().length === 0) return
                  try {
                    localStorage.setItem(
                      `lf_strategy_name_override_${strategyId}`,
                      editName.trim()
                    )
                    localStorage.setItem(
                      `lf_strategy_description_${strategyId}`,
                      editDescription
                    )
                  } catch {}
                  setIsEditOpen(false)
                  setStrategyMetaTick(t => t + 1)
                }}
                disabled={editName.trim().length === 0}
                type="button"
              >
                Save
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* End processing editors */}

      {/* Retrieval and Embedding moved to project-level settings. */}
    </div>
  )
}

export default StrategyView
