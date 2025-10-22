import { useEffect, useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { defaultStrategies } from './strategies'
import ParserSettingsForm from './ParserSettingsForm'
import PatternEditor from './PatternEditor'
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
import { Checkbox } from '../ui/checkbox'
import { useActiveProject } from '../../hooks/useActiveProject'
import {
  useListDatasets,
  datasetKeys,
  useReIngestDataset,
} from '../../hooks/useDatasets'
import { useProject, useUpdateProject } from '../../hooks/useProjects'
import ConfigEditor from '../ConfigEditor/ConfigEditor'

function StrategyView() {
  const navigate = useNavigate()
  const { strategyId } = useParams()
  const { toast } = useToast()
  const [mode, setMode] = useState<Mode>('designer')
  const queryClient = useQueryClient()
  const activeProject = useActiveProject()
  const reIngestMutation = useReIngestDataset()
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )
  const updateProjectMutation = useUpdateProject()
  const { data: datasetsResp } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject }
  )

  // Track whether this strategy has changes that require reprocessing
  const [needsReprocess, setNeedsReprocess] = useState(false)
  useEffect(() => {
    if (!strategyId) return
    try {
      const raw = localStorage.getItem(
        `lf_strategy_needs_reprocess_${strategyId}`
      )
      setNeedsReprocess(raw === '1' || raw === 'true')
    } catch {}
  }, [strategyId])
  const markNeedsReprocess = () => {
    try {
      localStorage.setItem(`lf_strategy_needs_reprocess_${strategyId}`, '1')
    } catch {}
    setNeedsReprocess(true)
  }
  const clearNeedsReprocess = () => {
    try {
      localStorage.setItem(`lf_strategy_needs_reprocess_${strategyId}`, '0')
    } catch {}
    setNeedsReprocess(false)
  }

  // Combine server datasets with local fallback (if user has local-only datasets)
  const allDatasets = useMemo(() => {
    if (datasetsResp?.datasets && datasetsResp.datasets.length > 0) {
      return datasetsResp.datasets.map(d => {
        const { name } = d
        let ragStrategy = (d as any).rag_strategy as string | undefined
        // Overlay local per-dataset override if present
        try {
          const storedName = localStorage.getItem(
            `lf_dataset_strategy_name_${name}`
          )
          if (storedName && storedName.trim().length > 0) {
            ragStrategy = storedName
          }
        } catch {}
        return { name, rag_strategy: ragStrategy }
      })
    }
    // Local fallback: minimal name + strategy from localStorage, if present
    try {
      const raw = localStorage.getItem('lf_datasets')
      if (!raw) return [] as { name: string; rag_strategy?: string }[]
      const arr = JSON.parse(raw)
      if (!Array.isArray(arr)) return []
      return arr
        .map((d: any) => {
          const name = typeof d?.name === 'string' ? d.name : d?.id
          if (!name) return null
          const storedName = localStorage.getItem(
            `lf_dataset_strategy_name_${name}`
          )
          return { name, rag_strategy: storedName || 'auto' }
        })
        .filter(Boolean) as { name: string; rag_strategy?: string }[]
    } catch {
      return [] as { name: string; rag_strategy?: string }[]
    }
  }, [datasetsResp])

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

  // Datasets using this strategy (from API) -----------------------------------
  const assignedDatasets = useMemo(() => {
    if (!allDatasets || !strategyName) return [] as string[]
    return allDatasets
      .filter(d => d.rag_strategy === strategyName)
      .map(d => d.name)
  }, [allDatasets, strategyName])

  const canReprocess =
    assignedDatasets.length > 0 && needsReprocess && !reIngestMutation.isPending

  // Manage datasets modal state
  const [isManageOpen, setIsManageOpen] = useState(false)
  const [selectedDatasets, setSelectedDatasets] = useState<Set<string>>(
    new Set()
  )

  useEffect(() => {
    if (!isManageOpen) return
    // Initialize selection from currently assigned
    setSelectedDatasets(new Set(assignedDatasets))
  }, [isManageOpen, assignedDatasets])

  const toggleDataset = (name: string) => {
    // Do not allow unassigning from this strategy
    if (isDatasetLocked(name)) return
    setSelectedDatasets(prev => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  // Helper to check if dataset is locked to this strategy
  const isDatasetLocked = (name: string) => {
    const current = allDatasets.find(d => d.name === name)?.rag_strategy
    return current === strategyName
  }

  // Reprocess confirmation modal
  const [isReprocessOpen, setIsReprocessOpen] = useState(false)
  const [pendingAdded, setPendingAdded] = useState<string[]>([])
  const [reprocessErrors, setReprocessErrors] = useState<{
    [datasetId: string]: unknown
  }>({})

  const getDatasetNameById = (id: string) => {
    const f = allDatasets.find(d => d.name === id)
    return f ? f.name : id
  }

  const saveAssignments = async () => {
    if (!strategyId || !strategyName) return
    const prev = new Set(assignedDatasets)
    const next = new Set(selectedDatasets)
    const added: string[] = []
    // Add only new selections; no unassign/removal allowed
    for (const n of next) if (!prev.has(n)) added.push(n)

    // Attempt backend update by rewriting project.config.datasets[].data_processing_strategy
    const ns = activeProject?.namespace
    const proj = activeProject?.project
    const currentConfig = projectResp?.project?.config
    const currentDatasets: any[] = currentConfig?.datasets || []

    const performLocalFallback = () => {
      try {
        const key = `lf_strategy_datasets_${strategyId}`
        const prevRaw = localStorage.getItem(key)
        const prevList: string[] = prevRaw ? JSON.parse(prevRaw) : []
        const working = new Set(prevList)
        for (const n of added) working.add(n)
        localStorage.setItem(key, JSON.stringify(Array.from(working)))
        // Also set per-dataset overrides so UI reflects immediately
        for (const n of added) {
          localStorage.setItem(`lf_dataset_strategy_name_${n}`, strategyName)
        }
        toast({ message: 'Assignments saved locally', variant: 'default' })
      } catch {}
    }

    try {
      if (!ns || !proj || !currentConfig) {
        performLocalFallback()
      } else {
        const updatedDatasets = (currentDatasets || []).map(ds =>
          added.includes(ds.name) ? { ...ds, rag_strategy: strategyName } : ds
        )
        const nextConfig = { ...currentConfig, datasets: updatedDatasets }
        await updateProjectMutation.mutateAsync({
          namespace: ns,
          projectId: proj,
          request: { config: nextConfig },
        })
        // Mirror per-dataset overrides locally for instant UI feedback
        try {
          for (const n of added) {
            localStorage.setItem(`lf_dataset_strategy_name_${n}`, strategyName)
          }
        } catch {}
        // Refresh datasets list
        queryClient.invalidateQueries({ queryKey: datasetKeys.list(ns, proj) })
        toast({ message: 'Assignments saved', variant: 'default' })
      }
    } catch (e) {
      console.error(
        'Failed to save assignments, falling back to localStorage',
        e
      )
      performLocalFallback()
    }

    // Mark that reprocess is needed when datasets are added
    if (added.length > 0) {
      markNeedsReprocess()
      // Reprocess newly added datasets?
      setPendingAdded(added)
      setIsReprocessOpen(true)
    }

    setIsManageOpen(false)
  }

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

  // Patterns editors helpers --------------------------------------------------
  const getDefaultIncludePatternsForParser = (parserName: string): string[] => {
    switch (parserName) {
      case 'PDFParser_LlamaIndex':
      case 'PDFParser_PyPDF2':
        return ['*.pdf']
      case 'DocxParser_LlamaIndex':
      case 'DocxParser_PythonDocx':
        return ['*.docx', '*.doc']
      case 'MarkdownParser_Python':
      case 'MarkdownParser_LlamaIndex':
        return ['*.md', '*.markdown']
      case 'CSVParser_Pandas':
      case 'CSVParser_LlamaIndex':
        return ['*.csv']
      case 'ExcelParser_OpenPyXL':
      case 'ExcelParser_Pandas':
      case 'ExcelParser_LlamaIndex':
        return ['*.xlsx', '*.xls']
      case 'TextParser_Python':
      case 'TextParser_LlamaIndex':
        return ['*.txt', '*.text']
      default:
        return []
    }
  }

  // const getDefaultApplyPatternsForExtractor = (): string[] => ['*']

  const parsePatternsString = (s: string | undefined): string[] => {
    if (!s) return []
    try {
      const arr = JSON.parse(s)
      if (Array.isArray(arr)) {
        return Array.from(new Set(arr.filter(x => typeof x === 'string')))
      }
    } catch {
      const parts = s
        .split(',')
        .map(t => t.trim())
        .filter(Boolean)
      return Array.from(new Set(parts))
    }
    return []
  }

  const patternsToString = (arr: string[]): string => JSON.stringify(arr)

  const SAFE_PATTERNS = [
    'README',
    'LICENSE',
    'Makefile',
    'Dockerfile',
    'requirements.txt',
    'package.json',
    'yarn.lock',
    'Pipfile',
    'setup.py',
    'config',
    'env',
  ]

  const isSuspiciousPattern = (p: string): boolean => {
    const trimmed = p.trim()
    if (trimmed === '*' || trimmed === '') return false
    if (SAFE_PATTERNS.includes(trimmed)) return false
    if (/[.*]/.test(trimmed) || /\.[a-z0-9]+$/i.test(trimmed)) return false
    return true
  }

  const getFriendlyParserName = (parserName: string): string => {
    switch (parserName) {
      case 'PDFParser_LlamaIndex':
        return 'PDF Document Parser'
      case 'PDFParser_PyPDF2':
        return 'PDF Parser (PyPDF2)'
      case 'DocxParser_LlamaIndex':
      case 'Docx Parser Llama Index':
        return 'Word Document Parser'
      case 'DocxParser_PythonDocx':
      case 'Docx Parser Python Docx':
        return 'Word Document Parser (Alternative)'
      case 'MarkdownParser_Python':
        return 'Markdown Parser (Basic)'
      case 'MarkdownParser_LlamaIndex':
      case 'Markdown Parser Llama Index':
        return 'Markdown Parser (LlamaIndex)'
      case 'CSVParser_Pandas':
        return 'CSV Data Parser'
      case 'CSVParser_LlamaIndex':
      case 'CSVParser Llama Index':
        return 'CSV Data Parser (LlamaIndex)'
      case 'ExcelParser_Pandas':
      case 'ExcelParser_LlamaIndex':
      case 'Excel Parser Llama Index':
        return 'Excel Spreadsheet Parser (LlamaIndex)'
      case 'ExcelParser_OpenPyXL':
      case 'Excel Parser Open Py XL':
        return 'Excel Spreadsheet Parser (OpenPyXL)'
      case 'TextParser_Python':
        return 'Plain Text Parser'
      case 'TextParser_LlamaIndex':
      case 'Text Parser Llama Index':
        return 'Plain Text Parser (LlamaIndex)'
      default: {
        // Sensible fallback: split by underscores/camelcase
        try {
          const spaced = parserName
            .replace(/_/g, ' ')
            .replace(/([a-z])([A-Z])/g, '$1 $2')
          return spaced.trim()
        } catch {
          return parserName
        }
      }
    }
  }

  const getFriendlyExtractorName = (name: string): string => {
    switch (name) {
      case 'ContentStatisticsExtractor':
        return 'Content Statistics Extractor'
      case 'EntityExtractor':
        return 'Entity Extractor'
      case 'KeywordExtractor':
        return 'Keyword Extractor'
      case 'TableExtractor':
        return 'Table Extractor'
      case 'DateTimeExtractor':
        return 'Date & Time Extractor'
      case 'PatternExtractor':
        return 'Pattern Extractor'
      case 'HeadingExtractor':
        return 'Heading Extractor'
      case 'LinkExtractor':
        return 'Link Extractor'
      case 'SummaryExtractor':
        return 'Text Summary Extractor'
      case 'TextSummaryExtractor':
        return 'Text Summary Extractor (Alternative)'
      case 'StatisticsExtractor':
        return 'Text Statistics Extractor'
      case 'YAKEExtractor':
        return 'YAKE Keyword Extractor'
      case 'SentimentExtractor':
        return 'Sentiment Analysis Extractor'
      default:
        try {
          return name.replace(/([a-z])([A-Z])/g, '$1 $2').trim()
        } catch {
          return name
        }
    }
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
  const [newParserPriorityError, setNewParserPriorityError] = useState(false)
  const [newParserIncludes, setNewParserIncludes] = useState<string[]>([])

  const slugify = (str: string) =>
    str
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')

  const MAX_PRIORITY = 1000

  const handleCreateParser = () => {
    const name = newParserType.trim()
    if (!name) return
    const prio = Number(newParserPriority)
    if (!Number.isInteger(prio)) {
      setNewParserPriorityError(true)
      toast({
        message: `Priority must be an integer between 0 and ${MAX_PRIORITY}`,
        variant: 'destructive',
      })
      return
    }
    if (prio < 0 || prio > MAX_PRIORITY) {
      setNewParserPriorityError(true)
      toast({
        message: `Priority must be between 0 and ${MAX_PRIORITY}`,
        variant: 'destructive',
      })
      return
    }
    const idBase = slugify(name) || 'parser'
    const id = `${idBase}-${Date.now()}`
    const next: ParserRow = {
      id,
      name,
      priority: prio,
      include: patternsToString(newParserIncludes),
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
    setNewParserIncludes([])
    markNeedsReprocess()
  }

  // Edit/Delete Parser modals -------------------------------------------------
  const [isEditParserOpen, setIsEditParserOpen] = useState(false)
  const [editParserId, setEditParserId] = useState<string>('')
  const [editParserConfig, setEditParserConfig] = useState<
    Record<string, unknown>
  >({})
  const [editParserPriority, setEditParserPriority] = useState<string>('1')
  const [editParserPriorityError, setEditParserPriorityError] = useState(false)
  const [editParserIncludes, setEditParserIncludes] = useState<string[]>([])

  const openEditParser = (id: string) => {
    const found = parserRows.find(p => p.id === id)
    if (!found) return
    setEditParserId(found.id)
    setEditParserConfig(found.config || getDefaultConfigForParser(found.name))
    setEditParserPriority(String(found.priority))
    setEditParserIncludes(parsePatternsString(found.include))
    setIsEditParserOpen(true)
  }

  const handleUpdateParser = () => {
    if (!editParserId) return
    const prio = Number(editParserPriority)
    if (!Number.isInteger(prio)) {
      setEditParserPriorityError(true)
      toast({
        message: `Priority must be an integer between 0 and ${MAX_PRIORITY}`,
        variant: 'destructive',
      })
      return
    }
    if (prio < 0 || prio > MAX_PRIORITY) {
      setEditParserPriorityError(true)
      toast({
        message: `Priority must be between 0 and ${MAX_PRIORITY}`,
        variant: 'destructive',
      })
      return
    }
    const next = parserRows.map(p =>
      p.id === editParserId
        ? {
            ...p,
            config: editParserConfig,
            priority: prio,
            include: patternsToString(editParserIncludes),
          }
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
    markNeedsReprocess()
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
    markNeedsReprocess()
  }

  // Add/Edit/Delete Extractor modals -----------------------------------------
  const [isAddExtractorOpen, setIsAddExtractorOpen] = useState(false)
  const [newExtractorType, setNewExtractorType] = useState<string>('')
  const [newExtractorPriority, setNewExtractorPriority] = useState<string>('1')
  const [newExtractorPriorityError, setNewExtractorPriorityError] =
    useState(false)
  const [newExtractorApplies, setNewExtractorApplies] = useState<string[]>([])
  const [newExtractorConfig, setNewExtractorConfig] = useState<
    Record<string, unknown>
  >({})

  const handleCreateExtractor = () => {
    const name = newExtractorType.trim()
    const prio = Number(newExtractorPriority)
    if (!name || !Number.isFinite(prio)) return
    if (!Number.isInteger(prio) || prio < 0 || prio > MAX_PRIORITY) {
      setNewExtractorPriorityError(true)
      toast({
        message: `Priority must be an integer between 0 and ${MAX_PRIORITY}`,
        variant: 'destructive',
      })
      return
    }
    if (newExtractorApplies.length === 0) {
      toast({
        message: 'Please enter or select an applies pattern for the extractor.',
        variant: 'default',
      })
      return
    }
    const idBase = slugify(name) || 'extractor'
    const id = `${idBase}-${Date.now()}`
    const next: ExtractorRow = {
      id,
      name,
      priority: prio,
      applyTo: patternsToString(newExtractorApplies),
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
    setNewExtractorApplies([])
    markNeedsReprocess()
  }

  const [isEditExtractorOpen, setIsEditExtractorOpen] = useState(false)
  const [editExtractorId, setEditExtractorId] = useState<string>('')
  const [editExtractorPriority, setEditExtractorPriority] =
    useState<string>('1')
  const [editExtractorPriorityError, setEditExtractorPriorityError] =
    useState(false)
  const [editExtractorConfig, setEditExtractorConfig] = useState<
    Record<string, unknown>
  >({})
  const [editExtractorApplies, setEditExtractorApplies] = useState<string[]>([])

  const openEditExtractor = (id: string) => {
    const found = extractorRows.find(e => e.id === id)
    if (!found) return
    setEditExtractorId(found.id)
    setEditExtractorPriority(String(found.priority))
    setEditExtractorConfig(
      found.config || getDefaultConfigForExtractor(found.name)
    )
    setEditExtractorApplies(parsePatternsString(found.applyTo))
    setIsEditExtractorOpen(true)
  }
  const handleUpdateExtractor = () => {
    const prio = Number(editExtractorPriority)
    if (!editExtractorId || !Number.isFinite(prio)) return
    if (!Number.isInteger(prio) || prio < 0 || prio > MAX_PRIORITY) {
      setEditExtractorPriorityError(true)
      toast({
        message: `Priority must be an integer between 0 and ${MAX_PRIORITY}`,
        variant: 'destructive',
      })
      return
    }
    const next = extractorRows.map(e =>
      e.id === editExtractorId
        ? {
            ...e,
            priority: prio,
            config: editExtractorConfig,
            applyTo: patternsToString(editExtractorApplies),
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
    markNeedsReprocess()
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
    markNeedsReprocess()
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
    markNeedsReprocess()
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
    <div
      className={`w-full h-full flex flex-col ${mode === 'designer' ? 'gap-3 pb-20' : ''}`}
    >
      {/* Header / Breadcrumbs */}
      {mode === 'designer' ? (
        <>
          <div className="flex items-center justify-between mb-1">
            <nav className="text-sm md:text-base flex items-center gap-1.5">
              <button
                className="text-teal-600 dark:text-teal-400 hover:underline"
                onClick={() => navigate('/chat/data')}
              >
                Data
              </button>
              <span className="text-muted-foreground px-1">/</span>
              <span className="text-foreground hidden sm:inline">
                Processing strategies
              </span>
              <span className="text-foreground sm:hidden">…</span>
              <span className="text-muted-foreground px-1">/</span>
              <span className="text-foreground">{strategyName}</span>
            </nav>
            <PageActions mode={mode} onModeChange={setMode} />
          </div>
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
        </>
      ) : (
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-2xl">Config editor</h2>
          <PageActions mode={mode} onModeChange={setMode} />
        </div>
      )}

      {mode !== 'designer' ? (
        <div className="flex-1 min-h-0 overflow-hidden">
          <ConfigEditor className="h-full" />
        </div>
      ) : (
        <>
          {/* Used by + Actions */}
          <div className="flex items-center justify-between gap-2 flex-wrap mb-3">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="text-xs text-muted-foreground">Used by</div>
              {assignedDatasets.length === 0 ? (
                <>
                  <div className="text-xs text-muted-foreground">
                    No datasets yet
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsManageOpen(true)}
                  >
                    Assign to dataset(s)
                  </Button>
                </>
              ) : (
                <>
                  {assignedDatasets.map(u => (
                    <Badge
                      key={u}
                      variant="default"
                      size="sm"
                      className="rounded-xl bg-teal-600 text-white dark:bg-teal-500 dark:text-slate-900"
                    >
                      {u}
                    </Badge>
                  ))}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setIsManageOpen(true)}
                  >
                    Manage datasets
                  </Button>
                </>
              )}
            </div>
            <div className="flex items-stretch gap-2 ml-auto w-full sm:w-auto basis-full sm:basis-auto flex-wrap">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsAddParserOpen(true)}
                className="w-[48%] sm:w-auto"
              >
                <Plus className="w-4 h-4" /> Add Parser
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setIsAddExtractorOpen(true)}
                className="w-[48%] sm:w-auto"
              >
                <Plus className="w-4 h-4" /> Add Extractor
              </Button>
              <Button
                variant="outline"
                size="sm"
                className={`w-[48%] sm:w-auto ${
                  canReprocess
                    ? 'bg-slate-900 text-white dark:bg-white dark:text-slate-900 border-transparent'
                    : ''
                }`}
                onClick={async () => {
                  if (!activeProject?.namespace || !activeProject?.project)
                    return
                  const failures: string[] = []
                  for (const n of assignedDatasets) {
                    try {
                      await reIngestMutation.mutateAsync({
                        namespace: activeProject.namespace!,
                        project: activeProject.project!,
                        dataset: n,
                      })
                      toast({
                        message: `Reprocessing ${n}…`,
                        variant: 'default',
                      })
                    } catch (e) {
                      console.error('Failed to start reprocessing', n, e)
                      toast({
                        message: `Failed to start reprocessing ${n}`,
                        variant: 'destructive',
                      })
                      failures.push(n)
                    }
                  }
                  if (failures.length === 0) {
                    clearNeedsReprocess()
                  }
                }}
                disabled={
                  assignedDatasets.length === 0 ||
                  !needsReprocess ||
                  reIngestMutation.isPending
                }
              >
                Reprocess datasets
              </Button>
              {isUniversal ? (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={handleResetDefaults}
                  className="w-[48%] sm:w-auto"
                >
                  Reset to defaults
                </Button>
              ) : null}
            </div>
          </div>

          {/* Manage Datasets Modal */}
          <Dialog open={isManageOpen} onOpenChange={setIsManageOpen}>
            <DialogContent className="sm:max-w-2xl p-0">
              <div className="flex flex-col max-h-[70vh]">
                <DialogHeader className="bg-background p-4 border-b">
                  <DialogTitle className="text-lg text-foreground">
                    Assign to dataset(s)
                  </DialogTitle>
                </DialogHeader>
                <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-2">
                  <div className="text-xs text-muted-foreground">
                    Datasets always have a processing strategy. You can't
                    unassign here. Select additional datasets to assign to this
                    strategy. You can also assign datasets to other strategies
                    from those strategy pages.
                  </div>
                  {!allDatasets || allDatasets.length === 0 ? (
                    <div className="text-sm text-muted-foreground">
                      No datasets found in this project.
                    </div>
                  ) : (
                    <ul className="rounded-md border border-border divide-y divide-border">
                      {allDatasets.map(ds => {
                        const name = ds.name
                        const current = (ds as any).rag_strategy
                        const selected =
                          selectedDatasets.has(name) || current === strategyName
                        const assignedElsewhere =
                          current &&
                          current !== 'auto' &&
                          current !== strategyName
                        return (
                          <li
                            key={name}
                            className={`flex items-center gap-3 px-3 py-3 hover:bg-muted/30 ${
                              current === strategyName
                                ? 'opacity-70 cursor-not-allowed'
                                : 'cursor-pointer'
                            }`}
                            aria-disabled={current === strategyName}
                            onClick={() => {
                              if (current === strategyName) return
                              toggleDataset(name)
                            }}
                          >
                            <Checkbox
                              checked={selected}
                              onCheckedChange={() => toggleDataset(name)}
                              onClick={e => e.stopPropagation()}
                              disabled={current === strategyName}
                              title={
                                current === strategyName
                                  ? 'This dataset cannot be unassigned from this strategy.'
                                  : assignedElsewhere
                                    ? 'This dataset is already assigned to another strategy and cannot be assigned here.'
                                    : undefined
                              }
                            />
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium text-foreground truncate">
                                {name}
                              </div>
                              <div className="text-xs text-muted-foreground truncate">
                                {assignedElsewhere
                                  ? `Currently: ${current}`
                                  : current
                                    ? `Currently: ${current}`
                                    : ''}
                              </div>
                            </div>
                            {selected ? (
                              <Badge
                                variant="default"
                                size="sm"
                                className="rounded-xl"
                              >
                                Selected
                              </Badge>
                            ) : null}
                          </li>
                        )
                      })}
                    </ul>
                  )}
                </div>
                <DialogFooter className="bg-background p-4 border-t flex items-center gap-2">
                  <button
                    className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                    onClick={() => setIsManageOpen(false)}
                    type="button"
                  >
                    Cancel
                  </button>
                  <button
                    className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
                    onClick={saveAssignments}
                    type="button"
                  >
                    Save assignments
                  </button>
                </DialogFooter>
              </div>
            </DialogContent>
          </Dialog>

          {/* Reprocess Confirmation Modal */}
          <Dialog open={isReprocessOpen} onOpenChange={setIsReprocessOpen}>
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle className="text-lg text-foreground">
                  Reprocess now?
                </DialogTitle>
              </DialogHeader>
              <div className="text-sm text-muted-foreground">
                Reprocess these dataset(s) with "{strategyName}" now?
              </div>
              {/* Show per-dataset errors if any */}
              {reprocessErrors && Object.keys(reprocessErrors).length > 0 ? (
                <div className="mt-2 rounded-md border border-destructive bg-destructive/10 p-2 text-sm">
                  <div className="font-semibold text-destructive mb-1">
                    Some datasets failed to reprocess:
                  </div>
                  <ul className="list-disc ml-4">
                    {Object.entries(reprocessErrors).map(
                      ([datasetId, errorMsg]) => (
                        <li key={datasetId}>
                          <span className="font-medium">
                            {getDatasetNameById(datasetId)}:
                          </span>{' '}
                          {String(errorMsg)}
                        </li>
                      )
                    )}
                  </ul>
                </div>
              ) : null}
              {pendingAdded.length > 0 ? (
                <div className="mt-2 rounded-md border border-border bg-accent/10 p-2 text-sm">
                  {pendingAdded.map(n => (
                    <div key={n} className="py-0.5">
                      {n}
                    </div>
                  ))}
                </div>
              ) : null}
              <DialogFooter className="flex items-center gap-2">
                <button
                  className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                  onClick={() => setIsReprocessOpen(false)}
                  type="button"
                >
                  Skip for now
                </button>
                <button
                  className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
                  onClick={async () => {
                    if (!activeProject?.namespace || !activeProject?.project) {
                      setIsReprocessOpen(false)
                      return
                    }
                    const errors: { [datasetId: string]: unknown } = {}
                    await Promise.all(
                      pendingAdded.map(async n => {
                        try {
                          await reIngestMutation.mutateAsync({
                            namespace: activeProject.namespace!,
                            project: activeProject.project!,
                            dataset: n,
                          })
                        } catch (err) {
                          errors[n] = (err as any)?.message || 'Unknown error'
                        }
                      })
                    )
                    if (Object.keys(errors).length > 0) {
                      setReprocessErrors(errors)
                    } else {
                      toast({
                        message: 'Reprocessing started…',
                        variant: 'default',
                      })
                      setIsReprocessOpen(false)
                    }
                  }}
                  type="button"
                >
                  Reprocess now
                </button>
              </DialogFooter>
            </DialogContent>
          </Dialog>

          {/* Processing editors */}
          {/* Tabs header outside card */}
          <Tabs
            activeTab={activeTab}
            setActiveTab={t => setActiveTab(t as 'parsers' | 'extractors')}
            tabs={[
              { id: 'parsers', label: `Parsers (${parserRows.length})` },
              {
                id: 'extractors',
                label: `Extractors (${extractorRows.length})`,
              },
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
                        <div className="text-sm font-medium w-[280px] max-w-[38vw] truncate">
                          {getFriendlyParserName(row.name)}
                        </div>
                        <div className="md:hidden">
                          <Badge
                            variant={getPriorityVariant(row.priority)}
                            size="sm"
                            className="rounded-xl"
                          >
                            Priority: {row.priority}
                          </Badge>
                        </div>
                        <div className="hidden md:block flex-1 min-w-[220px] text-left pr-4">
                          <span className="text-sm text-muted-foreground">
                            Good for:
                          </span>{' '}
                          <span className="text-sm font-medium text-foreground align-bottom whitespace-normal break-words">
                            {summarizeFileTypes(row.include) || '—'}
                          </span>
                        </div>
                        <Badge
                          variant={getPriorityVariant(row.priority)}
                          size="sm"
                          className="rounded-xl mr-2 ml-6 hidden md:inline-flex"
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
                              Parser type:
                            </span>{' '}
                            {row.name}
                          </div>
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
                        <div className="text-sm font-medium w-[280px] max-w-[38vw] truncate">
                          {getFriendlyExtractorName(row.name)}
                        </div>
                        <div className="md:hidden">
                          <Badge
                            variant={getPriorityVariant(row.priority)}
                            size="sm"
                            className="rounded-xl"
                          >
                            Priority: {row.priority}
                          </Badge>
                        </div>
                        <div className="hidden md:block flex-1 min-w-[220px] text-left pr-4">
                          <span className="text-sm text-muted-foreground">
                            Applies to:
                          </span>{' '}
                          <span className="text-sm font-medium text-foreground whitespace-normal break-words">
                            {row.applyTo || 'All (*)'}
                          </span>
                        </div>
                        <Badge
                          variant={getPriorityVariant(row.priority)}
                          size="sm"
                          className="rounded-xl mr-2 ml-6 hidden md:inline-flex"
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

          {activeTab === 'parsers' ? (
            <div className="mt-2 text-sm text-muted-foreground">
              Parsers convert different file formats (PDF, Word, Excel, etc.)
              into text that AI systems can read and understand.
            </div>
          ) : null}
          {activeTab === 'extractors' ? (
            <div className="mt-2 text-sm text-muted-foreground">
              Extractors pull out specific types of information (like dates,
              names, tables, or keywords) from the parsed text to make it more
              useful for AI retrieval and analysis.
            </div>
          ) : null}

          {/* Add Parser Modal */}
          <Dialog open={isAddParserOpen} onOpenChange={setIsAddParserOpen}>
            <DialogContent className="sm:max-w-3xl lg:max-w-4xl p-0 h-[100dvh] sm:h-auto max-h-[100vh] md:max-h-[85vh] overflow-hidden flex flex-col">
              <div className="flex flex-col flex-1 min-h-0">
                <DialogHeader className="bg-background p-4 border-b">
                  <DialogTitle className="text-lg text-foreground">
                    Add parser
                  </DialogTitle>
                </DialogHeader>
                <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-3">
                  <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <Label className="text-xs text-muted-foreground">
                        Parser type
                      </Label>
                      <Label className="text-xs text-muted-foreground">
                        Priority
                      </Label>
                    </div>
                    <div className="mt-1 grid grid-cols-1 md:grid-cols-2 gap-3 items-start">
                      <div>
                        {(() => {
                          const existing = new Set(parserRows.map(p => p.name))
                          const available = ORDERED_PARSER_TYPES.filter(
                            t => !existing.has(t) && PARSER_SCHEMAS[t]
                          )
                          if (!newParserType && available.length > 0) {
                            const first = available[0]
                            setTimeout(() => {
                              setNewParserType(first)
                              setNewParserConfig(
                                getDefaultConfigForParser(first)
                              )
                              setNewParserIncludes(
                                getDefaultIncludePatternsForParser(first)
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
                                  {newParserType
                                    ? getFriendlyParserName(newParserType)
                                    : available[0]
                                      ? getFriendlyParserName(available[0])
                                      : 'No parsers available'}
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
                                        setNewParserIncludes(
                                          getDefaultIncludePatternsForParser(t)
                                        )
                                      }}
                                    >
                                      {getFriendlyParserName(t)}
                                    </DropdownMenuItem>
                                  ))
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                          )
                        })()}
                        {newParserType && PARSER_SCHEMAS[newParserType] ? (
                          <div className="text-xs text-muted-foreground mt-1">
                            {PARSER_SCHEMAS[newParserType].description}
                          </div>
                        ) : null}
                      </div>
                      <div>
                        <Input
                          type="number"
                          className="bg-background w-40"
                          value={newParserPriority}
                          min={0}
                          onChange={e => {
                            const raw = e.target.value
                            setNewParserPriority(raw)
                            const n = Number(raw)
                            setNewParserPriorityError(
                              !Number.isInteger(n) || n < 0 || n > MAX_PRIORITY
                            )
                          }}
                        />
                        {newParserPriorityError ? (
                          <div className="text-xs text-destructive mt-1">
                            Priority must be an integer between 0 and{' '}
                            {MAX_PRIORITY}
                          </div>
                        ) : null}
                      </div>
                    </div>
                  </div>
                  {newParserType && PARSER_SCHEMAS[newParserType] ? (
                    <>
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
                      <PatternEditor
                        label="Included files / file types"
                        description="Specify which files this parser should process - file patterns, extensions, or specific filenames"
                        placeholder="*.pdf, data_*, report.docx"
                        value={newParserIncludes}
                        onChange={setNewParserIncludes}
                        isSuspicious={isSuspiciousPattern}
                      />
                    </>
                  ) : null}
                </div>
                <DialogFooter className="sticky bottom-0 bg-background p-4 border-t flex-col sm:flex-row sm:justify-end gap-2">
                  <button
                    className="px-3 py-2 rounded-md text-sm text-primary hover:underline w-full sm:w-auto"
                    onClick={() => setIsAddParserOpen(false)}
                    type="button"
                  >
                    Cancel
                  </button>
                  <button
                    className={`px-3 py-2 rounded-md text-sm w-full sm:w-auto ${
                      newParserType.trim().length > 0 && !newParserPriorityError
                        ? 'bg-primary text-primary-foreground hover:opacity-90'
                        : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'
                    }`}
                    onClick={handleCreateParser}
                    disabled={
                      newParserType.trim().length === 0 ||
                      newParserPriorityError
                    }
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
              className="sm:max-w-3xl lg:max-w-4xl p-0 h-[100dvh] sm:h-auto max-h-[100vh] md:max-h-[85vh] overflow-hidden flex flex-col"
              onOpenAutoFocus={e => e.preventDefault()}
            >
              <div className="flex flex-col flex-1 min-h-0">
                <DialogHeader className="bg-background p-4 border-b">
                  <DialogTitle className="text-lg text-foreground">
                    Edit parser
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
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-start">
                          <div>
                            <div className="text-sm font-medium text-foreground">
                              {getFriendlyParserName(found.name)}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {schema.description}
                            </div>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground mb-0.5">
                              Priority
                            </Label>
                            <Input
                              type="number"
                              className="bg-background w-40"
                              aria-label="Priority"
                              value={editParserPriority}
                              min={0}
                              onChange={e => {
                                const raw = e.target.value
                                setEditParserPriority(raw)
                                const n = Number(raw)
                                setEditParserPriorityError(
                                  !Number.isInteger(n) ||
                                    n < 0 ||
                                    n > MAX_PRIORITY
                                )
                              }}
                            />
                            {editParserPriorityError ? (
                              <div className="text-xs text-destructive mt-1">
                                Priority must be an integer between 0 and{' '}
                                {MAX_PRIORITY}
                              </div>
                            ) : null}
                          </div>
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
                        <PatternEditor
                          label="Included files / file types"
                          description="Specify which files this parser should process - file patterns, extensions, or specific filenames"
                          placeholder="*.pdf, data_*, report.docx"
                          value={editParserIncludes}
                          onChange={setEditParserIncludes}
                          isSuspicious={isSuspiciousPattern}
                        />
                      </>
                    )
                  })()}
                </div>
                <DialogFooter className="sticky bottom-0 bg-background p-4 border-t flex-col sm:flex-row sm:justify-end gap-2">
                  <button
                    className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm w-full sm:w-auto"
                    onClick={() => {
                      if (!editParserId) return
                      openDeleteParser(editParserId)
                    }}
                    type="button"
                  >
                    Remove
                  </button>
                  <button
                    className="px-3 py-2 rounded-md text-sm text-primary hover:underline w-full sm:w-auto"
                    onClick={() => setIsEditParserOpen(false)}
                    type="button"
                  >
                    Cancel
                  </button>
                  <button
                    className={`px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90 w-full sm:w-auto ${
                      editParserPriorityError
                        ? 'opacity-50 cursor-not-allowed'
                        : ''
                    }`}
                    onClick={handleUpdateParser}
                    type="button"
                    disabled={editParserPriorityError}
                  >
                    Save changes
                  </button>
                </DialogFooter>
              </div>
            </DialogContent>
          </Dialog>

          {/* Delete Parser Modal */}
          <Dialog
            open={isDeleteParserOpen}
            onOpenChange={setIsDeleteParserOpen}
          >
            <DialogContent className="sm:max-w-md">
              <DialogHeader>
                <DialogTitle className="text-lg text-foreground">
                  Delete parser
                </DialogTitle>
              </DialogHeader>
              <div className="text-sm text-muted-foreground">
                Are you sure you want to delete this parser? This action cannot
                be undone.
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
          <Dialog
            open={isAddExtractorOpen}
            onOpenChange={setIsAddExtractorOpen}
          >
            <DialogContent className="sm:max-w-3xl lg:max-w-4xl p-0 h-[100dvh] sm:h-auto max-h-[100vh] md:max-h-[85vh] overflow-hidden flex flex-col">
              <div className="flex flex-col flex-1 min-h-0">
                <DialogHeader className="bg-background p-4 border-b">
                  <DialogTitle className="text-lg text-foreground">
                    Add extractor
                  </DialogTitle>
                </DialogHeader>
                <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-3">
                  <div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <Label className="text-xs text-muted-foreground">
                        Extractor type
                      </Label>
                      <Label className="text-xs text-muted-foreground">
                        Priority
                      </Label>
                    </div>
                    <div className="mt-1 grid grid-cols-1 md:grid-cols-2 gap-3 items-start">
                      <div>
                        {(() => {
                          const existing = new Set(
                            extractorRows.map(e => e.name)
                          )
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
                                  {newExtractorType
                                    ? getFriendlyExtractorName(newExtractorType)
                                    : available[0]
                                      ? getFriendlyExtractorName(available[0])
                                      : 'No extractors available'}
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
                                      {getFriendlyExtractorName(t)}
                                    </DropdownMenuItem>
                                  ))
                                )}
                              </DropdownMenuContent>
                            </DropdownMenu>
                          )
                        })()}
                      </div>
                      <div>
                        <Input
                          type="number"
                          className="bg-background w-40"
                          value={newExtractorPriority}
                          min={0}
                          onChange={e => {
                            const raw = e.target.value
                            setNewExtractorPriority(raw)
                            const n = Number(raw)
                            setNewExtractorPriorityError(
                              !Number.isInteger(n) || n < 0 || n > MAX_PRIORITY
                            )
                          }}
                        />
                        {newExtractorPriorityError ? (
                          <div className="text-xs text-destructive mt-1">
                            Priority must be an integer between 0 and{' '}
                            {MAX_PRIORITY}
                          </div>
                        ) : null}
                      </div>
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
                          className="bg-background w-40"
                          value={newExtractorPriority}
                          min={0}
                          onChange={e => {
                            const raw = e.target.value
                            setNewExtractorPriority(raw)
                            const n = Number(raw)
                            setNewExtractorPriorityError(
                              !Number.isInteger(n) || n < 0 || n > MAX_PRIORITY
                            )
                          }}
                        />
                        {newExtractorPriorityError ? (
                          <div className="text-xs text-destructive mt-1">
                            Priority must be an integer between 0 and{' '}
                            {MAX_PRIORITY}
                          </div>
                        ) : null}
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
                      <PatternEditor
                        label="Applies to files / file types"
                        description="Specify which files this extractor should run on - use '*' for all parsed content, or limit by patterns"
                        placeholder="*, *.pdf, *.txt, data_*"
                        value={newExtractorApplies}
                        onChange={setNewExtractorApplies}
                        isSuspicious={isSuspiciousPattern}
                      />
                    </>
                  ) : null}
                </div>
                <DialogFooter className="sticky bottom-0 bg-background p-4 border-t flex-col sm:flex-row sm:justify-end gap-2">
                  <button
                    className="px-3 py-2 rounded-md text-sm text-primary hover:underline w-full sm:w-auto"
                    onClick={() => setIsAddExtractorOpen(false)}
                    type="button"
                  >
                    Cancel
                  </button>
                  <button
                    className={`px-3 py-2 rounded-md text-sm w-full sm:w-auto ${
                      newExtractorType.trim().length > 0 &&
                      !newExtractorPriorityError
                        ? 'bg-primary text-primary-foreground hover:opacity-90'
                        : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'
                    }`}
                    onClick={handleCreateExtractor}
                    disabled={
                      newExtractorType.trim().length === 0 ||
                      newExtractorPriorityError
                    }
                    type="button"
                  >
                    Add extractor
                  </button>
                </DialogFooter>
              </div>
            </DialogContent>
          </Dialog>

          {/* Edit Extractor Modal (schema-driven) */}
          <Dialog
            open={isEditExtractorOpen}
            onOpenChange={setIsEditExtractorOpen}
          >
            <DialogContent
              className="sm:max-w-3xl lg:max-w-4xl p-0 h-[100dvh] sm:h-auto max-h-[100vh] md:max-h-[85vh] overflow-hidden flex flex-col"
              onOpenAutoFocus={e => e.preventDefault()}
            >
              <div className="flex flex-col flex-1 min-h-0">
                <DialogHeader className="bg-background p-4 border-b">
                  <DialogTitle className="text-lg text-foreground">
                    Edit extractor
                  </DialogTitle>
                </DialogHeader>
                <div className="flex-1 min-h-0 overflow-y-auto p-4 flex flex-col gap-3">
                  {(() => {
                    const found = extractorRows.find(
                      e => e.id === editExtractorId
                    )
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
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 items-start">
                          <div>
                            <div className="text-sm font-medium text-foreground">
                              {getFriendlyExtractorName(found.name)}
                            </div>
                            <div className="text-xs text-muted-foreground">
                              {schema.description}
                            </div>
                          </div>
                          <div>
                            <Label className="text-xs text-muted-foreground mb-0.5">
                              Priority
                            </Label>
                            <Input
                              type="number"
                              className="bg-background w-40"
                              value={editExtractorPriority}
                              min={0}
                              onChange={e => {
                                const raw = e.target.value
                                setEditExtractorPriority(raw)
                                const n = Number(raw)
                                setEditExtractorPriorityError(
                                  Number.isFinite(n) && n < 0
                                )
                              }}
                            />
                            {editExtractorPriorityError ? (
                              <div className="text-xs text-destructive mt-1">
                                Priority cannot be less than 0
                              </div>
                            ) : null}
                          </div>
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
                        <PatternEditor
                          label="Applies to files / file types"
                          description="Specify which files this extractor should run on - use '*' for all parsed content, or limit by patterns"
                          placeholder="*, *.pdf, *.txt, data_*"
                          value={editExtractorApplies}
                          onChange={setEditExtractorApplies}
                          isSuspicious={isSuspiciousPattern}
                        />
                      </>
                    )
                  })()}
                </div>
                <DialogFooter className="sticky bottom-0 bg-background p-4 border-t flex-col sm:flex-row sm:justify-end gap-2">
                  <button
                    className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm w-full sm:w-auto"
                    onClick={() => {
                      if (!editExtractorId) return
                      setDeleteExtractorId(editExtractorId)
                      setIsDeleteExtractorOpen(true)
                    }}
                    type="button"
                  >
                    Remove
                  </button>
                  <button
                    className="px-3 py-2 rounded-md text-sm text-primary hover:underline w-full sm:w-auto"
                    onClick={() => setIsEditExtractorOpen(false)}
                    type="button"
                  >
                    Cancel
                  </button>
                  <button
                    className={`px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90 w-full sm:w-auto ${
                      editExtractorPriorityError
                        ? 'opacity-50 cursor-not-allowed'
                        : ''
                    }`}
                    onClick={handleUpdateExtractor}
                    type="button"
                    disabled={editExtractorPriorityError}
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
                Are you sure you want to delete this extractor? This action
                cannot be undone.
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
        </>
      )}
    </div>
  )
}

export default StrategyView
