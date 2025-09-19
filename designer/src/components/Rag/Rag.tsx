import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import PageActions from '../common/PageActions'
import { Mode } from '../ModeToggle'
import { Badge } from '../ui/badge'
import { defaultStrategies } from './strategies'
import { useToast } from '../ui/toast'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import {
  getStoredArray,
  setStoredArray,
  getStoredSet,
  setStoredSet,
} from '../../utils/storage'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'

type RagStrategy = import('./strategies').RagStrategy

function Rag() {
  const navigate = useNavigate()
  // const [query, setQuery] = useState('') // removed search for strategies
  const { toast } = useToast()
  const [mode, setMode] = useState<Mode>('designer')

  const [metaTick, setMetaTick] = useState(0)
  // Strategy (Processing) modal state
  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editId, _setEditId] = useState<string>('')
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [createName, setCreateName] = useState('')
  const [createDescription, setCreateDescription] = useState('')
  const [copyFromId, setCopyFromId] = useState('')
  const [reembedOpen, setReembedOpen] = useState(false)

  // Project-level configs (Embeddings / Retrievals) -------------------------
  type EmbeddingItem = {
    id: string
    name: string
    isDefault: boolean
    enabled: boolean
  }
  type RetrievalItem = {
    id: string
    name: string
    isDefault: boolean
    enabled: boolean
  }

  const EMB_LIST_KEY = 'lf_project_embeddings'
  const RET_LIST_KEY = 'lf_project_retrievals'

  const getEmbeddings = (): EmbeddingItem[] => {
    const arr = getStoredArray(EMB_LIST_KEY)
    return arr
      .filter(
        (e: any) => e && typeof e.id === 'string' && typeof e.name === 'string'
      )
      .map((e: any) => ({
        id: e.id,
        name: e.name,
        isDefault: Boolean(e.isDefault),
        enabled: typeof e.enabled === 'boolean' ? e.enabled : true,
      })) as EmbeddingItem[]
  }
  const saveEmbeddings = (list: EmbeddingItem[]) =>
    setStoredArray(EMB_LIST_KEY, list)

  const getRetrievals = (): RetrievalItem[] => {
    const arr = getStoredArray(RET_LIST_KEY)
    return arr
      .filter(
        (e: any) => e && typeof e.id === 'string' && typeof e.name === 'string'
      )
      .map((e: any) => ({
        id: e.id,
        name: e.name,
        isDefault: Boolean(e.isDefault),
        enabled: typeof e.enabled === 'boolean' ? e.enabled : true,
      })) as RetrievalItem[]
  }
  const saveRetrievals = (list: RetrievalItem[]) =>
    setStoredArray(RET_LIST_KEY, list)

  // Seed defaults once
  useEffect(() => {
    try {
      if (getEmbeddings().length === 0) {
        saveEmbeddings([
          {
            id: 'default_embeddings',
            name: 'default_embeddings',
            isDefault: true,
            enabled: true,
          },
        ])
      }
      if (getRetrievals().length === 0) {
        saveRetrievals([
          {
            id: 'basic_search',
            name: 'basic_search',
            isDefault: true,
            enabled: true,
          },
        ])
      }
      // Ensure each embedding strategy has a default config for display
      const embeddings = getEmbeddings()
      embeddings.forEach(e => {
        const key = `lf_strategy_embedding_config_${e.id}`
        const raw = localStorage.getItem(key)
        if (!raw) {
          const payload = {
            runtime: 'local',
            provider: 'Ollama (remote)',
            modelId: 'nomic-embed-text',
            baseUrl: 'http://localhost:11434',
            batchSize: 16,
            dimension: 768,
            maxInputTokens: 8192,
            similarity: 'cosine',
            timeout: 60,
          }
          try {
            localStorage.setItem(key, JSON.stringify(payload))
          } catch {}
        }
      })
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Refresh on processing changes (parsers/extractors add/edit/delete)
  useEffect(() => {
    const handler = (_e: Event) => {
      setMetaTick(t => t + 1)
    }
    try {
      window.addEventListener('lf:processingUpdated', handler as EventListener)
    } catch {}
    return () => {
      try {
        window.removeEventListener(
          'lf:processingUpdated',
          handler as EventListener
        )
      } catch {}
    }
  }, [])

  // Derive display strategies with local overrides
  const strategies: RagStrategy[] = defaultStrategies

  // Validate that an object is a well-formed RagStrategy
  const isValidRagStrategy = (s: any): s is RagStrategy => {
    return (
      !!s &&
      typeof s.id === 'string' &&
      typeof s.name === 'string' &&
      typeof s.description === 'string' &&
      typeof s.isDefault === 'boolean' &&
      typeof s.datasetsUsing === 'number'
    )
  }

  const getCustomStrategies = (): RagStrategy[] => {
    try {
      const raw = localStorage.getItem('lf_custom_strategies')
      if (!raw) return []
      const arr = JSON.parse(raw) as RagStrategy[]
      if (!Array.isArray(arr)) return []
      return arr.filter(isValidRagStrategy)
    } catch {
      return []
    }
  }
  const saveCustomStrategies = (list: RagStrategy[]) => {
    try {
      localStorage.setItem('lf_custom_strategies', JSON.stringify(list))
    } catch {}
  }
  const addCustomStrategy = (s: RagStrategy) => {
    const list = getCustomStrategies()
    const exists = list.some(x => x.id === s.id)
    if (exists) {
      toast({ message: 'Strategy id already exists', variant: 'destructive' })
      return
    }
    list.push(s)
    saveCustomStrategies(list)
    setMetaTick(t => t + 1)
  }
  const removeCustomStrategy = (id: string) => {
    const list = getCustomStrategies().filter(s => s.id !== id)
    saveCustomStrategies(list)
  }
  // const resetStrategies = () => { /* no-op in project-level layout */ }
  const getDeletedSet = (): Set<string> => getStoredSet('lf_strategy_deleted')
  const saveDeletedSet = (s: Set<string>) =>
    setStoredSet('lf_strategy_deleted', s)
  const markDeleted = (id: string) => {
    const set = getDeletedSet()
    set.add(id)
    saveDeletedSet(set)
    setMetaTick(t => t + 1)
  }
  const display = useMemo(() => {
    const deleted = getDeletedSet()
    const all = [...strategies, ...getCustomStrategies()]
    return all
      .filter(s => !deleted.has(s.id))
      .map(s => {
        let name = s.name
        let description = s.description
        try {
          const n = localStorage.getItem(`lf_strategy_name_override_${s.id}`)
          if (typeof n === 'string' && n.trim().length > 0) {
            name = n.trim()
          }
          const d = localStorage.getItem(`lf_strategy_description_${s.id}`)
          if (typeof d === 'string' && d.trim().length > 0) {
            description = d.trim()
          }
        } catch {}
        return { ...s, name, description }
      })
  }, [metaTick])

  // Note: processing strategies are derived from `display`

  const getEmbeddingSummary = (id: string): string => {
    try {
      const storedCfg = localStorage.getItem(
        `lf_strategy_embedding_config_${id}`
      )
      const storedModel = localStorage.getItem(
        `lf_strategy_embedding_model_${id}`
      )
      if (storedModel) return storedModel
      if (storedCfg) {
        const parsed = JSON.parse(storedCfg)
        if (parsed?.modelId) return parsed.modelId
      }
    } catch {}
    return 'Not set'
  }
  const getEmbeddingProvider = (id: string): string | null => {
    try {
      const raw = localStorage.getItem(`lf_strategy_embedding_config_${id}`)
      if (!raw) return null
      const cfg = JSON.parse(raw)
      const p = cfg?.provider || cfg?.runtime || null
      if (!p) return null
      if (typeof p === 'string') {
        if (p.includes('Ollama')) return 'Ollama'
        if (p.includes('OpenAI')) return 'OpenAI'
        if (p.includes('Cohere')) return 'Cohere'
        if (p.includes('Google')) return 'Google'
        if (p.includes('Azure')) return 'Azure OpenAI'
        if (p.includes('Bedrock')) return 'AWS Bedrock'
      }
      return String(p)
    } catch {
      return null
    }
  }
  // kept for future use if needed; currently not used after card redesign
  const getEmbeddingDimension = (id: string): number | null => {
    try {
      const raw = localStorage.getItem(`lf_strategy_embedding_config_${id}`)
      if (!raw) return null
      const cfg = JSON.parse(raw)
      return typeof cfg?.dimension === 'number' ? cfg.dimension : null
    } catch {
      return null
    }
  }
  const getEmbeddingLocation = (id: string): string | null => {
    try {
      let raw = localStorage.getItem(`lf_strategy_embedding_config_${id}`)
      if (!raw) {
        raw = localStorage.getItem('lf_last_embedding_provider_config')
      }
      if (!raw) return null
      const cfg = JSON.parse(raw)
      const baseUrl = cfg?.baseUrl || cfg?.base_url
      if (typeof baseUrl === 'string' && baseUrl.trim().length > 0) {
        try {
          const u = new URL(baseUrl)
          return `${u.hostname}${u.port ? `:${u.port}` : ''}`
        } catch {
          return baseUrl
        }
      }
      if (cfg?.endpoint) {
        try {
          const u = new URL(cfg.endpoint)
          return `${u.hostname}${u.port ? `:${u.port}` : ''}`
        } catch {
          return String(cfg.endpoint)
        }
      }
      if (cfg?.region) return String(cfg.region)
      if (cfg?.deployment) return String(cfg.deployment)
      if (
        (cfg?.provider &&
          String(cfg.provider).toLowerCase().includes('ollama')) ||
        cfg?.runtime === 'local'
      ) {
        return 'localhost:11434'
      }
      return null
    } catch {
      return null
    }
  }
  const getEmbeddingRuntime = (id: string): 'Local' | 'Cloud' | null => {
    try {
      const raw = localStorage.getItem(`lf_strategy_embedding_config_${id}`)
      if (!raw) return null
      const cfg = JSON.parse(raw)
      if (cfg?.runtime === 'local') return 'Local'
      if (cfg?.runtime === 'cloud') return 'Cloud'
      const p = String(cfg?.provider || '').toLowerCase()
      if (p.includes('ollama')) return 'Local'
      return 'Cloud'
    } catch {
      return null
    }
  }
  // (removed) summary helper was unused
  const getRetrievalDescription = (_id: string): string => {
    return 'Vector search with configurable extraction pipeline'
  }
  const getRetrievalMeta = (rid: string): string => {
    if (rid.includes('filtered')) return 'MetadataFilteredStrategy'
    return 'BasicSimilarityStrategy'
  }

  // Processing strategy helpers ----------------------------------------------
  const getStrategyDatasets = (sid: string): string[] => {
    try {
      const raw = localStorage.getItem(`lf_strategy_datasets_using_${sid}`)
      if (!raw) return []
      const arr = JSON.parse(raw)
      if (!Array.isArray(arr)) return []
      return arr
        .filter((d: any) => typeof d === 'string' && d.trim().length > 0)
        .map((d: string) => d.trim())
    } catch {
      return []
    }
  }

  const getParsersCount = (sid: string): number => {
    try {
      const raw = localStorage.getItem(`lf_strategy_parsers_${sid}`)
      if (!raw) return 7 // default seed
      const arr = JSON.parse(raw)
      return Array.isArray(arr) ? arr.length : 7
    } catch {
      return 7
    }
  }
  const getExtractorsCount = (sid: string): number => {
    try {
      const raw = localStorage.getItem(`lf_strategy_extractors_${sid}`)
      if (!raw) return 8 // default seed
      const arr = JSON.parse(raw)
      return Array.isArray(arr) ? arr.length : 8
    } catch {
      return 8
    }
  }

  // Create handlers for new cards --------------------------------------------
  // const createEmbedding = () => {
  //   const name = prompt('Enter embedding strategy name')?.trim()
  //   if (!name) return
  //   const slug = name
  //     .toLowerCase()
  //     .replace(/[^a-z0-9]+/g, '-')
  //     .replace(/^-+|-+$/g, '')
  //   const id = `emb-${slug || Date.now()}`
  //   const list = getEmbeddings()
  //   list.push({ id, name, isDefault: list.length === 0, enabled: true })
  //   saveEmbeddings(list)
  //   setMetaTick(t => t + 1)
  //   navigate(`/chat/rag/${id}/change-embedding`)
  // }
  // const createRetrieval = () => {
  //   const name = prompt('Enter retrieval strategy name')?.trim()
  //   if (!name) return
  //   const slug = name
  //     .toLowerCase()
  //     .replace(/[^a-z0-9]+/g, '-')
  //     .replace(/^-+|-+$/g, '')
  //   const id = `ret-${slug || Date.now()}`
  //   const list = getRetrievals()
  //   list.push({ id, name, isDefault: list.length === 0, enabled: true })
  //   saveRetrievals(list)
  //   setMetaTick(t => t + 1)
  //   navigate(`/chat/rag/${id}/retrieval`)
  // }

  const setDefaultEmbedding = (id: string) => {
    const list = getEmbeddings().map(e => ({ ...e, isDefault: e.id === id }))
    saveEmbeddings(list)
    setMetaTick(t => t + 1)
  }
  const setDefaultRetrieval = (id: string) => {
    const list = getRetrievals().map(r => ({ ...r, isDefault: r.id === id }))
    saveRetrievals(list)
    setMetaTick(t => t + 1)
  }
  const toggleRetrievalEnabled = (id: string) => {
    const list = getRetrievals().map(r =>
      r.id === id ? { ...r, enabled: !r.enabled } : r
    )
    saveRetrievals(list)
    setMetaTick(t => t + 1)
  }
  const duplicateRetrieval = (id: string) => {
    const list = getRetrievals()
    const found = list.find(r => r.id === id)
    if (!found) return
    const base = `${found.name} (copy)`
    const slug = base
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
    const newId = `ret-${slug}-${Date.now()}`
    list.push({ ...found, id: newId, name: base, isDefault: false })
    saveRetrievals(list)
    setMetaTick(t => t + 1)
  }

  const sortedEmbeddings = useMemo(() => {
    const list = getEmbeddings()
    return [...list].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [metaTick])
  const sortedRetrievals = useMemo(() => {
    const list = getRetrievals()
    return [...list].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [metaTick])
  const embeddingCount = sortedEmbeddings.length
  const retrievalCount = sortedRetrievals.length

  return (
    <>
      <div className="w-full flex flex-col gap-4 pb-32">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-2xl">
            {mode === 'designer' ? 'RAG' : 'Config editor'}
          </h2>
          <PageActions mode={mode} onModeChange={setMode} />
        </div>
        {mode !== 'designer' ? (
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="text-sm text-muted-foreground mb-1">
              Edit config
            </div>
            <div className="rounded-md overflow-hidden">
              <div className="h-[70vh]">
                <ConfigEditor />
              </div>
            </div>
          </div>
        ) : (
          <>
            {/* Page helper subtitle removed per request */}
            {/* Processing strategy - title outside card */}
            <div className="flex items-center justify-between mt-3 mb-1">
              <div>
                <div className="text-sm font-medium">Processing strategies</div>
                <div className="h-1" />
                <div className="text-xs text-muted-foreground">
                  Processing strategies are applied to datasets.
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  size="sm"
                  onClick={() => {
                    setCreateName('')
                    setCreateDescription('')
                    setCopyFromId('')
                    setIsCreateOpen(true)
                  }}
                >
                  Create new
                </Button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {display.map(s => {
                const datasets = getStrategyDatasets(s.id)
                const usageLabel = (() => {
                  if (datasets.length === 1) return datasets[0]
                  if (datasets.length > 1)
                    return `${datasets.length} datasets using`
                  if (
                    typeof s.datasetsUsing === 'number' &&
                    s.datasetsUsing > 1
                  )
                    return `${s.datasetsUsing} datasets using`
                  if (s.datasetsUsing === 1) return '1 dataset using'
                  return ''
                })()
                return (
                  <div
                    key={s.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${display.length === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() => navigate(`/chat/rag/${s.id}`)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/rag/${s.id}`)
                      }
                    }}
                  >
                    <div className="absolute top-2 right-2">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                            onClick={e => e.stopPropagation()}
                          >
                            <FontIcon type="overflow" className="w-4 h-4" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="min-w-[12rem] w-[12rem]"
                        >
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              _setEditId(s.id)
                              setEditName(s.name)
                              setEditDescription(s.description)
                              setIsEditOpen(true)
                            }}
                          >
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              setCreateName(`${s.name} (copy)`) // prefill
                              setCreateDescription(s.description)
                              setCopyFromId(s.id)
                              setIsCreateOpen(true)
                            }}
                          >
                            Duplicate
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onClick={e => {
                              e.stopPropagation()
                              const ok = confirm(
                                'Delete this processing strategy?'
                              )
                              if (!ok) return
                              // Mark as deleted and remove if custom
                              try {
                                removeCustomStrategy(s.id)
                                const set = getDeletedSet()
                                set.add(s.id)
                                saveDeletedSet(set)
                                setMetaTick(t => t + 1)
                              } catch {}
                            }}
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="text-sm font-medium">{s.name}</div>
                    <div className="text-xs text-primary text-left w-fit">
                      {s.description ||
                        'Unified processor for PDFs, Word docs, CSVs, Markdown, and text files.'}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap pt-0.5">
                      {usageLabel ? (
                        <Badge
                          variant="secondary"
                          size="sm"
                          className="rounded-xl bg-teal-700 text-white dark:bg-teal-400 dark:text-gray-900"
                        >
                          {usageLabel}
                        </Badge>
                      ) : null}
                      <Badge
                        variant="secondary"
                        size="sm"
                        className="rounded-xl"
                      >
                        Active
                      </Badge>
                    </div>
                    <div className="flex items-center justify-between mt-1">
                      <div className="text-xs text-muted-foreground">
                        {getParsersCount(s.id)} parsers •{' '}
                        {getExtractorsCount(s.id)} extractors
                      </div>
                      <Button
                        variant="outline"
                        size="sm"
                        className="px-3 h-7"
                        onClick={e => {
                          e.stopPropagation()
                          navigate(`/chat/rag/${s.id}`)
                        }}
                      >
                        Configure
                      </Button>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Embedding and Retrieval strategies - title outside card */}
            <div className="text-sm font-medium mb-1 mt-6">
              Project Embedding and retrieval strategies
            </div>
            {/* Embeddings card */}
            <div className="rounded-lg border border-border bg-card p-4 pt-3">
              {/* Embeddings block header with Add new inline */}
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-foreground font-medium">
                  Embedding strategies ({embeddingCount})
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigate('/chat/rag/add-embedding')}
                >
                  Add new
                </Button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sortedEmbeddings.map(ei => (
                  <div
                    key={ei.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${ei.enabled ? '' : 'opacity-70'} ${embeddingCount === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() =>
                      navigate(`/chat/rag/${ei.id}/change-embedding`)
                    }
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/rag/${ei.id}/change-embedding`)
                      }
                    }}
                  >
                    <div className="absolute top-2 right-2">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                            onClick={e => e.stopPropagation()}
                          >
                            <FontIcon type="overflow" className="w-4 h-4" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="min-w-[12rem] w-[12rem]"
                        >
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              navigate(`/chat/rag/${ei.id}/change-embedding`)
                            }}
                          >
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              setDefaultEmbedding(ei.id)
                              setReembedOpen(true)
                            }}
                          >
                            Make default
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onClick={e => {
                              e.stopPropagation()
                              const ok = confirm(
                                'Remove this embedding strategy?'
                              )
                              if (!ok) return
                              // hard delete config and list entry
                              try {
                                localStorage.removeItem(
                                  `lf_strategy_embedding_config_${ei.id}`
                                )
                                localStorage.removeItem(
                                  `lf_strategy_embedding_model_${ei.id}`
                                )
                              } catch {}
                              let list = getEmbeddings().filter(
                                x => x.id !== ei.id
                              )
                              if (
                                list.length > 0 &&
                                !list.some(x => x.isDefault)
                              ) {
                                list[0].isDefault = true
                              }
                              saveEmbeddings(list)
                              setMetaTick(t => t + 1)
                            }}
                          >
                            Remove
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="text-sm font-medium">
                          {getEmbeddingProvider(ei.id) || 'Provider'}
                        </div>
                        <div className="text-base font-semibold font-mono truncate">
                          {getEmbeddingSummary(ei.id)}
                        </div>
                        <div className="text-xs text-muted-foreground w-full truncate">
                          {(() => {
                            const loc = getEmbeddingLocation(ei.id)
                            return loc ? loc : ''
                          })()}
                        </div>
                      </div>
                      <div className="flex items-center gap-2 flex-wrap" />
                    </div>
                    <div className="text-xs text-muted-foreground truncate">
                      {ei.name}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap justify-between pt-0.5">
                      <div className="flex items-center gap-2 flex-wrap">
                        {(() => {
                          const dim = getEmbeddingDimension(ei.id)
                          return dim ? (
                            <Badge
                              variant="secondary"
                              size="sm"
                              className="rounded-xl"
                            >
                              {dim}-d
                            </Badge>
                          ) : null
                        })()}
                        {(() => {
                          const runtime = getEmbeddingRuntime(ei.id)
                          return runtime ? (
                            <Badge
                              variant="secondary"
                              size="sm"
                              className="rounded-xl"
                            >
                              {runtime}
                            </Badge>
                          ) : null
                        })()}
                        {ei.isDefault ? (
                          <Badge
                            variant="default"
                            size="sm"
                            className="rounded-xl"
                          >
                            Default
                          </Badge>
                        ) : null}
                      </div>
                      <div className="ml-auto">
                        <Button
                          variant="outline"
                          size="sm"
                          className="px-3 h-7"
                          onClick={e => {
                            e.stopPropagation()
                            navigate(`/chat/rag/${ei.id}/change-embedding`)
                          }}
                        >
                          Configure
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Retrievals card */}
            <div className="rounded-lg border border-border bg-card p-4 pt-3 mt-3">
              <div className="flex items-center justify-between mb-3">
                <div className="text-sm text-foreground font-medium">
                  Retrieval strategies ({retrievalCount})
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => navigate('/chat/rag/add-retrieval')}
                >
                  Add new
                </Button>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sortedRetrievals.map(ri => (
                  <div
                    key={ri.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${ri.enabled ? '' : 'opacity-70'} ${retrievalCount === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() => navigate(`/chat/rag/${ri.id}/retrieval`)}
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/rag/${ri.id}/retrieval`)
                      }
                    }}
                  >
                    <div className="absolute top-2 right-2">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
                            onClick={e => e.stopPropagation()}
                          >
                            <FontIcon type="overflow" className="w-4 h-4" />
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent
                          align="end"
                          className="min-w-[12rem] w-[12rem]"
                        >
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              const name = prompt(
                                'Rename retrieval strategy',
                                ri.name
                              )?.trim()
                              if (!name) return
                              const list = getRetrievals().map(x =>
                                x.id === ri.id ? { ...x, name } : x
                              )
                              saveRetrievals(list)
                              setMetaTick(t => t + 1)
                            }}
                          >
                            Rename
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              duplicateRetrieval(ri.id)
                            }}
                          >
                            Duplicate
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              setDefaultRetrieval(ri.id)
                            }}
                          >
                            Set as default
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              toggleRetrievalEnabled(ri.id)
                            }}
                          >
                            {ri.enabled ? 'Disable' : 'Enable'}
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onClick={e => {
                              e.stopPropagation()
                              const ok = confirm(
                                'Delete this retrieval strategy?'
                              )
                              if (!ok) return
                              let list = getRetrievals().filter(
                                x => x.id !== ri.id
                              )
                              if (
                                list.length > 0 &&
                                !list.some(x => x.isDefault)
                              ) {
                                list[0].isDefault = true
                              }
                              saveRetrievals(list)
                              setMetaTick(t => t + 1)
                            }}
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="text-sm font-medium">{ri.name}</div>
                    <div className="text-xs text-primary text-left w-full truncate">
                      {getRetrievalDescription(ri.id)}
                    </div>
                    <div className="text-xs text-muted-foreground w-full truncate">
                      {getRetrievalMeta(ri.id)}
                    </div>
                    <div className="flex items-center gap-2 flex-wrap justify-between pt-0.5">
                      <div className="flex items-center gap-2 flex-wrap">
                        {ri.isDefault ? (
                          <Badge
                            variant="default"
                            size="sm"
                            className="rounded-xl"
                          >
                            Default
                          </Badge>
                        ) : null}
                      </div>
                      <div className="ml-auto">
                        <Button
                          variant="outline"
                          size="sm"
                          className="px-3 h-7"
                          onClick={e => {
                            e.stopPropagation()
                            navigate(`/chat/rag/${ri.id}/retrieval`)
                          }}
                        >
                          Configure
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>

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
          <DialogFooter className="flex items-center justify-between gap-2">
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={() => {
                if (!editId) return
                const ok = confirm(
                  'Are you sure you want to delete this strategy?'
                )
                if (ok) {
                  try {
                    localStorage.removeItem(
                      `lf_strategy_name_override_${editId}`
                    )
                    localStorage.removeItem(`lf_strategy_description_${editId}`)
                  } catch {}
                  removeCustomStrategy(editId)
                  markDeleted(editId)
                  setIsEditOpen(false)
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
                  if (!editId || editName.trim().length === 0) return
                  try {
                    localStorage.setItem(
                      `lf_strategy_name_override_${editId}`,
                      editName.trim()
                    )
                    localStorage.setItem(
                      `lf_strategy_description_${editId}`,
                      editDescription
                    )
                  } catch {}
                  setIsEditOpen(false)
                  setMetaTick(t => t + 1)
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

      {/* Create Strategy Modal */}
      <Dialog
        open={isCreateOpen}
        onOpenChange={open => {
          setIsCreateOpen(open)
          if (!open) {
            setCreateName('')
            setCreateDescription('')
            setCopyFromId('')
          }
        }}
      >
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Create new processing strategy
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
                value={createName}
                onChange={e => setCreateName(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Copy from existing
              </label>
              <select
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                value={copyFromId}
                onChange={e => {
                  const v = e.target.value
                  setCopyFromId(v)
                  const found = display.find(x => x.id === v)
                  if (found) {
                    setCreateDescription(found.description || '')
                    if (createName.trim().length === 0) {
                      setCreateName(`${found.name} (copy)`)
                    }
                  }
                }}
              >
                <option value="">None</option>
                {display.map(s => (
                  <option key={s.id} value={s.id}>
                    {s.name}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <textarea
                rows={4}
                className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground"
                placeholder="Add a brief description"
                value={createDescription}
                onChange={e => setCreateDescription(e.target.value)}
              />
            </div>
          </div>
          <DialogFooter className="flex items-center justify-between gap-2">
            <button
              className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
              onClick={() => setIsCreateOpen(false)}
              type="button"
            >
              Cancel
            </button>
            <button
              className={`px-3 py-2 rounded-md text-sm ${createName.trim().length > 0 ? 'bg-primary text-primary-foreground hover:opacity-90' : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'}`}
              onClick={() => {
                const name = createName.trim()
                if (name.length === 0) return
                const slugify = (str: string) =>
                  str
                    .toLowerCase()
                    .replace(/[^a-z0-9]+/g, '-')
                    .replace(/^-+|-+$/g, '')
                const baseId = `custom-${slugify(name)}`
                const existingIds = new Set(
                  [...defaultStrategies, ...getCustomStrategies()].map(
                    s => s.id
                  )
                )
                let newId = baseId
                if (existingIds.has(newId)) {
                  newId = `${baseId}-${Date.now()}`
                }
                const newStrategy: RagStrategy = {
                  id: newId,
                  name,
                  description: createDescription,
                  isDefault: false,
                  datasetsUsing: 0,
                }
                addCustomStrategy(newStrategy)
                try {
                  localStorage.setItem(
                    `lf_strategy_name_override_${newId}`,
                    name
                  )
                  localStorage.setItem(
                    `lf_strategy_description_${newId}`,
                    createDescription
                  )
                } catch {}
                setIsCreateOpen(false)
                setCreateName('')
                setCreateDescription('')
                setCopyFromId('')
                setMetaTick(t => t + 1)
                toast({ message: 'Strategy created', variant: 'default' })
                // Stay on home so the new card is visible immediately
              }}
              disabled={createName.trim().length === 0}
              type="button"
            >
              Create
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Re-embed confirmation modal (after setting default embedding) */}
      <Dialog open={reembedOpen} onOpenChange={setReembedOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Re-embed project data?
            </DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            To keep your project running smoothly, this change requires
            re-embedding project data. Would you like to proceed now?
          </div>
          <DialogFooter>
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={() => {
                setReembedOpen(false)
                toast({
                  message: 'Default strategy updated',
                  variant: 'default',
                })
              }}
              type="button"
            >
              I'll do it later
            </button>
            <button
              className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
              onClick={() => {
                setReembedOpen(false)
                toast({
                  message: 'Default strategy updated',
                  variant: 'default',
                })
              }}
              type="button"
            >
              Yes, proceed with re-embed
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

export default Rag
