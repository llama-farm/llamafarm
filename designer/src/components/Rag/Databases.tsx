import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import PageActions from '../common/PageActions'
import { Badge } from '../ui/badge'
import { useToast } from '../ui/toast'
import { useModeWithReset } from '../../hooks/useModeWithReset'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { useActiveProject } from '../../hooks/useActiveProject'
import { useProject, useUpdateProject } from '../../hooks/useProjects'
import { useListDatasets } from '../../hooks/useDatasets'
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from '../ui/tooltip'

type Database = {
  name: string
  type?: string
  config?: Record<string, any>
}

function Databases() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [mode, setMode] = useModeWithReset('designer')
  const activeProject = useActiveProject()
  const { data: projectResp } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject
  )

  const { data: datasetsResp } = useListDatasets(
    activeProject?.namespace || '',
    activeProject?.project || '',
    { enabled: !!activeProject }
  )

  const updateProject = useUpdateProject()

  const [reembedOpen, setReembedOpen] = useState(false)

  // Database management -------------------------------------------------
  const databases = useMemo((): Database[] => {
    // Read databases from project config (single source of truth)
    const cfgDbs = (projectResp as any)?.project?.config?.rag?.databases
    if (Array.isArray(cfgDbs) && cfgDbs.length > 0) {
      return cfgDbs.map((db: any) => ({
        name: db?.name || 'unnamed',
        type: db?.type,
        config: db?.config,
      }))
    }

    // No fallback - config is the single source of truth
    return []
  }, [projectResp])

  // Active database state (namespaced per project)
  const projectKey = useMemo(() => {
    const ns = activeProject?.namespace || 'global'
    const proj = activeProject?.project || 'global'
    return `${ns}__${proj}`
  }, [activeProject?.namespace, activeProject?.project])

  const ACTIVE_DB_KEY = useMemo(
    () => `lf_ui_${projectKey}_active_database`,
    [projectKey]
  )

  const [activeDatabase, setActiveDatabase] = useState<string>(() => {
    try {
      const stored = localStorage.getItem(ACTIVE_DB_KEY)
      return stored || 'main_database'
    } catch {
      return 'main_database'
    }
  })

  // Persist active database selection
  useEffect(() => {
    try {
      localStorage.setItem(ACTIVE_DB_KEY, activeDatabase)
    } catch {}
  }, [activeDatabase, ACTIVE_DB_KEY])

  // Reload selection when project changes (validate against available list)
  useEffect(() => {
    try {
      const stored = localStorage.getItem(ACTIVE_DB_KEY)
      if (stored) {
        // ensure it's in the current databases list
        const exists = databases.some(d => d.name === stored)
        setActiveDatabase(exists ? stored : activeDatabase)
      }
    } catch {}
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ACTIVE_DB_KEY, databases])

  // Ensure active database exists in the list
  useEffect(() => {
    if (
      databases.length > 0 &&
      !databases.find(db => db.name === activeDatabase)
    ) {
      setActiveDatabase(databases[0].name)
    }
  }, [databases, activeDatabase])

  // Connected datasets for the active database
  const connectedDatasets = useMemo(() => {
    if (!datasetsResp?.datasets) return []
    return datasetsResp.datasets.filter(
      (d: any) => d.database === activeDatabase
    )
  }, [datasetsResp, activeDatabase])

  // Database-level configs (Embeddings / Retrievals) -------------------------
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



  // Current config database (for embedding strategies)
  const currentConfigDb = useMemo(() => {
    const cfgDbs = (projectResp as any)?.project?.config?.rag?.databases
    if (!Array.isArray(cfgDbs)) return null
    return cfgDbs.find((d: any) => d?.name === activeDatabase) || null
  }, [projectResp, activeDatabase])

  // Embeddings from config (single source of truth)
  const configEmbeddings: EmbeddingItem[] = useMemo(() => {
    if (!currentConfigDb) return []
    const list = Array.isArray(currentConfigDb.embedding_strategies)
      ? (currentConfigDb.embedding_strategies as any[])
      : []
    const def = currentConfigDb.default_embedding_strategy
    return list.map((e: any) => ({
      id: String(e?.name ?? 'embedding'),
      name: String(e?.name ?? 'embedding'),
      isDefault: def ? String(def) === String(e?.name) : false,
      enabled: true,
    }))
  }, [currentConfigDb])

  // Retrievals from config (single source of truth)
  const configRetrievals: RetrievalItem[] = useMemo(() => {
    if (!currentConfigDb) return []
    const list = Array.isArray(currentConfigDb.retrieval_strategies)
      ? (currentConfigDb.retrieval_strategies as any[])
      : []
    const def = currentConfigDb.default_retrieval_strategy
    return list.map((r: any) => ({
      id: String(r?.name ?? 'retrieval'),
      name: String(r?.name ?? 'retrieval'),
      isDefault: def ? String(def) === String(r?.name) : false,
      enabled: true,
    }))
  }, [currentConfigDb])

  // Seed defaults if no strategies exist in config
  useEffect(() => {
    // Only seed if we have a database but no strategies
    if (!currentConfigDb) return
    if (configEmbeddings.length > 0 && configRetrievals.length > 0) return

    const needsSeeding = async () => {
      try {
        const currentConfig = projectResp?.project?.config
        if (!currentConfig) return

        const rag = currentConfig.rag || {}
        const databases = rag.databases || []
        const dbIndex = databases.findIndex((d: any) => d.name === activeDatabase)
        if (dbIndex === -1) return

        let needsUpdate = false
        const db = databases[dbIndex]

        // Seed embedding strategies if missing
        if (!db.embedding_strategies || db.embedding_strategies.length === 0) {
          db.embedding_strategies = [{
            name: 'default_embeddings',
            type: 'UniversalEmbedder',
            config: {
              model: 'nomic-embed-text',
              dimension: 768,
              batch_size: 16,
              timeout: 60,
            }
          }]
          db.default_embedding_strategy = 'default_embeddings'
          needsUpdate = true
        }

        // Seed retrieval strategies if missing
        if (!db.retrieval_strategies || db.retrieval_strategies.length === 0) {
          db.retrieval_strategies = [{
            name: 'basic_search',
            type: 'BasicSimilarityStrategy',
            config: {
              top_k: 10,
              distance_metric: 'cosine'
            }
          }]
          db.default_retrieval_strategy = 'basic_search'
          needsUpdate = true
        }

        if (needsUpdate) {
          const updatedConfig = {
            ...currentConfig,
            rag: {
              ...rag,
              databases
            }
          }

          await updateProject.mutateAsync({
            namespace: activeProject!.namespace,
            projectId: activeProject!.project,
            request: { config: updatedConfig }
          })
        }
      } catch (error) {
        console.error('Failed to seed default strategies:', error)
      }
    }

    needsSeeding()
  }, [currentConfigDb, configEmbeddings.length, configRetrievals.length, activeDatabase, projectResp, updateProject, activeProject])

  const getEmbeddingSummary = (id: string): string => {
    try {
      const storedCfg = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
      const storedModel = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_model_${id}`
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
      const raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
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
      const raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
      if (!raw) return null
      const cfg = JSON.parse(raw)
      return typeof cfg?.dimension === 'number' ? cfg.dimension : null
    } catch {
      return null
    }
  }
  const getEmbeddingLocation = (id: string): string | null => {
    try {
      let raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
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
      const raw = localStorage.getItem(
        `lf_db_${activeDatabase}_embedding_config_${id}`
      )
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
  // Deprecated: old usage metric helper replaced by per-strategy dataset badges

  // Save embedding strategies back to config
  const saveEmbeddingStrategiesToConfig = async (embeddingStrategies: EmbeddingItem[]) => {
    if (!activeProject || !projectResp?.project?.config) return

    try {
      const currentConfig = projectResp.project.config
      const rag = currentConfig.rag || {}
      const databases = rag.databases || []

      // Find the current database
      const dbIndex = databases.findIndex((d: any) => d.name === activeDatabase)
      if (dbIndex === -1) return

      // Convert EmbeddingItem[] to config format
      const configStrategies = embeddingStrategies.map(e => ({
        name: e.name,
        type: 'UniversalEmbedder', // Default type, can be made configurable later
        config: {
          model: 'nomic-embed-text', // Default model, can be made configurable
          dimension: 768,
          batch_size: 16,
          timeout: 60,
        }
      }))

      // Update the database with new embedding strategies
      databases[dbIndex] = {
        ...databases[dbIndex],
        embedding_strategies: configStrategies,
        default_embedding_strategy: embeddingStrategies.find(e => e.isDefault)?.name || embeddingStrategies[0]?.name
      }

      const updatedConfig = {
        ...currentConfig,
        rag: {
          ...rag,
          databases
        }
      }

      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: updatedConfig }
      })

      toast({ message: 'Embedding strategies saved to config', variant: 'default' })
    } catch (error) {
      console.error('Failed to save embedding strategies:', error)
      toast({ message: 'Failed to save embedding strategies', variant: 'destructive' })
    }
  }

  // Save retrieval strategies back to config
  const saveRetrievalStrategiesToConfig = async (retrievalStrategies: RetrievalItem[]) => {
    if (!activeProject || !projectResp?.project?.config) return

    try {
      const currentConfig = projectResp.project.config
      const rag = currentConfig.rag || {}
      const databases = rag.databases || []

      // Find the current database
      const dbIndex = databases.findIndex((d: any) => d.name === activeDatabase)
      if (dbIndex === -1) return

      // Convert RetrievalItem[] to config format
      const configStrategies = retrievalStrategies.map(r => ({
        name: r.name,
        type: 'BasicSimilarityStrategy', // Default type, can be made configurable later
        config: {
          top_k: 10,
          distance_metric: 'cosine'
        }
      }))

      // Update the database with new retrieval strategies
      databases[dbIndex] = {
        ...databases[dbIndex],
        retrieval_strategies: configStrategies,
        default_retrieval_strategy: retrievalStrategies.find(r => r.isDefault)?.name || retrievalStrategies[0]?.name
      }

      const updatedConfig = {
        ...currentConfig,
        rag: {
          ...rag,
          databases
        }
      }

      await updateProject.mutateAsync({
        namespace: activeProject.namespace,
        projectId: activeProject.project,
        request: { config: updatedConfig }
      })

      toast({ message: 'Retrieval strategies saved to config', variant: 'default' })
    } catch (error) {
      console.error('Failed to save retrieval strategies:', error)
      toast({ message: 'Failed to save retrieval strategies', variant: 'destructive' })
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
  //   navigate(`/chat/databases/${id}/change-embedding`)
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
  //   navigate(`/chat/databases/${id}/retrieval`)
  // }

  const setDefaultEmbedding = async (id: string) => {
    const list = configEmbeddings.map(e => ({ ...e, isDefault: e.id === id }))
    // Save to config (single source of truth)
    await saveEmbeddingStrategiesToConfig(list)
  }
  const setDefaultRetrieval = async (id: string) => {
    const list = configRetrievals.map(r => ({ ...r, isDefault: r.id === id }))
    // Save to config (single source of truth)
    await saveRetrievalStrategiesToConfig(list)
  }
  const toggleRetrievalEnabled = async (id: string) => {
    const list = configRetrievals.map(r =>
      r.id === id ? { ...r, enabled: !r.enabled } : r
    )
    // Save to config (single source of truth)
    await saveRetrievalStrategiesToConfig(list)
  }
  const duplicateRetrieval = async (id: string) => {
    const found = configRetrievals.find(r => r.id === id)
    if (!found) return
    const base = `${found.name} (copy)`
    const slug = base
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-+|-+$/g, '')
    const newId = `ret-${slug}-${Date.now()}`
    const newList = [...configRetrievals, { ...found, id: newId, name: base, isDefault: false }]
    // Save to config (single source of truth)
    await saveRetrievalStrategiesToConfig(newList)
  }

  const sortedEmbeddings = useMemo(() => {
    // Use config data as single source of truth
    return [...configEmbeddings].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [configEmbeddings])
  const sortedRetrievals = useMemo(() => {
    // Use config data as single source of truth
    return [...configRetrievals].sort((a, b) => {
      if (a.isDefault !== b.isDefault) return a.isDefault ? -1 : 1
      if (a.enabled !== b.enabled) return a.enabled ? -1 : 1
      return a.name.localeCompare(b.name)
    })
  }, [configRetrievals])
  const embeddingCount = sortedEmbeddings.length
  const retrievalCount = sortedRetrievals.length

  return (
    <>
      <div
        className={`w-full h-full flex flex-col ${mode === 'designer' ? 'gap-4 pb-32' : ''}`}
      >
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <h2 className="text-2xl">
              {mode === 'designer' ? 'Databases' : 'Config editor'}
            </h2>
            {mode === 'designer' && (
              <TooltipProvider>
                <Tooltip>
                  <TooltipTrigger asChild>
                    <button className="text-muted-foreground hover:text-foreground transition-colors">
                      <FontIcon type="info" className="w-4 h-4" />
                    </button>
                  </TooltipTrigger>
                  <TooltipContent className="max-w-xs">
                    <p className="text-sm">
                      Databases store and organize your embedded data for AI
                      search. Create multiple databases to handle different
                      content types with specialized embedding and retrieval
                      strategies.
                    </p>
                  </TooltipContent>
                </Tooltip>
              </TooltipProvider>
            )}
          </div>
          <PageActions mode={mode} onModeChange={setMode} />
        </div>
        {mode === 'designer' && databases.length > 0 && (
          <div className="mb-2 border-b border-border">
            <div className="flex items-center gap-1">
              {databases.map(db => (
                <button
                  key={db.name}
                  onClick={() => setActiveDatabase(db.name)}
                  className={`px-4 py-2 font-medium transition-colors relative ${
                    activeDatabase === db.name
                      ? 'text-primary'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {db.name}
                  {activeDatabase === db.name && (
                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-primary" />
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
        {mode !== 'designer' ? (
          <div className="flex-1 min-h-0 overflow-hidden pb-6">
            <ConfigEditor className="h-full" />
          </div>
        ) : (
          <>
            {/* Database name header */}
            <div className="text-xl font-semibold mb-4 mt-2">
              {activeDatabase}
            </div>

            {/* Embedding and Retrieval strategies - title outside card */}
            <div className="text-sm font-medium mb-1">
              Project Embedding and retrieval strategies
            </div>
            {/* Embeddings card */}
            <div className="rounded-lg border border-border bg-card p-4 pt-3">
              {/* Embeddings block header with Add new inline */}
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm text-foreground font-medium">
                  Embedding strategies ({embeddingCount})
                </div>
                <span className="text-xs text-muted-foreground">
                  Managed by config
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sortedEmbeddings.map(ei => (
                  <div
                    key={ei.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${ei.enabled ? '' : 'opacity-70'} ${embeddingCount === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() =>
                      navigate(`/chat/databases/${ei.id}/change-embedding`)
                    }
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/databases/${ei.id}/change-embedding`)
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
                                navigate(
                                  `/chat/databases/${ei.id}/change-embedding`
                                )
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
                              onClick={async (e) => {
                                e.stopPropagation()
                                const ok = confirm(
                                  'Remove this embedding strategy?'
                                )
                                if (!ok) return
                                // Remove from config
                                let list = configEmbeddings.filter(
                                  x => x.id !== ei.id
                                )
                                if (
                                  list.length > 0 &&
                                  !list.some(x => x.isDefault)
                                ) {
                                  list[0].isDefault = true
                                }
                                // Save to config (single source of truth)
                                await saveEmbeddingStrategiesToConfig(list)
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
                            navigate(
                              `/chat/databases/${ei.id}/change-embedding`
                            )
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
                <span className="text-xs text-muted-foreground">
                  Managed by config
                </span>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {sortedRetrievals.map(ri => (
                  <div
                    key={ri.id}
                    className={`w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors ${ri.enabled ? '' : 'opacity-70'} ${retrievalCount === 1 ? 'md:col-span-2' : ''}`}
                    onClick={() =>
                      navigate(`/chat/databases/${ri.id}/retrieval`)
                    }
                    role="button"
                    tabIndex={0}
                    onKeyDown={e => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.preventDefault()
                        navigate(`/chat/databases/${ri.id}/retrieval`)
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
                              onClick={async (e) => {
                                e.stopPropagation()
                                const name = prompt(
                                  'Rename retrieval strategy',
                                  ri.name
                                )?.trim()
                                if (!name) return
                                const list = configRetrievals.map(x =>
                                  x.id === ri.id ? { ...x, name } : x
                                )
                                // Save to config (single source of truth)
                                await saveRetrievalStrategiesToConfig(list)
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
                              onClick={async (e) => {
                                e.stopPropagation()
                                const ok = confirm(
                                  'Delete this retrieval strategy?'
                                )
                                if (!ok) return
                                let list = configRetrievals.filter(
                                  x => x.id !== ri.id
                                )
                                if (
                                  list.length > 0 &&
                                  !list.some(x => x.isDefault)
                                ) {
                                  list[0].isDefault = true
                                }
                                // Save to config (single source of truth)
                                await saveRetrievalStrategiesToConfig(list)
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
                            navigate(`/chat/databases/${ri.id}/retrieval`)
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

            {/* Connected datasets section */}
            <div className="text-sm font-medium mb-1 mt-6">
              Connected datasets
            </div>
            <div className="rounded-lg border border-border bg-card p-4">
              {connectedDatasets.length === 0 ? (
                <div className="text-sm text-muted-foreground text-center py-4">
                  No datasets connected to this database yet
                </div>
              ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border">
                        <th className="text-left py-2 px-2 font-medium">
                          Name
                        </th>
                        <th className="text-left py-2 px-2 font-medium">
                          Files
                        </th>
                        <th className="text-left py-2 px-2 font-medium">
                          Processing Strategy
                        </th>
                        <th className="w-20"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {connectedDatasets.map((dataset: any) => (
                        <tr
                          key={dataset.name}
                          className="border-b border-border last:border-0 hover:bg-accent/20"
                        >
                          <td className="py-2 px-2">{dataset.name}</td>
                          <td className="py-2 px-2">
                            {Array.isArray(dataset.files)
                              ? dataset.files.length
                              : 0}
                          </td>
                          <td className="py-2 px-2">
                            <Badge
                              variant="default"
                              size="sm"
                              className="rounded-xl"
                            >
                              {dataset.data_processing_strategy || 'default'}
                            </Badge>
                          </td>
                          <td className="py-2 px-2">
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 px-3"
                              onClick={e => {
                                e.stopPropagation()
                                navigate(`/chat/data/${dataset.name}`)
                              }}
                            >
                              View
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}
      </div>

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

export default Databases
