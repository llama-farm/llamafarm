import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { SampleProject } from '../../data/sampleProjects'
import {
  useExamples,
  useImportExampleProject,
  useImportExampleData,
} from '../../hooks/useExamples'
import SampleCard from './SampleCard'
import SamplePreviewModal from './SamplePreviewModal'
import ImportSampleProjectModal from './ImportSampleProjectModal'
import ImportSampleDataModal from './ImportSampleDataModal'
import { useProjectModalContext } from '../../contexts/ProjectModalContext'
import { useToast } from '../ui/toast'
// import { useCreateProject } from '../../hooks/useProjects'
import { getCurrentNamespace } from '../../utils/namespaceUtils'
import { setActiveProject } from '../../utils/projectUtils'
import { useProjects } from '../../hooks/useProjects'
import { getProjectsList } from '../../utils/projectConstants'

function SampleProjects() {
  const navigate = useNavigate()
  const projectModal = useProjectModalContext()
  const { toast } = useToast()
  // Retained for offline fallback, but not used in server-backed import flows
  // const createProjectMutation = useCreateProject()
  const namespace = getCurrentNamespace()
  const { data: projectsResponse } = useProjects(namespace)
  const importProject = useImportExampleProject()
  const importData = useImportExampleData()
  const [search, setSearch] = useState('')
  const [selected, setSelected] = useState<SampleProject | null>(null)
  const [open, setOpen] = useState(false)
  const [importOpen, setImportOpen] = useState(false)
  const [importing, setImporting] = useState(false)
  const [dataOpen, setDataOpen] = useState(false)
  const [sortKey, setSortKey] = useState<'' | 'projectSize' | 'dataSize'>('')
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc')
  const [modelFilter, setModelFilter] = useState<'all' | string>('all')
  const [itemsToShow, setItemsToShow] = useState(12)
  const {
    data: examplesData,
    isLoading: isExamplesLoading,
    isError: isExamplesError,
    refetch: refetchExamples,
  } = useExamples()

  // Cycle sort: first click desc, second asc, third off
  const cycleSort = (key: 'projectSize' | 'dataSize') => {
    if (sortKey !== key) {
      setSortKey(key)
      setSortDir('desc')
      return
    }
    if (sortDir === 'desc') {
      setSortDir('asc')
    } else {
      setSortKey('')
      setSortDir('desc')
    }
  }

  const sizeToBytes = (v?: string): number => {
    if (!v) {
      console.warn('sizeToBytes: input is empty or undefined')
      return 0
    }
    const match = v.trim().match(/([0-9]+(?:\.[0-9]+)?)\s*(KB|MB|GB|TB)/i)
    if (!match) {
      console.warn(`sizeToBytes: malformed input "${v}"`)
      return 0
    }
    const value = parseFloat(match[1])
    const unit = match[2].toUpperCase()
    let pow: number
    switch (unit) {
      case 'KB':
        pow = 1
        break
      case 'MB':
        pow = 2
        break
      case 'GB':
        pow = 3
        break
      case 'TB':
        pow = 4
        break
      default:
        console.warn(`sizeToBytes: unsupported unit "${unit}" in input "${v}"`)
        return 0
    }
    return value * Math.pow(1024, pow)
  }

  const exampleProjects: SampleProject[] = useMemo(() => {
    const ex = examplesData?.examples || []
    return ex.map((e: any) => ({
      id: e.id,
      slug: e.slug || e.id,
      title: e.title,
      description: e.description || '',
      updatedAt: e.updated_at || new Date().toISOString(),
      downloadSize: e.project_size_human || '-',
      dataSize: e.data_size_human || '-',
      primaryModel: e.primaryModel,
      models: e.primaryModel ? [e.primaryModel] : [],
      tags: e.tags,
      datasetCount: e.dataset_count ?? undefined,
    }))
  }, [examplesData])

  // Use only dynamic examples. Do NOT fall back to legacy hardcoded list.
  const allProjects = useMemo(() => exampleProjects, [exampleProjects])

  const uniqueModels = useMemo(() => {
    const set = new Set<string>()
    allProjects.forEach(p => p.primaryModel && set.add(p.primaryModel))
    return Array.from(set)
  }, [allProjects])

  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase()
    let base = term
      ? allProjects.filter(p => {
          const inTitle = p.title.toLowerCase().includes(term)
          const inTags = (p.tags || []).some(t =>
            t.toLowerCase().includes(term)
          )
          return inTitle || inTags
        })
      : allProjects

    if (modelFilter !== 'all') {
      base = base.filter(
        p => (p.primaryModel || '').toLowerCase() === modelFilter.toLowerCase()
      )
    }

    if (sortKey) {
      base = [...base].sort((a, b) => {
        const aVal =
          sortKey === 'projectSize'
            ? sizeToBytes(a.downloadSize)
            : sizeToBytes(a.dataSize)
        const bVal =
          sortKey === 'projectSize'
            ? sizeToBytes(b.downloadSize)
            : sizeToBytes(b.dataSize)
        return sortDir === 'asc' ? aVal - bVal : bVal - aVal
      })
    }

    return base.slice(0, itemsToShow)
  }, [search, modelFilter, sortKey, sortDir, itemsToShow, allProjects])

  const existingProjects = useMemo(
    () => getProjectsList(projectsResponse),
    [projectsResponse]
  )

  return (
    <div className="w-full flex flex-col gap-3 pb-20 px-6 md:px-8 pt-16 md:pt-20 max-w-6xl mx-auto">
      {/* Breadcrumb + Actions */}
      <div className="flex items-center justify-between mb-1">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/')}
          >
            LlamaFarm home
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">Sample projects</span>
        </nav>
        <div className="flex items-center gap-2">
          <button
            className="px-3 py-2 rounded-md bg-primary text-primary-foreground hover:opacity-90"
            onClick={projectModal.openCreateModal}
          >
            New blank project
          </button>
          <button
            className="px-3 py-2 rounded-md border border-input hover:bg-accent/30"
            onClick={() => (navigate(-1), undefined)}
          >
            Back
          </button>
        </div>
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-lg md:text-xl font-medium">Sample Projects</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Explore and import sample projects and sample project data.
          </p>
        </div>
      </div>

      {/* Search + placeholder filters */}
      <div className="mb-4">
        <div className="w-full flex items-center bg-card rounded-lg px-3 py-2 border border-input">
          <FontIcon type="search" className="w-4 h-4 text-foreground" />
          <input
            className="w-full bg-transparent border-none focus:outline-none px-2 text-sm text-foreground"
            placeholder="Search projects or tags (e.g. maintenance)"
            value={search}
            onChange={e => setSearch(e.target.value)}
            aria-label="Search sample projects"
          />
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted-foreground">Sort</label>
            <button
              type="button"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-2xl border text-xs transition-colors ${
                sortKey === 'projectSize'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background text-foreground border-input hover:bg-accent/30'
              }`}
              onClick={() => cycleSort('projectSize')}
            >
              <span>Project size</span>
              <FontIcon
                type="chevron-down"
                className={`w-3.5 h-3.5 transition-transform ${
                  sortKey === 'projectSize' && sortDir === 'asc'
                    ? 'rotate-180'
                    : ''
                }`}
              />
            </button>
            <button
              type="button"
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-2xl border text-xs transition-colors ${
                sortKey === 'dataSize'
                  ? 'bg-primary text-primary-foreground border-primary'
                  : 'bg-background text-foreground border-input hover:bg-accent/30'
              }`}
              onClick={() => cycleSort('dataSize')}
            >
              <span>Data size</span>
              <FontIcon
                type="chevron-down"
                className={`w-3.5 h-3.5 transition-transform ${
                  sortKey === 'dataSize' && sortDir === 'asc'
                    ? 'rotate-180'
                    : ''
                }`}
              />
            </button>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted-foreground">Model</label>
            <select
              className="px-2 py-1.5 rounded-md border border-input bg-background text-xs"
              value={modelFilter}
              onChange={e => setModelFilter(e.target.value as any)}
            >
              <option value="all">All</option>
              {uniqueModels.map(m => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* List / Empty states */}
      <div className="flex flex-col gap-3">
        {isExamplesLoading ? (
          <div className="text-sm text-muted-foreground">
            Loading sample projects…
          </div>
        ) : isExamplesError ? (
          <div className="rounded-lg border border-input bg-card p-6 text-center">
            <div className="text-foreground font-medium">
              Could not load sample projects
            </div>
            <div className="mt-1 text-sm text-muted-foreground">
              Make sure the API server is running on localhost:8000 and try
              again.
            </div>
            <div className="mt-4 flex items-center justify-center gap-2">
              <button
                type="button"
                className="px-3 py-2 rounded-md border border-input hover:bg-accent/30"
                onClick={() => refetchExamples()}
              >
                Retry
              </button>
            </div>
          </div>
        ) : filtered.length === 0 ? (
          <div className="rounded-lg border border-input bg-card p-6 text-center">
            <div className="text-foreground font-medium">
              No sample projects available
            </div>
            <div className="mt-1 text-sm text-muted-foreground">
              Start the LlamaFarm server or adjust filters, then refresh.
            </div>
            <div className="mt-4 flex items-center justify-center gap-2">
              <button
                type="button"
                className="px-3 py-2 rounded-md border border-input hover:bg-accent/30"
                onClick={() => refetchExamples()}
              >
                Refresh
              </button>
            </div>
          </div>
        ) : (
          filtered.map(s => (
            <SampleCard
              key={s.id}
              sample={s}
              onPreview={sp => {
                setSelected(sp)
                setOpen(true)
              }}
            />
          ))
        )}
      </div>

      {/* Load more for simple pagination */}
      {filtered.length >= itemsToShow && (
        <div className="mt-2">
          <button
            type="button"
            className="px-3 py-2 rounded-md border border-input hover:bg-accent/30"
            onClick={() => setItemsToShow(prev => prev + 12)}
          >
            Load more
          </button>
        </div>
      )}

      {/* Preview modal */}
      <SamplePreviewModal
        open={open}
        sample={selected}
        onOpenChange={setOpen}
        onImportProject={sp => {
          setSelected(sp)
          setOpen(false)
          setImportOpen(true)
        }}
        onImportData={sp => {
          setSelected(sp)
          setOpen(false)
          setDataOpen(true)
        }}
      />

      {/* Import modal */}
      <ImportSampleProjectModal
        open={importOpen}
        sample={selected}
        isSubmitting={importing}
        onOpenChange={setImportOpen}
        onSubmit={async ({ name }) => {
          try {
            setImporting(true)
            if (!selected) throw new Error('No sample selected')
            // Call server to import full project from manifest
            await importProject.mutateAsync({
              exampleId: selected.id,
              namespace,
              name,
              process: true,
            })
            setActiveProject(name)
            try {
              const raw = localStorage.getItem('lf_custom_projects')
              const arr: string[] = raw ? JSON.parse(raw) : []
              if (!arr.includes(name)) {
                localStorage.setItem(
                  'lf_custom_projects',
                  JSON.stringify([...arr, name])
                )
              }
            } catch (storageErr) {
              console.warn(
                'Failed to persist custom project to localStorage',
                storageErr
              )
            }
            setImportOpen(false)
            toast({ message: `Project "${name}" created` })
            navigate('/chat/dashboard')
          } catch (err) {
            // Fallback to local-only creation when API is unavailable
            try {
              const raw = localStorage.getItem('lf_custom_projects')
              const arr: string[] = raw ? JSON.parse(raw) : []
              if (!arr.includes(name)) {
                localStorage.setItem(
                  'lf_custom_projects',
                  JSON.stringify([...arr, name])
                )
              }
            } catch (storageErr) {
              console.warn(
                'Failed to persist custom project to localStorage (offline)',
                storageErr
              )
            }
            setActiveProject(name)
            setImportOpen(false)
            toast({ message: `Project "${name}" created (local)` })
            navigate('/chat/dashboard')
          } finally {
            setImporting(false)
          }
        }}
      />

      {/* Import data only modal */}
      <ImportSampleDataModal
        open={dataOpen}
        sample={selected}
        isSubmitting={importing}
        projects={existingProjects}
        onOpenChange={setDataOpen}
        onSubmit={async payload => {
          try {
            setImporting(true)
            if (!selected) throw new Error('No sample selected')
            const targetProjectName = payload.name

            if (payload.target === 'new') {
              // Create and import data in one go using server import
              await importProject.mutateAsync({
                exampleId: selected.id,
                namespace,
                name: targetProjectName,
                process: true,
              })
            } else {
              // Import data into existing project
              await importData.mutateAsync({
                exampleId: selected.id,
                namespace,
                project: targetProjectName,
                include_strategies: payload.includeStrategies,
                process: true,
              })
            }

            setActiveProject(targetProjectName)
            setDataOpen(false)
            toast({
              message: payload.includeStrategies
                ? 'Sample data imported successfully'
                : 'Raw data imported successfully',
            })
            navigate('/chat/data')
          } catch (e) {
            // Offline/local fallback
            try {
              const raw = localStorage.getItem('lf_custom_projects')
              const arr: string[] = raw ? JSON.parse(raw) : []
              if (!arr.includes(payload.name)) {
                localStorage.setItem(
                  'lf_custom_projects',
                  JSON.stringify([...arr, payload.name])
                )
              }
            } catch (storageErr) {
              console.warn(
                'Failed to persist custom project to localStorage for local fallback',
                storageErr
              )
            }
            setActiveProject(payload.name)
            setDataOpen(false)
            toast({
              message: payload.includeStrategies
                ? 'Sample data imported successfully (local)'
                : 'Raw data imported successfully (local)',
            })
            navigate('/chat/data')
          } finally {
            setImporting(false)
          }
        }}
      />
    </div>
  )
}

export default SampleProjects
