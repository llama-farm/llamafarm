import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import PageActions from '../common/PageActions'
import { Mode } from '../ModeToggle'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { useToast } from '../ui/toast'

function Versions() {
  const navigate = useNavigate()
  const { toast } = useToast()
  const [mode, setMode] = useState<Mode>('designer')
  const [expanded, setExpanded] = useState<Record<string, boolean>>({})
  const [toDelete, setToDelete] = useState<VersionRow | null>(null)
  const [toRestore, setToRestore] = useState<VersionRow | null>(null)

  type VersionRow = {
    id: string
    name: string
    description: string
    date: string
    isCurrent?: boolean
    size: string
    path: string
    models?: {
      name: string
      provider?: string
      version?: string
      notes?: string
    }[]
    dataChunks?: { dataset: string; chunks: number }[]
    ragStrategies?: { id: string; name: string }[]
    status?: 'packaging' | 'success' | 'stopped'
  }

  const [rows, setRows] = useState<VersionRow[]>(() => {
    try {
      const raw = localStorage.getItem('lf_versions')
      if (raw) {
        const arr = JSON.parse(raw)
        if (Array.isArray(arr)) return arr
      }
    } catch {}
    return [
      {
        id: 'v1.0.1',
        name: 'v1.0.1',
        description: 'Bug fixes and prompt tweaks for staging rollout',
        date: '2025-09-09 10:22 AM',
        isCurrent: true,
        size: '2.34GB',
        path: '/Users/you/Downloads/',
        models: [
          { name: 'gpt-4o-mini', provider: 'OpenAI', version: '2025-09-01' },
          { name: 'text-embedding-3-large', provider: 'OpenAI' },
        ],
        dataChunks: [
          { dataset: 'dataset_telemetry_2025.csv', chunks: 1248 },
          { dataset: 'support_docs_pdf', chunks: 842 },
        ],
        ragStrategies: [
          { id: 'pdf-simple', name: 'PDF Simple' },
          { id: 'hybrid-bm25-embed', name: 'Hybrid BM25 + Embedding' },
        ],
      },
      {
        id: 'v1.0.0',
        name: 'v1.0.0',
        description: 'Initial packaged version for team handoff',
        date: '2025-09-07 4:18 PM',
        size: '2.31GB',
        path: '/Users/you/Downloads/',
        models: [
          { name: 'gpt-4o-mini', provider: 'OpenAI' },
          { name: 'text-embedding-3-small', provider: 'OpenAI' },
        ],
        dataChunks: [{ dataset: 'dataset_telemetry_2025.csv', chunks: 1207 }],
        ragStrategies: [{ id: 'pdf-simple', name: 'PDF Simple' }],
      },
      {
        id: 'v0.9.5',
        name: 'v0.9.5',
        description: 'Model config iteration with new dataset',
        date: '2025-09-03 1:04 PM',
        size: '2.28GB',
        path: '/Users/you/Downloads/',
        models: [
          { name: 'gpt-4o-mini', provider: 'OpenAI' },
          { name: 'text-embedding-3-small', provider: 'OpenAI' },
        ],
        dataChunks: [{ dataset: 'support_docs_pdf', chunks: 740 }],
        ragStrategies: [
          { id: 'hybrid-bm25-embed', name: 'Hybrid BM25 + Embedding' },
        ],
      },
    ]
  })

  useEffect(() => {
    try {
      localStorage.setItem('lf_versions', JSON.stringify(rows))
    } catch {}
  }, [rows])

  // Listen for package completion updates
  useEffect(() => {
    const handler = () => {
      try {
        const raw = localStorage.getItem('lf_versions')
        if (raw) {
          const arr = JSON.parse(raw)
          if (Array.isArray(arr)) setRows(arr)
        }
      } catch {}
    }
    window.addEventListener('lf_versions_updated', handler as EventListener)
    return () => {
      window.removeEventListener(
        'lf_versions_updated',
        handler as EventListener
      )
    }
  }, [])

  const toggleRow = (id: string) =>
    setExpanded(prev => ({ ...prev, [id]: !prev[id] }))

  const handleRestore = (v: VersionRow) => {
    setRows(prev => {
      // Mark only this version as current
      const updated = prev.map(item => ({
        ...item,
        isCurrent: item.id === v.id,
      }))
      // Move restored version to the top
      const target = updated.find(item => item.id === v.id)!
      const others = updated.filter(item => item.id !== v.id)
      return [target, ...others]
    })
  }

  const handleDelete = (v: VersionRow) => {
    setToDelete(v)
  }

  return (
    <div className="w-full flex flex-col gap-3 pb-20">
      {/* Breadcrumb + Actions */}
      <div className="flex items-center justify-between mb-1">
        <nav className="text-sm md:text-base flex items-center gap-1.5">
          <button
            className="text-teal-600 dark:text-teal-400 hover:underline"
            onClick={() => navigate('/chat/dashboard')}
          >
            Dashboard
          </button>
          <span className="text-muted-foreground px-1">/</span>
          <span className="text-foreground">Project versions</span>
        </nav>
        <PageActions mode={mode} onModeChange={setMode} />
      </div>

      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg md:text-xl font-medium">Project versions</h2>
        <div className="flex items-center gap-2">
          {/* Placeholder for future action buttons */}
        </div>
      </div>

      {/* Versions table */}
      <section className="rounded-md overflow-hidden border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted">
            <tr>
              <th className="text-left px-4 py-2 w-[40%]">Name</th>
              <th className="text-left px-4 py-2">Package data</th>
              <th className="text-left px-4 py-2 whitespace-nowrap">
                Package date
              </th>
              <th className="text-right px-4 py-2 w-[1%]">Actions</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((v, idx) => (
              <>
                <tr
                  key={v.id}
                  className="bg-card border-t border-border cursor-pointer hover:bg-accent/20"
                  onClick={() => toggleRow(v.id)}
                >
                  <td className="px-4 py-3 align-top">
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        className="p-1 rounded-md hover:bg-accent/40 text-muted-foreground"
                        aria-label={
                          expanded[v.id] ? 'Collapse row' : 'Expand row'
                        }
                        onClick={e => {
                          e.stopPropagation()
                          toggleRow(v.id)
                        }}
                      >
                        <FontIcon
                          type="chevron-down"
                          className={`w-4 h-4 transition-transform ${expanded[v.id] ? 'rotate-180' : ''}`}
                        />
                      </button>
                      <div className="flex items-center gap-2">
                        <div className="text-sm text-foreground font-medium">
                          {v.name}
                        </div>
                        {v.isCurrent && idx === 0 ? (
                          <span className="px-2 py-0.5 rounded-2xl text-xs border border-teal-200 text-teal-700 bg-teal-50 dark:border-teal-800 dark:text-teal-300 dark:bg-teal-900/30">
                            current
                          </span>
                        ) : null}
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top text-muted-foreground">
                    {v.status === 'packaging' ? (
                      <button
                        type="button"
                        className="text-xs text-teal-700 dark:text-teal-300 underline"
                        onClick={e => {
                          e.stopPropagation()
                          // Bring up the package modal to foreground via event
                          try {
                            window.dispatchEvent(
                              new CustomEvent('lf_open_package_modal')
                            )
                          } catch {}
                        }}
                      >
                        Packaging… (view)
                      </button>
                    ) : v.status === 'stopped' ? (
                      <div className="text-xs">Packaging stopped</div>
                    ) : (
                      <div className="line-clamp-2 max-w-[52ch]">
                        {v.description}
                      </div>
                    )}
                  </td>
                  <td className="px-4 py-3 align-top whitespace-nowrap text-muted-foreground">
                    {v.date}
                  </td>
                  <td className="px-4 py-3 align-top">
                    <div className="flex justify-end">
                      <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                          <button
                            type="button"
                            className="h-8 w-8 inline-flex items-center justify-center rounded-md border border-input hover:bg-accent/30"
                            aria-label="More actions"
                            onClick={e => e.stopPropagation()}
                          >
                            ⋯
                          </button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent align="end">
                          <DropdownMenuItem
                            onClick={() => setToRestore(v)}
                            disabled={v.isCurrent && idx === 0}
                          >
                            Restore
                          </DropdownMenuItem>
                          <DropdownMenuSeparator />
                          <DropdownMenuItem
                            onClick={() => handleDelete(v)}
                            disabled={v.isCurrent && idx === 0}
                            className="text-red-600 dark:text-red-300"
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>
                  </td>
                </tr>
                {expanded[v.id] ? (
                  <tr className="bg-background border-t border-border">
                    <td colSpan={4} className="px-4 py-3">
                      <div className="rounded-md border border-border bg-card p-3 text-xs">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <div className="text-muted-foreground">File</div>
                            <div className="font-mono text-foreground">{`${v.name}.llamafarm (${v.size})`}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">
                              Saved to
                            </div>
                            <div className="truncate text-foreground">
                              {v.path}
                            </div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Notes</div>
                            <div className="text-foreground">
                              {v.description}
                            </div>
                          </div>
                        </div>
                        <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
                          <div>
                            <div className="mb-1 font-medium text-foreground">
                              Models
                            </div>
                            {v.models && v.models.length > 0 ? (
                              <ul className="list-disc pl-5 space-y-1">
                                {v.models.map((m, i) => (
                                  <li key={i}>
                                    <span className="text-foreground">
                                      {m.name}
                                    </span>
                                    {m.provider ? (
                                      <span className="text-muted-foreground">{` · ${m.provider}`}</span>
                                    ) : null}
                                    {m.version ? (
                                      <span className="text-muted-foreground">{` · ${m.version}`}</span>
                                    ) : null}
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <div className="text-muted-foreground">—</div>
                            )}
                          </div>
                          <div>
                            <div className="mb-1 font-medium text-foreground">
                              Data chunks
                            </div>
                            {v.dataChunks && v.dataChunks.length > 0 ? (
                              <ul className="list-disc pl-5 space-y-1">
                                {v.dataChunks.map((d, i) => (
                                  <li key={i}>
                                    <span className="font-mono text-foreground">
                                      {d.dataset}
                                    </span>
                                    <span className="text-muted-foreground">{` · ${d.chunks} chunks`}</span>
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <div className="text-muted-foreground">—</div>
                            )}
                          </div>
                          <div>
                            <div className="mb-1 font-medium text-foreground">
                              RAG strategies
                            </div>
                            {v.ragStrategies && v.ragStrategies.length > 0 ? (
                              <ul className="list-disc pl-5 space-y-1">
                                {v.ragStrategies.map((r, i) => (
                                  <li key={i}>
                                    <span className="text-foreground">
                                      {r.name}
                                    </span>
                                    <span className="text-muted-foreground">{` · ${r.id}`}</span>
                                  </li>
                                ))}
                              </ul>
                            ) : (
                              <div className="text-muted-foreground">—</div>
                            )}
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                ) : null}
              </>
            ))}
          </tbody>
        </table>
      </section>
      {/* Delete confirmation dialog */}
      <Dialog
        open={!!toDelete}
        onOpenChange={open => !open && setToDelete(null)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete version</DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            {toDelete ? (
              <>
                Are you sure you want to delete{' '}
                <span className="font-mono text-foreground">
                  {toDelete.name}
                </span>
                ? This action cannot be undone.
              </>
            ) : null}
          </div>
          <DialogFooter>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm border border-input bg-background hover:bg-accent/30"
              onClick={() => setToDelete(null)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm bg-destructive text-destructive-foreground hover:opacity-90"
              onClick={() => {
                if (toDelete) {
                  setRows(prev => prev.filter(r => r.id !== toDelete.id))
                  setExpanded(prev => {
                    const next = { ...prev }
                    delete next[toDelete.id]
                    return next
                  })
                  setToDelete(null)
                }
              }}
            >
              Delete
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Restore confirmation dialog */}
      <Dialog
        open={!!toRestore}
        onOpenChange={open => !open && setToRestore(null)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Restore version</DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            {toRestore ? (
              <>
                Are you sure you want to restore{' '}
                <span className="font-mono text-foreground">
                  {toRestore.name}
                </span>
                ? This will set it as the current working version.
              </>
            ) : null}
          </div>
          <DialogFooter>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm border border-input bg-background hover:bg-accent/30"
              onClick={() => setToRestore(null)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="px-3 py-2 rounded-md text-sm bg-primary text-primary-foreground hover:opacity-90"
              onClick={() => {
                if (toRestore) {
                  handleRestore(toRestore)
                  toast({ message: `${toRestore.name} restored successfully` })
                  setToRestore(null)
                }
              }}
            >
              Restore
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Versions
