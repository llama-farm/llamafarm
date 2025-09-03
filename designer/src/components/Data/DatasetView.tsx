import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import SearchInput from '../ui/search-input'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogClose,
} from '../ui/dialog'
import { Textarea } from '../ui/textarea'
import { useToast } from '../ui/toast'

type Dataset = {
  id: string
  name: string
  lastRun: string | Date
  embedModel: string
  numChunks: number
  processedPercent: number
  version: string
  description?: string
}

function DatasetView() {
  const navigate = useNavigate()
  const { datasetId } = useParams()
  const { toast } = useToast()

  const [dataset, setDataset] = useState<Dataset | null>(null)
  const datasetName = useMemo(
    () => dataset?.name || datasetId || 'dataset',
    [dataset?.name, datasetId]
  )

  type RawFile = {
    id: string
    name: string
    size: number
    lastModified: number
    type?: string
  }

  const [files, setFiles] = useState<RawFile[]>([])
  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [searchValue, setSearchValue] = useState('')
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [uploadStatus, setUploadStatus] = useState<
    Record<string, 'processing' | 'done'>
  >({})
  const [isDeleteOpen, setIsDeleteOpen] = useState(false)
  const [fileToDelete, setFileToDelete] = useState<RawFile | null>(null)

  useEffect(() => {
    try {
      const storedFiles = localStorage.getItem('lf_raw_files')
      const storedAssignments = localStorage.getItem('lf_file_assignments')
      if (!storedFiles || !storedAssignments || !datasetId) return
      const all: RawFile[] = JSON.parse(storedFiles)
      const assignments: Record<string, string[]> =
        JSON.parse(storedAssignments)
      const filtered = all.filter(r =>
        (assignments[r.id] ?? []).includes(datasetId)
      )
      setFiles(filtered)
    } catch {}
  }, [datasetId])

  useEffect(() => {
    try {
      const storedDatasets = localStorage.getItem('lf_datasets')
      if (!storedDatasets || !datasetId) return
      const list = JSON.parse(storedDatasets) as Dataset[]
      const current = list.find(d => d.id === datasetId) || null
      setDataset(current)
    } catch {}
  }, [datasetId])

  const openEdit = () => {
    setEditName(dataset?.name ?? '')
    setEditDescription(dataset?.description ?? '')
    setIsEditOpen(true)
  }

  const saveDatasets = (arr: Dataset[]) => {
    try {
      localStorage.setItem(
        'lf_datasets',
        JSON.stringify(
          arr.map(d => ({ ...d, lastRun: new Date(d.lastRun).toISOString() }))
        )
      )
    } catch {}
  }

  const handleSaveEdit = () => {
    if (!dataset || !datasetId) return
    try {
      const stored = localStorage.getItem('lf_datasets')
      const list = stored ? (JSON.parse(stored) as Dataset[]) : []
      const updated = list.map(d =>
        d.id === datasetId
          ? {
              ...d,
              name: editName.trim() || d.name,
              description: editDescription,
            }
          : d
      )
      saveDatasets(updated)
      const current = updated.find(d => d.id === datasetId) || null
      setDataset(current)
      setIsEditOpen(false)
    } catch {}
  }

  const handleDelete = () => {
    if (!datasetId) return
    try {
      const stored = localStorage.getItem('lf_datasets')
      const list = stored ? (JSON.parse(stored) as Dataset[]) : []
      const updated = list.filter(d => d.id !== datasetId)
      saveDatasets(updated)
      const storedAssignments = localStorage.getItem('lf_file_assignments')
      if (storedAssignments) {
        const assignments: Record<string, string[]> =
          JSON.parse(storedAssignments)
        const cleaned: Record<string, string[]> = {}
        for (const [k, arr] of Object.entries(assignments)) {
          const next = arr.filter(id => id !== datasetId)
          if (next.length > 0) cleaned[k] = next
        }
        localStorage.setItem('lf_file_assignments', JSON.stringify(cleaned))
      }
      navigate('/chat/data')
    } catch {}
  }

  // Derived RAG strategy info for this dataset
  const strategyName = useMemo(() => {
    if (!datasetId) return 'PDF Simple'
    try {
      return (
        localStorage.getItem(`lf_dataset_strategy_name_${datasetId}`) ||
        'PDF Simple'
      )
    } catch {
      return 'PDF Simple'
    }
  }, [datasetId])

  const slugify = (v: string) =>
    v
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')

  const strategyId = useMemo(() => slugify(strategyName), [strategyName])

  const docType = useMemo(() => {
    const id = strategyId
    if (id.includes('pdf')) return 'PDFs'
    if (id.includes('csv')) return 'CSVs'
    if (id.includes('chat')) return 'Conversations'
    if (id.includes('json')) return 'JSON'
    return 'Documents'
  }, [strategyId])

  const embeddingModel = useMemo(() => {
    try {
      return (
        localStorage.getItem(`lf_strategy_embedding_model_${strategyId}`) ||
        'text-embedding-3-large'
      )
    } catch {
      return 'text-embedding-3-large'
    }
  }, [strategyId])

  return (
    <div className="h-full w-full flex flex-col gap-3 pb-40">
      <nav className="text-sm md:text-base flex items-center gap-1.5 mb-3">
        <button
          className="text-teal-600 dark:text-teal-400 hover:underline"
          onClick={() => navigate('/chat/data')}
        >
          Data
        </button>
        <span className="text-muted-foreground px-1">\</span>
        <span className="text-foreground">{datasetName}</span>
      </nav>

      {/* Header row */}
      <div className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-start justify-between gap-4">
          <div className="flex flex-col gap-1">
            <div className="flex items-center gap-2">
              <h2 className="text-xl md:text-2xl font-medium">{datasetName}</h2>
              <button
                className="p-1 rounded-md hover:bg-accent text-muted-foreground"
                onClick={openEdit}
                aria-label="Edit dataset"
                title="Edit dataset"
              >
                <FontIcon type="edit" className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-muted-foreground max-w-[640px]">
              {dataset?.description && dataset.description.trim().length > 0
                ? dataset.description
                : 'Add a short description so teammates know what this dataset is for.'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="secondary" size="sm" className="rounded-xl">
              {dataset?.numChunks?.toLocaleString?.() || '—'} chunks •{' '}
              {dataset?.processedPercent ?? 0}% processed
            </Badge>
            <Button
              size="sm"
              onClick={() => {
                if (!datasetId) return
                // Reprocess: bump lastRun timestamp
                try {
                  const stored = localStorage.getItem('lf_datasets')
                  const arr = stored ? (JSON.parse(stored) as Dataset[]) : []
                  const updated = arr.map(d =>
                    d.id === datasetId
                      ? {
                          ...d,
                          lastRun: new Date().toISOString(),
                        }
                      : d
                  )
                  localStorage.setItem('lf_datasets', JSON.stringify(updated))
                  setDataset(updated.find(d => d.id === datasetId) || null)
                  toast({ message: 'Reprocess started', variant: 'default' })
                } catch {}
              }}
            >
              Reprocess
            </Button>
          </div>
        </div>
      </div>

      {/* RAG Strategy card */}
      <section className="rounded-lg border border-border bg-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">RAG strategy</h3>
          <Button
            variant="outline"
            size="sm"
            onClick={() => navigate(`/chat/rag/${strategyId}`)}
          >
            Configure
          </Button>
        </div>
        <div className="flex items-center gap-2 flex-wrap mb-2">
          <Badge variant="default" size="sm" className="rounded-xl">
            {strategyName}
          </Badge>
          <Badge variant="secondary" size="sm" className="rounded-xl">
            Last processed{' '}
            {dataset?.lastRun
              ? new Date(dataset.lastRun).toLocaleString()
              : '—'}
          </Badge>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Document type</div>
            <Input value={docType} readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Search</div>
            <Input value="Hybrid" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Embedding model</div>
            <Input value={embeddingModel} readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Chunks</div>
            <Input
              value={dataset?.numChunks?.toLocaleString?.() || '—'}
              readOnly
              className="bg-background"
            />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Results</div>
            <Input value="8" readOnly className="bg-background" />
          </div>
          <div className="flex flex-col gap-1">
            <div className="text-xs text-muted-foreground">Avg query time</div>
            <Input value="180ms" readOnly className="bg-background" />
          </div>
        </div>
      </section>

      <Dialog open={isEditOpen} onOpenChange={setIsEditOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Edit dataset</DialogTitle>
          </DialogHeader>
          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">Name</label>
              <Input
                autoFocus
                value={editName}
                onChange={e => setEditName(e.target.value)}
                placeholder="Dataset name"
              />
            </div>
            <div className="flex flex-col gap-1">
              <label className="text-xs text-muted-foreground">
                Description
              </label>
              <Textarea
                value={editDescription}
                onChange={e => setEditDescription(e.target.value)}
                placeholder="Optional description"
                rows={3}
              />
            </div>
          </div>
          <div className="mt-4 flex flex-col-reverse gap-2 sm:flex-row sm:items-center sm:justify-between">
            <Button variant="destructive" onClick={handleDelete}>
              Delete
            </Button>
            <div className="flex items-center gap-2">
              <DialogClose asChild>
                <Button variant="secondary">Cancel</Button>
              </DialogClose>
              <Button onClick={handleSaveEdit} disabled={!editName.trim()}>
                Save
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Raw data */}
      <section className="rounded-lg border border-border bg-card p-4 mb-40">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-medium">Raw data</h3>
          <Button size="sm" onClick={() => fileInputRef.current?.click()}>
            Upload data
          </Button>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          className="hidden"
          multiple
          onChange={e => {
            if (!datasetId) return
            const list = e.target.files ? Array.from(e.target.files) : []
            if (list.length === 0) return
            const converted: RawFile[] = list.map(f => ({
              id: `${f.name}:${f.size}:${f.lastModified}`,
              name: f.name,
              size: f.size,
              lastModified: f.lastModified,
              type: f.type,
            }))
            try {
              setUploadStatus(prev => ({
                ...prev,
                ...Object.fromEntries(
                  converted.map(rf => [rf.id, 'processing' as const])
                ),
              }))
              const storedRaw = localStorage.getItem('lf_raw_files')
              const existing: RawFile[] = storedRaw ? JSON.parse(storedRaw) : []
              const existingIds = new Set(existing.map(r => r.id))
              const deduped = converted.filter(r => !existingIds.has(r.id))
              const updatedRaw = [...existing, ...deduped]
              localStorage.setItem('lf_raw_files', JSON.stringify(updatedRaw))

              const storedAssign = localStorage.getItem('lf_file_assignments')
              const assignments: Record<string, string[]> = storedAssign
                ? JSON.parse(storedAssign)
                : {}
              for (const rf of converted) {
                const arr = assignments[rf.id] ?? []
                if (!arr.includes(datasetId)) {
                  assignments[rf.id] = [...arr, datasetId]
                }
              }
              localStorage.setItem(
                'lf_file_assignments',
                JSON.stringify(assignments)
              )

              const nowAssigned = converted.filter(rf =>
                (assignments[rf.id] ?? []).includes(datasetId)
              )
              setFiles(prev => {
                const seen = new Set(prev.map(p => p.id))
                const add = nowAssigned.filter(n => !seen.has(n.id))
                return [...prev, ...add]
              })
              setTimeout(() => {
                setUploadStatus(prev => ({
                  ...prev,
                  ...Object.fromEntries(
                    converted.map(rf => [rf.id, 'done' as const])
                  ),
                }))
                setTimeout(() => {
                  setUploadStatus(prev => {
                    const next = { ...prev }
                    for (const rf of converted) delete (next as any)[rf.id]
                    return next
                  })
                }, 1500)
              }, 1500)
            } catch {}
            e.currentTarget.value = ''
          }}
        />
        <div className="flex items-center gap-2 mb-2">
          <div className="w-1/2">
            <SearchInput
              placeholder="Search raw files"
              value={searchValue}
              onChange={e => setSearchValue(e.target.value)}
            />
          </div>
        </div>
        <div className="rounded-md border border-input bg-background p-0 text-xs">
          {files.length === 0 ? (
            <div className="p-3 text-muted-foreground">
              No files assigned yet.
            </div>
          ) : (
            <ul>
              {files
                .filter(f =>
                  f.name.toLowerCase().includes(searchValue.toLowerCase())
                )
                .map(f => (
                  <li
                    key={f.id}
                    className="flex items-center justify-between px-3 py-3 border-b last:border-b-0 border-border/60"
                  >
                    <span className="font-mono text-xs text-muted-foreground truncate max-w-[60%]">
                      {f.name}
                    </span>
                    <div className="w-1/2 flex items-center justify-between gap-4">
                      <div className="text-xs text-muted-foreground">
                        {Math.ceil(f.size / 1024)} KB
                      </div>
                      <div className="flex items-center gap-6">
                        {uploadStatus[f.id] === 'processing' && (
                          <div className="flex items-center gap-1 text-muted-foreground">
                            <FontIcon type="fade" className="w-4 h-4" />
                            <span className="text-xs">Processing</span>
                          </div>
                        )}
                        {uploadStatus[f.id] === 'done' && (
                          <FontIcon
                            type="checkmark-outline"
                            className="w-4 h-4 text-teal-600 dark:text-teal-400"
                          />
                        )}
                        <button
                          className="w-4 h-4 grid place-items-center text-muted-foreground hover:text-foreground"
                          onClick={() => {
                            setFileToDelete(f)
                            setIsDeleteOpen(true)
                          }}
                          aria-label={`Remove ${f.name} from this dataset`}
                        >
                          <FontIcon type="trashcan" className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </li>
                ))}
            </ul>
          )}
        </div>
      </section>

      {/* Delete from dataset dialog */}
      <Dialog open={isDeleteOpen} onOpenChange={setIsDeleteOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Remove file from this dataset</DialogTitle>
          </DialogHeader>
          <div className="text-sm">
            <div className="mb-2 text-muted-foreground">
              This will remove the file from this dataset only. It will remain
              available in the project.
            </div>
            <div className="font-mono text-xs break-all">
              {fileToDelete?.name}
            </div>
          </div>
          <div className="mt-4 flex items-center justify-end gap-2">
            <DialogClose asChild>
              <Button variant="secondary">Cancel</Button>
            </DialogClose>
            <Button
              variant="destructive"
              onClick={() => {
                if (!fileToDelete || !datasetId) return
                try {
                  const stored = localStorage.getItem('lf_file_assignments')
                  const assignments: Record<string, string[]> = stored
                    ? JSON.parse(stored)
                    : {}
                  const arr = assignments[fileToDelete.id] ?? []
                  assignments[fileToDelete.id] = arr.filter(
                    id => id !== datasetId
                  )
                  localStorage.setItem(
                    'lf_file_assignments',
                    JSON.stringify(assignments)
                  )
                } catch {}
                setFiles(prev => prev.filter(x => x.id !== fileToDelete.id))
                setIsDeleteOpen(false)
                setFileToDelete(null)
                toast({
                  message: 'File removed from dataset',
                  variant: 'default',
                })
              }}
            >
              Yes, remove
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default DatasetView
