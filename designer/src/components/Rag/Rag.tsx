import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import ConfigEditor from '../ConfigEditor/ConfigEditor'
import PageActions from '../common/PageActions'
import { Mode } from '../ModeToggle'
import { Badge } from '../ui/badge'
import SearchInput from '../ui/search-input'
import { defaultStrategies } from './strategies'
import { useToast } from '../ui/toast'
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

type RagStrategy = import('./strategies').RagStrategy

function Rag() {
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const { toast } = useToast()
  const [mode, setMode] = useState<Mode>('designer')

  const [metaTick, setMetaTick] = useState(0)
  const [isEditOpen, setIsEditOpen] = useState(false)
  const [editId, setEditId] = useState<string>('')
  const [editName, setEditName] = useState('')
  const [editDescription, setEditDescription] = useState('')
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const [createName, setCreateName] = useState('')
  const [createDescription, setCreateDescription] = useState('')
  const [copyFromId, setCopyFromId] = useState('')

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
  const resetStrategies = () => {
    try {
      localStorage.removeItem('lf_strategy_deleted')
      defaultStrategies.forEach(s => {
        localStorage.removeItem(`lf_strategy_name_override_${s.id}`)
        localStorage.removeItem(`lf_strategy_description_${s.id}`)
      })
      localStorage.removeItem('lf_custom_strategies')
    } catch {}
    setMetaTick(t => t + 1)
    toast({ message: 'Strategies reset', variant: 'default' })
  }
  const getDeletedSet = (): Set<string> => {
    try {
      const raw = localStorage.getItem('lf_strategy_deleted')
      if (!raw) return new Set()
      const arr = JSON.parse(raw) as string[]
      return new Set(arr)
    } catch {
      return new Set()
    }
  }
  const saveDeletedSet = (s: Set<string>) => {
    try {
      localStorage.setItem('lf_strategy_deleted', JSON.stringify(Array.from(s)))
    } catch {}
  }
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

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return display
    return display.filter(s =>
      [s.name, s.description].some(v => v.toLowerCase().includes(q))
    )
  }, [query, display])

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
            <div className="text-sm text-muted-foreground mb-1">
              simple description – these can be applied to datasets
            </div>

            <div className="rounded-lg border border-border bg-card p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="text-sm font-medium">
                  Default RAG strategies
                </div>
                <div className="flex items-center gap-2">
                  <Button variant="outline" size="sm" onClick={resetStrategies}>
                    Reset list
                  </Button>
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

              <div className="mb-3 max-w-xl">
                <SearchInput
                  placeholder="Search RAG Strategies"
                  value={query}
                  onChange={e => setQuery(e.target.value)}
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {filtered.map(s => (
                  <div
                    key={s.id}
                    className="w-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative hover:bg-accent/20 hover:cursor-pointer transition-colors"
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
                          className="min-w-[10rem] w-[10rem]"
                        >
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              e.preventDefault()
                              setEditId(s.id)
                              setEditName(s.name)
                              setEditDescription(s.description)
                              setIsEditOpen(true)
                            }}
                          >
                            Edit
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={() => navigate(`/chat/rag/${s.id}`)}
                          >
                            Configure
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            onClick={e => {
                              e.stopPropagation()
                              e.preventDefault()
                              setCopyFromId(s.id)
                              setCreateName(`${s.name} (copy)`)
                              setCreateDescription(s.description || '')
                              setIsCreateOpen(true)
                            }}
                          >
                            Duplicate
                          </DropdownMenuItem>
                          <DropdownMenuItem
                            className="text-destructive focus:text-destructive"
                            onClick={e => {
                              e.stopPropagation()
                              e.preventDefault()
                              const ok = confirm(
                                'Are you sure you want to delete this strategy?'
                              )
                              if (ok) {
                                try {
                                  localStorage.removeItem(
                                    `lf_strategy_name_override_${s.id}`
                                  )
                                  localStorage.removeItem(
                                    `lf_strategy_description_${s.id}`
                                  )
                                } catch {}
                                removeCustomStrategy(s.id)
                                markDeleted(s.id)
                                toast({
                                  message: 'Strategy deleted',
                                  variant: 'default',
                                })
                              }
                            }}
                          >
                            Delete
                          </DropdownMenuItem>
                        </DropdownMenuContent>
                      </DropdownMenu>
                    </div>

                    <div className="text-sm font-medium">{s.name}</div>
                    <div className="text-xs text-primary text-left w-fit">
                      {s.description}
                    </div>

                    <div className="flex items-center gap-2 flex-wrap">
                      {s.isDefault ? (
                        <Badge
                          variant="default"
                          size="sm"
                          className="rounded-xl"
                        >
                          Default
                        </Badge>
                      ) : (
                        <Badge
                          variant="secondary"
                          size="sm"
                          className="rounded-xl"
                        >
                          Custom
                        </Badge>
                      )}
                      <Badge
                        variant={s.datasetsUsing > 0 ? 'default' : 'secondary'}
                        size="sm"
                        className={`rounded-xl ${s.datasetsUsing > 0 ? 'bg-emerald-600 text-emerald-50 dark:bg-emerald-400 dark:text-emerald-900' : ''}`}
                      >
                        {`${s.datasetsUsing} datasets using`}
                      </Badge>
                    </div>

                    <div className="flex justify-end pt-1">
                      <Button
                        variant="outline"
                        size="sm"
                        className="px-3"
                        onClick={e => {
                          e.stopPropagation()
                          navigate(`/chat/rag/${s.id}`)
                        }}
                      >
                        Configure
                      </Button>
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
              Create new RAG strategy
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
                navigate(`/chat/rag/${newId}`)
              }}
              disabled={createName.trim().length === 0}
              type="button"
            >
              Create
            </button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

export default Rag
