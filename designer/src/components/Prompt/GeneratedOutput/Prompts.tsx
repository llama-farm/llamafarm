import { useMemo, useState } from 'react'
import FontIcon from '../../../common/FontIcon'
import { Button } from '../../ui/button'
import { useActiveProject } from '../../../hooks/useActiveProject'
import { useProject } from '../../../hooks/useProjects'
import projectService from '../../../api/projectService'
import PromptModal, { PromptModalMode } from './PromptModal'
import { Input } from '../../ui/input'
import { Label } from '../../ui/label'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../../ui/dialog'
import { useToast } from '../../ui/toast'
import {
  parsePromptSets,
  serializePromptSets,
  type PromptSet,
} from '../../../utils/promptSets'

const Prompts = () => {
  const activeProject = useActiveProject()
  const { data: projectResponse, refetch } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )
  const { toast } = useToast()

  const promptSets: PromptSet[] = useMemo(() => {
    const prompts = projectResponse?.project?.config?.prompts as
      | Array<{
          name: string
          messages: Array<{ role?: string; content: string }>
        }>
      | undefined
    return parsePromptSets(prompts)
  }, [projectResponse])

  // (preview rows removed; table renders directly from sets)

  // Add prompt modal state
  const [isOpen, setIsOpen] = useState(false)
  const [mode, setMode] = useState<PromptModalMode>('create')
  const [initialText, setInitialText] = useState('')
  const [initialRole, setInitialRole] = useState<
    'system' | 'assistant' | 'user'
  >('system')
  const [isDeleteOpen, setIsDeleteOpen] = useState(false)
  const [deleteIndex, setDeleteIndex] = useState<number | null>(null)
  const [deleteSetIndex, setDeleteSetIndex] = useState<number | null>(null)
  const [editIndex, setEditIndex] = useState<number | null>(null)
  const [currentSetIndex, setCurrentSetIndex] = useState<number | null>(null)

  // Create set dialog
  const [isCreateSetOpen, setIsCreateSetOpen] = useState(false)
  const [newSetName, setNewSetName] = useState('')

  // Edit set modal
  const [isEditSetOpen, setIsEditSetOpen] = useState(false)
  const [editSetIndex, setEditSetIndex] = useState<number | null>(null)
  const [editSetName, setEditSetName] = useState('')
  const [isConfirmDeleteOpen, setIsConfirmDeleteOpen] = useState(false)

  const handleSavePrompt = async (
    text: string,
    role: 'system' | 'assistant' | 'user',
    selectedSetIdx?: number
  ) => {
    if (!activeProject || !projectResponse?.project?.config) return
    const { config } = projectResponse.project
    const sets = parsePromptSets(
      config.prompts as
        | Array<{
            name: string
            messages: Array<{ role?: string; content: string }>
          }>
        | undefined
    )
    // Use the selected set index from modal if provided (create mode), otherwise use current set
    const setIdx =
      selectedSetIdx !== undefined ? selectedSetIdx : (currentSetIndex ?? 0)
    const idx = setIdx >= 0 ? setIdx : 0
    const nextSets = [...sets]
    const target = { ...nextSets[idx] }
    target.items = [...target.items]

    if (
      mode === 'edit' &&
      editIndex != null &&
      editIndex >= 0 &&
      editIndex < target.items.length
    ) {
      target.items[editIndex] = { role, content: text }
    } else {
      target.items.unshift({ role, content: text })
    }
    nextSets[idx] = target

    const prompts = serializePromptSets(nextSets)
    const request = { config: { ...config, prompts } }
    try {
      await projectService.updateProject(
        activeProject.namespace,
        activeProject.project,
        request
      )
      await refetch()
      setIsOpen(false)
      setEditIndex(null)
      setCurrentSetIndex(null)
      toast({ message: 'Prompt saved', variant: 'default' })
    } catch (e) {
      console.error('Failed to save prompt', e)
      toast({ message: 'Failed to save prompt', variant: 'destructive' })
    }
  }

  // openCreatePrompt removed (unused)

  const openEditPrompt = (index: number, setIndex: number) => {
    const set = promptSets[setIndex]
    const item = set?.items[index]
    setMode('edit')
    setEditIndex(index)
    setCurrentSetIndex(setIndex)
    setInitialText(item?.content || '')
    setInitialRole((item?.role as 'system' | 'assistant' | 'user') || 'system')
    setIsOpen(true)
  }

  const performDeletePrompt = async (
    setIndex: number,
    index: number
  ): Promise<boolean> => {
    if (!activeProject || !projectResponse?.project?.config) return false
    const { config } = projectResponse.project
    const sets = parsePromptSets(
      config.prompts as
        | Array<{
            name: string
            messages: Array<{ role?: string; content: string }>
          }>
        | undefined
    )
    if (setIndex < 0 || setIndex >= sets.length) return false
    const nextSets = [...sets]
    const target = { ...nextSets[setIndex] }
    target.items = [...target.items]
    if (index < 0 || index >= target.items.length) return false
    target.items.splice(index, 1)
    nextSets[setIndex] = target

    const prompts = serializePromptSets(nextSets)
    const request = { config: { ...config, prompts } }
    try {
      await projectService.updateProject(
        activeProject.namespace,
        activeProject.project,
        request
      )
      await refetch()
      toast({ message: 'Prompt deleted', variant: 'default' })
      return true
    } catch (e) {
      console.error('Failed to delete prompt', e)
      toast({ message: 'Failed to delete prompt', variant: 'destructive' })
      return false
    }
  }

  const openDeletePrompt = (index: number, setIndex: number) => {
    setDeleteIndex(index)
    setDeleteSetIndex(setIndex)
    setIsDeleteOpen(true)
  }

  const confirmDeletePrompt = async () => {
    if (deleteIndex == null || deleteSetIndex == null) return
    const success = await performDeletePrompt(deleteSetIndex, deleteIndex)
    if (success) {
      setIsDeleteOpen(false)
      setDeleteIndex(null)
      setDeleteSetIndex(null)
    }
  }

  const createSet = async () => {
    if (!activeProject || !projectResponse?.project?.config) return
    const name = newSetName.trim() || 'Untitled'
    const { config } = projectResponse.project
    const sets = parsePromptSets(
      config.prompts as
        | Array<{
            name: string
            messages: Array<{ role?: string; content: string }>
          }>
        | undefined
    )
    const next = [...sets, { name, items: [] }]
    const prompts = serializePromptSets(next)
    const request = { config: { ...config, prompts } }
    try {
      await projectService.updateProject(
        activeProject.namespace,
        activeProject.project,
        request
      )
      await refetch()
      setIsCreateSetOpen(false)
      setNewSetName('')
      toast({ message: 'Prompt set created', variant: 'default' })
    } catch (e) {
      console.error('Failed to create prompt set', e)
      toast({ message: 'Failed to create set', variant: 'destructive' })
    }
  }

  const deleteSet = async (index: number) => {
    if (!activeProject || !projectResponse?.project?.config) return
    const { config } = projectResponse.project
    const sets = parsePromptSets(
      config.prompts as
        | Array<{
            name: string
            messages: Array<{ role?: string; content: string }>
          }>
        | undefined
    )
    if (index < 0 || index >= sets.length) return
    const next = sets.filter((_, i) => i !== index)
    const prompts = serializePromptSets(next)
    const request = { config: { ...config, prompts } }
    try {
      await projectService.updateProject(
        activeProject.namespace,
        activeProject.project,
        request
      )
      await refetch()
      toast({ message: 'Prompt set deleted', variant: 'default' })
    } catch (e) {
      console.error('Failed to delete set', e)
      toast({ message: 'Failed to delete set', variant: 'destructive' })
    }
  }

  const openEditSet = (index: number, currentName: string) => {
    setEditSetIndex(index)
    setEditSetName(currentName)
    setIsEditSetOpen(true)
  }

  const saveEditSet = async () => {
    if (
      editSetIndex == null ||
      !activeProject ||
      !projectResponse?.project?.config
    )
      return
    const { config } = projectResponse.project
    const sets = parsePromptSets(
      config.prompts as
        | Array<{
            name: string
            messages: Array<{ role?: string; content: string }>
          }>
        | undefined
    )
    if (editSetIndex < 0 || editSetIndex >= sets.length) return
    const next = [...sets]
    next[editSetIndex] = {
      ...next[editSetIndex],
      name: editSetName.trim() || 'Untitled',
    }
    const prompts = serializePromptSets(next)
    const request = { config: { ...config, prompts } }
    try {
      await projectService.updateProject(
        activeProject.namespace,
        activeProject.project,
        request
      )
      await refetch()
      setIsEditSetOpen(false)
      setEditSetIndex(null)
      setEditSetName('')
      toast({ message: 'Set updated', variant: 'default' })
    } catch (e) {
      console.error('Failed to update set', e)
      toast({ message: 'Failed to update set', variant: 'destructive' })
    }
  }

  const confirmDeleteSet = async () => {
    if (editSetIndex == null) return
    await deleteSet(editSetIndex)
    setIsConfirmDeleteOpen(false)
    setIsEditSetOpen(false)
    setEditSetIndex(null)
    setEditSetName('')
  }

  // Full page empty state when no sets exist
  if (promptSets.length === 0) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="text-center px-6 py-10 rounded-xl border border-border bg-card/40 max-w-md">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
            <FontIcon type="prompt" className="w-6 h-6 text-primary" />
          </div>
          <div className="text-lg font-medium text-foreground mb-2">
            No prompt sets yet
          </div>
          <div className="text-sm text-muted-foreground mb-6">
            Create your first prompt set to start building your AI assistant's
            behavior. Each set can contain system, user, and assistant prompts.
          </div>
          <Button
            onClick={() => setIsCreateSetOpen(true)}
            className="w-full sm:w-auto"
          >
            Create your first prompt set
          </Button>
        </div>
        {/* Dialogs still need to be rendered */}
        <Dialog
          open={isCreateSetOpen}
          onOpenChange={v => (!v ? setIsCreateSetOpen(false) : undefined)}
        >
          <DialogContent className="sm:max-w-md">
            <DialogHeader>
              <DialogTitle className="text-lg text-foreground">
                Create new prompt set
              </DialogTitle>
            </DialogHeader>
            <div className="space-y-3">
              <div>
                <Label htmlFor="set-name" className="text-sm text-foreground">
                  Prompt set name
                </Label>
                <Input
                  id="set-name"
                  value={newSetName}
                  onChange={e => setNewSetName(e.target.value)}
                  placeholder="e.g., default, customer-service"
                  className="mt-1"
                />
              </div>
            </div>
            <DialogFooter className="flex gap-2 justify-end">
              <Button
                variant="outline"
                onClick={() => {
                  setIsCreateSetOpen(false)
                  setNewSetName('')
                }}
              >
                Cancel
              </Button>
              <Button onClick={createSet}>Create</Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>
    )
  }

  return (
    <div className="w-full h-full">
      <div className="w-full flex flex-col sm:flex-row items-start sm:items-center justify-between mb-5 gap-3">
        <p className="text-sm text-muted-foreground w-full">
          Manage multiple named prompt sets. Each set contains role:prompt pairs
          you can add, edit, or delete.
        </p>
        <div className="flex gap-2 w-full sm:w-auto">
          <Button
            variant="outline"
            onClick={() => setIsCreateSetOpen(true)}
            className="whitespace-nowrap w-full sm:w-auto"
          >
            New prompt set
          </Button>
          {/* Removed top-level New pair per updated UX */}
        </div>
      </div>

      {promptSets.map((set, sIdx) => (
        <div
          key={sIdx}
          className="w-full rounded-lg border border-border bg-card mb-4"
        >
          <div className="flex items-center justify-between px-3 py-3 border-b border-border/60 bg-muted/50">
            <div className="text-lg font-medium">{set.name}</div>
            <div className="flex items-center gap-3">
              <FontIcon
                type="edit"
                isButton
                className="w-4 h-4 text-muted-foreground inline-block mr-2"
                handleOnClick={() => openEditSet(sIdx, set.name)}
              />
              <Button
                size="sm"
                onClick={() => {
                  setMode('create')
                  setInitialText('')
                  setInitialRole('system')
                  setEditIndex(null)
                  setCurrentSetIndex(sIdx)
                  setIsOpen(true)
                }}
              >
                Add prompt
              </Button>
            </div>
          </div>

          <ul className="w-full">
            {set.items.map((prompt, index) => (
              <li
                key={index}
                className="relative px-4 py-4 text-sm font-mono leading-4 tracking-[0.32px] border-b border-border last:border-b-0"
              >
                {/* Actions - top right corner */}
                <div className="absolute top-3 right-4 flex items-center gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="h-7 px-2.5 text-xs font-sans font-normal"
                    onClick={() => openEditPrompt(index, sIdx)}
                  >
                    <FontIcon
                      type="edit"
                      className="w-3.5 h-3.5 mr-1.5"
                    />
                    Edit
                  </Button>
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-7 w-7 p-0 text-muted-foreground hover:text-destructive"
                    onClick={() => openDeletePrompt(index, sIdx)}
                  >
                    <FontIcon
                      type="trashcan"
                      className="w-4 h-4"
                    />
                  </Button>
                </div>
                {/* Content */}
                <div className="font-mono text-xs text-muted-foreground flex flex-col gap-1 pr-28">
                  <span>{prompt.role || '—'}</span>
                  <div className="whitespace-pre-line break-words line-clamp-6 text-foreground">
                    {prompt.content}
                  </div>
                </div>
              </li>
            ))}
            {set.items.length === 0 && (
              <li className="flex items-center justify-center py-12">
                <div className="text-center px-6">
                  <div className="mx-auto mb-3 flex h-10 w-10 items-center justify-center rounded-full bg-primary/15 border border-primary/30">
                    <FontIcon type="prompt" className="w-5 h-5 text-primary" />
                  </div>
                  <div className="text-base font-medium text-foreground mb-2">
                    No prompts yet
                  </div>
                  <div className="text-sm text-muted-foreground mb-4">
                    Add your first prompt to this set
                  </div>
                  <Button
                    size="sm"
                    onClick={() => {
                      setMode('create')
                      setInitialText('')
                      setInitialRole('system')
                      setEditIndex(null)
                      setCurrentSetIndex(sIdx)
                      setIsOpen(true)
                    }}
                  >
                    Add prompt
                  </Button>
                </div>
              </li>
            )}
          </ul>
        </div>
      ))}
      <PromptModal
        isOpen={isOpen}
        mode={mode}
        initialText={initialText}
        initialRole={initialRole}
        promptSets={promptSets}
        selectedSetIndex={currentSetIndex ?? 0}
        onClose={() => setIsOpen(false)}
        onSave={handleSavePrompt}
      />

      <Dialog
        open={isDeleteOpen}
        onOpenChange={v => (!v ? setIsDeleteOpen(false) : undefined)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Delete prompt
            </DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            This will remove the prompt from your project configuration. This
            action cannot be undone.
          </div>
          <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2">
            <div />
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsDeleteOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
                onClick={confirmDeletePrompt}
                type="button"
              >
                Delete
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={isCreateSetOpen}
        onOpenChange={v => (!v ? setIsCreateSetOpen(false) : undefined)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              New prompt set
            </DialogTitle>
          </DialogHeader>
          <Label className="text-sm text-muted-foreground mb-0">Name</Label>
          <Input
            value={newSetName}
            onChange={e => setNewSetName(e.target.value)}
            placeholder="Default"
            className="mb-4"
          />
          <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2">
            <div />
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsCreateSetOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="px-3 py-2 rounded-md bg-primary text-primary-foreground hover:opacity-90 text-sm"
                onClick={createSet}
                type="button"
              >
                Create
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={isEditSetOpen}
        onOpenChange={v => (!v ? setIsEditSetOpen(false) : undefined)}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Edit prompt set
            </DialogTitle>
          </DialogHeader>
          <Label className="text-sm text-muted-foreground mb-0">Name</Label>
          <Input
            value={editSetName}
            onChange={e => setEditSetName(e.target.value)}
            placeholder="Untitled"
            className="mb-4"
          />
          <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2">
            <div>
              <button
                className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
                onClick={() => setIsConfirmDeleteOpen(true)}
                type="button"
              >
                Delete set
              </button>
            </div>
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => setIsEditSetOpen(false)}
                type="button"
              >
                Cancel
              </button>
              <button
                className="px-3 py-2 rounded-md bg-primary text-primary-foreground hover:opacity-90 text-sm"
                onClick={saveEditSet}
                type="button"
              >
                Save
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={isConfirmDeleteOpen}
        onOpenChange={v => {
          if (!v) {
            setIsConfirmDeleteOpen(false)
            setIsEditSetOpen(false)
          }
        }}
      >
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="text-lg text-foreground">
              Delete prompt set
            </DialogTitle>
          </DialogHeader>
          <div className="text-sm text-muted-foreground">
            This will remove this prompt set and all prompts within it. This
            action cannot be undone.
          </div>
          <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2">
            <div />
            <div className="flex items-center gap-2 ml-auto">
              <button
                className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
                onClick={() => {
                  setIsConfirmDeleteOpen(false)
                  setIsEditSetOpen(false)
                }}
                type="button"
              >
                Cancel
              </button>
              <button
                className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
                onClick={confirmDeleteSet}
                type="button"
              >
                Delete
              </button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Prompts
