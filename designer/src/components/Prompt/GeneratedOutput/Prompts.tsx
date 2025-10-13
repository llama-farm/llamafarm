import { useMemo, useState } from 'react'
import FontIcon from '../../../common/FontIcon'
import { Button } from '../../ui/button'
import { useActiveProject } from '../../../hooks/useActiveProject'
import { useProject } from '../../../hooks/useProjects'
import projectService from '../../../api/projectService'
import PromptModal, { PromptModalMode } from './PromptModal'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../../ui/dialog'
import { useToast } from '../../ui/toast'

interface PromptRow {
  role?: string
  preview: string
}

const Prompts = () => {
  const activeProject = useActiveProject()
  const { data: projectResponse, refetch } = useProject(
    activeProject?.namespace || '',
    activeProject?.project || '',
    !!activeProject?.namespace && !!activeProject?.project
  )
  const { toast } = useToast()

  const rows: PromptRow[] = useMemo(() => {
    const prompts = projectResponse?.project?.config?.prompts as
      | Array<{ role?: string; content: string }>
      | undefined
    if (!prompts || prompts.length === 0) return []
    return prompts.map(p => ({ role: p.role, preview: p.content }))
  }, [projectResponse])

  // Add prompt modal state
  const [isOpen, setIsOpen] = useState(false)
  const [mode, setMode] = useState<PromptModalMode>('create')
  const [initialText, setInitialText] = useState('')
  const [initialRole, setInitialRole] = useState<
    'system' | 'assistant' | 'user'
  >('system')
  const [isDeleteOpen, setIsDeleteOpen] = useState(false)
  const [deleteIndex, setDeleteIndex] = useState<number | null>(null)
  const [editIndex, setEditIndex] = useState<number | null>(null)

  const handleSavePrompt = async (
    text: string,
    role: 'system' | 'assistant' | 'user'
  ) => {
    if (!activeProject || !projectResponse?.project?.config) return
    const { config } = projectResponse.project
    const prompts = Array.isArray(config.prompts) ? [...config.prompts] : []
    if (
      mode === 'edit' &&
      editIndex != null &&
      editIndex >= 0 &&
      editIndex < prompts.length
    ) {
      const existing = prompts[editIndex] as any
      prompts[editIndex] = { ...existing, role, content: text }
    } else {
      prompts.unshift({ role, content: text })
    }
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
      toast({ message: 'Prompt saved', variant: 'default' })
    } catch (e) {
      console.error('Failed to save prompt', e)
      toast({ message: 'Failed to save prompt', variant: 'destructive' })
    }
  }

  const openCreatePrompt = () => {
    setMode('create')
    setInitialText('')
    setInitialRole('system')
    setEditIndex(null)
    setIsOpen(true)
  }

  const openEditPrompt = (index: number) => {
    const item = rows[index]
    setMode('edit')
    setEditIndex(index)
    setInitialText(item?.preview || '')
    setInitialRole((item?.role as 'system' | 'assistant' | 'user') || 'system')
    setIsOpen(true)
  }

  const performDeletePrompt = async (index: number): Promise<boolean> => {
    if (!activeProject || !projectResponse?.project?.config) return false
    const { config } = projectResponse.project
    const prompts = Array.isArray(config.prompts) ? [...config.prompts] : []
    if (index < 0 || index >= prompts.length) return false
    prompts.splice(index, 1)
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

  const openDeletePrompt = (index: number) => {
    setDeleteIndex(index)
    setIsDeleteOpen(true)
  }

  const confirmDeletePrompt = async () => {
    if (deleteIndex == null) return
    const success = await performDeletePrompt(deleteIndex)
    if (success) {
      setIsDeleteOpen(false)
      setDeleteIndex(null)
    }
  }

  return (
    <div className="w-full h-full">
      <div className="w-full flex flex-col sm:flex-row items-start sm:items-center justify-between mb-2 gap-3">
        <p className="text-sm text-muted-foreground w-full">
          Prompts are instructions that tell your model how to behave. The
          following prompts have been saved for this project so far. You can
          view the prompts defined in your project configuration.
        </p>
        <Button
          onClick={openCreatePrompt}
          className="whitespace-nowrap w-full sm:w-auto"
        >
          New prompt
        </Button>
      </div>
      <table className="w-full">
        <thead className="bg-white dark:bg-blue-600 font-normal">
          <tr>
            <th className="text-left w-[15%] py-2 px-3 font-normal">Role</th>
            <th className="text-left py-2 px-3 font-normal">Preview</th>
            <th className="text-right w-[1%] py-2 px-3 font-normal">Actions</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((prompt, index) => (
            <tr
              key={index}
              className={`border-b border-solid border-white dark:border-blue-600 text-sm font-mono leading-4 tracking-[0.32px]${
                index === rows.length - 1 ? 'border-b-0' : 'border-b'
              }`}
            >
              <td className="align-top p-3">{prompt.role || '—'}</td>
              <td className="align-top p-3">
                <div className="whitespace-pre-line break-words line-clamp-6">
                  {prompt.preview}
                </div>
              </td>
              <td className="align-top p-3 text-right whitespace-nowrap">
                <FontIcon
                  type="edit"
                  isButton
                  className="w-4 h-4 text-muted-foreground inline-block mr-3"
                  handleOnClick={() => openEditPrompt(index)}
                />
                <FontIcon
                  type="trashcan"
                  isButton
                  className="w-4 h-4 text-muted-foreground inline-block"
                  handleOnClick={() => openDeletePrompt(index)}
                />
              </td>
            </tr>
          ))}
          {rows.length === 0 && (
            <tr>
              <td
                colSpan={3}
                className="align-top p-3 text-sm text-muted-foreground"
              >
                No prompts found in project configuration.
              </td>
            </tr>
          )}
        </tbody>
      </table>
      <PromptModal
        isOpen={isOpen}
        mode={mode}
        initialText={initialText}
        initialRole={initialRole}
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
    </div>
  )
}

export default Prompts
