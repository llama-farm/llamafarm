import { useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../ui/dialog'
import { Label } from '../ui/label'
import { Input } from '../ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import FontIcon from '../../common/FontIcon'

interface CreateProjectModalProps {
  isOpen: boolean
  availableProjects: string[]
  copyFromProject?: string | null
  onClose: () => void
  onCreate: (
    name: string,
    copyFrom?: string | null,
    deployment?: 'local' | 'cloud' | 'unsure'
  ) => void
  isLoading?: boolean
  projectError?: string | null
  onNameChange?: (name: string) => boolean
}

const CreateProjectModal: React.FC<CreateProjectModalProps> = ({
  isOpen,
  availableProjects,
  copyFromProject = null,
  onClose,
  onCreate,
  isLoading = false,
  projectError = null,
  onNameChange,
}) => {
  const [name, setName] = useState('')
  const [selectedSource, setSelectedSource] = useState<string>('')
  const [deployment, setDeployment] = useState<'local' | 'cloud' | 'unsure'>(
    'local'
  )
  const [hasAttemptedValidation, setHasAttemptedValidation] = useState(false)

  useEffect(() => {
    if (isOpen) {
      setName('')
      // Set default to "scratch" or pre-filled copyFromProject
      setSelectedSource(copyFromProject || '')
      setDeployment('local')
      setHasAttemptedValidation(false)
      // Don't trigger validation on open - only validate on save attempt
    }
  }, [isOpen, copyFromProject])

  // Handle Escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen && !isLoading) {
        onClose()
      }
    }

    if (isOpen) {
      document.addEventListener('keydown', handleEscape)
      return () => document.removeEventListener('keydown', handleEscape)
    }
  }, [isOpen, isLoading, onClose])

  const isValid = name.trim().length > 0 && !isLoading
  const copyFromValue = selectedSource || null

  const handleCancel = (e: React.MouseEvent) => {
    e.preventDefault()
    e.stopPropagation()
    onClose()
  }

  const handleOpenChange = (open: boolean) => {
    if (!open) {
      onClose()
    }
  }

  const handleCreate = (e: React.MouseEvent) => {
    e.preventDefault()
    setHasAttemptedValidation(true)

    // Validate name if validator is provided
    if (onNameChange) {
      const isValidName = onNameChange(name.trim())
      if (!isValidName) {
        return // Validation failed, error will be shown via projectError prop
      }
    }

    // Only proceed if name is valid
    if (name.trim().length > 0 && !isLoading) {
      onCreate(name.trim(), copyFromValue, deployment)
    }
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent
        className="sm:max-w-xl"
        onEscapeKeyDown={e => {
          e.preventDefault()
          if (!isLoading) onClose()
        }}
        onPointerDownOutside={e => {
          // Allow clicking outside to close modal when not loading
          if (!isLoading) return
          e.preventDefault()
        }}
        onInteractOutside={e => {
          // Allow clicking outside to close modal when not loading
          if (!isLoading) return
          e.preventDefault()
        }}
      >
        <DialogHeader>
          <DialogTitle className="text-lg text-foreground">
            New project
          </DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-4 pt-1">
          <div className="grid gap-2.5">
            <Label htmlFor="projectName">Project name</Label>
            <Input
              id="projectName"
              value={name}
              onChange={e => {
                const newName = e.target.value
                setName(newName)
                // Clear error state when user starts typing after validation attempt
                if (hasAttemptedValidation && projectError) {
                  setHasAttemptedValidation(false)
                }
              }}
              placeholder="my-project"
              disabled={isLoading}
              className={
                hasAttemptedValidation && projectError
                  ? 'border-destructive'
                  : ''
              }
            />
            {hasAttemptedValidation && projectError ? (
              <p className="text-xs text-destructive">{projectError}</p>
            ) : (
              <p className="text-xs text-muted-foreground">
                Only letters, numbers, underscores (_) and hyphens (-) allowed.
                No spaces.
              </p>
            )}
          </div>

          <div className="grid gap-2.5">
            <Label htmlFor="copyFrom">Copy configuration from (optional)</Label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button
                  id="copyFrom"
                  type="button"
                  disabled={isLoading}
                  className="w-full h-9 rounded-lg border border-input bg-card px-3 text-left flex items-center justify-between disabled:opacity-50 disabled:cursor-not-allowed text-sm text-foreground"
                >
                  <span className="truncate">
                    {selectedSource || 'None, start from scratch'}
                  </span>
                  <FontIcon type="chevron-down" className="w-4 h-4 shrink-0" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-[var(--radix-dropdown-menu-trigger-width)] max-h-64 overflow-auto">
                <DropdownMenuItem
                  onClick={() => setSelectedSource('')}
                  className="cursor-pointer"
                >
                  None, start from scratch
                </DropdownMenuItem>
                {availableProjects.map(projectName => (
                  <DropdownMenuItem
                    key={projectName}
                    onClick={() => setSelectedSource(projectName)}
                    className="cursor-pointer"
                  >
                    {projectName}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
            <p className="text-xs text-muted-foreground">
              {selectedSource
                ? `Configuration will be copied from ${selectedSource} (runtime, prompts, RAG settings)`
                : 'If you prefer, you can copy the config from an existing project into the new one.'}
            </p>
          </div>

          <div className="grid gap-2.5">
            <Label>Where do you plan to deploy this?</Label>
            <div className="flex flex-row gap-3">
              <label className="flex-1 inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-10 hover:bg-accent/20 cursor-pointer">
                <input
                  type="radio"
                  name="deploy"
                  className="h-4 w-4"
                  checked={deployment === 'local'}
                  onChange={() => setDeployment('local')}
                  disabled={isLoading}
                />
                <span className="text-sm">Local machine</span>
              </label>
              <label className="flex-1 inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-10 hover:bg-accent/20 cursor-pointer">
                <input
                  type="radio"
                  name="deploy"
                  className="h-4 w-4"
                  checked={deployment === 'cloud'}
                  onChange={() => setDeployment('cloud')}
                  disabled={isLoading}
                />
                <span className="text-sm">Cloud</span>
              </label>
              <label className="flex-1 inline-flex items-center gap-2 rounded-md border border-input bg-background px-3 h-10 hover:bg-accent/20 cursor-pointer">
                <input
                  type="radio"
                  name="deploy"
                  className="h-4 w-4"
                  checked={deployment === 'unsure'}
                  onChange={() => setDeployment('unsure')}
                  disabled={isLoading}
                />
                <span className="text-sm">Not sure</span>
              </label>
            </div>
          </div>
        </div>

        <DialogFooter className="flex flex-row items-center justify-end gap-2">
          <button
            className="px-3 py-2 rounded-md text-sm text-primary hover:underline disabled:opacity-50"
            onClick={handleCancel}
            disabled={isLoading}
            type="button"
          >
            Cancel
          </button>
          <button
            className={`px-3 py-2 rounded-md text-sm ${
              isValid
                ? 'bg-primary text-primary-foreground hover:opacity-90'
                : 'opacity-50 cursor-not-allowed bg-primary text-primary-foreground'
            }`}
            onClick={handleCreate}
            disabled={!isValid}
            type="button"
          >
            {isLoading ? 'Creating...' : 'Create project'}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default CreateProjectModal
