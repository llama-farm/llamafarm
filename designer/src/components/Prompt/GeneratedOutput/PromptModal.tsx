import { useEffect, useState } from 'react'
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '../../ui/dialog'
import FontIcon from '../../../common/FontIcon'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../../ui/dropdown-menu'

export type PromptModalMode = 'create' | 'edit'

interface PromptModalProps {
  isOpen: boolean
  mode: PromptModalMode
  initialText?: string
  initialRole?: 'system' | 'assistant' | 'user'
  promptSets?: Array<{ name: string }>
  selectedSetIndex?: number
  onClose: () => void
  onSave: (
    text: string,
    role: 'system' | 'assistant' | 'user',
    setIndex?: number
  ) => void
  onDelete?: () => void
}

const PromptModal: React.FC<PromptModalProps> = ({
  isOpen,
  mode,
  initialText = '',
  initialRole = 'system',
  promptSets = [],
  selectedSetIndex = 0,
  onClose,
  onSave,
  onDelete,
}) => {
  const [text, setText] = useState(initialText)
  const [role, setRole] = useState<'system' | 'assistant' | 'user'>(initialRole)
  const [setIndex, setSetIndex] = useState(selectedSetIndex)

  useEffect(() => {
    if (isOpen) {
      setText(initialText)
      setRole(initialRole)
      setSetIndex(selectedSetIndex)
    }
  }, [isOpen, mode, initialText, initialRole, selectedSetIndex])

  const title = mode === 'create' ? 'Add prompt' : 'Edit prompt'
  const cta = mode === 'create' ? 'Add' : 'Save'
  const isValid = text.trim().length > 0

  const handleDelete = () => {
    if (!onDelete) return
    const ok = confirm('Delete this prompt?')
    if (ok) onDelete()
  }

  return (
    <Dialog open={isOpen} onOpenChange={v => (!v ? onClose() : undefined)}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle className="text-lg text-foreground">{title}</DialogTitle>
        </DialogHeader>

        <div className="flex flex-col gap-3 pt-1">
          {mode === 'create' && (
            <div className="flex items-start justify-between gap-3 p-3 rounded-md bg-secondary/40 border border-border">
              <p className="text-xs text-muted-foreground">
                Explain how the model should use context chunks and what to do
                when no documents are found. Keep instructions concise to
                preserve tokens and avoid conflicting guidance.
              </p>
              <a
                href="https://docs.llamafarm.dev/docs/prompts"
                target="_blank"
                rel="noreferrer"
                className="shrink-0 inline-flex items-center px-2 py-1 rounded-md border border-input text-xs hover:bg-accent/30"
              >
                Learn more
              </a>
            </div>
          )}
          {mode === 'create' && promptSets.length > 0 && (
            <div>
              <label className="text-xs text-muted-foreground mb-0">
                Prompt set
              </label>
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <button className="w-full h-9 rounded-md border border-input bg-background px-3 text-left flex items-center justify-between mt-1">
                    <span className="text-sm">
                      {promptSets[setIndex]?.name || 'Select set'}
                    </span>
                    <FontIcon type="chevron-down" className="w-4 h-4" />
                  </button>
                </DropdownMenuTrigger>
                <DropdownMenuContent className="w-full">
                  {promptSets.map((set, idx) => (
                    <DropdownMenuItem
                      key={idx}
                      className="w-full justify-start text-left"
                      onClick={() => setSetIndex(idx)}
                    >
                      {set.name}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>
            </div>
          )}
          <div>
            <label className="text-xs text-muted-foreground mb-0">Role</label>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="w-full h-9 rounded-md border border-input bg-background px-3 text-left flex items-center justify-between mt-1">
                  <span className="text-sm">{role}</span>
                  <FontIcon type="chevron-down" className="w-4 h-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="w-64">
                {(['system', 'assistant', 'user'] as const).map(opt => (
                  <DropdownMenuItem
                    key={opt}
                    className="w-full justify-start text-left"
                    onClick={() => setRole(opt)}
                  >
                    {opt}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <div>
            <label className="text-xs text-muted-foreground mb-0">
              Prompt text
            </label>
            <textarea
              rows={10}
              className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-input text-foreground font-mono text-sm"
              placeholder="You are a helpful assistant. When context is provided, cite sources by title. If no relevant information is found, answer from general knowledge."
              value={text}
              onChange={e => setText(e.target.value)}
            />
          </div>
        </div>

        <DialogFooter className="flex flex-row items-center justify-between sm:justify-between gap-2">
          {mode === 'edit' ? (
            <button
              className="px-3 py-2 rounded-md bg-destructive text-destructive-foreground hover:opacity-90 text-sm"
              onClick={handleDelete}
              type="button"
            >
              Delete
            </button>
          ) : (
            <div />
          )}
          <div className="flex items-center gap-2 ml-auto">
            <button
              className="px-3 py-2 rounded-md text-sm text-primary hover:underline"
              onClick={onClose}
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
              disabled={!isValid}
              onClick={() =>
                onSave(
                  text.trim(),
                  role,
                  mode === 'create' ? setIndex : undefined
                )
              }
            >
              {cta}
            </button>
          </div>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

export default PromptModal
