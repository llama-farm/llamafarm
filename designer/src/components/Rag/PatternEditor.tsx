import { AlertTriangle, Trash2 } from 'lucide-react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { useState } from 'react'

type PatternEditorProps = {
  label: string
  description?: string
  placeholder?: string
  value: string[]
  onChange: (next: string[]) => void
  isSuspicious?: (p: string) => boolean
}

export default function PatternEditor({
  label,
  description,
  placeholder,
  value,
  onChange,
  isSuspicious,
}: PatternEditorProps) {
  const [text, setText] = useState('')

  const add = () => {
    const v = text.trim()
    if (!v) return
    const next = Array.from(new Set([...value, v]))
    onChange(next)
    setText('')
  }

  const removeAt = (idx: number) => {
    onChange(value.filter((_, i) => i !== idx))
  }

  return (
    <div>
      <Label className="text-xs text-muted-foreground">{label}</Label>
      {description ? (
        <div className="text-xs text-muted-foreground mt-1">{description}</div>
      ) : null}
      <div className="mt-2 rounded-md border border-border">
        <div className="p-2 border-b border-border flex items-center gap-2">
          <div className="relative flex-1">
            <Input
              className="bg-background w-full pr-2"
              placeholder={placeholder}
              value={text}
              onChange={e => setText(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') {
                  e.preventDefault()
                  add()
                }
              }}
            />
          </div>
          <Button variant="outline" size="sm" onClick={add}>
            Add entry
          </Button>
        </div>
        <div className="p-2 flex flex-col gap-1">
          {value.length === 0 ? (
            <div className="text-xs text-muted-foreground">No entries yet</div>
          ) : (
            value.map((pat, idx) => (
              <div
                key={`${pat}-${idx}`}
                className="flex items-center justify-between gap-2 rounded-sm bg-accent/10 px-2 py-1"
              >
                <div className="flex items-center gap-1 text-sm">
                  {isSuspicious && isSuspicious(pat) ? (
                    <AlertTriangle className="w-3.5 h-3.5 text-yellow-500" />
                  ) : null}
                  <span className="truncate max-w-[60vw]">{pat}</span>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeAt(idx)}
                  aria-label={`Delete ${pat}`}
                >
                  <Trash2 className="w-4 h-4" />
                </Button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
