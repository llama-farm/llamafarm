import { Checkbox } from '../ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import FontIcon from '../../common/FontIcon'

interface PromptSetSelectorProps {
  promptSetNames: string[]
  selectedPromptSets: string[]
  onTogglePromptSet: (name: string, checked: boolean | string) => void
  onClearPromptSets: () => void
  disabled?: boolean
  triggerId?: string
  label?: string
}

export function PromptSetSelector({
  promptSetNames,
  selectedPromptSets,
  onTogglePromptSet,
  onClearPromptSets,
  disabled = false,
  triggerId,
  label = 'Prompt sets',
}: PromptSetSelectorProps) {
  return (
    <div>
      {label && (
        <label
          className="text-xs text-muted-foreground mb-1 block"
          htmlFor={triggerId}
        >
          {label}
        </label>
      )}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            id={triggerId}
            disabled={disabled}
            className="w-full h-9 rounded-lg border border-input bg-background px-3 text-left flex items-center justify-between disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className="truncate text-sm flex items-center gap-2">
              {selectedPromptSets.length > 0 ? (
                <>
                  <span className="inline-flex items-center px-2 py-0.5 text-[10px] rounded-full bg-secondary text-secondary-foreground">
                    {selectedPromptSets.length}
                  </span>
                  <span className="truncate">
                    {selectedPromptSets.join(', ')}
                  </span>
                </>
              ) : (
                <span className="text-muted-foreground">All sets</span>
              )}
            </span>
            <FontIcon type="chevron-down" className="w-4 h-4" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-64 max-h-64 overflow-auto">
          {promptSetNames.map(name => (
            <DropdownMenuItem
              key={name}
              className="w-full justify-start text-left"
              onSelect={e => e.preventDefault()}
            >
              <label className="flex items-center gap-2 w-full">
                <Checkbox
                  checked={selectedPromptSets.includes(name)}
                  onCheckedChange={v => onTogglePromptSet(name, v)}
                />
                <span className="text-sm">{name}</span>
              </label>
            </DropdownMenuItem>
          ))}
          <div className="h-px bg-border my-1" />
          <DropdownMenuItem onClick={onClearPromptSets}>
            <span className="text-xs text-muted-foreground">
              Clear selection
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

