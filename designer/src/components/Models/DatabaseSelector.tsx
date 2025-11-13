import { Checkbox } from '../ui/checkbox'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '../ui/dropdown-menu'
import FontIcon from '../../common/FontIcon'

interface DatabaseSelectorProps {
  databases: Array<{ name: string; type: string; is_default: boolean }>
  selectedDatabases: string[]
  onToggleDatabase: (name: string, checked: boolean | string) => void
  onClearDatabases: () => void
  disabled?: boolean
  triggerId?: string
  label?: string
}

export function DatabaseSelector({
  databases,
  selectedDatabases,
  onToggleDatabase,
  onClearDatabases,
  disabled = false,
  triggerId,
  label = 'Databases',
}: DatabaseSelectorProps) {
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
            className={`${label ? 'w-full h-9' : 'h-auto'} rounded-lg border border-input bg-background ${label ? 'px-3' : 'px-2 py-1'} text-left flex items-center justify-between disabled:opacity-50 disabled:cursor-not-allowed`}
          >
            <span className={`truncate ${label ? 'text-sm' : 'text-xs'} flex items-center gap-2`}>
              {selectedDatabases.length > 0 ? (
                <>
                  <span className="inline-flex items-center px-2 py-0.5 text-[10px] rounded-full bg-secondary text-secondary-foreground">
                    {selectedDatabases.length}
                  </span>
                  <span className="truncate">
                    {selectedDatabases.join(', ')}
                  </span>
                </>
              ) : (
                <span className="text-muted-foreground">All databases</span>
              )}
            </span>
            <FontIcon type="chevron-down" className="w-4 h-4" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className="w-64 max-h-64 overflow-auto">
          {databases.map(database => (
            <DropdownMenuItem
              key={database.name}
              className="w-full justify-start text-left"
              onSelect={e => e.preventDefault()}
            >
              <label className="flex items-center gap-2 w-full">
                <Checkbox
                  checked={selectedDatabases.includes(database.name)}
                  onCheckedChange={v => onToggleDatabase(database.name, v)}
                />
                <span className="text-sm flex-1">{database.name}</span>
                {database.is_default && (
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/20 text-primary">
                    default
                  </span>
                )}
              </label>
            </DropdownMenuItem>
          ))}
          <div className="h-px bg-border my-1" />
          <DropdownMenuItem onClick={onClearDatabases}>
            <span className="text-xs text-muted-foreground">
              Clear selection
            </span>
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}
