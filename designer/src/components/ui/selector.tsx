import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from './dropdown-menu'
import FontIcon from '../../common/FontIcon'

export interface SelectorOption {
  value: string
  label: string
  description?: string
}

export interface SelectorProps {
  value: string | null
  options: SelectorOption[]
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  loading?: boolean
  emptyMessage?: string
  label?: string
  className?: string
}

export function Selector({
  value,
  options,
  onChange,
  placeholder = 'Select...',
  disabled = false,
  loading = false,
  emptyMessage = 'No options available',
  label,
  className = '',
}: SelectorProps) {
  const selectedOption = options.find(opt => opt.value === value)
  const displayText = selectedOption?.label || placeholder

  return (
    <div className={className}>
      {label && (
        <label className="text-xs text-muted-foreground mb-1 block">
          {label}
        </label>
      )}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button
            type="button"
            disabled={disabled || loading}
            className="w-full h-9 rounded-lg border border-input bg-background px-3 text-left flex items-center justify-between gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <span className={`truncate text-sm ${!selectedOption ? 'text-muted-foreground' : ''}`}>
              {loading ? 'Loading...' : displayText}
            </span>
            {loading ? (
              <span className="w-4 h-4 flex-shrink-0 inline-block animate-spin rounded-full border-2 border-current border-t-transparent" />
            ) : (
              <FontIcon type="chevron-down" className="w-4 h-4 flex-shrink-0" />
            )}
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="max-h-64 overflow-auto">
          {options.length === 0 ? (
            <div className="px-2 py-2 text-xs text-muted-foreground text-center">
              {emptyMessage}
            </div>
          ) : (
            options.map(option => (
              <DropdownMenuItem
                key={option.value}
                className="w-full justify-start text-left cursor-pointer"
                onClick={() => onChange(option.value)}
              >
                <div className="flex flex-col">
                  <span className="text-sm">{option.label}</span>
                  {option.description && (
                    <span className="text-xs text-muted-foreground">
                      {option.description}
                    </span>
                  )}
                </div>
              </DropdownMenuItem>
            ))
          )}
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  )
}

export default Selector
