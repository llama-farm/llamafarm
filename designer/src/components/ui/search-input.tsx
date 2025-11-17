import * as React from 'react'
import { Input } from './input'
import FontIcon from '../../common/FontIcon'
import { cn } from '@/lib/utils'

export interface SearchInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  containerClassName?: string
  onClear?: () => void
}

export function SearchInput({
  className,
  containerClassName,
  onClear,
  value,
  ...props
}: SearchInputProps) {
  const hasValue = value !== undefined && value !== null && String(value).length > 0

  return (
    <div className={cn('relative w-full', containerClassName)}>
      <FontIcon
        type="search"
        className="absolute left-2 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none"
      />
      <Input className={cn('pl-8 pr-8', className)} value={value} {...props} />
      {hasValue && onClear && (
        <button
          type="button"
          onClick={onClear}
          className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-md text-muted-foreground hover:text-foreground hover:bg-accent transition-colors"
          aria-label="Clear search"
        >
          <FontIcon type="close" className="w-3.5 h-3.5" />
        </button>
      )}
    </div>
  )
}

export default SearchInput
