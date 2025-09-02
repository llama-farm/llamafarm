import * as React from 'react'
import { Input } from './input'
import FontIcon from '../../common/FontIcon'
import { cn } from '@/lib/utils'

export interface SearchInputProps
  extends React.InputHTMLAttributes<HTMLInputElement> {
  containerClassName?: string
}

export function SearchInput({
  className,
  containerClassName,
  ...props
}: SearchInputProps) {
  return (
    <div className={cn('relative w-full', containerClassName)}>
      <FontIcon
        type="search"
        className="absolute left-2 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none"
      />
      <Input className={cn('pl-8', className)} {...props} />
    </div>
  )
}

export default SearchInput
