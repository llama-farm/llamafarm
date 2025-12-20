import * as React from 'react'

import { cn } from '@/lib/utils'

const Select = React.forwardRef<
  HTMLSelectElement,
  React.ComponentProps<'select'>
>(({ className, children, style, ...props }, ref) => {
  const isInteractive = !props.disabled
  const focusRingClass = isInteractive
    ? 'focus-visible:ring-1 focus-visible:ring-white'
    : 'focus-visible:ring-0'

  return (
    <select
      className={cn(
        'flex h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-base shadow-sm transition-colors focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
        'appearance-none pr-10',
        focusRingClass,
        className
      )}
      style={{
        backgroundImage: `url("data:image/svg+xml;charset=US-ASCII,%3Csvg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='0 0 24 24' fill='none' stroke='%239ca3af' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'%3E%3C/polyline%3E%3C/svg%3E")`,
        backgroundSize: '1rem',
        backgroundPosition: 'right 0.75rem center',
        backgroundRepeat: 'no-repeat',
        ...style,
      }}
      ref={ref}
      {...props}
    >
      {children}
    </select>
  )
})
Select.displayName = 'Select'

export { Select }
