import * as React from 'react'

import { cn } from '@/lib/utils'

const Textarea = React.forwardRef<
  HTMLTextAreaElement,
  React.ComponentProps<'textarea'>
>(({ className, ...props }, ref) => {
  const isInteractive = !props.readOnly && !props.disabled
  const isInvalid = (props as any)['aria-invalid'] || (props as any)['data-invalid']
  const focusRingClass = isInteractive
    ? isInvalid
      ? 'focus-visible:ring-1 focus-visible:ring-red-500 dark:focus-visible:ring-red-300'
      : 'focus-visible:ring-1 focus-visible:ring-white'
    : 'focus-visible:ring-0'
  return (
    <textarea
      className={cn(
        'flex min-h-[60px] w-full rounded-md border border-input bg-transparent px-3 py-2 text-base shadow-sm placeholder:text-muted-foreground focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50 md:text-sm',
        focusRingClass,
        className
      )}
      ref={ref}
      {...props}
    />
  )
})
Textarea.displayName = 'Textarea'

export { Textarea }
