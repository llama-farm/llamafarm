# Radix UI Patterns

Best practices for Radix UI primitives in the Designer subsystem.

## Overview

The Designer uses Radix UI primitives for accessible, unstyled components:
- Dialog, Tooltip, Dropdown Menu
- Checkbox, Switch, Label
- Collapsible
- Slot for component composition

All primitives are wrapped in `components/ui/` with Tailwind styling.

## Component Wrapping Pattern

```typescript
import * as React from 'react'
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { cn } from '@/lib/utils'

const Dialog = DialogPrimitive.Root
const DialogTrigger = DialogPrimitive.Trigger
const DialogPortal = DialogPrimitive.Portal

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, children, ...props }, ref) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'fixed left-[50%] top-[50%] translate-x-[-50%] translate-y-[-50%]',
        'z-50 w-full max-w-lg rounded-lg border bg-background p-6 shadow-lg',
        'data-[state=open]:animate-in data-[state=closed]:animate-out',
        className
      )}
      {...props}
    >
      {children}
    </DialogPrimitive.Content>
  </DialogPortal>
))
DialogContent.displayName = DialogPrimitive.Content.displayName
```

## Checklist Items

### 1. ForwardRef on All Wrapped Components

| Attribute | Value |
|-----------|-------|
| Description | Radix wrapper components must use forwardRef |
| Search Pattern | `grep -L "forwardRef" components/ui/*.tsx` |
| Pass Criteria | All components wrapping Radix primitives use React.forwardRef |
| Severity | Critical |
| Recommendation | Wrap with `React.forwardRef<ElementRef, ComponentProps>()` |

### 2. DisplayName Set on ForwardRef Components

| Attribute | Value |
|-----------|-------|
| Description | ForwardRef components must have displayName for DevTools |
| Search Pattern | `grep -A1 "forwardRef" components/ui/*.tsx \| grep -v "displayName"` |
| Pass Criteria | Every forwardRef component has .displayName set |
| Severity | Critical |
| Recommendation | Add `Component.displayName = Primitive.displayName` |

### 3. Dialog Has Accessible Title

| Attribute | Value |
|-----------|-------|
| Description | Dialog components must include DialogTitle for screen readers |
| Search Pattern | `grep -A20 "<Dialog" components/**/*.tsx \| grep "DialogTitle"` |
| Pass Criteria | Every Dialog usage includes DialogTitle (visible or sr-only) |
| Severity | Critical |
| Recommendation | Add `<DialogTitle>` or `<DialogTitle className="sr-only">` |

### 4. Portal Used for Overlays

| Attribute | Value |
|-----------|-------|
| Description | Dialog, Tooltip, Dropdown must render in Portal |
| Search Pattern | `grep -E "DialogPortal|TooltipPortal" components/ui/*.tsx` |
| Pass Criteria | Overlay content wrapped in Portal primitive |
| Severity | High |
| Recommendation | Wrap content in `<DialogPrimitive.Portal>` |

### 5. Proper Z-Index Layering

| Attribute | Value |
|-----------|-------|
| Description | Overlays must use consistent z-index values |
| Search Pattern | `grep -E "z-50|z-40" components/ui/*.tsx` |
| Pass Criteria | Overlays use z-50, overlay backgrounds use z-40 |
| Severity | High |
| Recommendation | Use `z-50` for content, ensure overlay is below content |

### 6. Keyboard Navigation Works

| Attribute | Value |
|-----------|-------|
| Description | Interactive components must support keyboard navigation |
| Search Pattern | `grep -E "onKeyDown|aria-" components/ui/*.tsx` |
| Pass Criteria | Radix handles this by default; custom handlers don't break it |
| Severity | High |
| Recommendation | Don't override onKeyDown unless extending functionality |

### 7. Animation States Use Data Attributes

| Attribute | Value |
|-----------|-------|
| Description | Entry/exit animations should use Radix data-state attributes |
| Search Pattern | `grep -E "data-\[state=" components/ui/*.tsx` |
| Pass Criteria | Animations use `data-[state=open]:` and `data-[state=closed]:` |
| Severity | High |
| Recommendation | Use `data-[state=open]:animate-in data-[state=closed]:animate-out` |

### 8. Close Button Has Screen Reader Label

| Attribute | Value |
|-----------|-------|
| Description | Close buttons must have accessible labels |
| Search Pattern | `grep -A5 "DialogClose\|Close" components/ui/*.tsx` |
| Pass Criteria | Close buttons have aria-label or sr-only text |
| Severity | Medium |
| Recommendation | Add `<span className="sr-only">Close</span>` |

### 9. Tooltip Has Delay Configuration

| Attribute | Value |
|-----------|-------|
| Description | Tooltips should have appropriate show/hide delays |
| Search Pattern | `grep -E "delayDuration|skipDelayDuration" components/ui/*.tsx` |
| Pass Criteria | TooltipProvider has delayDuration configured |
| Severity | Medium |
| Recommendation | Use `<TooltipProvider delayDuration={300}>` |

### 10. Checkbox/Switch Paired with Label

| Attribute | Value |
|-----------|-------|
| Description | Form controls must be associated with labels |
| Search Pattern | `grep -B5 -A5 "<Checkbox\|<Switch" components/**/*.tsx` |
| Pass Criteria | Controls have id and associated Label with htmlFor |
| Severity | Medium |
| Recommendation | Add `id` prop and wrap with `<Label htmlFor={id}>` |

### 11. Slot Used for Polymorphic Components

| Attribute | Value |
|-----------|-------|
| Description | Use Radix Slot for asChild pattern |
| Search Pattern | `grep -E "asChild|Slot" components/ui/*.tsx` |
| Pass Criteria | Polymorphic components use Slot when asChild is true |
| Severity | Low |
| Recommendation | `const Comp = asChild ? Slot : 'button'` |

## Patterns

### Dialog Component

```tsx
import * as DialogPrimitive from '@radix-ui/react-dialog'
import { X } from 'lucide-react'
import { cn } from '@/lib/utils'

const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content> & {
    hideCloseButton?: boolean
  }
>(({ className, children, hideCloseButton, ...props }, ref) => (
  <DialogPrimitive.Portal>
    <DialogPrimitive.Overlay
      className={cn(
        'fixed inset-0 z-50 bg-black/80',
        'data-[state=open]:animate-in data-[state=closed]:animate-out',
        'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0'
      )}
    />
    <DialogPrimitive.Content
      ref={ref}
      className={cn(
        'fixed left-[50%] top-[50%] z-50 translate-x-[-50%] translate-y-[-50%]',
        'w-full max-w-lg rounded-lg border bg-background p-6 shadow-lg',
        'data-[state=open]:animate-in data-[state=closed]:animate-out',
        'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
        'data-[state=closed]:zoom-out-95 data-[state=open]:zoom-in-95',
        className
      )}
      {...props}
    >
      {children}
      {!hideCloseButton && (
        <DialogPrimitive.Close className="absolute right-4 top-4 rounded-sm opacity-70 hover:opacity-100 focus:outline-none focus:ring-2 focus:ring-ring">
          <X className="h-4 w-4" />
          <span className="sr-only">Close</span>
        </DialogPrimitive.Close>
      )}
    </DialogPrimitive.Content>
  </DialogPrimitive.Portal>
))
DialogContent.displayName = DialogPrimitive.Content.displayName
```

### Tooltip Component

```tsx
import * as TooltipPrimitive from '@radix-ui/react-tooltip'

const TooltipProvider = TooltipPrimitive.Provider
const Tooltip = TooltipPrimitive.Root
const TooltipTrigger = TooltipPrimitive.Trigger

const TooltipContent = React.forwardRef<
  React.ElementRef<typeof TooltipPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof TooltipPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <TooltipPrimitive.Content
    ref={ref}
    sideOffset={sideOffset}
    className={cn(
      'z-50 overflow-hidden rounded-md border px-3 py-1.5 text-sm shadow-md',
      'bg-popover text-popover-foreground',
      'animate-in fade-in-0 zoom-in-95',
      'data-[state=closed]:animate-out data-[state=closed]:fade-out-0',
      'data-[side=bottom]:slide-in-from-top-2',
      'data-[side=top]:slide-in-from-bottom-2',
      className
    )}
    {...props}
  />
))
TooltipContent.displayName = TooltipPrimitive.Content.displayName
```

### Checkbox with Label

```tsx
import * as CheckboxPrimitive from '@radix-ui/react-checkbox'
import { Check } from 'lucide-react'

const Checkbox = React.forwardRef<
  React.ElementRef<typeof CheckboxPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof CheckboxPrimitive.Root>
>(({ className, ...props }, ref) => (
  <CheckboxPrimitive.Root
    ref={ref}
    className={cn(
      'peer h-4 w-4 shrink-0 rounded-sm border border-input shadow',
      'focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring',
      'disabled:cursor-not-allowed disabled:opacity-50',
      'data-[state=checked]:bg-primary data-[state=checked]:text-primary-foreground',
      className
    )}
    {...props}
  >
    <CheckboxPrimitive.Indicator className="flex items-center justify-center text-current">
      <Check className="h-3.5 w-3.5" />
    </CheckboxPrimitive.Indicator>
  </CheckboxPrimitive.Root>
))
Checkbox.displayName = CheckboxPrimitive.Root.displayName

// Usage
<div className="flex items-center gap-2">
  <Checkbox id="terms" />
  <Label htmlFor="terms">Accept terms and conditions</Label>
</div>
```

### Polymorphic Button with Slot

```tsx
import { Slot } from '@radix-ui/react-slot'

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return (
      <Comp
        className={cn(buttonVariants({ className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'

// Usage - renders as anchor
<Button asChild>
  <a href="/dashboard">Go to Dashboard</a>
</Button>
```

### Dropdown Menu

```tsx
import * as DropdownMenuPrimitive from '@radix-ui/react-dropdown-menu'

const DropdownMenu = DropdownMenuPrimitive.Root
const DropdownMenuTrigger = DropdownMenuPrimitive.Trigger

const DropdownMenuContent = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Content>
>(({ className, sideOffset = 4, ...props }, ref) => (
  <DropdownMenuPrimitive.Portal>
    <DropdownMenuPrimitive.Content
      ref={ref}
      sideOffset={sideOffset}
      className={cn(
        'z-50 min-w-[8rem] overflow-hidden rounded-md border bg-popover p-1 text-popover-foreground shadow-md',
        'data-[state=open]:animate-in data-[state=closed]:animate-out',
        'data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0',
        'data-[side=bottom]:slide-in-from-top-2',
        className
      )}
      {...props}
    />
  </DropdownMenuPrimitive.Portal>
))
DropdownMenuContent.displayName = DropdownMenuPrimitive.Content.displayName

const DropdownMenuItem = React.forwardRef<
  React.ElementRef<typeof DropdownMenuPrimitive.Item>,
  React.ComponentPropsWithoutRef<typeof DropdownMenuPrimitive.Item>
>(({ className, ...props }, ref) => (
  <DropdownMenuPrimitive.Item
    ref={ref}
    className={cn(
      'relative flex cursor-default select-none items-center rounded-sm px-2 py-1.5 text-sm outline-none',
      'focus:bg-accent focus:text-accent-foreground',
      'data-[disabled]:pointer-events-none data-[disabled]:opacity-50',
      className
    )}
    {...props}
  />
))
DropdownMenuItem.displayName = DropdownMenuPrimitive.Item.displayName
```

## Anti-Patterns

### Avoid: Missing ForwardRef

```tsx
// Bad - no ref forwarding
const DialogContent = ({ className, ...props }) => (
  <DialogPrimitive.Content className={cn('...', className)} {...props} />
)

// Good - with forwardRef
const DialogContent = React.forwardRef<
  React.ElementRef<typeof DialogPrimitive.Content>,
  React.ComponentPropsWithoutRef<typeof DialogPrimitive.Content>
>(({ className, ...props }, ref) => (
  <DialogPrimitive.Content ref={ref} className={cn('...', className)} {...props} />
))
DialogContent.displayName = DialogPrimitive.Content.displayName
```

### Avoid: Missing Accessible Labels

```tsx
// Bad - no accessible label
<DialogPrimitive.Close>
  <X className="h-4 w-4" />
</DialogPrimitive.Close>

// Good - with sr-only label
<DialogPrimitive.Close>
  <X className="h-4 w-4" />
  <span className="sr-only">Close</span>
</DialogPrimitive.Close>
```

### Avoid: Breaking Keyboard Navigation

```tsx
// Bad - prevents default keyboard handling
<DialogPrimitive.Content onKeyDown={(e) => e.stopPropagation()}>

// Good - extend without breaking
<DialogPrimitive.Content onKeyDown={(e) => {
  if (e.key === 'Enter') handleSubmit()
  // Don't prevent default or stop propagation
}}>
```
