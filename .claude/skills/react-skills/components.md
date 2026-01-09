# React Component Patterns

Guidelines for building maintainable, accessible React components.

## Component Structure

### Function Components

Always use function components with TypeScript interfaces:

```typescript
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost'
  size?: 'sm' | 'md' | 'lg'
}

export function Button({ variant = 'primary', size = 'md', className, ...props }: ButtonProps) {
  return (
    <button
      className={cn(buttonVariants({ variant, size }), className)}
      {...props}
    />
  )
}
```

### forwardRef Components

Use `forwardRef` when wrapping DOM elements or Radix primitives:

```typescript
const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : 'button'
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  }
)
Button.displayName = 'Button'
```

## Checklist Items

### 1. Components Must Use TypeScript Interfaces

| Attribute | Value |
|-----------|-------|
| Description | All components should have explicit TypeScript prop interfaces |
| Search Pattern | `grep -E "function\s+[A-Z]\w+\s*\(" --include="*.tsx"` |
| Pass Criteria | Props are typed via interface or type alias, not inline |
| Severity | High |
| Fix | Extract inline types to named interfaces above component |

### 2. forwardRef Components Need displayName

| Attribute | Value |
|-----------|-------|
| Description | forwardRef components must set displayName for DevTools |
| Search Pattern | `grep -B5 -A1 "forwardRef" --include="*.tsx"` |
| Pass Criteria | Every forwardRef has corresponding `.displayName = 'ComponentName'` |
| Severity | Medium |
| Fix | Add `ComponentName.displayName = 'ComponentName'` after definition |

### 3. Spread Props Last

| Attribute | Value |
|-----------|-------|
| Description | Spread `...props` should come last to allow overrides |
| Search Pattern | `grep -E "\{\.\.\.props.*," --include="*.tsx"` |
| Pass Criteria | No destructured props after `...props` spread |
| Severity | Medium |
| Fix | Reorder to put `{...props}` at the end of JSX attributes |

### 4. Use Composition Over Conditionals

| Attribute | Value |
|-----------|-------|
| Description | Prefer compound components over complex conditional rendering |
| Search Pattern | `grep -E "^\s*\{.*\?.*:.*\?.*:.*\}" --include="*.tsx"` |
| Pass Criteria | No deeply nested ternaries (max 1 level) |
| Severity | Medium |
| Fix | Extract conditions into separate components or use early returns |

### 5. Avoid Inline Function Definitions in JSX

| Attribute | Value |
|-----------|-------|
| Description | Event handlers should be defined outside JSX for readability |
| Search Pattern | `grep -E "on[A-Z]\w+=\{.*=>" --include="*.tsx"` |
| Pass Criteria | Complex handlers (>1 expression) are extracted to named functions |
| Severity | Low |
| Fix | Extract to `const handleClick = useCallback(() => {...}, [deps])` |

### 6. Use cn() for Class Merging

| Attribute | Value |
|-----------|-------|
| Description | Use cn() utility for Tailwind class merging, not template literals |
| Search Pattern | `grep -E "className=\{\`" --include="*.tsx"` |
| Pass Criteria | Template literals only for simple string interpolation, not conditionals |
| Severity | Low |
| Fix | Replace with `cn('base', condition && 'conditional')` |

### 7. Accessible Interactive Elements

| Attribute | Value |
|-----------|-------|
| Description | Buttons and interactive elements need accessible labels |
| Search Pattern | `grep -E "<button[^>]*>" --include="*.tsx"` |
| Pass Criteria | Buttons have visible text, aria-label, or aria-labelledby |
| Severity | High |
| Fix | Add `aria-label` for icon-only buttons |

### 8. Key Props on List Items

| Attribute | Value |
|-----------|-------|
| Description | Mapped elements must have stable, unique key props |
| Search Pattern | `grep -E "\.map\(" --include="*.tsx"` |
| Pass Criteria | All .map() callbacks return elements with key prop using stable IDs |
| Severity | Critical |
| Fix | Use unique identifier (id, not index) as key |

## Component Variants with CVA

Use `class-variance-authority` for component variants:

```typescript
import { cva, type VariantProps } from 'class-variance-authority'

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md font-medium transition-colors',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
      },
      size: {
        default: 'h-10 px-4 py-2',
        sm: 'h-8 px-3 text-xs',
        lg: 'h-11 px-8',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}
```

## Compound Components

For complex components, use the compound component pattern:

```typescript
const Tabs = ({ children }: { children: React.ReactNode }) => (
  <div className="tabs">{children}</div>
)

const TabList = ({ children }: { children: React.ReactNode }) => (
  <div role="tablist">{children}</div>
)

const TabPanel = ({ children }: { children: React.ReactNode }) => (
  <div role="tabpanel">{children}</div>
)

Tabs.List = TabList
Tabs.Panel = TabPanel

// Usage
<Tabs>
  <Tabs.List>...</Tabs.List>
  <Tabs.Panel>...</Tabs.Panel>
</Tabs>
```

## Provider Composition

Compose providers from the outside in (outermost wraps everything):

```typescript
// main.tsx - correct order
<React.StrictMode>
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </ThemeProvider>
  </QueryClientProvider>
</React.StrictMode>
```
