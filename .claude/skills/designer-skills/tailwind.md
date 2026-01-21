# TailwindCSS Patterns

Best practices for TailwindCSS in the Designer subsystem.

## Configuration

The Designer uses TailwindCSS 3.3 with:
- Dark mode via `class` strategy
- CSS variables for theming (HSL colors)
- `tailwind-merge` for class deduplication
- `class-variance-authority` (CVA) for component variants
- Custom scrollbar utilities

## Class Merging Utility

Always use `cn()` to merge Tailwind classes:

```typescript
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// Usage
cn('px-4 py-2', isActive && 'bg-primary', className)
```

## Checklist Items

### 1. Always Use cn() for Class Merging

| Attribute | Value |
|-----------|-------|
| Description | All dynamic className combinations must use cn() utility |
| Search Pattern | `grep -E "className=\{.*\+" components/**/*.tsx` |
| Pass Criteria | No string concatenation for classNames |
| Severity | Critical |
| Recommendation | Replace `className={base + ' ' + extra}` with `cn(base, extra)` |

### 2. Dark Mode Classes Present

| Attribute | Value |
|-----------|-------|
| Description | Components with colors must have dark mode variants |
| Search Pattern | `grep -E "bg-|text-|border-" components/**/*.tsx \| grep -v "dark:"` |
| Pass Criteria | Color utilities paired with dark: variants or use CSS variables |
| Severity | Critical |
| Recommendation | Add `dark:bg-slate-800` or use `bg-background` (CSS variable) |

### 3. Use Semantic Color Variables

| Attribute | Value |
|-----------|-------|
| Description | Prefer semantic CSS variable colors over raw values |
| Search Pattern | `grep -E "bg-slate-|bg-gray-|text-slate-|text-gray-" components/**/*.tsx` |
| Pass Criteria | Use `bg-background`, `text-foreground`, `border-border` instead |
| Severity | High |
| Recommendation | Replace `bg-slate-900` with `bg-background` for automatic theming |

### 4. CVA for Component Variants

| Attribute | Value |
|-----------|-------|
| Description | Components with variants should use class-variance-authority |
| Search Pattern | `grep -E "variant.*===|size.*===" components/**/*.tsx` |
| Pass Criteria | Conditional variant classes use CVA, not ternaries |
| Severity | High |
| Recommendation | Define variants with `cva()` and `VariantProps<typeof variants>` |

### 5. Responsive Breakpoints Used Correctly

| Attribute | Value |
|-----------|-------|
| Description | Use mobile-first responsive design |
| Search Pattern | `grep -E "sm:|md:|lg:|xl:" components/**/*.tsx` |
| Pass Criteria | Base styles are mobile, breakpoints add desktop styles |
| Severity | High |
| Recommendation | Write `text-sm md:text-base` not `text-base sm:text-sm` |

### 6. No Inline Styles

| Attribute | Value |
|-----------|-------|
| Description | Avoid inline style attributes when Tailwind utilities exist |
| Search Pattern | `grep -E "style=\{\{" components/**/*.tsx` |
| Pass Criteria | No style={{ }} except for dynamic calculated values |
| Severity | High |
| Recommendation | Use Tailwind utilities: `style={{width: '100px'}}` -> `w-[100px]` |

### 7. Consistent Spacing Scale

| Attribute | Value |
|-----------|-------|
| Description | Use Tailwind spacing scale consistently |
| Search Pattern | `grep -E "p-\[|m-\[|gap-\[|space-" components/**/*.tsx` |
| Pass Criteria | Arbitrary values only when scale values don't fit |
| Severity | Medium |
| Recommendation | Prefer `p-4` over `p-[16px]`, use `p-[18px]` only if needed |

### 8. Focus States Defined

| Attribute | Value |
|-----------|-------|
| Description | Interactive elements must have visible focus states |
| Search Pattern | `grep -E "focus:|focus-visible:" components/**/*.tsx` |
| Pass Criteria | Buttons, inputs, links have focus-visible styles |
| Severity | Medium |
| Recommendation | Add `focus-visible:ring-2 focus-visible:ring-ring` |

### 9. Transition Utilities for Animations

| Attribute | Value |
|-----------|-------|
| Description | CSS transitions should use Tailwind utilities |
| Search Pattern | `grep -E "transition-|duration-|ease-" components/**/*.tsx` |
| Pass Criteria | Transitions use utilities, not custom CSS |
| Severity | Medium |
| Recommendation | Use `transition-colors duration-200` for color changes |

### 10. Hover States on Interactive Elements

| Attribute | Value |
|-----------|-------|
| Description | Clickable elements should have hover feedback |
| Search Pattern | `grep -E "onClick=\{" components/**/*.tsx` |
| Pass Criteria | Elements with onClick have hover: classes |
| Severity | Medium |
| Recommendation | Add `hover:bg-accent` or `hover:opacity-80` |

### 11. No Tailwind in JavaScript Logic

| Attribute | Value |
|-----------|-------|
| Description | Tailwind classes should be in JSX, not computed in JS |
| Search Pattern | `grep -E "const.*=.*'(bg-|text-|p-|m-)" components/**/*.tsx` |
| Pass Criteria | Classes defined inline or via cn(), not in variables |
| Severity | Low |
| Recommendation | Use cn() directly in className, not intermediate variables |

## Patterns

### Button with CVA

```typescript
import { cva, type VariantProps } from 'class-variance-authority'
import { cn } from '@/lib/utils'

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-md text-sm font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50',
  {
    variants: {
      variant: {
        default: 'bg-primary text-primary-foreground hover:bg-primary/90',
        destructive: 'bg-destructive text-destructive-foreground hover:bg-destructive/90',
        outline: 'border border-input bg-background hover:bg-accent',
        secondary: 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
        ghost: 'hover:bg-accent hover:text-accent-foreground',
        link: 'text-primary underline-offset-4 hover:underline',
      },
      size: {
        default: 'px-4 py-2',
        sm: 'h-8 rounded-md px-3 text-xs',
        lg: 'h-10 rounded-md px-8',
        icon: 'h-9 w-9',
      },
    },
    defaultVariants: {
      variant: 'default',
      size: 'default',
    },
  }
)

interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

function Button({ className, variant, size, ...props }: ButtonProps) {
  return (
    <button
      className={cn(buttonVariants({ variant, size, className }))}
      {...props}
    />
  )
}
```

### Responsive Layout

```tsx
<div className="flex flex-col gap-4 md:flex-row md:gap-6">
  <aside className="w-full md:w-64 lg:w-80">
    {/* Sidebar */}
  </aside>
  <main className="flex-1 min-w-0">
    {/* Content */}
  </main>
</div>
```

### Dark Mode Component

```tsx
<div className="bg-background text-foreground border border-border rounded-lg p-4">
  <h2 className="text-lg font-semibold text-foreground">
    Title
  </h2>
  <p className="text-sm text-muted-foreground">
    Description text
  </p>
</div>
```

### Conditional Classes with cn()

```tsx
<button
  className={cn(
    'px-4 py-2 rounded-md transition-colors',
    isActive
      ? 'bg-primary text-primary-foreground'
      : 'bg-secondary text-secondary-foreground hover:bg-secondary/80',
    isDisabled && 'opacity-50 cursor-not-allowed',
    className
  )}
>
  {children}
</button>
```

### Custom Scrollbar

```tsx
<div className="overflow-y-auto scrollbar-thin max-h-96">
  {/* Scrollable content */}
</div>

<div className="overflow-y-auto scrollbar-custom h-[400px]">
  {/* Editor with custom scrollbar */}
</div>
```

## CSS Variable Color System

```css
/* Theme colors defined in index.css */
:root {
  --background: 0 0% 100%;
  --foreground: 222.2 84% 4.9%;
  --primary: 222.2 47.4% 11.2%;
  --primary-foreground: 210 40% 98%;
  --secondary: 210 40% 96%;
  --secondary-foreground: 222.2 47.4% 11.2%;
  --muted: 210 40% 96%;
  --muted-foreground: 215.4 16.3% 46.9%;
  --accent: 210 40% 96%;
  --accent-foreground: 222.2 47.4% 11.2%;
  --destructive: 0 84.2% 60.2%;
  --destructive-foreground: 210 40% 98%;
  --border: 214.3 31.8% 91.4%;
  --input: 214.3 31.8% 91.4%;
  --ring: 222.2 84% 4.9%;
}

.dark {
  --background: 222.2 84% 4.9%;
  --foreground: 210 40% 98%;
  /* ... dark theme values */
}
```

## Anti-Patterns

### Avoid: String Concatenation

```tsx
// Bad
className={'px-4 ' + (isActive ? 'bg-primary' : 'bg-secondary')}

// Good
className={cn('px-4', isActive ? 'bg-primary' : 'bg-secondary')}
```

### Avoid: Hardcoded Colors Without Dark Mode

```tsx
// Bad - no dark mode support
className="bg-white text-black"

// Good - uses CSS variables
className="bg-background text-foreground"

// Good - explicit dark mode
className="bg-white dark:bg-slate-900 text-black dark:text-white"
```

### Avoid: Desktop-First Responsive

```tsx
// Bad - desktop first
className="text-lg sm:text-base xs:text-sm"

// Good - mobile first
className="text-sm md:text-base lg:text-lg"
```
