---
name: react-skills
description: React 18 patterns for LlamaFarm Designer. Covers components, hooks, TanStack Query, and testing.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# React Skills for LlamaFarm Designer

Best practices and patterns for React 18 development in the Designer subsystem.

## Tech Stack

- **React 18.2** with StrictMode
- **TypeScript 5.2+** for type safety
- **TanStack Query v5** for server state management
- **React Router v7** for client-side routing
- **TailwindCSS** with `tailwind-merge` and `clsx` for styling
- **Radix UI** primitives for accessible components
- **Vite** for build tooling
- **Vitest** + React Testing Library for testing

## Directory Structure

```
designer/src/
  api/          # API service functions
  components/   # React components (feature-organized)
  contexts/     # React context providers
  hooks/        # Custom hooks
  lib/          # Utility functions (cn, etc.)
  types/        # TypeScript type definitions
  utils/        # Helper functions
  test/         # Test utilities and mocks
```

## Core Patterns

### Component Composition

- Use composition over inheritance
- Prefer small, focused components
- Use `forwardRef` for components that wrap DOM elements
- Apply `displayName` to forwardRef components for DevTools

### State Management

- **Local UI state**: `useState`, `useReducer`
- **Server state**: TanStack Query (`useQuery`, `useMutation`)
- **Shared UI state**: React Context with custom hooks
- **Form state**: Controlled components with validation

### Hooks

- Follow Rules of Hooks (top-level, consistent order)
- Create custom hooks for reusable logic
- Use query key factories for TanStack Query
- Memoize expensive computations with `useMemo`
- Stabilize callbacks with `useCallback`

### Styling

- Use `cn()` from `lib/utils` to merge Tailwind classes
- Use `cva` (class-variance-authority) for component variants
- Follow dark mode conventions with `dark:` prefix

## Related Guides

- [components.md](./components.md) - Component patterns
- [hooks.md](./hooks.md) - Hook patterns and rules
- [state.md](./state.md) - State management patterns
- [performance.md](./performance.md) - Performance optimization
- [security.md](./security.md) - Security best practices

## Quick Reference

```typescript
// Utility for merging Tailwind classes
import { cn } from '@/lib/utils'
cn('base-class', condition && 'conditional-class', className)

// Query key factory pattern
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  list: (ns: string) => [...projectKeys.lists(), ns] as const,
}

// Context with validation hook
const MyContext = createContext<MyContextType | undefined>(undefined)
export function useMyContext() {
  const ctx = useContext(MyContext)
  if (!ctx) throw new Error('useMyContext must be within MyProvider')
  return ctx
}
```

## Testing

```typescript
import { renderWithProviders } from '@/test/utils'
import { screen } from '@testing-library/react'

test('renders component', () => {
  renderWithProviders(<MyComponent />)
  expect(screen.getByText('Hello')).toBeInTheDocument()
})
```
