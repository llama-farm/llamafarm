---
name: designer-skills
description: Designer subsystem patterns for LlamaFarm. Covers React 18, TanStack Query, TailwindCSS, and Radix UI.
allowed-tools: Read, Grep, Glob
user-invocable: false
---

# Designer Skills for LlamaFarm

Framework-specific patterns and checklists for the Designer subsystem (React 18 + TanStack Query + TailwindCSS + Radix UI).

## Overview

The Designer is a browser-based project workbench for building AI applications. It provides config editing, chat testing, dataset management, RAG configuration, and model selection.

## Tech Stack

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.2 | UI framework with StrictMode |
| TypeScript | 5.2+ | Type safety |
| TanStack Query | v5 | Server state management |
| TailwindCSS | 3.3 | Utility-first CSS |
| Radix UI | 1.x | Accessible component primitives |
| Vite | 6.x | Build tooling and dev server |
| React Router | v7 | Client-side routing |
| Vitest | 1.x | Testing framework |
| axios | 1.x | HTTP client |
| framer-motion | 12.x | Animations |

## Directory Structure

```
designer/src/
  api/          # API service modules (axios-based)
  assets/       # Static assets and icons
  components/   # Feature-organized React components
    ui/         # Radix-based primitive components
  contexts/     # React Context providers
  hooks/        # Custom hooks (TanStack Query wrappers)
  lib/          # Utilities (cn, etc.)
  types/        # TypeScript type definitions
  utils/        # Helper functions
  test/         # Test utilities, factories, mocks
```

## Prerequisites: Shared Skills

Before applying Designer-specific patterns, ensure compliance with:

- [TypeScript Skills](../typescript-skills/SKILL.md) - Strict typing, patterns, security
- [React Skills](../react-skills/SKILL.md) - Component patterns, hooks, state management

## Framework-Specific Guides

| Guide | Description |
|-------|-------------|
| [tanstack-query.md](./tanstack-query.md) | Query/Mutation patterns, caching, invalidation |
| [tailwind.md](./tailwind.md) | TailwindCSS patterns, theming, responsive design |
| [radix.md](./radix.md) | Radix UI component patterns, accessibility |
| [performance.md](./performance.md) | Frontend optimizations, bundle size, lazy loading |

## Key Patterns

### API Client Configuration

```typescript
// Centralized client with interceptors
export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: { 'Content-Type': 'application/json' },
  timeout: 60000,
})

// Error handling interceptor
apiClient.interceptors.response.use(
  response => response,
  (error: AxiosError) => {
    if (error.response?.status === 422) {
      throw new ValidationError('Validation error', error.response.data)
    }
    throw new NetworkError('Request failed', error)
  }
)
```

### Query Client Configuration

```typescript
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,
      gcTime: 5 * 60_000,
      retry: 2,
      retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30_000),
      refetchOnWindowFocus: false,
    },
    mutations: { retry: 1 },
  },
})
```

### Class Merging Utility

```typescript
// lib/utils.ts - Always use cn() for Tailwind classes
import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}
```

### Theme Provider Pattern

```typescript
const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) throw new Error('useTheme must be used within ThemeProvider')
  return context
}

// Apply via Tailwind dark mode class strategy
useEffect(() => {
  document.documentElement.classList.toggle('dark', theme === 'dark')
}, [theme])
```

## Component Conventions

### Feature Components

- Located in `components/{Feature}/` directories
- One component per file, named after the component
- Co-located with feature-specific types and utilities

### UI Primitives

- Located in `components/ui/`
- Wrap Radix UI primitives with Tailwind styling
- Use `forwardRef` for ref forwarding
- Set `displayName` for DevTools

### Icons

- Located in `assets/icons/`
- Functional components accepting SVG props
- Use `lucide-react` for standard icons

## Testing

```typescript
// Use MSW for API mocking
import { server } from '@/test/mocks/server'
import { renderWithProviders } from '@/test/utils'

beforeAll(() => server.listen())
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

test('renders with query data', async () => {
  renderWithProviders(<MyComponent />)
  await screen.findByText('Expected text')
})
```

## Checklist Summary

| Category | Critical | High | Medium | Low |
|----------|----------|------|--------|-----|
| TanStack Query | 3 | 4 | 3 | 2 |
| TailwindCSS | 2 | 3 | 4 | 2 |
| Radix UI | 3 | 3 | 2 | 1 |
| Performance | 2 | 4 | 3 | 2 |
