# State Management Patterns

Guidelines for managing state in React applications with TanStack Query and Context.

## State Categories

| Type | Tool | Example |
|------|------|---------|
| Server state | TanStack Query | API data, user profiles |
| Local UI state | useState | Form inputs, toggles |
| Shared UI state | Context | Theme, auth status |
| URL state | React Router | Filters, pagination |
| Form state | Controlled components | Input values, validation |

## TanStack Query Patterns

### QueryClient Configuration

```typescript
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 60_000,           // 1 minute
      gcTime: 5 * 60_000,          // 5 minutes (was cacheTime)
      retry: 2,
      retryDelay: attemptIndex => Math.min(1000 * 2 ** attemptIndex, 30_000),
      refetchOnWindowFocus: false,
      refetchOnMount: true,
      refetchOnReconnect: true,
    },
    mutations: {
      retry: 1,
    },
  },
})
```

### Query Key Factory Pattern

```typescript
export const chatCompletionsKeys = {
  all: ['chatCompletions'] as const,
  completions: (namespace: string, projectId: string) =>
    [...chatCompletionsKeys.all, namespace, projectId] as const,
  session: (namespace: string, projectId: string, sessionId: string) =>
    [...chatCompletionsKeys.completions(namespace, projectId), 'session', sessionId] as const,
}
```

## Checklist Items

### 1. Server State Uses TanStack Query

| Attribute | Value |
|-----------|-------|
| Description | API data should use useQuery/useMutation, not useState+useEffect |
| Search Pattern | `grep -E "useState.*fetch\|useEffect.*fetch" --include="*.ts" --include="*.tsx"` |
| Pass Criteria | No manual fetch+setState patterns for server data |
| Severity | High |
| Fix | Convert to useQuery with queryFn |

### 2. Query Keys Use Factory Pattern

| Attribute | Value |
|-----------|-------|
| Description | Query keys should use factory functions for consistency |
| Search Pattern | `grep -E "queryKey:\s*\[" --include="*.ts"` |
| Pass Criteria | Query keys reference factory (e.g., `projectKeys.list(ns)`) |
| Severity | Medium |
| Fix | Create key factory and reference it |

### 3. Mutations Invalidate Related Queries

| Attribute | Value |
|-----------|-------|
| Description | Mutations should invalidate or update related query caches |
| Search Pattern | `grep -A20 "useMutation" --include="*.ts"` |
| Pass Criteria | onSuccess includes invalidateQueries or setQueryData |
| Severity | High |
| Fix | Add `queryClient.invalidateQueries({ queryKey: ... })` in onSuccess |

### 4. Queries Have Proper enabled Condition

| Attribute | Value |
|-----------|-------|
| Description | Queries with dependencies should use enabled option |
| Search Pattern | `grep -A10 "useQuery" --include="*.ts"` |
| Pass Criteria | Queries depending on params have `enabled: !!param` |
| Severity | Medium |
| Fix | Add `enabled: !!dependency && !!otherDep` |

### 5. Context Providers Have Proper Memoization

| Attribute | Value |
|-----------|-------|
| Description | Context values should be memoized to prevent unnecessary rerenders |
| Search Pattern | `grep -B5 -A15 "Provider value=" --include="*.tsx"` |
| Pass Criteria | Context value uses useMemo or is a stable object |
| Severity | Medium |
| Fix | Wrap value with `useMemo(() => ({ ... }), [deps])` |

### 6. Context Has Validation Hook

| Attribute | Value |
|-----------|-------|
| Description | Context should have a custom hook that throws if used outside provider |
| Search Pattern | `grep -A10 "useContext" --include="*.tsx"` |
| Pass Criteria | Custom hook checks for undefined and throws descriptive error |
| Severity | High |
| Fix | Add `if (!ctx) throw new Error('useX must be within XProvider')` |

### 7. Avoid Prop Drilling

| Attribute | Value |
|-----------|-------|
| Description | Props passed through 3+ levels should use Context |
| Search Pattern | Manual review of component hierarchies |
| Pass Criteria | No props passed unchanged through intermediate components |
| Severity | Medium |
| Fix | Create Context for deeply shared state |

### 8. Optimistic Updates for Mutations

| Attribute | Value |
|-----------|-------|
| Description | User-facing mutations should use optimistic updates |
| Search Pattern | `grep -A30 "useMutation" --include="*.ts"` |
| Pass Criteria | Critical mutations have onMutate with optimistic update |
| Severity | Low |
| Fix | Add onMutate to update cache optimistically, onError to rollback |

## Context Provider Pattern

```typescript
type Theme = 'light' | 'dark'

interface ThemeContextType {
  theme: Theme
  toggleTheme: () => void
  setTheme: (theme: Theme) => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function useTheme() {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [theme, setThemeState] = useState<Theme>(() => {
    const saved = localStorage.getItem('theme') as Theme
    if (saved) return saved
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? 'dark'
      : 'light'
  })

  const setTheme = useCallback((newTheme: Theme) => {
    setThemeState(newTheme)
    localStorage.setItem('theme', newTheme)
  }, [])

  const toggleTheme = useCallback(() => {
    setTheme(theme === 'light' ? 'dark' : 'light')
  }, [theme, setTheme])

  const value = useMemo(
    () => ({ theme, toggleTheme, setTheme }),
    [theme, toggleTheme, setTheme]
  )

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}
```

## Test QueryClient Pattern

```typescript
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  })
}

function AllTheProviders({ children }: { children: React.ReactNode }) {
  // Memoize to prevent recreation on re-renders
  const [queryClient] = useState(() => createTestQueryClient())

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>{children}</BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  )
}
```

## Combined Mutation Hooks

For related mutations, expose a unified interface:

```typescript
export function useProjectMutations() {
  const createMutation = useCreateProject()
  const updateMutation = useUpdateProject()
  const deleteMutation = useDeleteProject()

  return {
    create: createMutation,
    update: updateMutation,
    delete: deleteMutation,
    isLoading:
      createMutation.isPending ||
      updateMutation.isPending ||
      deleteMutation.isPending,
    error:
      createMutation.error ||
      updateMutation.error ||
      deleteMutation.error,
  }
}
```
