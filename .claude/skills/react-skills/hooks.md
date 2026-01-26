# React Hooks Patterns

Guidelines for writing and using React hooks effectively.

## Rules of Hooks

1. **Only call hooks at the top level** - Never inside loops, conditions, or nested functions
2. **Only call hooks from React functions** - Components or custom hooks only
3. **Use the `use` prefix** - Custom hooks must start with `use`

## Custom Hook Patterns

### Basic Custom Hook

```typescript
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false)

  useEffect(() => {
    const mql = window.matchMedia(query)
    const onChange = (e: MediaQueryListEvent) => setMatches(e.matches)

    setMatches(mql.matches)
    mql.addEventListener('change', onChange)
    return () => mql.removeEventListener('change', onChange)
  }, [query])

  return matches
}
```

### Composed Hooks

```typescript
export function useIsMobile(): boolean {
  return useMediaQuery('(max-width: 767px)')
}
```

## Checklist Items

### 1. Hooks Called at Top Level

| Attribute | Value |
|-----------|-------|
| Description | Hooks must not be called conditionally or in loops |
| Search Pattern | `grep -E "if\s*\(.*\)\s*\{[^}]*use[A-Z]" --include="*.ts" --include="*.tsx"` |
| Pass Criteria | No hooks inside if/for/while blocks |
| Severity | Critical |
| Fix | Move hook call to top level, conditionally use its return value instead |

### 2. Custom Hooks Use `use` Prefix

| Attribute | Value |
|-----------|-------|
| Description | Custom hooks must be named with `use` prefix |
| Search Pattern | `grep -l "export.*function.*use" hooks/*.ts` |
| Pass Criteria | All exported hook functions start with `use` |
| Severity | High |
| Fix | Rename function to start with `use` |

### 3. useEffect Has Cleanup

| Attribute | Value |
|-----------|-------|
| Description | Effects with subscriptions/timers must return cleanup function |
| Search Pattern | `grep -B2 -A10 "useEffect\(" --include="*.ts" --include="*.tsx"` |
| Pass Criteria | addEventListener, setInterval, setTimeout have corresponding cleanup |
| Severity | High |
| Fix | Return cleanup function: `return () => { removeEventListener(...) }` |

### 4. useEffect Dependencies Are Complete

| Attribute | Value |
|-----------|-------|
| Description | All variables used in useEffect should be in dependency array |
| Search Pattern | `grep -A15 "useEffect\(" --include="*.ts" --include="*.tsx"` |
| Pass Criteria | ESLint exhaustive-deps rule passes |
| Severity | High |
| Fix | Add missing deps or wrap with useCallback/useMemo if intentional |

### 5. useCallback for Handler Props

| Attribute | Value |
|-----------|-------|
| Description | Callbacks passed to children should be memoized |
| Search Pattern | `grep -E "on[A-Z]\w+=\{[a-z]\w+\}" --include="*.tsx"` |
| Pass Criteria | Handler functions passed to child components use useCallback |
| Severity | Medium |
| Fix | Wrap handler with `useCallback(handler, [deps])` |

### 6. useMemo for Expensive Computations

| Attribute | Value |
|-----------|-------|
| Description | Expensive computations should be memoized |
| Search Pattern | `grep -E "\.(filter|map|reduce|sort)\(" --include="*.tsx"` |
| Pass Criteria | Large data transformations wrapped in useMemo |
| Severity | Medium |
| Fix | Wrap with `useMemo(() => computation, [deps])` |

### 7. useState Initializer for Expensive Initial Values

| Attribute | Value |
|-----------|-------|
| Description | Expensive initial state should use lazy initializer |
| Search Pattern | `grep -E "useState\([^)]+\(" --include="*.ts" --include="*.tsx"` |
| Pass Criteria | Function calls in useState use lazy form: `useState(() => fn())` |
| Severity | Medium |
| Fix | Change `useState(expensiveFn())` to `useState(() => expensiveFn())` |

### 8. useRef for Mutable Values That Don't Trigger Rerenders

| Attribute | Value |
|-----------|-------|
| Description | Values that change but shouldn't trigger rerenders use useRef |
| Search Pattern | `grep -E "useRef\(" --include="*.ts" --include="*.tsx"` |
| Pass Criteria | Refs used for DOM elements, timers, previous values, mutable flags |
| Severity | Low |
| Fix | Use `useRef` for values that change without needing rerender |

## TanStack Query Hook Patterns

### Query Key Factories

```typescript
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  list: (namespace: string) => [...projectKeys.lists(), namespace] as const,
  details: () => [...projectKeys.all, 'detail'] as const,
  detail: (namespace: string, id: string) =>
    [...projectKeys.details(), namespace, id] as const,
}
```

### Query Hook

```typescript
export function useProjects(namespace: string) {
  return useQuery({
    queryKey: projectKeys.list(namespace),
    queryFn: () => projectService.listProjects(namespace),
    enabled: !!namespace,
    staleTime: 5 * 60 * 1000,
    retry: 1,
  })
}
```

### Mutation Hook

```typescript
export function useCreateProject() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ namespace, request }) =>
      projectService.createProject(namespace, request),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({
        queryKey: projectKeys.list(variables.namespace)
      })
      queryClient.setQueryData(
        projectKeys.detail(variables.namespace, data.project.name),
        { project: data.project }
      )
    },
    onError: (error) => {
      console.error('Failed to create project:', error)
    },
  })
}
```

## Context Hook Pattern

```typescript
const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function useTheme() {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider')
  }
  return context
}
```

## Cleanup Patterns

### Timer Cleanup

```typescript
useEffect(() => {
  const timer = setTimeout(() => {
    setExpired(true)
  }, 5000)

  return () => clearTimeout(timer)
}, [])
```

### Event Listener Cleanup

```typescript
useEffect(() => {
  const handleResize = () => setWidth(window.innerWidth)
  window.addEventListener('resize', handleResize)
  return () => window.removeEventListener('resize', handleResize)
}, [])
```

### AbortController for Fetch

```typescript
useEffect(() => {
  const controller = new AbortController()

  fetch(url, { signal: controller.signal })
    .then(res => res.json())
    .then(setData)
    .catch(err => {
      if (err.name !== 'AbortError') throw err
    })

  return () => controller.abort()
}, [url])
```

### RAF Cleanup

```typescript
useEffect(() => {
  const rafRef = { current: null as number | null }

  const animate = () => {
    // animation logic
    rafRef.current = requestAnimationFrame(animate)
  }
  rafRef.current = requestAnimationFrame(animate)

  return () => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current)
  }
}, [])
```
