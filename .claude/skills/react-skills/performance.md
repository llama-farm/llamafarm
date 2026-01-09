# React Performance Optimization

Guidelines for optimizing React application performance.

## Core Principles

1. **Measure first** - Use React DevTools Profiler before optimizing
2. **Avoid premature optimization** - Only optimize when there's a measurable problem
3. **Minimize rerenders** - Prevent unnecessary component updates
4. **Code split** - Load code only when needed

## Memoization Patterns

### useMemo for Expensive Computations

```typescript
const { thinking, contentWithoutThinking } = useMemo(
  () => parseMessageContentMemo(content, type, message.id),
  [content, type, message.id]
)
```

### useCallback for Stable References

```typescript
const handleSendClick = useCallback(async () => {
  const messageContent = inputValue.trim()
  if (!canSend || !messageContent) return

  setWantsAutoScroll(true)
  const success = await sendMessage(messageContent)
  if (success) updateInput('')
}, [inputValue, canSend, sendMessage, updateInput])
```

### React.memo for Pure Components

```typescript
const Message = React.memo(function Message({ message }: MessageProps) {
  // Component only rerenders if message prop changes
  return <div>{message.content}</div>
})
```

## Checklist Items

### 1. Expensive Computations Are Memoized

| Attribute | Value |
|-----------|-------|
| Description | Array transformations and parsing should use useMemo |
| Search Pattern | `grep -E "\.(filter|map|reduce|sort|parse)" --include="*.tsx"` |
| Pass Criteria | Expensive operations inside render wrapped with useMemo |
| Severity | Medium |
| Fix | Wrap with `useMemo(() => computation, [deps])` |

### 2. Callbacks Passed to Children Are Stable

| Attribute | Value |
|-----------|-------|
| Description | Functions passed as props should use useCallback |
| Search Pattern | `grep -E "on[A-Z]\w+=\{" --include="*.tsx"` |
| Pass Criteria | Callback props use useCallback or are defined outside component |
| Severity | Medium |
| Fix | Wrap with `useCallback(fn, [deps])` |

### 3. Context Values Are Memoized

| Attribute | Value |
|-----------|-------|
| Description | Context provider values should be memoized |
| Search Pattern | `grep -B5 "Provider value=" --include="*.tsx"` |
| Pass Criteria | Value prop uses useMemo or is a stable reference |
| Severity | High |
| Fix | Wrap value with `useMemo(() => ({ ... }), [deps])` |

### 4. Large Lists Use Virtualization

| Attribute | Value |
|-----------|-------|
| Description | Lists with 100+ items should use virtualization |
| Search Pattern | `grep -E "\.map\(" --include="*.tsx"` |
| Pass Criteria | Large lists use react-window or similar |
| Severity | Medium |
| Fix | Implement virtualization with react-window |

### 5. Images Are Lazy Loaded

| Attribute | Value |
|-----------|-------|
| Description | Off-screen images should use lazy loading |
| Search Pattern | `grep -E "<img" --include="*.tsx"` |
| Pass Criteria | Images have `loading="lazy"` or use Intersection Observer |
| Severity | Low |
| Fix | Add `loading="lazy"` attribute |

### 6. Code Splitting at Route Level

| Attribute | Value |
|-----------|-------|
| Description | Routes should use React.lazy for code splitting |
| Search Pattern | `grep -E "import.*from.*components" App.tsx` |
| Pass Criteria | Large route components use React.lazy |
| Severity | Medium |
| Fix | Use `const Component = React.lazy(() => import('./Component'))` |

**Current State**: The Designer currently uses static imports for all routes. This is a future optimization opportunity for large bundles.

### 7. Avoid Inline Object/Array Literals in JSX

| Attribute | Value |
|-----------|-------|
| Description | Inline objects create new references on every render |
| Search Pattern | `grep -E "style=\{\{" --include="*.tsx"` |
| Pass Criteria | Static styles extracted to constants or CSS |
| Severity | Low |
| Fix | Extract to const outside component or use Tailwind |

### 8. useTransition for Non-Urgent Updates

| Attribute | Value |
|-----------|-------|
| Description | Large state updates should use useTransition |
| Search Pattern | `grep -E "useState.*\[\]" --include="*.tsx"` |
| Pass Criteria | Bulk updates that affect many components use startTransition |
| Severity | Low |
| Fix | Wrap update with `startTransition(() => setState(...))` |

## Lazy Loading Patterns

### React.lazy with Suspense

```typescript
const Dashboard = React.lazy(() => import('./components/Dashboard'))
const Models = React.lazy(() => import('./components/Models'))

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/models" element={<Models />} />
      </Routes>
    </Suspense>
  )
}
```

### Named Exports with Lazy

```typescript
const Dashboard = React.lazy(() =>
  import('./components/Dashboard').then(module => ({
    default: module.Dashboard
  }))
)
```

## Scroll Performance

### RAF Debouncing

```typescript
const rafRef = useRef<number | null>(null)

const handleScroll = useCallback(() => {
  if (rafRef.current) {
    cancelAnimationFrame(rafRef.current)
  }

  rafRef.current = requestAnimationFrame(() => {
    const atBottom = checkIfAtBottom()
    setIsUserAtBottom(atBottom)
  })
}, [checkIfAtBottom])

useEffect(() => {
  return () => {
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current)
    }
  }
}, [])
```

### Smooth Scrolling

```typescript
useEffect(() => {
  if (wantsAutoScroll && listRef.current) {
    listRef.current.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: 'auto', // 'auto' prevents jank during streaming
    })
  }
}, [messages, wantsAutoScroll])
```

## Avoiding Unnecessary Rerenders

### Split Context by Update Frequency

```typescript
// Separate frequently-changing state from stable state
const ThemeContext = createContext<Theme>('light')
const ThemeActionsContext = createContext<ThemeActions>(null!)

// Components that only need actions don't rerender on theme change
function ThemeToggle() {
  const { toggleTheme } = useContext(ThemeActionsContext)
  return <button onClick={toggleTheme}>Toggle</button>
}
```

### Stable Callback References

```typescript
// Bad - creates new function on every render
<Button onClick={() => handleClick(id)} />

// Good - stable reference
const handleItemClick = useCallback((id: string) => {
  // handle click
}, [])

<Button onClick={() => handleItemClick(id)} />

// Best - if child is memoized
const handleItemClick = useCallback(() => {
  handleClick(id)
}, [id, handleClick])

<MemoizedButton onClick={handleItemClick} />
```

## Bundle Size Optimization

### Import Only What You Need

```typescript
// Bad - imports entire library
import * as _ from 'lodash'

// Good - tree-shakeable imports
import debounce from 'lodash/debounce'

// Best - use native alternatives when possible
const debounce = (fn: Function, ms: number) => {
  let timer: NodeJS.Timeout
  return (...args: any[]) => {
    clearTimeout(timer)
    timer = setTimeout(() => fn(...args), ms)
  }
}
```

### Analyze Bundle

```bash
# Use rollup-plugin-visualizer (already in devDependencies)
nx build designer --mode=analyze
```
