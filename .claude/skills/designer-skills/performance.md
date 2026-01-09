# Frontend Performance Patterns

Best practices for performance optimization in the Designer subsystem.

## Overview

Performance optimization focuses on:
- Bundle size reduction
- Render optimization
- Network efficiency
- Memory management

## Checklist Items

### 1. Lazy Loading for Routes

| Attribute | Value |
|-----------|-------|
| Description | Route components should be lazy loaded |
| Search Pattern | `grep -E "import.*from.*components/" App.tsx` |
| Pass Criteria | Heavy route components use React.lazy() |
| Severity | Critical |
| Recommendation | Use `const Component = React.lazy(() => import('./Component'))` |

### 2. Code Splitting at Route Level

| Attribute | Value |
|-----------|-------|
| Description | Each major route should be a separate chunk |
| Search Pattern | `grep -E "React.lazy\(" App.tsx` |
| Pass Criteria | Dashboard, Data, Models, RAG pages are split |
| Severity | Critical |
| Recommendation | Wrap route imports with `React.lazy()` and `<Suspense>` |

### 3. Images Use Proper Loading

| Attribute | Value |
|-----------|-------|
| Description | Images should have loading="lazy" for below-fold content |
| Search Pattern | `grep -E "<img" components/**/*.tsx` |
| Pass Criteria | Non-critical images have loading="lazy" |
| Severity | High |
| Recommendation | Add `loading="lazy"` to images not in initial viewport |

### 4. useMemo for Expensive Computations

| Attribute | Value |
|-----------|-------|
| Description | Large data transformations should be memoized |
| Search Pattern | `grep -E "\.(filter|map|reduce|sort)\(" components/**/*.tsx` |
| Pass Criteria | Operations on large arrays wrapped in useMemo |
| Severity | High |
| Recommendation | Wrap with `useMemo(() => data.filter(...), [data])` |

### 5. useCallback for Handler Props

| Attribute | Value |
|-----------|-------|
| Description | Callbacks passed to memoized children should be stable |
| Search Pattern | `grep -E "on[A-Z]\w+=\{[a-z]" components/**/*.tsx` |
| Pass Criteria | Handlers passed to React.memo components use useCallback |
| Severity | High |
| Recommendation | Wrap with `useCallback((e) => handle(e), [deps])` |

### 6. React.memo for Pure Components

| Attribute | Value |
|-----------|-------|
| Description | Components receiving only primitive props should be memoized |
| Search Pattern | `grep -E "export (const\|function)" components/**/*.tsx` |
| Pass Criteria | List item components and frequently re-rendered components use memo |
| Severity | High |
| Recommendation | Wrap with `React.memo(Component)` |

### 7. Virtualization for Long Lists

| Attribute | Value |
|-----------|-------|
| Description | Lists with 50+ items should use virtualization |
| Search Pattern | `grep -E "\.map\(" components/**/*.tsx` |
| Pass Criteria | Long lists use react-window or similar |
| Severity | High |
| Recommendation | Use `react-window` FixedSizeList for large lists |

### 8. Tree-Shakeable Imports

| Attribute | Value |
|-----------|-------|
| Description | Import only what you need from libraries |
| Search Pattern | `grep -E "import \* as" components/**/*.tsx` |
| Pass Criteria | No `import *` except for Radix primitives |
| Severity | Medium |
| Recommendation | Use `import { specific } from 'library'` |

### 9. Dynamic Imports for Heavy Libraries

| Attribute | Value |
|-----------|-------|
| Description | Large libraries should be dynamically imported |
| Search Pattern | `grep -E "import.*codemirror\|import.*framer" components/**/*.tsx` |
| Pass Criteria | CodeMirror, chart libraries use dynamic import |
| Severity | Medium |
| Recommendation | Use `const Editor = React.lazy(() => import('./Editor'))` |

### 10. Debounced Search/Input Handlers

| Attribute | Value |
|-----------|-------|
| Description | Search inputs should debounce API calls |
| Search Pattern | `grep -E "onChange.*search\|onInput" components/**/*.tsx` |
| Pass Criteria | Search handlers use debounce (300-500ms) |
| Severity | Medium |
| Recommendation | Use `useDebouncedCallback` or `lodash.debounce` |

### 11. Avoid Inline Object/Array Props

| Attribute | Value |
|-----------|-------|
| Description | Inline objects cause unnecessary re-renders |
| Search Pattern | `grep -E "=\{\{" components/**/*.tsx` |
| Pass Criteria | No inline style/object literals in render (except static) |
| Severity | Medium |
| Recommendation | Extract to useMemo or constant outside component |

### 12. Query Stale Time Optimized

| Attribute | Value |
|-----------|-------|
| Description | Queries should have appropriate staleTime |
| Search Pattern | `grep -E "staleTime:" hooks/*.ts` |
| Pass Criteria | Static data: 5min+, dynamic: 30s-2min |
| Severity | Medium |
| Recommendation | Set `staleTime: 5 * 60 * 1000` for config data |

### 13. Bundle Analyzer Check

| Attribute | Value |
|-----------|-------|
| Description | Regularly analyze bundle for unexpected size |
| Search Pattern | Check `rollup-plugin-visualizer` in vite.config |
| Pass Criteria | No unexpected large dependencies in main bundle |
| Severity | Medium |
| Recommendation | Run `nx build designer` and check stats.html |

### 14. Preload Critical Assets

| Attribute | Value |
|-----------|-------|
| Description | Critical fonts/images should be preloaded |
| Search Pattern | `grep -E "<link.*preload" index.html` |
| Pass Criteria | Primary font preloaded in index.html |
| Severity | Low |
| Recommendation | Add `<link rel="preload" href="font.woff2" as="font">` |

### 15. Service Worker for Caching

| Attribute | Value |
|-----------|-------|
| Description | Static assets should be cached via service worker |
| Search Pattern | `grep -E "serviceWorker\|workbox" vite.config` |
| Pass Criteria | Production build includes SW for asset caching |
| Severity | Low |
| Recommendation | Configure vite-plugin-pwa for asset caching |

## Patterns

### Lazy Loading Routes

```tsx
import React, { Suspense, lazy } from 'react'
import { Routes, Route } from 'react-router-dom'
import LoadingSpinner from './components/LoadingSpinner'

// Lazy load heavy route components
const Dashboard = lazy(() => import('./components/Dashboard/Dashboard'))
const Data = lazy(() => import('./components/Data/Data'))
const Models = lazy(() => import('./components/Models/Models'))
const Databases = lazy(() => import('./components/Rag/Databases'))

function App() {
  return (
    <Suspense fallback={<LoadingSpinner />}>
      <Routes>
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/data" element={<Data />} />
        <Route path="/models" element={<Models />} />
        <Route path="/databases" element={<Databases />} />
      </Routes>
    </Suspense>
  )
}
```

### Memoized List Item

```tsx
import React, { memo, useCallback } from 'react'

interface ProjectItemProps {
  project: Project
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

const ProjectItem = memo(function ProjectItem({
  project,
  onSelect,
  onDelete,
}: ProjectItemProps) {
  const handleSelect = useCallback(() => {
    onSelect(project.id)
  }, [project.id, onSelect])

  const handleDelete = useCallback(() => {
    onDelete(project.id)
  }, [project.id, onDelete])

  return (
    <div onClick={handleSelect}>
      <span>{project.name}</span>
      <button onClick={handleDelete}>Delete</button>
    </div>
  )
})
```

### Debounced Search

```tsx
import { useState, useMemo, useDeferredValue } from 'react'

function SearchableList({ items }: { items: Item[] }) {
  const [query, setQuery] = useState('')
  const deferredQuery = useDeferredValue(query)

  const filteredItems = useMemo(
    () => items.filter(item =>
      item.name.toLowerCase().includes(deferredQuery.toLowerCase())
    ),
    [items, deferredQuery]
  )

  return (
    <>
      <input
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="Search..."
      />
      <ul>
        {filteredItems.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </>
  )
}
```

### Virtualized List

```tsx
import { FixedSizeList as List } from 'react-window'

interface VirtualizedListProps {
  items: Item[]
  height: number
}

function VirtualizedList({ items, height }: VirtualizedListProps) {
  const Row = ({ index, style }: { index: number; style: React.CSSProperties }) => (
    <div style={style} className="flex items-center px-4">
      {items[index].name}
    </div>
  )

  return (
    <List
      height={height}
      itemCount={items.length}
      itemSize={40}
      width="100%"
    >
      {Row}
    </List>
  )
}
```

### Dynamic Import for Heavy Components

```tsx
import React, { Suspense, lazy } from 'react'

// Only load CodeMirror when needed
const CodeMirrorEditor = lazy(() => import('./CodeMirrorEditor'))

function ConfigEditor({ config }: { config: string }) {
  return (
    <Suspense fallback={<div className="h-96 bg-muted animate-pulse" />}>
      <CodeMirrorEditor value={config} />
    </Suspense>
  )
}
```

### Optimized Context Provider

```tsx
import React, { createContext, useContext, useMemo, useState } from 'react'

interface ThemeContextType {
  theme: 'light' | 'dark'
  toggleTheme: () => void
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  const [theme, setTheme] = useState<'light' | 'dark'>('dark')

  // Memoize context value to prevent unnecessary re-renders
  const value = useMemo(
    () => ({
      theme,
      toggleTheme: () => setTheme(t => t === 'light' ? 'dark' : 'light'),
    }),
    [theme]
  )

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  )
}
```

### Preloading Critical Routes

```tsx
// Preload on hover/focus for faster navigation
function NavLink({ to, children }: { to: string; children: React.ReactNode }) {
  const preload = () => {
    // Vite supports import() for preloading
    if (to === '/data') {
      import('./components/Data/Data')
    }
  }

  return (
    <Link
      to={to}
      onMouseEnter={preload}
      onFocus={preload}
    >
      {children}
    </Link>
  )
}
```

## Bundle Optimization

### Vite Config for Code Splitting

```typescript
// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { visualizer } from 'rollup-plugin-visualizer'

export default defineConfig({
  plugins: [
    react(),
    visualizer({ open: true, gzipSize: true }),
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],
          'vendor-query': ['@tanstack/react-query'],
          'vendor-ui': ['@radix-ui/react-dialog', '@radix-ui/react-tooltip'],
          'vendor-editor': ['@codemirror/state', '@codemirror/view'],
        },
      },
    },
  },
})
```

## Anti-Patterns

### Avoid: Inline Objects in Props

```tsx
// Bad - creates new object every render
<Component style={{ marginTop: 10 }} config={{ debug: true }} />

// Good - stable references
const style = useMemo(() => ({ marginTop: 10 }), [])
const config = useMemo(() => ({ debug: true }), [])
<Component style={style} config={config} />

// Good - for truly static values, define outside component
const STYLE = { marginTop: 10 } as const
```

### Avoid: Functions in Render

```tsx
// Bad - creates new function every render
<List items={items.filter(i => i.active)} />

// Good - memoized
const activeItems = useMemo(() => items.filter(i => i.active), [items])
<List items={activeItems} />
```

### Avoid: Large Component Files

```tsx
// Bad - 500+ line component
function Dashboard() {
  // ... hundreds of lines
}

// Good - split into smaller components
function Dashboard() {
  return (
    <>
      <DashboardHeader />
      <DashboardMetrics />
      <DashboardCharts />
      <DashboardActivity />
    </>
  )
}
```
