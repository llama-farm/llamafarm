# TypeScript Patterns for LlamaFarm

Idiomatic patterns for React and Electron TypeScript code.

## React Component Patterns

### Functional Components with Props Interface
```tsx
interface ButtonProps {
  readonly label: string
  readonly variant?: 'primary' | 'secondary'
  readonly disabled?: boolean
  readonly onClick?: () => void
}

function Button({ label, variant = 'primary', disabled, onClick }: ButtonProps): JSX.Element {
  return (
    <button
      className={`btn btn-${variant}`}
      disabled={disabled}
      onClick={onClick}
    >
      {label}
    </button>
  )
}
```

### Props with Children
```tsx
interface CardProps {
  readonly title: string
  readonly children: React.ReactNode
}

function Card({ title, children }: CardProps): JSX.Element {
  return (
    <div className="card">
      <h2>{title}</h2>
      {children}
    </div>
  )
}
```

## Hook Patterns

### Custom Hook with Return Type
```typescript
interface UseToggleReturn {
  isOpen: boolean
  open: () => void
  close: () => void
  toggle: () => void
}

function useToggle(initial = false): UseToggleReturn {
  const [isOpen, setIsOpen] = useState(initial)

  const open = useCallback(() => setIsOpen(true), [])
  const close = useCallback(() => setIsOpen(false), [])
  const toggle = useCallback(() => setIsOpen(prev => !prev), [])

  return { isOpen, open, close, toggle }
}
```

### Query Key Factory Pattern
```typescript
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  list: (namespace: string) => [...projectKeys.lists(), namespace] as const,
  details: () => [...projectKeys.all, 'detail'] as const,
  detail: (namespace: string, id: string) => [...projectKeys.details(), namespace, id] as const,
}
```

### TanStack Query Hook Pattern
```typescript
export function useProjects(namespace: string) {
  return useQuery({
    queryKey: projectKeys.list(namespace),
    queryFn: () => projectService.listProjects(namespace),
    enabled: !!namespace,
    staleTime: 5 * 60 * 1000,
    retry: 1,
    refetchOnWindowFocus: false,
  })
}

export function useCreateProject() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ namespace, request }: { namespace: string; request: CreateProjectRequest }) =>
      projectService.createProject(namespace, request),
    onSuccess: (data, variables) => {
      queryClient.invalidateQueries({ queryKey: projectKeys.list(variables.namespace) })
    },
  })
}
```

## API Service Patterns

### Service Module with Typed Functions
```typescript
export async function getProject(
  namespace: string,
  projectId: string
): Promise<GetProjectResponse> {
  const { data } = await apiClient.get<GetProjectResponse>(
    `/projects/${encodeURIComponent(namespace)}/${encodeURIComponent(projectId)}`
  )
  return data
}

const projectService = {
  listProjects,
  getProject,
  createProject,
  updateProject,
  deleteProject,
}

export default projectService
```

### Custom Error Classes
```typescript
export class ApiError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly response?: unknown
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

export class ValidationError extends Error {
  constructor(
    message: string,
    public readonly validationErrors: unknown
  ) {
    super(message)
    this.name = 'ValidationError'
  }
}
```

## Electron Patterns

### Class-Based Architecture
```typescript
export class WindowManager {
  private mainWindow: BrowserWindow | null = null
  private splashWindow: BrowserWindow | null = null

  createMainWindow(): BrowserWindow {
    this.mainWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      webPreferences: {
        nodeIntegration: false,
        contextIsolation: true,
        preload: path.join(__dirname, '../preload/index.js'),
      },
    })
    return this.mainWindow
  }

  cleanup(): void {
    this.mainWindow?.close()
    this.splashWindow?.close()
  }
}
```

### IPC Handler Pattern
```typescript
private setupIPCHandlers(): void {
  ipcMain.handle('cli:info', async (): Promise<CLIInfo> => {
    const isInstalled = await this.cliInstaller.isInstalled()
    return {
      isInstalled,
      path: isInstalled ? this.cliInstaller.getCLIPath() : null,
    }
  })
}
```

---

## Checklist

### PATTERNS-001: Use query key factories
- **Description**: Query keys should use factory pattern for consistency
- **Search**: `grep -r "queryKey:" designer/src/hooks/`
- **Pass**: All queryKey values use factory functions (e.g., `projectKeys.detail()`)
- **Fail**: Inline arrays like `queryKey: ['projects', id]`
- **Severity**: High
- **Fix**: Create key factory objects and use them consistently

### PATTERNS-002: Avoid inline object types in function signatures
- **Description**: Extract inline types to named interfaces
- **Search**: `grep -rE "function \w+\([^)]*\{[^}]+\}[^)]*\)" designer/src/`
- **Pass**: No matches (all params use named types)
- **Fail**: Functions with inline object types in parameters
- **Severity**: Medium
- **Fix**: Extract to interface above function

### PATTERNS-003: Use readonly for immutable props
- **Description**: Props interfaces should use readonly modifier
- **Search**: `grep -rE "interface \w+Props" designer/src/`
- **Pass**: Props use `readonly` for all fields
- **Fail**: Mutable props without readonly
- **Severity**: Low
- **Fix**: Add `readonly` to interface fields

### PATTERNS-004: Prefer named exports for hooks
- **Description**: Custom hooks should be named exports, not default
- **Search**: `grep -r "export default function use" designer/src/hooks/`
- **Pass**: No default exports for hooks
- **Fail**: Hooks exported as default
- **Severity**: Medium
- **Fix**: Use `export function useX()` instead

### PATTERNS-005: Mutation hooks should invalidate queries
- **Description**: useMutation hooks should invalidate related queries on success
- **Search**: `grep -A10 "useMutation" designer/src/hooks/`
- **Pass**: All mutations have `onSuccess` with `invalidateQueries`
- **Fail**: Mutations without cache invalidation
- **Severity**: High
- **Fix**: Add `onSuccess` handler with `queryClient.invalidateQueries()`

### PATTERNS-006: Use callback refs for event handlers
- **Description**: Event handlers should use useCallback for stable references
- **Search**: `grep -rE "onClick=\{[^}]+\}" designer/src/components/`
- **Pass**: Complex handlers wrapped in useCallback
- **Fail**: Inline arrow functions in JSX for handlers
- **Severity**: Low
- **Fix**: Extract to useCallback or stable function reference

### PATTERNS-007: Electron classes use private fields
- **Description**: Class fields should use private modifier
- **Search**: `grep -r "class.*{" electron-app/src/`
- **Pass**: Instance fields marked private
- **Fail**: Public fields without explicit access modifier
- **Severity**: Medium
- **Fix**: Add `private` modifier to internal fields

### PATTERNS-008: API functions return typed promises
- **Description**: API service functions must have explicit return types
- **Search**: `grep -rE "async function|async \(" designer/src/api/`
- **Pass**: All async functions have `Promise<T>` return type
- **Fail**: Missing return type annotation
- **Severity**: High
- **Fix**: Add explicit `Promise<ResponseType>` return annotation

### PATTERNS-009: Context providers use typed value
- **Description**: React contexts should define value type explicitly
- **Search**: `grep -r "createContext" designer/src/contexts/`
- **Pass**: Context created with typed default value
- **Fail**: Context with undefined or any type
- **Severity**: Medium
- **Fix**: Define context value interface and use it

### PATTERNS-010: useMemo/useCallback have correct dependencies
- **Description**: Memoization hooks must list all dependencies
- **Search**: `grep -rE "useMemo|useCallback" designer/src/`
- **Pass**: All referenced variables in dependency array
- **Fail**: Missing dependencies (eslint-plugin-react-hooks should catch)
- **Severity**: High
- **Fix**: Add missing dependencies or restructure code
