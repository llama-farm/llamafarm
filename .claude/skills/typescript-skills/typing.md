# TypeScript Typing Guidelines

Strict typing, generics, and utility types for LlamaFarm.

## Strict Mode Requirements

Both `designer/` and `electron-app/` use strict TypeScript:
```json
{
  "strict": true,
  "noUnusedLocals": true,
  "noUnusedParameters": true,
  "noFallthroughCasesInSwitch": true
}
```

## Type vs Interface

### Use `interface` for:
- Object shapes (props, API responses, entities)
- Extendable contracts
- Declaration merging needs

```typescript
interface Project {
  namespace: string
  name: string
  config: Record<string, unknown>
  validation_error?: string | null
}

interface CreateProjectRequest {
  name: string
  config_template?: string
}
```

### Use `type` for:
- Union types
- Intersection types
- Mapped types
- Function types

```typescript
type MessageRole = 'system' | 'user' | 'assistant' | 'tool'
type HealthStatus = 'healthy' | 'degraded' | 'unhealthy'
type StreamHandler = (chunk: ChatStreamChunk) => void
type ProjectWithMeta = Project & { meta: ProjectMeta }
```

## Generic Patterns

### Generic API Response
```typescript
interface ApiResponse<T> {
  data: T
  status: number
  message?: string
}

async function fetchData<T>(url: string): Promise<ApiResponse<T>> {
  const response = await apiClient.get<T>(url)
  return { data: response.data, status: response.status }
}
```

### Generic Hook
```typescript
function useLocalStorage<T>(key: string, initialValue: T): [T, (value: T) => void] {
  const [storedValue, setStoredValue] = useState<T>(() => {
    const item = localStorage.getItem(key)
    return item ? JSON.parse(item) : initialValue
  })

  const setValue = (value: T) => {
    setStoredValue(value)
    localStorage.setItem(key, JSON.stringify(value))
  }

  return [storedValue, setValue]
}
```

### Constrained Generics
```typescript
interface HasId {
  id: string
}

function findById<T extends HasId>(items: T[], id: string): T | undefined {
  return items.find(item => item.id === id)
}
```

## Utility Types

### Common Patterns
```typescript
// Partial for optional updates
interface UpdateProjectRequest {
  config: Partial<ProjectConfig>
}

// Pick for subset of properties
type ProjectSummary = Pick<Project, 'namespace' | 'name'>

// Omit for excluding properties
type CreateProject = Omit<Project, 'id' | 'createdAt'>

// Record for dictionaries
type ConfigMap = Record<string, unknown>

// Required to make optional fields required
type RequiredProject = Required<Project>

// Readonly for immutable data
type ImmutableProject = Readonly<Project>
```

### Custom Utility Types
```typescript
// Make specific properties optional
type PartialBy<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

// Make specific properties required
type RequiredBy<T, K extends keyof T> = Omit<T, K> & Required<Pick<T, K>>

// Extract non-nullable type
type NonNullableFields<T> = {
  [K in keyof T]: NonNullable<T[K]>
}
```

## Const Assertions

### Query Keys
```typescript
export const projectKeys = {
  all: ['projects'] as const,
  lists: () => [...projectKeys.all, 'list'] as const,
  detail: (id: string) => [...projectKeys.all, 'detail', id] as const,
}
// Type: readonly ['projects', 'detail', string]
```

### Configuration Objects
```typescript
const ALLOWED_TYPES = ['OllamaEmbedder', 'OpenAIEmbedder', 'HuggingFaceEmbedder'] as const
type EmbedderType = typeof ALLOWED_TYPES[number]
// Type: 'OllamaEmbedder' | 'OpenAIEmbedder' | 'HuggingFaceEmbedder'
```

## Discriminated Unions

### API Response States
```typescript
type ApiState<T> =
  | { status: 'idle' }
  | { status: 'loading' }
  | { status: 'success'; data: T }
  | { status: 'error'; error: Error }

function handleState<T>(state: ApiState<T>): void {
  switch (state.status) {
    case 'idle':
      // No data yet
      break
    case 'loading':
      // Show spinner
      break
    case 'success':
      // state.data is T here
      console.log(state.data)
      break
    case 'error':
      // state.error is Error here
      console.error(state.error.message)
      break
  }
}
```

### Message Types
```typescript
interface BaseMessage {
  content: string
}

interface UserMessage extends BaseMessage {
  role: 'user'
}

interface AssistantMessage extends BaseMessage {
  role: 'assistant'
  tool_calls?: ToolCall[]
}

interface ToolMessage extends BaseMessage {
  role: 'tool'
  tool_call_id: string
}

type ChatMessage = UserMessage | AssistantMessage | ToolMessage
```

## Null Safety

### Optional Chaining
```typescript
const hostname = config?.server?.hostname ?? 'localhost'
const port = project?.config?.runtime?.port
```

### Type Guards
```typescript
function isProject(value: unknown): value is Project {
  return (
    typeof value === 'object' &&
    value !== null &&
    'namespace' in value &&
    'name' in value &&
    'config' in value
  )
}

function assertProject(value: unknown): asserts value is Project {
  if (!isProject(value)) {
    throw new Error('Invalid project object')
  }
}
```

### Avoiding Non-Null Assertions
```typescript
// BAD: Using non-null assertion
const name = project!.name

// GOOD: Explicit null check
if (!project) {
  throw new Error('Project is required')
}
const name = project.name

// GOOD: Optional chaining with default
const name = project?.name ?? 'Unnamed'
```

---

## Checklist

### TYPING-001: No implicit any
- **Description**: All variables and parameters must have explicit or inferred types
- **Search**: `grep -rn ": any" designer/src/ electron-app/src/`
- **Pass**: No `any` types, or justified with comment
- **Fail**: Unexplained `any` usage
- **Severity**: Critical
- **Fix**: Replace with specific type or `unknown`

### TYPING-002: No non-null assertions
- **Description**: Avoid `!` operator; use proper null checks
- **Search**: `grep -rE "\w+!" designer/src/ | grep -v "\.test\." | grep -v node_modules`
- **Pass**: No non-null assertions in production code
- **Fail**: Using `value!` instead of null checks
- **Severity**: Critical
- **Fix**: Add null check or use optional chaining

### TYPING-003: Explicit function return types
- **Description**: Public functions should have explicit return types
- **Search**: `grep -rE "export (async )?function \w+\([^)]*\)[^:]" designer/src/`
- **Pass**: All exported functions have return type
- **Fail**: Missing return type on exported function
- **Severity**: High
- **Fix**: Add explicit return type annotation

### TYPING-004: Use unknown instead of any for external data
- **Description**: Data from external sources should be `unknown`, not `any`
- **Search**: `grep -rE "as any|: any" designer/src/api/`
- **Pass**: API responses typed as specific types or unknown
- **Fail**: Using `any` for API responses
- **Severity**: High
- **Fix**: Define proper response interface

### TYPING-005: Const assertions for literal types
- **Description**: Use `as const` for objects that should be readonly literals
- **Search**: `grep -r "queryKey:" designer/src/hooks/`
- **Pass**: Query key arrays use `as const`
- **Fail**: Missing `as const` on literal arrays
- **Severity**: Medium
- **Fix**: Add `as const` to readonly arrays/objects

### TYPING-006: Interfaces for object shapes
- **Description**: Object types should use interface, not inline types
- **Search**: `grep -rE ":\s*\{[^}]+\}" designer/src/types/`
- **Pass**: All object shapes are named interfaces
- **Fail**: Inline object types in type definitions
- **Severity**: Medium
- **Fix**: Extract to named interface

### TYPING-007: Discriminated unions for state
- **Description**: Use discriminated unions for complex state
- **Search**: `grep -r "type.*State" designer/src/types/`
- **Pass**: State types use discriminated unions where appropriate
- **Fail**: Optional properties instead of unions
- **Severity**: Medium
- **Fix**: Refactor to discriminated union with status field

### TYPING-008: Readonly for immutable data
- **Description**: Props and constants should use readonly
- **Search**: `grep -rE "interface \w+Props" designer/src/`
- **Pass**: Props interfaces use readonly modifier
- **Fail**: Mutable props without readonly
- **Severity**: Low
- **Fix**: Add `readonly` to interface fields

### TYPING-009: Proper generic constraints
- **Description**: Generics should have appropriate constraints
- **Search**: `grep -rE "<T>" designer/src/`
- **Pass**: Generics constrained where needed (e.g., `T extends object`)
- **Fail**: Unconstrained generics that should be limited
- **Severity**: High
- **Fix**: Add `extends` constraint to generic

### TYPING-010: Type guards for runtime checks
- **Description**: Use type guards for runtime type checking
- **Search**: `grep -r "is Project\|is Error\|is Array" designer/src/`
- **Pass**: Type guards defined for complex type checks
- **Fail**: Type assertions without validation
- **Severity**: Critical
- **Fix**: Create type guard function with runtime checks
