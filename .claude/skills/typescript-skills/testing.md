# Testing Patterns for LlamaFarm TypeScript

Vitest testing patterns for Designer React application.

## Test Configuration

Designer uses Vitest with jsdom environment:
```typescript
// vitest.config.ts
export default mergeConfig(viteConfig, defineConfig({
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.{test,spec}.{ts,tsx}'],
  },
}))
```

## Test Setup

```typescript
// src/test/setup.ts
import { afterEach, beforeAll, afterAll, vi } from 'vitest'
import { cleanup } from '@testing-library/react'
import '@testing-library/jest-dom/vitest'
import { server } from './mocks/server'

beforeAll(() => {
  server.listen({ onUnhandledRequest: 'warn' })
})

afterEach(() => {
  cleanup()
  server.resetHandlers()
})

afterAll(() => {
  server.close()
})
```

## MSW Mock Server

```typescript
// src/test/mocks/server.ts
import { setupServer } from 'msw/node'
import { handlers } from './handlers'

export const server = setupServer(...handlers)
```

## Test Factory Pattern

```typescript
// src/test/factories/projectFactory.ts
import { Project, ListProjectsResponse } from '../../types/project'

interface MockProjectOptions {
  namespace?: string
  name?: string
  config?: Record<string, unknown>
  validation_error?: string | null
}

export function createMockProject(options: MockProjectOptions = {}): Project {
  const {
    namespace = 'default',
    name = 'test-project',
    config = {
      version: 'v1',
      name,
      namespace,
      runtime: { provider: 'ollama', model: 'llama3.2:3b' },
      prompts: [],
    },
    validation_error = null,
  } = options

  return { namespace, name, config, validation_error }
}

export function createMockProjectsList(
  namespace = 'default',
  count = 2
): ListProjectsResponse {
  const projects = Array.from({ length: count }, (_, i) =>
    createMockProject({ namespace, name: `project-${i + 1}` })
  )
  return { total: projects.length, projects }
}
```

## API Service Tests

```typescript
import { describe, it, expect, beforeEach } from 'vitest'
import { http, HttpResponse } from 'msw'
import { server } from '../../test/mocks/server'
import { listProjects, createProject } from '../projectService'
import { createMockProject, createMockProjectsList } from '../../test/factories/projectFactory'

const API_BASE = 'http://localhost:14345/v1'

describe('projectService', () => {
  beforeEach(() => {
    server.resetHandlers()
  })

  describe('listProjects', () => {
    it('should list projects for namespace', async () => {
      const mockResponse = createMockProjectsList('default', 2)
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json(mockResponse)
        })
      )

      const result = await listProjects('default')

      expect(result).toEqual(mockResponse)
      expect(result.total).toBe(2)
    })

    it('should handle API errors', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.json({ detail: 'Server error' }, { status: 500 })
        })
      )

      await expect(listProjects('default')).rejects.toThrow()
    })

    it('should handle network errors', async () => {
      server.use(
        http.get(`${API_BASE}/projects/:namespace`, () => {
          return HttpResponse.error()
        })
      )

      await expect(listProjects('default')).rejects.toThrow()
    })
  })
})
```

## Component Tests

```typescript
import { describe, it, expect } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import MyComponent from './MyComponent'

function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  })

  return render(
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        {ui}
      </BrowserRouter>
    </QueryClientProvider>
  )
}

describe('MyComponent', () => {
  it('should render title', () => {
    renderWithProviders(<MyComponent title="Test" />)
    expect(screen.getByText('Test')).toBeInTheDocument()
  })

  it('should handle click events', async () => {
    const user = userEvent.setup()
    const onClick = vi.fn()

    renderWithProviders(<MyComponent title="Test" onClick={onClick} />)

    await user.click(screen.getByRole('button'))
    expect(onClick).toHaveBeenCalledTimes(1)
  })
})
```

## Hook Tests

```typescript
import { describe, it, expect } from 'vitest'
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useProjects } from './useProjects'

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  })
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  )
}

describe('useProjects', () => {
  it('should fetch projects', async () => {
    const { result } = renderHook(() => useProjects('default'), {
      wrapper: createWrapper(),
    })

    await waitFor(() => expect(result.current.isSuccess).toBe(true))
    expect(result.current.data?.projects).toBeDefined()
  })
})
```

## Mocking Patterns

### Mock Functions
```typescript
const mockFn = vi.fn()
mockFn.mockReturnValue('value')
mockFn.mockResolvedValue('async value')
mockFn.mockImplementation((arg) => arg * 2)
```

### Mock Modules
```typescript
vi.mock('../api/projectService', () => ({
  default: {
    listProjects: vi.fn().mockResolvedValue({ projects: [], total: 0 }),
    getProject: vi.fn(),
  },
}))
```

### Mock localStorage
```typescript
const localStorageMock = (() => {
  let store: Record<string, string> = {}
  return {
    getItem: (key: string) => store[key] ?? null,
    setItem: (key: string, value: string) => { store[key] = value },
    removeItem: (key: string) => { delete store[key] },
    clear: () => { store = {} },
  }
})()

Object.defineProperty(window, 'localStorage', { value: localStorageMock })
```

---

## Checklist

### TESTING-001: Tests use factories for mock data
- **Description**: Test data should come from factory functions
- **Search**: `grep -r "createMock" designer/src/**/*.test.ts`
- **Pass**: Tests use factory functions for mock data
- **Fail**: Inline mock objects repeated across tests
- **Severity**: High
- **Fix**: Create factory function in src/test/factories/

### TESTING-002: MSW for API mocking
- **Description**: API tests should use MSW, not axios mocks
- **Search**: `grep -r "vi.mock.*axios" designer/src/`
- **Pass**: No direct axios mocks; using MSW handlers
- **Fail**: Mocking axios directly
- **Severity**: Critical
- **Fix**: Use MSW http handlers instead

### TESTING-003: Reset handlers between tests
- **Description**: MSW handlers must reset in beforeEach/afterEach
- **Search**: `grep -r "server.resetHandlers" designer/src/`
- **Pass**: resetHandlers called in setup or beforeEach
- **Fail**: Missing handler reset between tests
- **Severity**: Critical
- **Fix**: Add `server.resetHandlers()` to afterEach

### TESTING-004: Use userEvent over fireEvent
- **Description**: Prefer userEvent for realistic interactions
- **Search**: `grep -r "fireEvent\." designer/src/**/*.test.tsx`
- **Pass**: Using userEvent for user interactions
- **Fail**: Using fireEvent for clicks, typing
- **Severity**: Medium
- **Fix**: Replace with `userEvent.setup()` and async methods

### TESTING-005: Async assertions use waitFor
- **Description**: Async state changes need waitFor
- **Search**: `grep -r "await.*result.current" designer/src/`
- **Pass**: Using waitFor for async hook results
- **Fail**: Direct assertions on async results
- **Severity**: High
- **Fix**: Wrap in `await waitFor(() => expect(...))`

### TESTING-006: QueryClient in test wrappers
- **Description**: Query tests need fresh QueryClient
- **Search**: `grep -r "renderHook" designer/src/hooks/`
- **Pass**: Each test uses new QueryClient
- **Fail**: Shared QueryClient across tests
- **Severity**: High
- **Fix**: Create wrapper function that instantiates new QueryClient

### TESTING-007: Test error states
- **Description**: Tests should cover error scenarios
- **Search**: `grep -r "rejects.toThrow\|status: 500\|HttpResponse.error" designer/src/`
- **Pass**: API tests include error cases
- **Fail**: Only happy path tested
- **Severity**: Medium
- **Fix**: Add tests for 400, 404, 500, network errors

### TESTING-008: Cleanup after each test
- **Description**: DOM cleanup must run after each test
- **Search**: `grep -r "cleanup" designer/src/test/setup.ts`
- **Pass**: cleanup() called in afterEach
- **Fail**: Missing cleanup call
- **Severity**: Medium
- **Fix**: Add `cleanup()` to afterEach in setup.ts
