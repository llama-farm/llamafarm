# Testing Guide for Designer

This document provides comprehensive guidance on testing the Designer React TypeScript application.

## Overview

The Designer application uses a modern testing stack:

- **Vitest**: Fast, Vite-native test runner with Jest-compatible API
- **React Testing Library**: Testing utilities for React components
- **MSW (Mock Service Worker)**: API mocking at the network level
- **jsdom**: Browser-like environment for testing

## Table of Contents

- [Getting Started](#getting-started)
- [Running Tests](#running-tests)
- [Writing Tests](#writing-tests)
- [Test Utilities](#test-utilities)
- [MSW and API Mocking](#msw-and-api-mocking)
- [Mock Data Factories](#mock-data-factories)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Getting Started

### Installation

Dependencies are already configured in `package.json`. Install them with:

```bash
npm install
```

### Project Structure

```
designer/
├── src/
│   ├── test/
│   │   ├── setup.ts              # Global test setup
│   │   ├── utils.tsx              # Custom render utilities
│   │   ├── mocks/
│   │   │   ├── handlers.ts        # MSW request handlers
│   │   │   └── server.ts          # MSW server setup
│   │   └── factories/
│   │       ├── projectFactory.ts
│   │       ├── datasetFactory.ts
│   │       └── embeddingStrategyFactory.ts
│   ├── components/
│   │   └── __tests__/             # Component tests
│   └── App.test.tsx               # Example smoke test
└── vitest.config.ts               # Vitest configuration
```

## Running Tests

### Available Commands

```bash
# Run tests in watch mode (interactive)
npm test

# Run tests once (CI mode)
npm run test:run

# Run tests with UI (visual test runner)
npm run test:ui

# Generate coverage report
npm run test:coverage
```

### Watch Mode

In watch mode, Vitest will:
- Re-run tests when files change
- Show which tests are affected
- Allow filtering tests interactively

### Coverage Reports

Coverage reports are generated in:
- `coverage/` - HTML reports (open `coverage/index.html` in browser)
- Terminal - Text summary

## Writing Tests

### Basic Test Structure

```tsx
import { describe, it, expect } from 'vitest'
import { screen } from '@testing-library/react'
import { renderWithProviders } from '@/test/utils'
import { MyComponent } from './MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    renderWithProviders(<MyComponent />)
    expect(screen.getByText('Hello')).toBeInTheDocument()
  })
})
```

### Testing User Interactions

```tsx
import { describe, it, expect } from 'vitest'
import { screen, userEvent } from '@/test/utils'
import { renderWithProviders } from '@/test/utils'
import { Button } from './Button'

describe('Button', () => {
  it('handles click events', async () => {
    const user = userEvent.setup()
    const handleClick = vi.fn()
    
    renderWithProviders(<Button onClick={handleClick}>Click me</Button>)
    
    await user.click(screen.getByRole('button'))
    
    expect(handleClick).toHaveBeenCalledTimes(1)
  })
})
```

### Testing Async Behavior

```tsx
import { describe, it, expect } from 'vitest'
import { screen, waitFor } from '@testing-library/react'
import { renderWithProviders } from '@/test/utils'
import { ProjectsList } from './ProjectsList'

describe('ProjectsList', () => {
  it('loads and displays projects', async () => {
    renderWithProviders(<ProjectsList namespace="default" />)
    
    // Initially shows loading state
    expect(screen.getByText(/loading/i)).toBeInTheDocument()
    
    // Wait for projects to load
    await waitFor(() => {
      expect(screen.getByText('project-1')).toBeInTheDocument()
    })
  })
})
```

## Test Utilities

### renderWithProviders

The custom render function wraps components with all necessary providers:

```tsx
import { renderWithProviders } from '@/test/utils'

// Automatically includes:
// - QueryClientProvider (React Query)
// - BrowserRouter (React Router)

const { getByText, container } = renderWithProviders(<MyComponent />)
```

### createTestQueryClient

Create a custom QueryClient for specific testing scenarios:

```tsx
import { createTestQueryClient } from '@/test/utils'

const queryClient = createTestQueryClient()

// Pre-populate cache
queryClient.setQueryData(['projects', 'default'], mockProjects)

// Use in test
const { getByText } = render(
  <QueryClientProvider client={queryClient}>
    <MyComponent />
  </QueryClientProvider>
)
```

## MSW and API Mocking

### How MSW Works

MSW intercepts HTTP requests at the network level, providing realistic API mocking without modifying application code.

### Available Handlers

Handlers are defined in `src/test/mocks/handlers.ts`:

- `GET /api/v1/projects/:namespace` - List projects
- `POST /api/v1/projects/:namespace` - Create project
- `GET /api/v1/projects/:namespace/:projectId` - Get project
- `PUT /api/v1/projects/:namespace/:projectId` - Update project
- `DELETE /api/v1/projects/:namespace/:projectId` - Delete project
- `GET /api/v1/projects/:namespace/:project/datasets/` - List datasets
- `POST /api/v1/projects/:namespace/:project/datasets/` - Create dataset
- And more...

### Customizing Responses

Override handlers for specific tests:

```tsx
import { server } from '@/test/mocks/server'
import { http, HttpResponse } from 'msw'

describe('Error handling', () => {
  it('handles API errors', async () => {
    // Override handler to return error
    server.use(
      http.get('/api/v1/projects/:namespace', () => {
        return HttpResponse.json(
          { detail: 'Server error' },
          { status: 500 }
        )
      })
    )
    
    renderWithProviders(<ProjectsList namespace="default" />)
    
    await waitFor(() => {
      expect(screen.getByText(/error/i)).toBeInTheDocument()
    })
  })
})
```

### Request Verification

Verify that requests are made correctly:

```tsx
import { server } from '@/test/mocks/server'
import { http, HttpResponse } from 'msw'

it('sends correct request body', async () => {
  let requestBody: any = null
  
  server.use(
    http.post('/api/v1/projects/:namespace', async ({ request }) => {
      requestBody = await request.json()
      return HttpResponse.json({ project: createMockProject() })
    })
  )
  
  // Trigger API call...
  
  await waitFor(() => {
    expect(requestBody).toEqual({
      name: 'new-project',
      config_template: 'default',
    })
  })
})
```

## Mock Data Factories

Factories generate consistent test data.

### Project Factory

```tsx
import {
  createMockProject,
  createMockProjectsList,
  createMockProjectWithConfig,
  createMockProjectWithError,
} from '@/test/factories/projectFactory'

// Single project
const project = createMockProject({
  namespace: 'default',
  name: 'my-project',
})

// List of projects
const projectsList = createMockProjectsList('default', 5) // 5 projects

// Project with custom config
const configuredProject = createMockProjectWithConfig('my-project', {
  runtime: { provider: 'openai', model: 'gpt-4' },
})

// Project with validation error
const errorProject = createMockProjectWithError('bad-project', 'Invalid config')
```

### Dataset Factory

```tsx
import {
  createMockDataset,
  createMockDatasetsList,
  createMockDatasetWithFiles,
  createMockProcessingDataset,
  createMockTaskStatus,
} from '@/test/factories/datasetFactory'

// Single dataset
const dataset = createMockDataset({
  name: 'my-dataset',
  data_processing_strategy: 'pdf_ingest',
  database: 'main_db',
})

// List of datasets
const datasetsList = createMockDatasetsList(3)

// Dataset with files
const datasetWithFiles = createMockDatasetWithFiles('my-dataset', 5) // 5 files

// Processing dataset
const processingDataset = createMockProcessingDataset('processing-dataset')

// Task status (matches TaskStatusResponse interface)
const taskStatus = createMockTaskStatus('task-123', 'SUCCESS')
const failedTask = createMockTaskStatus('task-456', 'FAILURE', { 
  error: 'Custom error message',
  traceback: 'Custom traceback' 
})
```

### Embedding Strategy Factory

```tsx
import {
  createMockEmbeddingStrategies,
  createMockDatabases,
  createMockRetrievalStrategy,
} from '@/test/factories/embeddingStrategyFactory'

// Available strategies
const strategies = createMockEmbeddingStrategies()

// Available databases
const databases = createMockDatabases()

// Retrieval strategy
const retrievalStrategy = createMockRetrievalStrategy('strategy-1', 'semantic_search')
```

## Best Practices

### 1. Test User Behavior, Not Implementation

❌ **Don't:**
```tsx
expect(component.state.isLoading).toBe(true)
```

✅ **Do:**
```tsx
expect(screen.getByText(/loading/i)).toBeInTheDocument()
```

### 2. Use Accessible Queries

Prefer queries that match how users interact:

```tsx
// Good (in order of preference)
screen.getByRole('button', { name: /submit/i })
screen.getByLabelText(/username/i)
screen.getByPlaceholderText(/enter email/i)
screen.getByText(/welcome/i)

// Avoid when possible
screen.getByTestId('submit-button')
```

### 3. Wait for Async Changes

```tsx
import { waitFor } from '@testing-library/react'

// Wait for element to appear
await waitFor(() => {
  expect(screen.getByText('Success')).toBeInTheDocument()
})

// Or use findBy queries (implicit waitFor)
expect(await screen.findByText('Success')).toBeInTheDocument()
```

### 4. Clean Up Between Tests

The test setup automatically cleans up, but for complex scenarios:

```tsx
import { cleanup } from '@testing-library/react'

afterEach(() => {
  cleanup()
  // Additional cleanup if needed
})
```

### 5. Test Error States

```tsx
it('displays error message when API fails', async () => {
  server.use(
    http.get('/api/v1/projects/:namespace', () => {
      return HttpResponse.json(
        { detail: 'Failed to load' },
        { status: 500 }
      )
    })
  )
  
  renderWithProviders(<ProjectsList namespace="default" />)
  
  expect(await screen.findByText(/failed to load/i)).toBeInTheDocument()
})
```

### 6. Test Loading States

```tsx
it('shows loading spinner while fetching data', () => {
  renderWithProviders(<ProjectsList namespace="default" />)
  
  expect(screen.getByTestId('loading-spinner')).toBeInTheDocument()
})
```

### 7. Organize Tests Logically

```tsx
describe('ProjectModal', () => {
  describe('Create Mode', () => {
    it('shows create title')
    it('allows entering project name')
    it('creates project on submit')
  })
  
  describe('Edit Mode', () => {
    it('shows edit title')
    it('pre-fills project name')
    it('updates project on submit')
  })
  
  describe('Validation', () => {
    it('shows error for empty name')
    it('shows error for duplicate name')
  })
})
```

## Troubleshooting

### Tests Hang or Timeout

**Cause**: Async operations not resolved

**Solution**: Use `waitFor` or `findBy` queries:

```tsx
// Instead of:
expect(screen.getByText('Loaded')).toBeInTheDocument()

// Use:
expect(await screen.findByText('Loaded')).toBeInTheDocument()
```

### MSW Not Intercepting Requests

**Cause**: Handler not matching request

**Solution**: Check handler patterns and request URLs:

```tsx
// Make sure patterns match exactly
http.get('/api/v1/projects/:namespace', ...)  // ✓
http.get('/projects/:namespace', ...)          // ✗ (missing /api/v1)
```

### Tests Pass Individually but Fail Together

**Cause**: Shared state between tests

**Solution**: Ensure proper cleanup:

```tsx
afterEach(() => {
  cleanup()
  server.resetHandlers() // This is done automatically in setup
})
```

### TypeScript Errors with Test Utilities

**Cause**: Missing type definitions

**Solution**: Ensure `@testing-library/jest-dom/vitest` is imported in `setup.ts`

### Coverage Not Generating

**Cause**: Missing coverage provider

**Solution**: Install coverage package:

```bash
npm install -D @vitest/coverage-v8
```

## Advanced Topics

### Testing Context Providers

```tsx
import { renderWithProviders } from '@/test/utils'

// Custom wrapper for additional providers
function customRender(ui: ReactElement) {
  return renderWithProviders(
    <MyContextProvider value={mockValue}>
      {ui}
    </MyContextProvider>
  )
}

// Use in test
customRender(<MyComponent />)
```

### Testing Hooks

```tsx
import { renderHook, waitFor } from '@testing-library/react'
import { QueryClientProvider } from '@tanstack/react-query'
import { createTestQueryClient } from '@/test/utils'
import { useProjects } from './useProjects'

it('fetches projects', async () => {
  const queryClient = createTestQueryClient()
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  )
  
  const { result } = renderHook(() => useProjects('default'), { wrapper })
  
  await waitFor(() => {
    expect(result.current.data).toBeDefined()
  })
  
  expect(result.current.data?.projects).toHaveLength(2)
})
```

### Debugging Tests

```tsx
import { screen } from '@testing-library/react'

// Print current DOM
screen.debug()

// Print specific element
screen.debug(screen.getByRole('button'))

// Use getBy* queries in try-catch to see available roles
try {
  screen.getByRole('button')
} catch (e) {
  console.log(e.message) // Shows available roles
}
```

## Resources

- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [MSW Documentation](https://mswjs.io/)
- [Testing Library Cheat Sheet](https://testing-library.com/docs/react-testing-library/cheatsheet/)

## Next Steps

With the test infrastructure in place, you can now:

1. Write comprehensive tests for components
2. Add integration tests for user flows
3. Increase test coverage incrementally
4. Set up CI/CD with automated testing

Happy testing! 🧪

