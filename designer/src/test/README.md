# Test Infrastructure

This directory contains the test infrastructure for the Designer application.

## Structure

```
test/
├── setup.ts              # Global test setup (MSW, localStorage, matchMedia)
├── utils.tsx             # Custom render utilities and helpers
├── mocks/
│   ├── handlers.ts       # MSW request handlers for API mocking
│   └── server.ts         # MSW server instance
└── factories/
    ├── projectFactory.ts           # Mock project data generators
    ├── datasetFactory.ts           # Mock dataset data generators
    └── embeddingStrategyFactory.ts # Mock RAG data generators
```

## Key Files

### `setup.ts`

Global test configuration that runs before all tests:
- Starts MSW server for API mocking
- Mocks `localStorage` for browser APIs
- Mocks `matchMedia` for responsive design tests
- Cleans up after each test

### `utils.tsx`

Custom utilities for testing React components:
- `renderWithProviders()` - Renders components with all required providers
- `createTestQueryClient()` - Creates a test-specific QueryClient
- Re-exports all React Testing Library utilities

### `mocks/handlers.ts`

Defines MSW handlers for intercepting API calls:
- Project CRUD operations
- Dataset CRUD operations
- File upload/delete
- Task status
- System info

### `mocks/server.ts`

MSW server instance for Node environment (used in tests).

### `factories/`

Factory functions for generating consistent mock data:
- **projectFactory**: Create mock projects with various configurations
- **datasetFactory**: Create mock datasets with files and processing states
- **embeddingStrategyFactory**: Create mock RAG strategies and databases

## Usage

### Basic Component Test

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

### Using Mock Data Factories

```tsx
import { createMockProject } from '@/test/factories/projectFactory'

const project = createMockProject({
  namespace: 'test',
  name: 'my-project',
})
```

### Overriding API Responses

```tsx
import { server } from '@/test/mocks/server'
import { http, HttpResponse } from 'msw'

server.use(
  http.get('/api/v1/projects/:namespace', () => {
    return HttpResponse.json({ 
      projects: [], 
      total: 0 
    })
  })
)
```

## See Also

- [Full Testing Guide](../../docs/TESTING.md) - Comprehensive documentation
- [Vitest Documentation](https://vitest.dev/)
- [React Testing Library](https://testing-library.com/react)
- [MSW Documentation](https://mswjs.io/)

