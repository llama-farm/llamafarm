import React, { ReactElement, useState } from 'react'
import { render, RenderOptions } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter } from 'react-router-dom'
import { ThemeProvider } from '../contexts/ThemeContext'

/**
 * Create a new QueryClient instance for testing with retry disabled
 * This ensures tests fail fast instead of retrying failed requests
 */
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0, // Disable caching
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  })
}

interface AllTheProvidersProps {
  children: React.ReactNode
}

/**
 * Wrapper component that provides all necessary context providers
 * Used by renderWithProviders to wrap test components
 * 
 * Note: QueryClient is memoized with useState to prevent recreation on re-renders,
 * which would reset React Query caches and break tests relying on persisted state.
 */
function AllTheProviders({ children }: AllTheProvidersProps) {
  const [queryClient] = useState(() => createTestQueryClient())

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider>
        <BrowserRouter>{children}</BrowserRouter>
      </ThemeProvider>
    </QueryClientProvider>
  )
}

/**
 * Custom render function that wraps components with all necessary providers
 * Use this instead of @testing-library/react's render function
 * 
 * @example
 * ```tsx
 * import { renderWithProviders } from '@/test/utils'
 * import { MyComponent } from './MyComponent'
 * 
 * test('renders component', () => {
 *   const { getByText } = renderWithProviders(<MyComponent />)
 *   expect(getByText('Hello')).toBeInTheDocument()
 * })
 * ```
 */
export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>
) {
  return render(ui, { wrapper: AllTheProviders, ...options })
}

/**
 * Create a custom QueryClient with specific options
 * Useful when you need to test specific QueryClient behavior
 * 
 * @example
 * ```tsx
 * const queryClient = createTestQueryClient()
 * queryClient.setQueryData(['projects', 'default'], mockProjects)
 * ```
 */
export { createTestQueryClient }

// Re-export everything from React Testing Library
export * from '@testing-library/react'
export { default as userEvent } from '@testing-library/user-event'

