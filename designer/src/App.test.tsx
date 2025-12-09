import { describe, it, expect } from 'vitest'
import { renderWithProviders } from './test/utils'
import App from './App'

/**
 * Smoke test for the main App component
 * This test verifies that the test infrastructure is working correctly
 */
describe('App Component - Smoke Test', () => {
  it('renders without crashing', () => {
    renderWithProviders(<App />)
    
    // Verify that the main element is present
    expect(document.querySelector('main')).toBeInTheDocument()
  })

  it('renders header', () => {
    renderWithProviders(<App />)
    
    // The header should be present
    expect(document.querySelector('header')).toBeInTheDocument()
  })
})

