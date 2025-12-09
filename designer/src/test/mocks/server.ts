import { setupServer } from 'msw/node'
import { handlers } from './handlers'

/**
 * MSW Server for Node environment (used in tests)
 * This server intercepts HTTP requests during tests and returns mock responses
 */
export const server = setupServer(...handlers)

