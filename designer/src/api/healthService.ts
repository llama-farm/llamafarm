/**
 * Health Check API Service
 * 
 * Provides health status information from the LlamaFarm server.
 * Used for diagnosing connection issues and service availability.
 */

import { apiClient } from './client'
import { HealthResponse } from '../types/chat'

/**
 * Component health status (re-exported for convenience)
 */
export type HealthComponent = HealthResponse['components'][number]

/**
 * Seed project health status (re-exported for convenience)
 */
export type HealthSeed = HealthResponse['seeds'][number]

/**
 * Cached health result with timestamp
 */
interface CachedHealth {
  data: HealthResponse
  fetchedAt: number
}

// Cache health checks for 5 seconds
const CACHE_TTL_MS = 5000
let cachedHealth: CachedHealth | null = null

/**
 * Get server health status
 * 
 * @param skipCache - If true, bypass cache and fetch fresh data
 * @returns Promise<HealthResponse> - Health status of all server components
 */
export async function getHealth(skipCache: boolean = false): Promise<HealthResponse> {
  // Check cache first
  if (!skipCache && cachedHealth) {
    const age = Date.now() - cachedHealth.fetchedAt
    if (age < CACHE_TTL_MS) {
      return cachedHealth.data
    }
  }

  try {
    // Health endpoint is at root level (no /v1 prefix)
    const response = await apiClient.get<HealthResponse>('/health', {
      // Override base URL to remove /v1 prefix
      baseURL: apiClient.defaults.baseURL?.replace(/\/v1\/?$/, ''),
      timeout: 5000, // Quick timeout for health checks
    })

    // Cache the result
    cachedHealth = {
      data: response.data,
      fetchedAt: Date.now(),
    }

    return response.data
  } catch (error) {
    // If health check fails, clear cache and rethrow
    cachedHealth = null
    throw error
  }
}

/**
 * Clear the health check cache
 * Useful when you want to force a fresh check
 */
export function clearHealthCache(): void {
  cachedHealth = null
}

/**
 * Check if a specific component is healthy
 * 
 * @param health - Health response
 * @param componentName - Name of component to check (e.g., 'server', 'rag-service', 'ollama')
 * @returns boolean - True if component exists and is healthy
 */
export function isComponentHealthy(health: HealthResponse, componentName: string): boolean {
  const component = health.components.find(c => c.name === componentName)
  return component?.status === 'healthy'
}

/**
 * Get all unhealthy components
 * 
 * @param health - Health response
 * @returns Array of components that are degraded or unhealthy
 */
export function getUnhealthyComponents(health: HealthResponse): HealthComponent[] {
  return health.components.filter(c => c.status !== 'healthy')
}

/**
 * Get all unhealthy seeds
 * 
 * @param health - Health response
 * @returns Array of seeds that are degraded or unhealthy
 */
export function getUnhealthySeeds(health: HealthResponse): HealthSeed[] {
  return health.seeds.filter(s => s.status !== 'healthy')
}

export default {
  getHealth,
  clearHealthCache,
  isComponentHealthy,
  getUnhealthyComponents,
  getUnhealthySeeds,
}

