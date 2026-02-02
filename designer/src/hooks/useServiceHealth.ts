import { useState, useEffect, useCallback, useRef } from 'react'
import { getHealth } from '../api/healthService'
import { HealthResponse } from '../types/chat'

const POLL_INTERVAL_MS = 30000 // 30 seconds

/**
 * Service display information for the UI
 */
export interface ServiceDisplay {
  id: string
  displayName: string
  status: 'healthy' | 'degraded' | 'unhealthy'
  message: string
  latencyMs?: number
  host?: string
}

export type AggregateStatus = 'healthy' | 'degraded' | 'unhealthy' | null

/**
 * Mapping of backend component names to display names
 */
const SERVICE_DISPLAY_MAP: Record<string, string> = {
  server: 'LlamaFarm Server',
  'universal-runtime': 'Universal Runtime',
  'rag-service': 'RAG Service',
}

/**
 * Components to display in the panel (subset of all components)
 */
const DISPLAY_COMPONENTS = ['server', 'universal-runtime', 'rag-service']

/**
 * Compute aggregate status from components
 */
function computeAggregateStatus(
  components: HealthResponse['components']
): AggregateStatus {
  const displayComponents = components.filter((c) =>
    DISPLAY_COMPONENTS.includes(c.name)
  )
  if (displayComponents.length === 0) return null
  if (displayComponents.some((c) => c.status === 'unhealthy')) return 'unhealthy'
  if (displayComponents.some((c) => c.status === 'degraded')) return 'degraded'
  return 'healthy'
}

/**
 * Hook to manage service health data with polling
 *
 * @param isOpen - Whether the panel is open (controls polling)
 * @returns Health data, loading state, error state, and refresh function
 */
export function useServiceHealth(isOpen: boolean) {
  const [health, setHealth] = useState<HealthResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<Error | null>(null)
  const intervalRef = useRef<number | null>(null)

  const fetchHealth = useCallback(async () => {
    setIsLoading(true)
    setError(null)
    try {
      const data = await getHealth(true) // skipCache for fresh data
      setHealth(data)
    } catch (err) {
      setError(err instanceof Error ? err : new Error('Failed to fetch health'))
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Fetch on open + start polling
  useEffect(() => {
    if (isOpen) {
      fetchHealth() // immediate fetch
      intervalRef.current = window.setInterval(fetchHealth, POLL_INTERVAL_MS)
    } else {
      // Stop polling when closed
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
      }
    }
  }, [isOpen, fetchHealth])

  // Compute aggregate status
  const aggregateStatus = health
    ? computeAggregateStatus(health.components)
    : null

  // Filter, map, and sort services for display (consistent ordering)
  const services: ServiceDisplay[] =
    health?.components
      .filter((c) => DISPLAY_COMPONENTS.includes(c.name))
      .map((c) => ({
        id: c.name,
        displayName: SERVICE_DISPLAY_MAP[c.name] || c.name,
        status: c.status,
        message: c.message,
        latencyMs: c.latency_ms,
        host: c.details?.host,
      }))
      .sort((a, b) => DISPLAY_COMPONENTS.indexOf(a.id) - DISPLAY_COMPONENTS.indexOf(b.id)) ?? []

  return {
    services,
    aggregateStatus,
    isLoading,
    error,
    refresh: fetchHealth,
  }
}
