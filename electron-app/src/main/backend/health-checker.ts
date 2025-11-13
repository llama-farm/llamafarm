/**
 * Health Checker - Monitors backend service health
 */

import axios from 'axios'

export interface HealthStatus {
  status: 'healthy' | 'degraded' | 'unhealthy'
  components: ComponentHealth[]
  readyCount: number
  totalCount: number
  message?: string
}

export interface ComponentHealth {
  name: string
  status: string
  message: string
  latencyMs?: number
}

export class HealthChecker {
  private baseUrl: string

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl
  }

  /**
   * Check backend health
   */
  async check(): Promise<HealthStatus> {
    try {
      const response = await axios.get(`${this.baseUrl}/health`, {
        timeout: 5000
      })

      const data = response.data

      // Count ready components
      const components: ComponentHealth[] = data.components || []
      const readyCount = components.filter(c => c.status === 'healthy').length
      const totalCount = components.length

      // Determine overall status
      let status: 'healthy' | 'degraded' | 'unhealthy'
      if (readyCount === totalCount && totalCount > 0) {
        status = 'healthy'
      } else if (readyCount > 0) {
        status = 'degraded'
      } else {
        status = 'unhealthy'
      }

      return {
        status,
        components,
        readyCount,
        totalCount,
        message: data.summary || `${readyCount}/${totalCount} services ready`
      }
    } catch (error) {
      // If we can't reach the health endpoint, backend is unhealthy
      return {
        status: 'unhealthy',
        components: [],
        readyCount: 0,
        totalCount: 0,
        message: error instanceof Error ? error.message : 'Health check failed'
      }
    }
  }

  /**
   * Wait for backend to be healthy
   */
  async waitForHealthy(timeoutMs: number = 180000): Promise<HealthStatus> {
    const startTime = Date.now()
    const checkInterval = 3000 // 3 seconds

    while (Date.now() - startTime < timeoutMs) {
      const health = await this.check()

      if (health.status === 'healthy') {
        return health
      }

      await new Promise(resolve => setTimeout(resolve, checkInterval))
    }

    throw new Error('Health check timeout')
  }
}
