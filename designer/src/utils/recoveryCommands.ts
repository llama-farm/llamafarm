/**
 * Recovery Command Generator
 * 
 * Generates actionable recovery commands based on error type and health status.
 * Commands are platform-aware and provide clear steps for users to fix issues.
 */

import { ErrorType } from './errorClassifier'
import { HealthResponse } from '../types/chat'
import { getUnhealthyComponents, getUnhealthySeeds } from '../api/healthService'

/**
 * Recovery command with description
 */
export interface RecoveryCommand {
  description: string
  command: string
}

/**
 * Generate recovery commands for server down scenario
 */
function getServerDownCommands(): RecoveryCommand[] {
  return [
    {
      description: 'Start the LlamaFarm server',
      command: 'lf start',
    },
  ]
}

/**
 * Generate recovery commands based on unhealthy components
 */
function getDegradedCommands(healthStatus: HealthResponse): RecoveryCommand[] {
  const commands: RecoveryCommand[] = []
  const unhealthyComponents = getUnhealthyComponents(healthStatus)
  const unhealthySeeds = getUnhealthySeeds(healthStatus)

  // Check for specific component issues
  const hasOllamaIssue = unhealthyComponents.some(c => 
    c.name === 'ollama' || c.name.includes('ollama')
  )
  const hasRagIssue = unhealthyComponents.some(c => 
    c.name === 'rag-service' || c.name.includes('rag')
  )
  const hasStorageIssue = unhealthyComponents.some(c => 
    c.name === 'storage'
  )

  // Check for model issues in seeds
  const modelIssue = unhealthySeeds.find(s => 
    s.message.toLowerCase().includes('model') && 
    s.message.toLowerCase().includes('not found')
  )

  // Generate specific commands based on issues
  if (modelIssue && modelIssue.runtime?.model) {
    commands.push({
      description: `Pull the missing model '${modelIssue.runtime.model}'`,
      command: `ollama pull ${modelIssue.runtime.model}`,
    })
  }

  if (hasOllamaIssue) {
    // Check platform for appropriate command
    const isMac = typeof navigator !== 'undefined' && navigator.platform.toUpperCase().indexOf('MAC') >= 0
    commands.push({
      description: isMac ? 'Open Ollama app and restart services' : 'Start Ollama service',
      command: isMac ? 'open -a Ollama && lf start' : 'ollama serve',
    })
  }

  if (hasRagIssue) {
    commands.push({
      description: 'Restart the RAG service',
      command: 'docker restart llamafarm-rag',
    })
  }

  if (hasStorageIssue) {
    commands.push({
      description: 'Check data directory permissions',
      command: 'ls -la ~/.llamafarm',
    })
  }

  // Always suggest restarting all services as a catch-all
  if (commands.length > 0) {
    commands.push({
      description: 'Or restart all services',
      command: 'lf start',
    })
  } else {
    // If no specific issue identified, just suggest restart
    commands.push({
      description: 'Restart all services',
      command: 'lf start',
    })
  }

  return commands
}

/**
 * Generate recovery commands for timeout scenario
 */
function getTimeoutCommands(): RecoveryCommand[] {
  return [
    {
      description: 'Check server logs for errors',
      command: 'docker logs llamafarm-server',
    },
    {
      description: 'Restart if server is stuck',
      command: 'lf start',
    },
  ]
}

/**
 * Generate recovery commands for validation errors
 */
function getValidationCommands(): RecoveryCommand[] {
  return [
    {
      description: 'Check your project configuration',
      command: '# Review your llamafarm.yaml file',
    },
  ]
}

/**
 * Generate recovery commands for unknown errors
 */
function getUnknownCommands(): RecoveryCommand[] {
  return [
    {
      description: 'Check server logs',
      command: 'docker logs llamafarm-server',
    },
    {
      description: 'Restart services',
      command: 'lf start',
    },
  ]
}

/**
 * Generate appropriate recovery commands based on error type and health status
 * 
 * @param errorType - The classified error type
 * @param healthStatus - Optional health status from server
 * @returns Array of recovery commands with descriptions
 */
export function generateRecoveryCommands(
  errorType: ErrorType,
  healthStatus?: HealthResponse
): RecoveryCommand[] {
  switch (errorType) {
    case 'server_down':
      return getServerDownCommands()
    
    case 'degraded':
      if (healthStatus) {
        return getDegradedCommands(healthStatus)
      }
      // Fallback if no health status available
      return getServerDownCommands()
    
    case 'timeout':
      return getTimeoutCommands()
    
    case 'validation':
      return getValidationCommands()
    
    case 'unknown':
    default:
      return getUnknownCommands()
  }
}

/**
 * Get a brief summary of health issues for display
 */
export function getHealthSummary(healthStatus: HealthResponse): string {
  const unhealthyComponents = getUnhealthyComponents(healthStatus)
  const unhealthySeeds = getUnhealthySeeds(healthStatus)

  if (unhealthyComponents.length === 0 && unhealthySeeds.length === 0) {
    return 'All services are running.'
  }

  const issues: string[] = []

  // Summarize component issues
  for (const component of unhealthyComponents) {
    const name = component.name.replace(/-/g, ' ')
    issues.push(`${name} is ${component.status}`)
  }

  // Summarize seed issues
  for (const seed of unhealthySeeds) {
    if (seed.message.toLowerCase().includes('model') && seed.message.toLowerCase().includes('not found')) {
      issues.push(`model '${seed.runtime?.model || 'unknown'}' not found`)
    } else {
      issues.push(seed.message)
    }
  }

  return issues.join(', ')
}

export default {
  generateRecoveryCommands,
  getHealthSummary,
}

