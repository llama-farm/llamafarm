/**
 * Add-on types for LlamaFarm Designer
 *
 * Matches backend Pydantic models from server/api/routers/addons/types.py
 */

/**
 * Add-on metadata and installation status
 */
export interface AddonInfo {
  /** Unique addon identifier (e.g., 'stt', 'tts', 'onnxruntime') */
  name: string
  /** Human-readable display name */
  display_name: string
  /** Brief description of what the addon provides */
  description: string
  /** Target component (e.g., 'universal-runtime') */
  component: string
  /** Addon version */
  version: string
  /** List of addon dependencies (other addon names) */
  dependencies: string[]
  /** List of Python packages to install (empty for meta addons) */
  packages: string[]
  /** Whether the addon is currently installed */
  installed: boolean
  /** ISO timestamp when installed, null if not installed */
  installed_at: string | null
}

/**
 * Request to install an addon
 */
export interface AddonInstallRequest {
  /** Addon name to install */
  name: string
  /** Whether to restart the service after installation */
  restart_service: boolean
}

/**
 * Response after initiating addon installation
 */
export interface AddonInstallResponse {
  /** Background task ID for tracking installation progress */
  task_id: string
  /** Initial status ('in_progress') */
  status: string
  /** Name of addon being installed */
  addon: string
}

/**
 * Installation task status
 */
export interface AddonTaskStatus {
  /** Current task status */
  status: 'in_progress' | 'completed' | 'failed'
  /** Progress percentage (0-100) */
  progress: number
  /** Human-readable status message */
  message: string
  /** Error message if status is 'failed', null otherwise */
  error: string | null
}

/**
 * Request to uninstall an addon
 */
export interface AddonUninstallRequest {
  /** Addon name to uninstall */
  name: string
}
