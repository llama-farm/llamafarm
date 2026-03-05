/**
 * Add-ons API Service
 *
 * Provides functions for managing LlamaFarm add-ons (installation, status checking, etc.)
 */

import { apiClient } from './client'
import type {
  AddonInfo,
  AddonInstallRequest,
  AddonInstallResponse,
  AddonTaskStatus,
} from '../types/addons'

/**
 * List all available add-ons with installation status
 *
 * @returns Promise<AddonInfo[]> - List of all add-ons
 */
export async function listAddons(): Promise<AddonInfo[]> {
  const response = await apiClient.get<AddonInfo[]>('/addons')
  return response.data
}

/**
 * Install an add-on (initiates background task)
 *
 * @param request - Installation request with addon name and restart preference
 * @returns Promise<AddonInstallResponse> - Task ID for tracking progress
 */
export async function installAddon(
  request: AddonInstallRequest
): Promise<AddonInstallResponse> {
  const response = await apiClient.post<AddonInstallResponse>('/addons/install', request)
  return response.data
}

/**
 * Get installation task status
 *
 * @param taskId - Task ID from installation response
 * @returns Promise<AddonTaskStatus> - Current task status and progress
 */
export async function getTaskStatus(taskId: string): Promise<AddonTaskStatus> {
  const response = await apiClient.get<AddonTaskStatus>(
    `/addons/tasks/${encodeURIComponent(taskId)}`
  )
  return response.data
}

/**
 * Uninstall an add-on
 *
 * @param name - Addon name to uninstall
 * @returns Promise<void>
 */
export async function uninstallAddon(name: string): Promise<void> {
  await apiClient.post('/addons/uninstall', { name })
}

export default {
  listAddons,
  installAddon,
  getTaskStatus,
  uninstallAddon,
}
