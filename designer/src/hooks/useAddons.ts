import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import addonsService from '../api/addonsService'
import type {
  AddonInstallRequest,
  AddonTaskStatus,
} from '../types/addons'

export const DEFAULT_ADDON_LIST_STALE_TIME = 600_000 // 10 minutes

/**
 * Query keys for addon-related queries
 * Follows the hierarchical pattern used in useDatasets.ts
 */
export const addonKeys = {
  all: ['addons'] as const,
  list: () => [...addonKeys.all, 'list'] as const,
  task: (taskId: string) => [...addonKeys.all, 'task', taskId] as const,
}

/**
 * Hook to fetch list of all available addons with installation status
 *
 * @param options - Query options (enabled, staleTime)
 * @returns Query result with addons list
 */
export function useListAddons(options?: {
  enabled?: boolean
  staleTime?: number
}) {
  return useQuery({
    queryKey: addonKeys.list(),
    queryFn: () => addonsService.listAddons(),
    enabled: options?.enabled !== false,
    staleTime: options?.staleTime ?? DEFAULT_ADDON_LIST_STALE_TIME,
    gcTime: 0, // Don't cache failed queries or old data
    retry: false, // Don't retry failed requests
  })
}

/**
 * Hook to install an addon (initiates background task)
 *
 * @returns Mutation for installing addons
 */
export function useInstallAddon() {
  return useMutation({
    mutationFn: (request: AddonInstallRequest) =>
      addonsService.installAddon(request),
    // Note: Query invalidation happens in onComplete callback after installation finishes,
    // not here (which fires when task starts, not when it completes)
  })
}

/**
 * Hook to poll installation task status
 *
 * This hook implements smart polling that stops when the task reaches
 * a terminal state (completed or failed).
 *
 * Pattern: Based on useTaskStatus in useDatasets.ts (lines 292-321)
 *
 * @param taskId - Task ID to monitor (null to disable)
 * @param options - Query options
 * @returns Query result with task status
 */
export function useTaskStatus(
  taskId: string | null,
  options?: {
    enabled?: boolean
    refetchInterval?: number
  }
) {
  return useQuery({
    queryKey: addonKeys.task(taskId!),
    queryFn: () => addonsService.getTaskStatus(taskId!),
    enabled: !!taskId && options?.enabled !== false,
    refetchInterval: query => {
      // Stop polling if task completed, failed, or query disabled
      const data = query.state.data as AddonTaskStatus | undefined
      if (
        data?.status === 'completed' ||
        data?.status === 'failed'
      ) {
        return false
      }
      return options?.refetchInterval || 2000 // Poll every 2 seconds by default
    },
    refetchIntervalInBackground: true, // Continue polling even if tab loses focus
    staleTime: 0, // Always consider stale to ensure fresh polling
    gcTime: 5 * 60 * 1000, // Keep in cache for 5 minutes after unmount
  })
}

/**
 * Hook to uninstall an addon
 *
 * @returns Mutation for uninstalling addons
 */
export function useUninstallAddon() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: (name: string) => addonsService.uninstallAddon(name),
    onSuccess: () => {
      // Invalidate addon list to refetch installation status
      queryClient.invalidateQueries({
        queryKey: addonKeys.list(),
      })
    },
  })
}
