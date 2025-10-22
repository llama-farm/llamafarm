import { apiClient } from './client'

export type VersionCheckResponse = {
  latest_version: string
  name?: string
  release_notes?: string
  release_url?: string
  published_at?: string
  from_cache?: boolean
  install?: {
    mac_linux?: string
    windows?: string
  }
}

// (Avoid adding extra server info APIs in this commit)

export async function getVersionCheck(
  signal?: AbortSignal
): Promise<VersionCheckResponse> {
  let controller: AbortController | undefined
  let usedSignal: AbortSignal
  if (signal) {
    usedSignal = signal
  } else {
    controller = new AbortController()
    usedSignal = controller.signal
  }
  try {
    const res = await apiClient.get<VersionCheckResponse>(
      '/system/version-check',
      {
        signal: usedSignal as any,
      }
    )
    return res.data
  } finally {
    if (controller) controller.abort()
  }
}
