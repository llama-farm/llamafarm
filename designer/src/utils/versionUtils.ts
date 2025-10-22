import pkg from '../../package.json'

export type ReleaseInfo = {
  latestVersion: string
  htmlUrl: string
  publishedAt?: string
}

const STORAGE_KEYS = {
  latest: 'lf_latest_release',
  checkedAt: 'lf_latest_release_checked_at',
}

export function getCurrentVersion(): string {
  // Prefer runtime-provided image tag (dev/prod)
  try {
    const viteEnv = (import.meta as any)?.env?.VITE_APP_IMAGE_TAG as
      | string
      | undefined
    if (viteEnv && typeof viteEnv === 'string' && viteEnv.trim() !== '') {
      return viteEnv
    }
  } catch {}
  try {
    const winEnv = (window as any)?.ENV?.VITE_APP_IMAGE_TAG as
      | string
      | undefined
    if (winEnv && typeof winEnv === 'string' && winEnv.trim() !== '') {
      return winEnv
    }
  } catch {}

  // Fallback to package version (source build)
  const raw = (pkg as any)?.version as string | undefined
  return raw || '0.0.0'
}

/**
 * Returns the injected image tag if provided by the runtime via either
 * import.meta.env.VITE_APP_IMAGE_TAG (Vite) or window.ENV.VITE_APP_IMAGE_TAG (injected at runtime).
 * Returns null if not present.
 */
export function getInjectedImageTag(): string | null {
  try {
    const viteEnv = (import.meta as any)?.env?.VITE_APP_IMAGE_TAG as
      | string
      | undefined
    if (viteEnv && typeof viteEnv === 'string' && viteEnv.trim() !== '') {
      return viteEnv
    }
  } catch {}
  try {
    const winEnv = (window as any)?.ENV?.VITE_APP_IMAGE_TAG as
      | string
      | undefined
    if (winEnv && typeof winEnv === 'string' && winEnv.trim() !== '') {
      return winEnv
    }
  } catch {}
  return null
}

export function normalizeVersion(version: string | null | undefined): string {
  if (!version) return '0.0.0'
  return version.startsWith('v') ? version.slice(1) : version
}

export function compareSemver(a: string, b: string): number {
  const pa = normalizeVersion(a)
    .split('.')
    .map(n => parseInt(n || '0', 10))
  const pb = normalizeVersion(b)
    .split('.')
    .map(n => parseInt(n || '0', 10))
  for (let i = 0; i < 3; i++) {
    const da = pa[i] || 0
    const db = pb[i] || 0
    if (da > db) return 1
    if (da < db) return -1
  }
  return 0
}

export function storeLatestRelease(
  info: ReleaseInfo,
  nowMs: number = Date.now()
): void {
  try {
    localStorage.setItem(STORAGE_KEYS.latest, JSON.stringify(info))
    localStorage.setItem(STORAGE_KEYS.checkedAt, String(nowMs))
  } catch {}
}

export function getStoredLatestRelease(): {
  info: ReleaseInfo | null
  checkedAt: number | null
} {
  try {
    const raw = localStorage.getItem(STORAGE_KEYS.latest)
    const checkedAtRaw = localStorage.getItem(STORAGE_KEYS.checkedAt)
    const info = raw ? (JSON.parse(raw) as ReleaseInfo) : null
    const checkedAt = checkedAtRaw ? parseInt(checkedAtRaw, 10) : null
    return { info, checkedAt }
  } catch {
    return { info: null, checkedAt: null }
  }
}

export async function fetchLatestReleaseFromGithub(
  signal?: AbortSignal
): Promise<ReleaseInfo | null> {
  // Primary: releases/latest
  try {
    const res = await fetch(
      'https://api.github.com/repos/llama-farm/llamafarm/releases/latest',
      { signal }
    )
    if (res.ok) {
      const data = await res.json()
      const latestVersion = (data?.tag_name as string) || ''
      const htmlUrl =
        (data?.html_url as string) ||
        'https://github.com/llama-farm/llamafarm/releases'
      const publishedAt = (data?.published_at as string) || undefined
      if (latestVersion) {
        return { latestVersion, htmlUrl, publishedAt }
      }
    }
  } catch {}
  // Fallback: first non-draft from releases list
  try {
    const res = await fetch(
      'https://api.github.com/repos/llama-farm/llamafarm/releases',
      { signal }
    )
    if (res.ok) {
      const arr = (await res.json()) as any[]
      const item = arr?.find(x => !x?.draft)
      if (item) {
        return {
          latestVersion: (item.tag_name as string) || '',
          htmlUrl:
            (item.html_url as string) ||
            'https://github.com/llama-farm/llamafarm/releases',
          publishedAt: (item.published_at as string) || undefined,
        }
      }
    }
  } catch {}
  return null
}

export function shouldCheck(
  lastCheckedAtMs: number | null,
  intervalMs: number
): boolean {
  if (!lastCheckedAtMs) return true
  return Date.now() - lastCheckedAtMs > intervalMs
}

export type DismissContext = 'home' | 'project'

function dismissKey(version: string, ctx: DismissContext): string {
  const v = normalizeVersion(version)
  return `lf_upgrade_dismissed::v${v}::${ctx}`
}

export function isDismissed(version: string, ctx: DismissContext): boolean {
  try {
    return localStorage.getItem(dismissKey(version, ctx)) === '1'
  } catch {
    return false
  }
}

export function setDismissed(
  version: string,
  ctx: DismissContext,
  dismissed: boolean = true
): void {
  try {
    const key = dismissKey(version, ctx)
    if (dismissed) localStorage.setItem(key, '1')
    else localStorage.removeItem(key)
  } catch {}
}

export function getGithubReleasesUrl(): string {
  return 'https://github.com/llama-farm/llamafarm/releases'
}
