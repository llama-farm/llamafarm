export type ReleaseInfo = {
  latestVersion: string
  htmlUrl: string
  publishedAt?: string
}

const STORAGE_KEYS = {
  latest: 'lf_latest_release',
  checkedAt: 'lf_latest_release_checked_at',
  currentVersion: 'lf_current_version',
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

export function storeCurrentVersion(version: string): void {
  try {
    localStorage.setItem(STORAGE_KEYS.currentVersion, version)
  } catch {}
}

export function getStoredCurrentVersion(): string {
  try {
    return localStorage.getItem(STORAGE_KEYS.currentVersion) || '0.0.0'
  } catch {
    return '0.0.0'
  }
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
