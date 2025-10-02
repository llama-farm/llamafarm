import { useMemo } from 'react'
import { useUpgradeAvailability } from '@/hooks/useUpgradeAvailability'

export function HomeUpgradeBanner() {
  const {
    upgradeAvailable,
    latestVersion,
    isDismissedFor,
    dismiss,
    releasesUrl,
    _dismissCounter,
  } = useUpgradeAvailability()

  const shouldShow = useMemo(() => {
    if (!upgradeAvailable) return false
    return !isDismissedFor('home')
  }, [upgradeAvailable, isDismissedFor, _dismissCounter])

  if (!shouldShow || !latestVersion) return null

  return (
    <>
      <div className="fixed top-12 left-0 right-0 z-40 w-full border-b bg-teal-50 text-teal-900 dark:bg-teal-900/30 dark:text-teal-200 border-teal-200 dark:border-teal-800">
        <div className="w-full px-4 py-2 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2 text-sm">
            <span className="font-medium">Upgrade available</span>
            <span className="font-mono">v{latestVersion}</span>
            <a
              href={releasesUrl}
              target="_blank"
              rel="noreferrer"
              className="text-teal-700 underline dark:text-teal-300"
            >
              Details
            </a>
          </div>
          <div className="flex items-center gap-2">
            <a
              href={releasesUrl}
              target="_blank"
              rel="noreferrer"
              className="px-3 py-1 rounded-md bg-teal-600 text-white hover:bg-teal-700 text-sm"
            >
              Upgrade to latest
            </a>
            <button
              onClick={() => dismiss('home')}
              className="px-2 py-1 rounded-md border border-teal-200 dark:border-teal-800 text-sm hover:bg-teal-100 dark:hover:bg-teal-800/40"
            >
              Dismiss
            </button>
          </div>
        </div>
      </div>
      {/* Spacer to offset fixed banner height */}
      <div aria-hidden className="w-full h-10" />
    </>
  )
}

export function ProjectUpgradeBanner() {
  const {
    upgradeAvailable,
    latestVersion,
    isDismissedFor,
    dismiss,
    releasesUrl,
    _dismissCounter,
  } = useUpgradeAvailability()

  const shouldShow = useMemo(() => {
    if (!upgradeAvailable) return false
    return !isDismissedFor('project')
  }, [upgradeAvailable, isDismissedFor, _dismissCounter])

  if (!shouldShow || !latestVersion) return null

  return (
    <div className="fixed right-4 bottom-4 z-40">
      <div className="rounded-lg border border-teal-200 dark:border-teal-800 bg-teal-50 dark:bg-teal-900/30 text-teal-900 dark:text-teal-200 shadow-lg p-3 flex items-center gap-3">
        <div className="text-sm">
          <span className="font-medium">Upgrade available</span>{' '}
          <span className="font-mono">v{latestVersion}</span>
        </div>
        <a
          href={releasesUrl}
          target="_blank"
          rel="noreferrer"
          className="px-2 py-1 rounded-md bg-teal-600 text-white hover:bg-teal-700 text-xs"
        >
          Upgrade
        </a>
        <button
          onClick={() => dismiss('project')}
          className="px-2 py-1 rounded-md text-xs border border-teal-200 dark:border-teal-800 hover:bg-teal-100 dark:hover:bg-teal-800/40"
        >
          Dismiss
        </button>
      </div>
    </div>
  )
}
