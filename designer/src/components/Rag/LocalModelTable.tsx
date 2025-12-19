import { useState, Fragment } from 'react'
import FontIcon from '../../common/FontIcon'
import { Button } from '../ui/button'
import { Badge } from '../ui/badge'
import { Input } from '../ui/input'
import Loader from '../../common/Loader'

export type Variant = {
  id: string
  label: string
  dim: string
  quality: string
  download: string
  modelIdentifier?: string
  isDownloaded?: boolean
  diskSize?: number | null
}

export type LocalGroup = {
  id: number
  name: string
  dim: string
  quality: string
  ramVram: string
  download: string
  variants: Variant[]
}

interface LocalModelTableProps {
  filteredGroups: LocalGroup[]
  query: string
  onQueryChange: (query: string) => void
  selected?: {
    runtime: 'Local' | 'Cloud'
    provider: string
    modelId: string
  } | null
  downloadStates: Record<
    string,
    {
      state: 'idle' | 'downloading' | 'success' | 'error'
      progress: number
      downloadedBytes: number
      totalBytes: number
      error?: string
    }
  >
  onSelect: (variant: Variant) => void
  onDownloadRetry?: (variant: Variant) => void
  onRefresh?: () => void
  isRefreshing?: boolean
  isLoadingCachedModels?: boolean
}

export function LocalModelTable({
  filteredGroups,
  query,
  onQueryChange,
  selected,
  downloadStates,
  onSelect,
  onDownloadRetry,
  onRefresh,
  isRefreshing = false,
  isLoadingCachedModels = false,
}: LocalModelTableProps) {
  const [expandedGroupId, setExpandedGroupId] = useState<number | null>(null)

  return (
    <>
      <div className="flex items-center justify-between w-full">
        <div className="relative flex-1 max-w-md">
          <FontIcon
            type="search"
            className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2"
          />
          <Input
            placeholder="Search local options"
            value={query}
            onChange={e => onQueryChange(e.target.value)}
            className="pl-9 h-10"
          />
        </div>
        {onRefresh && (
          <Button
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={isRefreshing || isLoadingCachedModels}
            className="h-10 ml-2"
          >
            {(isRefreshing || isLoadingCachedModels) && (
              <Loader size={16} className="mr-2" />
            )}
            Refresh
          </Button>
        )}
      </div>
      <div className="w-full rounded-lg border border-border">
        <div className="hidden md:grid grid-cols-12 items-start md:items-center bg-secondary text-secondary-foreground text-xs px-3 py-2 gap-x-2">
          <div className="col-span-3 min-w-0">Model</div>
          <div className="col-span-2 min-w-0">dim</div>
          <div className="col-span-2 min-w-0">Quality</div>
          <div className="col-span-2 min-w-0">Size</div>
          <div className="col-span-3 min-w-0">Status</div>
        </div>
        {filteredGroups.map(group => {
          const isOpen = expandedGroupId === group.id
          return (
            <div key={group.id} className="border-t border-border">
              {/* Mobile layout */}
              <div
                className="md:hidden flex flex-col px-3 py-3 cursor-pointer hover:bg-accent/40"
                onClick={() =>
                  setExpandedGroupId(prev =>
                    prev === group.id ? null : group.id
                  )
                }
              >
                <div className="flex items-center gap-2">
                  <span
                    className="flex-shrink-0 flex-grow-0 inline-flex items-center justify-center"
                    style={{
                      width: '1rem',
                      height: '1rem',
                      minWidth: '1rem',
                      maxWidth: '1rem',
                      flexShrink: 0,
                      flexGrow: 0,
                    }}
                  >
                    <FontIcon
                      type="chevron-down"
                      className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    />
                  </span>
                  <span className="font-medium">{group.name}</span>
                </div>
                <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs text-muted-foreground">
                  <span>dim: {group.dim}</span>
                  <span>Quality: {group.quality}</span>
                  <span>Size: {group.download}</span>
                  <span>{group.ramVram}</span>
                </div>
              </div>
              {/* Desktop layout */}
              <div
                className="hidden md:grid grid-cols-12 items-start md:items-center px-3 py-4 md:py-3 text-sm cursor-pointer hover:bg-accent/40 gap-x-2"
                onClick={() =>
                  setExpandedGroupId(prev =>
                    prev === group.id ? null : group.id
                  )
                }
              >
                <div className="col-span-3 flex items-center gap-2 min-w-0">
                  <span
                    className="flex-shrink-0 flex-grow-0 inline-flex items-center justify-center"
                    style={{
                      width: '1rem',
                      height: '1rem',
                      minWidth: '1rem',
                      maxWidth: '1rem',
                      flexShrink: 0,
                      flexGrow: 0,
                    }}
                  >
                    <FontIcon
                      type="chevron-down"
                      className={`w-4 h-4 transition-transform ${isOpen ? 'rotate-180' : ''}`}
                    />
                  </span>
                  <span className="truncate font-medium min-w-0">
                    {group.name}
                  </span>
                </div>
                <div className="col-span-2 text-xs text-muted-foreground truncate min-w-0">
                  {group.dim}
                </div>
                <div className="col-span-2 text-xs text-muted-foreground truncate min-w-0">
                  {group.quality}
                </div>
                <div className="col-span-2 text-xs text-muted-foreground truncate min-w-0">
                  {group.download}
                </div>
                <div className="col-span-2 text-xs text-muted-foreground truncate min-w-0">
                  {group.ramVram}
                </div>
                <div className="col-span-3 min-w-0" />
              </div>
              {group.variants && isOpen && (
                <div className="px-3 pb-2">
                  {group.variants.map(v => {
                    const isUsing =
                      selected?.runtime === 'Local' &&
                      selected?.modelId === v.id
                    const downloadState = downloadStates[v.id]
                    const isDownloading =
                      downloadState?.state === 'downloading'
                    const hasError = downloadState?.state === 'error'
                    const isDownloaded =
                      v.isDownloaded || downloadState?.state === 'success'

                    return (
                      <Fragment key={v.id}>
                        {/* Mobile layout */}
                        <div
                          className="md:hidden flex flex-col px-3 py-3 rounded-md hover:bg-accent/40"
                        >
                          <div className="mb-2">
                            <span className="font-mono text-xs">{v.label}</span>
                          </div>
                          <div className="flex flex-wrap gap-x-4 gap-y-1 mb-2 text-xs text-muted-foreground">
                            <span>dim: {v.dim}</span>
                            <span>Quality: {v.quality}</span>
                            <span>Size: {v.download}</span>
                          </div>
                          <div className="flex flex-col gap-2">
                            <div className="flex flex-wrap items-center gap-2">
                              {isDownloaded && (
                                <Badge
                                  variant="outline"
                                  size="sm"
                                  className="rounded-xl text-muted-foreground border-muted flex-shrink-0"
                                >
                                  On disk
                                </Badge>
                              )}
                              {isUsing && (
                                <Badge
                                  variant="default"
                                  size="sm"
                                  className="rounded-xl flex-shrink-0"
                                >
                                  <FontIcon
                                    type="checkmark-filled"
                                    className="w-3 h-3 mr-1 flex-shrink-0"
                                  />
                                  Using
                                </Badge>
                              )}
                              {hasError && (
                                <Badge
                                  variant="secondary"
                                  size="sm"
                                  className="rounded-xl text-destructive border-destructive flex-shrink-0"
                                >
                                  Error
                                </Badge>
                              )}
                              {!isDownloaded && !isUsing && !hasError && (
                                <Badge
                                  variant="outline"
                                  size="sm"
                                  className="rounded-xl flex-shrink-0"
                                >
                                  Not downloaded
                                </Badge>
                              )}
                            </div>
                            <Button
                              size="sm"
                              className={`h-8 px-3 flex-shrink-0 w-fit ${
                                !isUsing &&
                                selected?.runtime === 'Local' &&
                                selected?.modelId
                                  ? 'opacity-60 hover:opacity-100'
                                  : ''
                              }`}
                              onClick={e => {
                                e.stopPropagation()
                                onSelect(v)
                              }}
                              disabled={isDownloading || hasError}
                              variant={
                                !isUsing &&
                                selected?.runtime === 'Local' &&
                                selected?.modelId
                                  ? 'outline'
                                  : 'default'
                              }
                            >
                              {isDownloading ? (
                                'Downloading...'
                              ) : isUsing ? (
                                <FontIcon
                                  type="checkmark-filled"
                                  className="w-4 h-4"
                                />
                              ) : selected?.runtime === 'Local' &&
                                selected?.modelId ? (
                                'Use instead'
                              ) : (
                                'Use'
                              )}
                            </Button>
                          </div>
                        </div>
                        {/* Desktop layout */}
                        <div
                          className="hidden md:grid grid-cols-12 items-start md:items-center px-3 py-4 md:py-3 text-sm rounded-md hover:bg-accent/40 gap-x-2 gap-y-2"
                        >
                          <div className="col-span-3 flex items-start md:items-center text-muted-foreground min-w-0 pt-1 md:pt-0">
                            <span
                              className="inline-block w-4 flex-shrink-0"
                              style={{
                                width: '1rem',
                                minWidth: '1rem',
                                flexShrink: 0,
                              }}
                            />
                            <span className="ml-2 font-mono text-xs truncate min-w-0">
                              {v.label}
                            </span>
                          </div>
                          <div className="col-span-2 text-xs text-muted-foreground truncate min-w-0 pt-1 md:pt-0">
                            {v.dim}
                          </div>
                          <div className="col-span-2 text-xs text-muted-foreground truncate min-w-0 pt-1 md:pt-0">
                            {v.quality}
                          </div>
                          <div className="col-span-2 text-xs text-muted-foreground truncate min-w-0 pt-1 md:pt-0">
                            {v.download}
                          </div>
                          <div className="col-span-3 flex flex-row items-center gap-2 min-w-0 flex-wrap">
                            {isDownloaded && (
                              <Badge
                                variant="outline"
                                size="sm"
                                className="rounded-xl text-muted-foreground border-muted flex-shrink-0"
                              >
                                On disk
                              </Badge>
                            )}
                            {isUsing && (
                              <Badge
                                variant="default"
                                size="sm"
                                className="rounded-xl flex-shrink-0"
                              >
                                <FontIcon
                                  type="checkmark-filled"
                                  className="w-3 h-3 mr-1 flex-shrink-0"
                                />
                                Using
                              </Badge>
                            )}
                            {hasError && (
                              <Badge
                                variant="secondary"
                                size="sm"
                                className="rounded-xl text-destructive border-destructive flex-shrink-0"
                              >
                                Error
                              </Badge>
                            )}
                            {!isDownloaded && !isUsing && !hasError && (
                              <Badge
                                variant="outline"
                                size="sm"
                                className="rounded-xl flex-shrink-0"
                              >
                                Not downloaded
                              </Badge>
                            )}
                            <Button
                              size="sm"
                              className={`h-8 px-3 flex-shrink-0 ${
                                !isUsing &&
                                selected?.runtime === 'Local' &&
                                selected?.modelId
                                  ? 'opacity-60 hover:opacity-100'
                                  : ''
                              }`}
                              onClick={e => {
                                e.stopPropagation()
                                onSelect(v)
                              }}
                              disabled={isDownloading || hasError}
                              variant={
                                !isUsing &&
                                selected?.runtime === 'Local' &&
                                selected?.modelId
                                  ? 'outline'
                                  : 'default'
                              }
                            >
                              {isDownloading ? (
                                'Downloading...'
                              ) : isUsing ? (
                                <FontIcon
                                  type="checkmark-filled"
                                  className="w-4 h-4"
                                />
                              ) : selected?.runtime === 'Local' &&
                                selected?.modelId ? (
                                'Use instead'
                              ) : (
                                'Use'
                              )}
                            </Button>
                          </div>
                        </div>
                        {hasError && downloadState?.error && onDownloadRetry && (
                          <>
                            <div className="md:hidden mt-2 px-3">
                              <div className="text-xs text-destructive">
                                {downloadState.error}
                              </div>
                              <Button
                                size="sm"
                                variant="outline"
                                className="mt-1 h-6 text-xs"
                                onClick={e => {
                                  e.stopPropagation()
                                  onDownloadRetry(v)
                                }}
                              >
                                Retry
                              </Button>
                            </div>
                            <div className="hidden md:block col-span-12 mt-2 px-3">
                              <div className="text-xs text-destructive">
                                {downloadState.error}
                              </div>
                              <Button
                                size="sm"
                                variant="outline"
                                className="mt-1 h-6 text-xs"
                                onClick={e => {
                                  e.stopPropagation()
                                  onDownloadRetry(v)
                                }}
                              >
                                Retry
                              </Button>
                            </div>
                          </>
                        )}
                      </Fragment>
                    )
                  })}
                  <div className="flex justify-end pr-3">
                    <button
                      className="text-xs text-muted-foreground hover:text-foreground"
                      onClick={() => setExpandedGroupId(null)}
                    >
                      Hide
                    </button>
                  </div>
                </div>
              )}
            </div>
          )
        })}
      </div>
    </>
  )
}

