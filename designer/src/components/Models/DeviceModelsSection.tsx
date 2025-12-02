import { useState, useEffect } from 'react'
import { Button } from '../ui/button'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'
import { SearchInput } from '../ui/search-input'
import { motion, AnimatePresence } from 'framer-motion'

interface DeviceModel {
  id: string
  name: string
  modelIdentifier: string
  meta: string
  badges: string[]
}

interface DeviceModelCardProps {
  model: DeviceModel
  onUse: () => void
  onDelete: () => void
  isInUse?: boolean
}

function DeviceModelCard({
  model,
  onUse,
  onDelete,
  isInUse,
}: DeviceModelCardProps) {
  return (
    <div className="w-full h-full bg-card rounded-lg border border-border flex flex-col gap-2 p-4 relative">
      <div className="absolute top-2 right-2">
        <button
          className="w-6 h-6 grid place-items-center rounded-md text-muted-foreground hover:bg-accent/30"
          onClick={onDelete}
          aria-label="Delete from disk"
        >
          <FontIcon type="overflow" className="w-4 h-4" />
        </button>
      </div>

      <div className="text-sm text-muted-foreground">
        {model.modelIdentifier}
      </div>

      <div className="text-lg font-medium">{model.name}</div>

      <div className="flex justify-between items-center gap-2 pt-1">
        <span className="text-sm text-muted-foreground">{model.meta}</span>
        <Button
          onClick={onUse}
          size="sm"
          disabled={isInUse}
          className="w-auto px-6 flex-shrink-0"
        >
          {isInUse ? 'Using' : 'Use'}
        </Button>
      </div>
    </div>
  )
}

interface DeviceModelsSectionProps {
  models: DeviceModel[]
  isLoading: boolean
  isRefreshing: boolean
  onUse: (model: DeviceModel) => void
  onDelete: (model: DeviceModel) => void
  onRefresh: () => void
  isModelInUse: (modelId: string) => boolean
}

export function DeviceModelsSection({
  models,
  isLoading,
  isRefreshing,
  onUse,
  onDelete,
  onRefresh,
  isModelInUse,
}: DeviceModelsSectionProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [currentPage, setCurrentPage] = useState(0)

  const ITEMS_PER_PAGE = 6

  // Filter models based on search query
  const filteredModels = models.filter(
    m =>
      m.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      m.modelIdentifier.toLowerCase().includes(searchQuery.toLowerCase())
  )

  // Calculate pagination
  const totalPages = Math.ceil(filteredModels.length / ITEMS_PER_PAGE)

  // Clamp currentPage to valid range when filtered results shrink
  useEffect(() => {
    if (totalPages > 0 && currentPage >= totalPages) {
      setCurrentPage(Math.max(0, totalPages - 1))
    }
  }, [totalPages, currentPage])

  const effectivePage =
    totalPages > 0 ? Math.min(currentPage, totalPages - 1) : 0

  const paginatedModels = filteredModels.slice(
    effectivePage * ITEMS_PER_PAGE,
    (effectivePage + 1) * ITEMS_PER_PAGE
  )

  const resetPage = () => setCurrentPage(0)

  return (
    <div className="flex flex-col gap-4 mb-8 md:mb-10">
      <div className="flex items-start justify-between">
        <div>
          <h3 className="font-medium">Models on device</h3>
          <div className="h-1" />
          <div className="text-sm text-muted-foreground">
            Models found on your local disk that are ready to use.
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={onRefresh}
          disabled={isLoading || isRefreshing}
          className="flex items-center gap-2"
        >
          {isLoading || isRefreshing ? (
            <Loader size={14} className="border-primary" />
          ) : (
            <FontIcon type="recently-viewed" className="w-4 h-4" />
          )}
          {isLoading || isRefreshing ? 'Refreshing...' : 'Refresh'}
        </Button>
      </div>

      {/* Search bar - only show when there are models */}
      {!isLoading && !isRefreshing && models.length > 0 && (
        <SearchInput
          placeholder="Search models..."
          value={searchQuery}
          onChange={e => {
            setSearchQuery(e.target.value)
            resetPage()
          }}
          onClear={() => {
            setSearchQuery('')
            resetPage()
          }}
        />
      )}

      {isLoading || isRefreshing ? (
        <div className="flex items-center justify-center py-8">
          <Loader size={24} className="border-primary" />
        </div>
      ) : models.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-8 flex items-center justify-center">
          <div className="text-sm text-muted-foreground text-center">
            No models found on disk. Download models below to get started.
          </div>
        </div>
      ) : filteredModels.length === 0 ? (
        <div className="rounded-xl border border-border bg-card p-8 flex items-center justify-center">
          <div className="text-sm text-muted-foreground text-center">
            {`No models found matching "${searchQuery}"`}
          </div>
        </div>
      ) : (
        <>
          <div className="min-h-[280px]">
            <AnimatePresence mode="wait">
              <motion.div
                key={effectivePage}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3 items-start"
              >
                {paginatedModels.map(m => (
                  <DeviceModelCard
                    key={m.id}
                    model={m}
                    onUse={() => onUse(m)}
                    onDelete={() => onDelete(m)}
                    isInUse={isModelInUse(m.id)}
                  />
                ))}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Pagination controls */}
          {filteredModels.length > ITEMS_PER_PAGE && (
            <div className="flex items-center justify-center gap-2 mt-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => setCurrentPage(p => Math.max(0, p - 1))}
                disabled={effectivePage === 0}
                className="flex items-center gap-1"
              >
                <FontIcon type="chevron-down" className="w-4 h-4 rotate-90" />
                Previous
              </Button>
              <span className="text-sm text-muted-foreground px-2">
                Page {effectivePage + 1} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  setCurrentPage(p => Math.min(totalPages - 1, p + 1))
                }
                disabled={effectivePage >= totalPages - 1}
                className="flex items-center gap-1"
              >
                Next
                <FontIcon type="chevron-down" className="w-4 h-4 -rotate-90" />
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  )
}

export type { DeviceModel }
