/**
 * HF Dataset Finder - inline component for onboarding wizard
 * Appears nested under 'need-data' option in DataStatusSelector
 */

import { useState, useCallback, useEffect } from 'react'
import { cn } from '@/lib/utils'
import { Search, Loader2, ExternalLink, Check, Download, ArrowUpRight, AlertCircle } from 'lucide-react'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { useHFDatasetSearch } from '../../hooks/useHFDatasets'
import { getDatasetConfigs, validateDatasetAccess } from '../../api/huggingface'
import type { HFDatasetSearchResult, SelectedHFDataset } from '../../types/huggingface'

interface HFDatasetFinderProps {
  onSelectDataset: (dataset: SelectedHFDataset | null) => void
  selectedDataset: SelectedHFDataset | null
  className?: string
}

export function HFDatasetFinder({
  onSelectDataset,
  selectedDataset,
  className,
}: HFDatasetFinderProps) {
  const [searchQuery, setSearchQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  const [loadingConfigFor, setLoadingConfigFor] = useState<string | null>(null)
  const [validationError, setValidationError] = useState<string | null>(null)

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedQuery(searchQuery), 300)
    return () => clearTimeout(timer)
  }, [searchQuery])

  const { data: searchResults, isLoading, error } = useHFDatasetSearch(debouncedQuery)

  const handleSelect = useCallback(
    async (dataset: HFDatasetSearchResult) => {
      // Clear any previous validation error
      setValidationError(null)
      setLoadingConfigFor(dataset.id)

      try {
        // First, get the actual configs for this dataset
        const configs = await getDatasetConfigs(dataset.id)
        const firstConfig = configs[0]
        // Prefer 'train' split if available, otherwise first split
        const split = firstConfig.splits.includes('train')
          ? 'train'
          : firstConfig.splits[0] || 'train'

        // Validate that we can actually access this dataset
        const validation = await validateDatasetAccess(dataset.id, firstConfig.config, split)

        if (!validation.valid) {
          setValidationError(validation.error || 'This dataset cannot be imported.')
          setLoadingConfigFor(null)
          return
        }

        onSelectDataset({
          id: dataset.id,
          name: dataset.cardData?.pretty_name || dataset.id.split('/').pop() || dataset.id,
          rowCount: 100, // HF datasets-server API limit is 100 rows per request
          config: firstConfig.config,
          split,
        })
      } catch (err) {
        console.error('Failed to get dataset configs:', err)
        // Try fallback with validation
        const validation = await validateDatasetAccess(dataset.id, 'default', 'train')

        if (!validation.valid) {
          setValidationError(validation.error || 'This dataset cannot be imported.')
          setLoadingConfigFor(null)
          return
        }

        onSelectDataset({
          id: dataset.id,
          name: dataset.cardData?.pretty_name || dataset.id.split('/').pop() || dataset.id,
          rowCount: 100, // HF datasets-server API limit is 100 rows per request
          config: 'default',
          split: 'train',
        })
      } finally {
        setLoadingConfigFor(null)
      }
    },
    [onSelectDataset]
  )

  return (
    <div className={cn('space-y-3', className)}>
      {/* Search input */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search datasets (e.g., customer support, FAQ, reviews)"
          value={searchQuery}
          onChange={e => setSearchQuery(e.target.value)}
          className="pl-10 h-10"
        />
      </div>

      {/* Results */}
      <div className="space-y-2 max-h-[240px] overflow-y-auto">
        {isLoading && debouncedQuery && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">Searching...</span>
          </div>
        )}

        {error && (
          <p className="text-sm text-destructive text-center py-4">
            Failed to search datasets. Please try again.
          </p>
        )}

        {validationError && (
          <div className="p-3 bg-destructive/10 border border-destructive/30 rounded-lg flex items-start gap-2">
            <AlertCircle className="h-4 w-4 text-destructive flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-destructive">{validationError}</p>
              <p className="text-xs text-muted-foreground mt-1">
                Try selecting a different dataset with a smaller size.
              </p>
            </div>
          </div>
        )}

        {!isLoading && !error && searchResults?.length === 0 && debouncedQuery && (
          <p className="text-sm text-muted-foreground text-center py-4">
            No datasets found. Try a different search term.
          </p>
        )}

        {!isLoading &&
          !error &&
          searchResults?.map(dataset => (
            <DatasetCard
              key={dataset.id}
              dataset={dataset}
              isSelected={selectedDataset?.id === dataset.id}
              isLoadingConfig={loadingConfigFor === dataset.id}
              onSelect={() => handleSelect(dataset)}
            />
          ))}
      </div>

      {/* Selected dataset indicator */}
      {selectedDataset && (
        <div className="p-3 bg-primary/10 border border-primary/30 rounded-lg flex items-center gap-2">
          <Check className="h-4 w-4 text-primary flex-shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium truncate">{selectedDataset.name}</p>
            <p className="text-xs text-muted-foreground">
              Will import {selectedDataset.rowCount.toLocaleString()} rows when you finish setup
            </p>
          </div>
        </div>
      )}

      {/* Explore on Hugging Face link */}
      <div className="pt-2 border-t border-border/50">
        <a
          href="https://huggingface.co/datasets"
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <span className="text-base">🤗</span>
          <span>Explore datasets on Hugging Face</span>
          <ArrowUpRight className="h-3.5 w-3.5" />
        </a>
        <p className="text-xs text-muted-foreground/70 mt-1 pl-6">
          Browse and copy dataset names to search above
        </p>
      </div>
    </div>
  )
}

/** Size categories that are safe to import */
const SAFE_SIZES = ['n<1K', '1K<n<10K', '10K<n<100K', '100K<n<1M']

/** Individual dataset card in search results */
function DatasetCard({
  dataset,
  isSelected,
  isLoadingConfig,
  onSelect,
}: {
  dataset: HFDatasetSearchResult
  isSelected: boolean
  isLoadingConfig: boolean
  onSelect: () => void
}) {
  const displayName = dataset.cardData?.pretty_name || dataset.id.split('/').pop() || dataset.id
  const sizeCategory = dataset.cardData?.size_categories?.[0]
  const isSafeSize = sizeCategory ? SAFE_SIZES.includes(sizeCategory) : false
  const isLargeSize = sizeCategory && !isSafeSize

  return (
    <div
      className={cn(
        'p-3 rounded-lg border transition-all duration-200',
        isSelected
          ? 'border-primary bg-primary/5 dark:bg-primary/10'
          : 'border-border bg-card/50 hover:bg-white dark:hover:bg-card hover:border-primary/40'
      )}
    >
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <span className="font-medium text-sm text-foreground truncate">{displayName}</span>
          </div>
          {dataset.description && (
            <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {dataset.description}
            </p>
          )}
          <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
            <span className="flex items-center gap-1">
              <Download className="h-3 w-3" />
              {dataset.downloads.toLocaleString()}
            </span>
            {sizeCategory && (
              <span
                className={cn(
                  'px-1.5 py-0.5 rounded',
                  isSafeSize && 'bg-green-500/10 text-green-700 dark:text-green-400',
                  isLargeSize && 'bg-amber-500/10 text-amber-700 dark:text-amber-400'
                )}
              >
                {sizeCategory}
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-col gap-1.5">
          <Button
            size="sm"
            variant="ghost"
            className="h-7 px-2 text-xs"
            asChild
          >
            <a
              href={`https://huggingface.co/datasets/${dataset.id}`}
              target="_blank"
              rel="noopener noreferrer"
              onClick={e => e.stopPropagation()}
            >
              <ExternalLink className="h-3 w-3 mr-1" />
              View
            </a>
          </Button>
          <Button
            size="sm"
            variant={isSelected ? 'default' : 'outline'}
            className="h-7 px-2 text-xs"
            disabled={isLoadingConfig}
            onClick={e => {
              e.stopPropagation()
              onSelect()
            }}
          >
            {isLoadingConfig ? (
              <>
                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                Loading...
              </>
            ) : isSelected ? (
              <>
                <Check className="h-3 w-3 mr-1" />
                Selected
              </>
            ) : (
              'Use this'
            )}
          </Button>
        </div>
      </div>
    </div>
  )
}
