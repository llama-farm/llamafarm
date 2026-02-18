import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { usePendingReviews, useReviewDecision } from '../../hooks/useVision'
import { PanelIntro } from './PanelIntro'
import FontIcon from '../../common/FontIcon'
import type { ReviewDecision } from '../../types/vision'

export function ReviewPanel() {
  const [page, setPage] = useState(1)
  const pageSize = 10

  const { data, isLoading, error, refetch } = usePendingReviews(page, pageSize)
  const reviewMutation = useReviewDecision()

  const handleDecision = (id: string, decision: ReviewDecision, correctedClass?: string) => {
    reviewMutation.mutate({ id, decision, corrected_class: correctedClass })
  }

  // Loading state with skeleton
  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <PanelIntro>
          Review low-confidence detections flagged by the system. Correct or confirm classifications to improve model accuracy over time.
        </PanelIntro>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1, 2, 3].map(i => (
            <div key={i} className="rounded-lg border border-border overflow-hidden">
              <div className="aspect-square bg-secondary/50 animate-pulse" />
              <div className="p-3 flex flex-col gap-2">
                <div className="h-4 bg-secondary/50 animate-pulse rounded w-2/3" />
                <div className="h-3 bg-secondary/50 animate-pulse rounded w-1/2" />
                <div className="h-8 bg-secondary/50 animate-pulse rounded mt-1" />
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  // Error state with retry
  if (error) {
    return (
      <div className="flex flex-col gap-6">
        <PanelIntro>
          Review low-confidence detections flagged by the system. Correct or confirm classifications to improve model accuracy over time.
        </PanelIntro>
        <div className="text-center py-12">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-destructive/15 border border-destructive/30">
            <FontIcon type="alert-triangle" className="w-6 h-6 text-destructive" />
          </div>
          <p className="text-sm font-medium mb-1">Unable to load review items</p>
          <p className="text-sm text-muted-foreground mb-4">
            The review service may not be available yet. This is normal if no detections have been flagged.
          </p>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            Retry
          </Button>
        </div>
      </div>
    )
  }

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.ceil(total / pageSize)

  // Empty state
  if (items.length === 0) {
    return (
      <div className="flex flex-col gap-6">
        <PanelIntro>
          Review low-confidence detections flagged by the system. Correct or confirm classifications to improve model accuracy over time.
        </PanelIntro>
        <div className="text-center py-12">
          <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-500/15 border border-green-500/30">
            <FontIcon type="checkmark-filled" className="w-6 h-6 text-green-600" />
          </div>
          <p className="text-sm font-medium mb-1">No items to review</p>
          <p className="text-sm text-muted-foreground">
            When the system flags low-confidence detections, they'll appear here for your review.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-6">
      <PanelIntro>
        Review low-confidence detections flagged by the system. Correct or confirm classifications to improve model accuracy over time.
      </PanelIntro>

      <div className="flex items-center justify-between">
        <p className="text-sm text-muted-foreground">{total} item{total !== 1 ? 's' : ''} pending review</p>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            disabled={page <= 1}
            onClick={() => setPage(p => p - 1)}
          >
            Previous
          </Button>
          <span className="text-sm text-muted-foreground">
            {page} / {totalPages}
          </span>
          <Button
            variant="outline"
            size="sm"
            disabled={page >= totalPages}
            onClick={() => setPage(p => p + 1)}
          >
            Next
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {items.map(item => (
          <ReviewCard
            key={item.id}
            item={item}
            onDecision={handleDecision}
            isSubmitting={reviewMutation.isPending}
          />
        ))}
      </div>
    </div>
  )
}

function ReviewCard({
  item,
  onDecision,
  isSubmitting,
}: {
  item: { id: string; image: string; detection: { class_name: string; confidence: number }; model: string; timestamp: string }
  onDecision: (id: string, decision: ReviewDecision, correctedClass?: string) => void
  isSubmitting: boolean
}) {
  const [correctedClass, setCorrectedClass] = useState('')

  return (
    <div className="rounded-lg border border-border overflow-hidden">
      <div className="aspect-square bg-secondary flex items-center justify-center">
        <img
          src={`data:image/jpeg;base64,${item.image}`}
          alt={item.detection.class_name}
          className="max-w-full max-h-full object-contain"
        />
      </div>
      <div className="p-3 flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <span className="text-sm font-medium">{item.detection.class_name}</span>
          <Badge variant="secondary">
            {(item.detection.confidence * 100).toFixed(1)}%
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground">Model: {item.model}</p>

        <Input
          value={correctedClass}
          onChange={e => setCorrectedClass(e.target.value)}
          placeholder="Correct class (if wrong)"
          className="text-sm"
        />

        <div className="flex gap-2">
          <Button
            size="sm"
            variant="outline"
            className="flex-1 text-green-600 border-green-600/30 hover:bg-green-600/10"
            onClick={() => onDecision(item.id, 'correct')}
            disabled={isSubmitting}
          >
            Correct
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="flex-1 text-red-600 border-red-600/30 hover:bg-red-600/10"
            onClick={() => onDecision(item.id, 'wrong', correctedClass || undefined)}
            disabled={isSubmitting}
          >
            Wrong
          </Button>
          <Button
            size="sm"
            variant="outline"
            className="flex-1"
            onClick={() => onDecision(item.id, 'skip')}
            disabled={isSubmitting}
          >
            Skip
          </Button>
        </div>
      </div>
    </div>
  )
}
