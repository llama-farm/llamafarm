import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { usePendingReviews, useReviewDecision } from '../../hooks/useVision'
import type { ReviewDecision } from '../../types/vision'

export function ReviewPanel() {
  const [page, setPage] = useState(1)
  const pageSize = 10

  const { data, isLoading, error } = usePendingReviews(page, pageSize)
  const reviewMutation = useReviewDecision()

  const handleDecision = (id: string, decision: ReviewDecision, correctedClass?: string) => {
    reviewMutation.mutate({ id, decision, corrected_class: correctedClass })
  }

  if (isLoading) {
    return <div className="text-sm text-muted-foreground py-8 text-center">Loading review items...</div>
  }

  if (error) {
    return (
      <div className="text-sm text-destructive py-8 text-center">
        Failed to load reviews: {(error as Error).message}
      </div>
    )
  }

  const items = data?.items ?? []
  const total = data?.total ?? 0
  const totalPages = Math.ceil(total / pageSize)

  if (items.length === 0) {
    return (
      <div className="text-center py-12">
        <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-green-500/15 border border-green-500/30">
          <span className="text-green-600 text-xl">✓</span>
        </div>
        <p className="text-sm text-muted-foreground">No pending reviews. All caught up!</p>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
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
