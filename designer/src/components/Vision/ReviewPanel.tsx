import { useState, useMemo } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { cn } from '@/lib/utils'
import { Check, Pencil, X, ArrowRight } from 'lucide-react'
import { usePendingReviews, useReviewDecision } from '../../hooks/useVision'
import { BoundingBoxCanvas } from './BoundingBoxCanvas'
import { PanelIntro } from './PanelIntro'
import type { ReviewDecision, ReviewItem, Detection } from '../../types/vision'

export function ReviewPanel() {
  const [page] = useState(1)
  const pageSize = 50 // fetch more for sequential review

  const { data, isLoading, error, refetch } = usePendingReviews(page, pageSize)
  const reviewMutation = useReviewDecision()

  const [currentIndex, setCurrentIndex] = useState(0)
  const [editMode, setEditMode] = useState(false)
  const [correctedClass, setCorrectedClass] = useState('')

  // Sort by lowest confidence first
  const items = useMemo(() => {
    const raw = data?.items ?? []
    return [...raw].sort((a, b) => a.detection.confidence - b.detection.confidence)
  }, [data?.items])

  const total = data?.total ?? 0
  const currentItem = items[currentIndex] as ReviewItem | undefined

  const handleDecision = (decision: ReviewDecision) => {
    if (!currentItem) return
    reviewMutation.mutate(
      { id: currentItem.id, decision, corrected_class: decision === 'wrong' && correctedClass ? correctedClass : undefined },
      {
        onSuccess: () => {
          setCorrectedClass('')
          setEditMode(false)
          if (currentIndex < items.length - 1) {
            setCurrentIndex(i => i + 1)
          } else {
            // Refetch next batch
            refetch()
            setCurrentIndex(0)
          }
        },
      }
    )
  }

  const handleSkip = () => {
    setEditMode(false)
    setCorrectedClass('')
    if (currentIndex < items.length - 1) {
      setCurrentIndex(i => i + 1)
    } else {
      refetch()
      setCurrentIndex(0)
    }
  }

  // Loading state
  if (isLoading) {
    return (
      <div className="flex flex-col gap-6">
        <PanelIntro title="Review">
          Review low-confidence detections flagged by the system. Correct or confirm classifications to improve model accuracy over time.
        </PanelIntro>
        <div className="flex flex-col items-center justify-center py-16">
          <div className="w-full max-w-lg aspect-video bg-secondary/50 animate-pulse rounded-lg" />
          <div className="h-8 bg-secondary/50 animate-pulse rounded w-48 mt-4" />
        </div>
      </div>
    )
  }

  // Empty state — shared between no-error and error-with-no-items
  if (items.length === 0) {
    return (
      <div className="flex flex-col gap-4">
        <PanelIntro title="Review">
          Correct the model's mistakes to make it smarter. When streaming flags a detection it isn't sure about, it shows up here for you to confirm, reject, or re-label.
        </PanelIntro>

        {error && (
          <div className="rounded-md border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-700">
            Could not reach the review queue — this feature requires the streaming backend to persist low-confidence detections.
          </div>
        )}

        {/* Visual flow diagram */}
        <div className="rounded-lg border border-border p-6">
          <p className="text-sm font-medium text-foreground mb-4">How the learning loop works</p>
          <div className="flex items-center gap-2 flex-wrap text-xs mb-6">
            <span className="px-3 py-1.5 rounded-md bg-blue-500/10 border border-blue-500/20 text-blue-700 font-medium">Stream</span>
            <span className="text-muted-foreground">→</span>
            <span className="px-3 py-1.5 rounded-md bg-yellow-500/10 border border-yellow-500/20 text-yellow-700 font-medium">Low-confidence flagged</span>
            <span className="text-muted-foreground">→</span>
            <span className="px-3 py-1.5 rounded-md bg-primary/10 border border-primary/20 text-primary font-medium">You review here</span>
            <span className="text-muted-foreground">→</span>
            <span className="px-3 py-1.5 rounded-md bg-green-500/10 border border-green-500/20 text-green-700 font-medium">Retrain with corrections</span>
          </div>

          <div className="space-y-4 text-sm text-muted-foreground">
            <div className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center mt-0.5">1</span>
              <div>
                <span className="text-foreground font-medium">Start a stream</span> in the Stream tab. As the model detects objects, anything below the confidence threshold gets saved for review.
              </div>
            </div>
            <div className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center mt-0.5">2</span>
              <div>
                <span className="text-foreground font-medium">Items appear here</span> — you'll see the image, what the model thinks it detected, and a confidence score. The least confident items come first.
              </div>
            </div>
            <div className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center mt-0.5">3</span>
              <div>
                <span className="text-foreground font-medium">You decide</span> — <span className="text-green-600">Accept</span> if the model got it right, <span className="text-red-600">Reject</span> if it's wrong, or <span className="text-foreground">Edit</span> to provide the correct label (e.g. "that's a van, not a truck").
              </div>
            </div>
            <div className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center mt-0.5">4</span>
              <div>
                <span className="text-foreground font-medium">Retrain</span> — your corrections become labeled training data. Go to the Train tab to fine-tune the model with this data, making it more accurate for your use case.
              </div>
            </div>
          </div>
        </div>

        <div className={cn(
          'rounded-lg border-2 border-dashed border-border py-8',
          'flex flex-col items-center justify-center text-center gap-2'
        )}>
          <p className="text-sm font-medium text-muted-foreground">No items to review yet</p>
          <p className="text-xs text-muted-foreground max-w-sm">
            Start a streaming session and detections below the confidence threshold will appear here automatically.
          </p>
        </div>
      </div>
    )
  }

  if (!currentItem) return null

  const reviewed = currentIndex
  const detection: Detection = {
    ...currentItem.detection,
    box: currentItem.detection.box ?? { x1: 0, y1: 0, x2: 0, y2: 0 },
  }
  const hasBox = detection.box.x2 > 0 || detection.box.y2 > 0

  return (
    <div className="flex flex-col gap-4">
      <PanelIntro title="Review">
        Review low-confidence detections flagged by the system. Sorted by lowest confidence first.
      </PanelIntro>

      {error && items.length > 0 && (
        <div className="rounded-md border border-yellow-500/30 bg-yellow-500/10 px-3 py-2 text-sm text-yellow-700">
          Failed to refresh review items. Showing cached data.
          <Button variant="link" size="sm" className="ml-2 h-auto p-0 text-yellow-700 underline" onClick={() => refetch()}>Retry</Button>
        </div>
      )}

      {/* Progress bar */}
      <div className="flex flex-col gap-1">
        <div className="flex items-center justify-between text-sm">
          <span className="text-muted-foreground">
            {reviewed} of {total} reviewed
          </span>
          <span className="text-muted-foreground text-xs">
            {total > 0 ? Math.round((reviewed / total) * 100) : 0}%
          </span>
        </div>
        <div className="w-full h-2 bg-secondary rounded-full overflow-hidden">
          <div
            className="h-full bg-primary rounded-full transition-all"
            style={{ width: total > 0 ? `${(reviewed / total) * 100}%` : '0%' }}
          />
        </div>
      </div>

      {/* Image with overlay */}
      <div className="flex flex-col items-center">
        {hasBox ? (
          <BoundingBoxCanvas
            imageSrc={`data:image/jpeg;base64,${currentItem.image}`}
            detections={[detection]}
            className="rounded-lg border border-border max-h-[500px]"
          />
        ) : (
          <img
            src={`data:image/jpeg;base64,${currentItem.image}`}
            alt={currentItem.detection.class_name}
            className="max-w-full max-h-[500px] rounded-lg border border-border object-contain"
          />
        )}
      </div>

      {/* Detection info */}
      <div className="flex items-center justify-center gap-3">
        <span className="text-lg font-medium">{currentItem.detection.class_name}</span>
        <Badge
          variant="secondary"
          className={cn(
            'text-sm',
            currentItem.detection.confidence >= 0.5 ? '' : 'bg-red-500/15 text-red-700 border-red-500/30'
          )}
        >
          {(currentItem.detection.confidence * 100).toFixed(1)}%
        </Badge>
        <span className="text-xs text-muted-foreground">Model: {currentItem.model}</span>
      </div>

      {/* Edit correction input */}
      {editMode && (
        <div className="flex gap-2 max-w-md mx-auto w-full">
          <Input
            value={correctedClass}
            onChange={e => setCorrectedClass(e.target.value)}
            placeholder="Enter correct class name..."
            autoFocus
          />
          <Button
            size="sm"
            onClick={() => handleDecision('wrong')}
            disabled={!correctedClass || reviewMutation.isPending}
          >
            Submit
          </Button>
          <Button
            size="sm"
            variant="outline"
            onClick={() => { setEditMode(false); setCorrectedClass('') }}
          >
            Cancel
          </Button>
        </div>
      )}

      {/* Action buttons */}
      {!editMode && (
        <div className="flex items-center justify-center gap-3">
          <Button
            variant="outline"
            className="gap-2 text-green-600 border-green-600/30 hover:bg-green-600/10"
            onClick={() => handleDecision('correct')}
            disabled={reviewMutation.isPending}
          >
            <Check className="w-4 h-4" />
            Accept
          </Button>
          <Button
            variant="outline"
            className="gap-2"
            onClick={() => setEditMode(true)}
            disabled={reviewMutation.isPending}
          >
            <Pencil className="w-4 h-4" />
            Edit
          </Button>
          <Button
            variant="outline"
            className="gap-2 text-red-600 border-red-600/30 hover:bg-red-600/10"
            onClick={() => handleDecision('wrong')}
            disabled={reviewMutation.isPending}
          >
            <X className="w-4 h-4" />
            Reject
          </Button>
          <Button
            variant="outline"
            className="gap-2"
            onClick={handleSkip}
            disabled={reviewMutation.isPending}
          >
            <ArrowRight className="w-4 h-4" />
            Skip
          </Button>
        </div>
      )}
    </div>
  )
}
