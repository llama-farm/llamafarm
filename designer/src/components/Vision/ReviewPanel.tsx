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
        <PanelIntro>
          Review low-confidence detections flagged by the system. Correct or confirm classifications to improve model accuracy over time.
        </PanelIntro>
        <div className="flex flex-col items-center justify-center py-16">
          <div className="w-full max-w-lg aspect-video bg-secondary/50 animate-pulse rounded-lg" />
          <div className="h-8 bg-secondary/50 animate-pulse rounded w-48 mt-4" />
        </div>
      </div>
    )
  }

  // Empty state (only show when no cached items AND no error, or no items at all)
  if (items.length === 0 && !error) {
    return (
      <div className="flex flex-col gap-4">
        <p className="text-sm text-muted-foreground">
          Review is the human-in-the-loop step that makes your models smarter over time.
        </p>
        <div className="py-8 max-w-lg mx-auto">
          <p className="text-sm font-medium text-foreground mb-4">Nothing to review yet</p>
          <p className="text-sm text-muted-foreground mb-4">
            Review items appear when streaming detects objects it isn't confident about. Here's how it works:
          </p>
          <ol className="text-sm text-muted-foreground space-y-3 list-none">
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">1</span>
              <span><span className="text-foreground font-medium">Start a stream</span> — connect a camera or video feed in the Stream tab and set a confidence threshold.</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">2</span>
              <span><span className="text-foreground font-medium">Uncertain detections get queued</span> — when the model detects something but isn't confident enough (e.g. 40% confidence vs your 70% threshold), it flags it for review instead of discarding it.</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">3</span>
              <span><span className="text-foreground font-medium">You confirm or correct</span> — accept ("yes, that's a truck"), reject ("no, that's a shadow"), or re-label ("that's actually a van"). These corrections become training data.</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">4</span>
              <span><span className="text-foreground font-medium">Retrain</span> — use your reviewed data to fine-tune the model. Each cycle makes it more accurate for your specific use case.</span>
            </li>
          </ol>
        </div>
      </div>
    )
  }

  // If error but no items at all, show the empty state with walkthrough
  if (error && items.length === 0) {
    return (
      <div className="flex flex-col gap-4">
        <p className="text-sm text-muted-foreground">
          Review is the human-in-the-loop step that makes your models smarter over time.
        </p>
        <div className="py-8 max-w-lg mx-auto">
          <p className="text-sm font-medium text-foreground mb-4">Nothing to review yet</p>
          <p className="text-sm text-muted-foreground mb-4">
            Review items appear when streaming detects objects it isn't confident about. Here's how it works:
          </p>
          <ol className="text-sm text-muted-foreground space-y-3 list-none">
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">1</span>
              <span><span className="text-foreground font-medium">Start a stream</span> — connect a camera or video feed in the Stream tab and set a confidence threshold.</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">2</span>
              <span><span className="text-foreground font-medium">Uncertain detections get queued</span> — when the model detects something but isn't confident enough, it flags it for review.</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">3</span>
              <span><span className="text-foreground font-medium">You confirm or correct</span> — accept, reject, or re-label. These corrections become training data.</span>
            </li>
            <li className="flex gap-3">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 text-primary text-xs font-medium flex items-center justify-center">4</span>
              <span><span className="text-foreground font-medium">Retrain</span> — use your reviewed data to fine-tune the model. Each cycle makes it more accurate.</span>
            </li>
          </ol>
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
      <PanelIntro>
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
