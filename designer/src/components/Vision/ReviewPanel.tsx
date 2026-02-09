import { useState, useEffect, useCallback } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { useToast } from '../ui/toast'
import FontIcon from '../../common/FontIcon'
import Loader from '../../common/Loader'

const RUNTIME_URL = ''

interface ReviewItem {
  id: string
  image_id: string
  image_data?: string  // base64 if available
  image_path?: string
  predicted_class: string
  confidence: number
  session_id?: string
  created_at: string
  box?: {
    x1: number
    y1: number
    x2: number
    y2: number
  }
}

export function ReviewPanel() {
  const { toast } = useToast()
  
  const [items, setItems] = useState<ReviewItem[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedItem, setSelectedItem] = useState<ReviewItem | null>(null)
  const [correctedLabel, setCorrectedLabel] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  
  // Fetch review queue
  const fetchReviewQueue = useCallback(async () => {
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/review-queue`)
      if (response.ok) {
        const data = await response.json()
        setItems(data.items || [])
      } else if (response.status === 404) {
        // Endpoint might not exist yet
        setItems([])
      }
    } catch (error) {
      console.error('Failed to fetch review queue:', error)
      setItems([])
    } finally {
      setIsLoading(false)
    }
  }, [])
  
  useEffect(() => {
    fetchReviewQueue()
    
    // Poll for updates
    const interval = setInterval(fetchReviewQueue, 10000)
    return () => clearInterval(interval)
  }, [fetchReviewQueue])
  
  // Select item for review
  const selectItem = (item: ReviewItem) => {
    setSelectedItem(item)
    setCorrectedLabel(item.predicted_class)
  }
  
  // Approve detection (label is correct)
  const approveItem = async (item: ReviewItem) => {
    setIsSubmitting(true)
    
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: item.image_id,
          corrected_class: item.predicted_class,  // Same as predicted = approve
          original_confidence: item.confidence,
          box: item.box,
          session_id: item.session_id,
        }),
      })
      
      if (!response.ok) throw new Error('Failed to approve')
      
      toast({ message: `Approved: ${item.predicted_class}` })
      
      // Remove from local list
      setItems(prev => prev.filter(i => i.id !== item.id))
      setSelectedItem(null)
      
    } catch (error) {
      toast({ message: 'Failed to approve', variant: 'destructive' })
    } finally {
      setIsSubmitting(false)
    }
  }
  
  // Submit correction
  const submitCorrection = async () => {
    if (!selectedItem || !correctedLabel.trim()) return
    
    setIsSubmitting(true)
    
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: selectedItem.image_id,
          corrected_class: correctedLabel.trim(),
          original_confidence: selectedItem.confidence,
          box: selectedItem.box,
          session_id: selectedItem.session_id,
        }),
      })
      
      if (!response.ok) throw new Error('Failed to submit correction')
      
      toast({ 
        message: correctedLabel === selectedItem.predicted_class 
          ? `Approved: ${correctedLabel}`
          : `Corrected: ${selectedItem.predicted_class} → ${correctedLabel}` 
      })
      
      // Remove from local list
      setItems(prev => prev.filter(i => i.id !== selectedItem.id))
      setSelectedItem(null)
      setCorrectedLabel('')
      
    } catch (error) {
      toast({ message: 'Failed to submit correction', variant: 'destructive' })
    } finally {
      setIsSubmitting(false)
    }
  }
  
  // Reject (remove from queue without adding to training)
  const rejectItem = async (item: ReviewItem) => {
    setIsSubmitting(true)
    try {
      const response = await fetch(`${RUNTIME_URL}/v1/vision/corrections`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: item.image_id,
          corrected_class: '__rejected__',
          original_confidence: item.confidence,
          session_id: item.session_id,
        }),
      })

      if (!response.ok) throw new Error('Failed to reject')

      setItems(prev => prev.filter(i => i.id !== item.id))
      if (selectedItem?.id === item.id) {
        setSelectedItem(null)
      }
      toast({ message: 'Removed from review queue' })
    } catch (error) {
      toast({ message: 'Failed to reject', variant: 'destructive' })
    } finally {
      setIsSubmitting(false)
    }
  }
  
  return (
    <div className="grid grid-cols-2 gap-6 h-full">
      {/* Queue List */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Review Queue</h3>
          <span className="text-sm text-muted-foreground">{items.length} items</span>
        </div>
        
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader className="w-8 h-8" />
          </div>
        ) : items.length === 0 ? (
          <div className="bg-card border rounded-lg p-8 text-center">
            <FontIcon type="checkmark-filled" className="w-12 h-12 mx-auto mb-4 text-green-500" />
            <h3 className="text-lg font-semibold mb-2">All Caught Up!</h3>
            <p className="text-muted-foreground">
              No uncertain detections need review. The cascade is working well.
            </p>
          </div>
        ) : (
          <div className="space-y-2 max-h-[600px] overflow-y-auto">
            {items.map(item => (
              <div 
                key={item.id}
                onClick={() => selectItem(item)}
                className={`bg-card border rounded-lg p-3 cursor-pointer transition-colors ${
                  selectedItem?.id === item.id ? 'border-primary' : 'hover:border-primary/50'
                }`}
              >
                <div className="flex items-center gap-3">
                  {/* Thumbnail */}
                  {item.image_data ? (
                    <img 
                      src={item.image_data} 
                      alt="" 
                      className="w-16 h-16 object-cover rounded"
                    />
                  ) : (
                    <div className="w-16 h-16 bg-muted rounded flex items-center justify-center">
                      <FontIcon type="eye" className="w-6 h-6 text-muted-foreground" />
                    </div>
                  )}
                  
                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <div className="font-medium truncate">{item.predicted_class}</div>
                    <div className="text-sm text-muted-foreground">
                      Confidence: {(item.confidence * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {new Date(item.created_at).toLocaleString()}
                    </div>
                  </div>
                  
                  {/* Quick actions */}
                  <div className="flex gap-1" onClick={e => e.stopPropagation()}>
                    <Button size="sm" variant="ghost" onClick={() => approveItem(item)} disabled={isSubmitting}>
                      <FontIcon type="checkmark-filled" className="w-4 h-4 text-green-500" />
                    </Button>
                    <Button size="sm" variant="ghost" onClick={() => rejectItem(item)}>
                      <FontIcon type="close" className="w-4 h-4 text-red-500" />
                    </Button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      
      {/* Review Detail */}
      <div className="bg-card border rounded-lg p-4">
        {selectedItem ? (
          <div className="space-y-4">
            <h3 className="text-lg font-semibold">Review Detection</h3>
            
            {/* Image */}
            <div className="aspect-video bg-muted rounded-lg overflow-hidden flex items-center justify-center">
              {selectedItem.image_data ? (
                <img 
                  src={selectedItem.image_data} 
                  alt="" 
                  className="max-w-full max-h-full object-contain"
                />
              ) : (
                <div className="text-center text-muted-foreground">
                  <FontIcon type="eye" className="w-12 h-12 mx-auto mb-2" />
                  <p>Image not available</p>
                </div>
              )}
            </div>
            
            {/* Prediction */}
            <div className="p-4 bg-muted rounded-lg">
              <div className="text-sm text-muted-foreground mb-1">Model predicted:</div>
              <div className="text-xl font-bold">{selectedItem.predicted_class}</div>
              <div className="text-sm text-muted-foreground">
                with {(selectedItem.confidence * 100).toFixed(1)}% confidence
              </div>
            </div>
            
            {/* Correction Input */}
            <div className="space-y-2">
              <Label>Correct Label (if wrong)</Label>
              <div className="flex gap-2">
                <Input
                  value={correctedLabel}
                  onChange={e => setCorrectedLabel(e.target.value)}
                  placeholder="Enter correct label..."
                  onKeyDown={e => e.key === 'Enter' && submitCorrection()}
                />
              </div>
            </div>
            
            {/* Actions */}
            <div className="flex gap-2 pt-4 border-t">
              <Button 
                onClick={() => approveItem(selectedItem)} 
                disabled={isSubmitting}
                className="flex-1"
              >
                <FontIcon type="checkmark-filled" className="w-4 h-4 mr-2" />
                Approve as "{selectedItem.predicted_class}"
              </Button>
              <Button 
                onClick={submitCorrection} 
                disabled={isSubmitting || !correctedLabel.trim() || correctedLabel === selectedItem.predicted_class}
                variant="secondary"
                className="flex-1"
              >
                <FontIcon type="edit" className="w-4 h-4 mr-2" />
                Correct to "{correctedLabel}"
              </Button>
            </div>
            
            <Button 
              onClick={() => rejectItem(selectedItem)} 
              variant="ghost" 
              className="w-full"
            >
              <FontIcon type="trashcan" className="w-4 h-4 mr-2" />
              Reject (bad image)
            </Button>
          </div>
        ) : (
          <div className="h-full flex items-center justify-center text-center">
            <div>
              <FontIcon type="eye" className="w-16 h-16 mx-auto mb-4 text-muted-foreground" />
              <h3 className="text-lg font-semibold mb-2">Select an Item</h3>
              <p className="text-muted-foreground">
                Click on an item from the queue to review it
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
