import { useRef, useEffect } from 'react'
import { useDevTools } from '../../contexts/DevToolsContext'
import DevToolsCollapsedBar from './DevToolsCollapsedBar'
import DevToolsExpandedPanel from './DevToolsExpandedPanel'

export default function DevToolsDrawer() {
  const {
    selectedRequest,
    isExpanded,
    activeTab,
    setIsExpanded,
    setActiveTab,
  } = useDevTools()

  const panelRef = useRef<HTMLDivElement>(null)

  // Click-outside detection to collapse the drawer
  useEffect(() => {
    if (!isExpanded) return

    const handleClickOutside = (event: MouseEvent) => {
      // Check if click is on the overlay (not the panel)
      const target = event.target as HTMLElement
      if (target.classList.contains('devtools-overlay')) {
        setIsExpanded(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [isExpanded, setIsExpanded])

  // Handle escape key to close
  useEffect(() => {
    if (!isExpanded) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setIsExpanded(false)
      }
    }

    document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [isExpanded, setIsExpanded])

  return (
    <>
      {/* Overlay when expanded - covers only the Test page content area */}
      {isExpanded && (
        <div
          className="devtools-overlay absolute inset-0 z-40 bg-black/60 animate-in fade-in-0 duration-200"
          aria-hidden="true"
        />
      )}

      {/* Drawer container - absolute positioned within the Test page content */}
      <div
        ref={panelRef}
        className="absolute bottom-0 left-0 right-0 z-50"
      >
        {isExpanded ? (
          <DevToolsExpandedPanel
            request={selectedRequest}
            activeTab={activeTab}
            onTabChange={setActiveTab}
            onClose={() => setIsExpanded(false)}
          />
        ) : (
          <DevToolsCollapsedBar
            request={selectedRequest}
            onClick={() => setIsExpanded(true)}
          />
        )}
      </div>
    </>
  )
}
