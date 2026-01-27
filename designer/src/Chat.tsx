import { useCallback, useEffect, useRef, useState } from 'react'
import Chatbox from './components/Chatbox/Chatbox'
import { Outlet, useLocation, useSearchParams } from 'react-router-dom'
import { ProjectUpgradeBanner } from './components/common/UpgradeBanners'
import { decodeMessageFromUrl } from './utils/homePageUtils'
// import { Button } from './components/ui/button'
import { useMobileView } from './contexts/MobileViewContext'

function Chat() {
  const [isPanelOpen, setIsPanelOpen] = useState<boolean>(true)
  const [initialMessage, setInitialMessage] = useState<string | null>(null)
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const prevPanelOpenRef = useRef<boolean>(true)
  const { isMobile, mobileView, setMobileView } = useMobileView()
  // Track whether we auto-collapsed due to mid-width constraint
  const autoCollapsedRef = useRef<boolean>(false)
  const chatPanelRef = useRef<HTMLDivElement | null>(null)
  const [chatWidthPx, setChatWidthPx] = useState<number | null>(null)
  const isDraggingRef = useRef<boolean>(false)
  const [isDragging, setIsDragging] = useState<boolean>(false)
  const dragRafRef = useRef<number | null>(null)
  const chatLeftRef = useRef<number>(0)
  const onDragFnRef = useRef<(e: MouseEvent) => void>(() => {})
  const stopDragFnRef = useRef<() => void>(() => {})

  // Handle initial message from URL parameters (from home page project creation)
  useEffect(() => {
    const initialMessageParam = searchParams.get('initialMessage')
    if (initialMessageParam) {
      const decodedMessage = decodeMessageFromUrl(initialMessageParam)
      if (decodedMessage) {
        setInitialMessage(decodedMessage)

        // Clear the URL parameter after handling to clean up the URL
        setSearchParams(params => {
          params.delete('initialMessage')
          return params
        })
      }
    }
  }, [searchParams, setSearchParams])

  // Close the designer chat panel when viewing the Test page, and restore previous state when leaving
  useEffect(() => {
    if (location.pathname.startsWith('/chat/test')) {
      // Save current state before closing
      prevPanelOpenRef.current = isPanelOpen
      setIsPanelOpen(false)
    } else {
      // Restore previous state when leaving /chat/test
      setIsPanelOpen(prevPanelOpenRef.current)
    }
  }, [location.pathname])

  // Close chat panel when a new project is selected (starts fresh)
  useEffect(() => {
    const handleProjectChange = () => {
      // Close the chat panel when switching projects
      setIsPanelOpen(false)
    }

    window.addEventListener('lf-active-project', handleProjectChange)
    return () => {
      window.removeEventListener('lf-active-project', handleProjectChange)
    }
  }, [])

  // Close chat panel when onboarding wizard starts or resets
  useEffect(() => {
    const handleOnboardingStart = () => {
      setIsPanelOpen(false)
    }

    window.addEventListener('lf-onboarding-started', handleOnboardingStart)
    window.addEventListener('lf-onboarding-reset', handleOnboardingStart)
    return () => {
      window.removeEventListener('lf-onboarding-started', handleOnboardingStart)
      window.removeEventListener('lf-onboarding-reset', handleOnboardingStart)
    }
  }, [])

  // Mobile detection is provided by context

  // Initialize default chat width (25% of viewport) with min/max bounds when panel opens on desktop
  useEffect(() => {
    if (isMobile || !isPanelOpen) return
    const minPx = 360
    const maxPx = 820
    const def = Math.round(window.innerWidth * 0.25)
    const clamped = Math.max(minPx, Math.min(maxPx, def))
    setChatWidthPx(prev => (prev == null ? clamped : prev))
  }, [isMobile, isPanelOpen])

  // Cleanup drag listeners and body styles on unmount
  useEffect(() => {
    return () => {
      if (dragRafRef.current != null) cancelAnimationFrame(dragRafRef.current)
      dragRafRef.current = null
      document.body.style.cursor = ''
      ;(document.body.style as any).userSelect = ''
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [])

  const handleMouseMove = useCallback((e: MouseEvent) => {
    onDragFnRef.current(e)
  }, [])

  const handleMouseUp = useCallback(() => {
    stopDragFnRef.current()
  }, [])

  useEffect(() => {
    onDragFnRef.current = (e: MouseEvent) => {
      if (!isDraggingRef.current || !chatPanelRef.current) return
      const minPx = 360
      const maxPx = 820
      const proposed = e.clientX - chatLeftRef.current
      const next = Math.max(minPx, Math.min(maxPx, proposed))
      if (dragRafRef.current != null) cancelAnimationFrame(dragRafRef.current)
      dragRafRef.current = requestAnimationFrame(() => {
        if (chatPanelRef.current) chatPanelRef.current.style.width = `${next}px`
      })
    }

    stopDragFnRef.current = () => {
      isDraggingRef.current = false
      setIsDragging(false)
      if (chatPanelRef.current) {
        const rect = chatPanelRef.current.getBoundingClientRect()
        const minPx = 360
        const maxPx = 820
        const finalW = Math.max(minPx, Math.min(maxPx, rect.width))
        setChatWidthPx(finalW)
      }
      if (dragRafRef.current != null) cancelAnimationFrame(dragRafRef.current)
      dragRafRef.current = null
      document.body.style.cursor = ''
      ;(document.body.style as any).userSelect = ''
      document.removeEventListener('mousemove', handleMouseMove)
      document.removeEventListener('mouseup', handleMouseUp)
    }
  }, [handleMouseMove, handleMouseUp])

  const startDrag: React.MouseEventHandler<HTMLDivElement> = useCallback(
    e => {
      e.preventDefault()
      if (isMobile || !isPanelOpen) return
      
      // Capture current width before setting isDragging
      if (chatPanelRef.current) {
        const currentWidth = chatPanelRef.current.getBoundingClientRect().width
        chatLeftRef.current = chatPanelRef.current.getBoundingClientRect().left
        
        // Set the width explicitly so it doesn't jump when isDragging becomes true
        if (chatWidthPx == null || Math.abs(currentWidth - chatWidthPx) > 1) {
          setChatWidthPx(currentWidth)
        }
      }
      
      isDraggingRef.current = true
      setIsDragging(true)
      document.body.style.cursor = 'col-resize'
      ;(document.body.style as any).userSelect = 'none'
      document.addEventListener('mousemove', handleMouseMove)
      document.addEventListener('mouseup', handleMouseUp)
    },
    [isMobile, isPanelOpen, handleMouseMove, handleMouseUp, chatWidthPx]
  )

  // Auto-collapse chat panel on mid-width screens to avoid over-squeezing project pane
  const wasOpenBeforeAutoCollapseRef = useRef<boolean>(true)
  useEffect(() => {
    const mql = window.matchMedia('(max-width: 1100px)')
    const onChange = (e: MediaQueryListEvent | MediaQueryList) => {
      if (isMobile) return
      const matches = 'matches' in e ? e.matches : (e as MediaQueryList).matches
      if (matches) {
        // Save the user's current preference before auto-collapsing
        wasOpenBeforeAutoCollapseRef.current = prevPanelOpenRef.current
        setIsPanelOpen(false)
        autoCollapsedRef.current = true
      } else {
        // Restore the prior preference when leaving the constrained width
        setIsPanelOpen(wasOpenBeforeAutoCollapseRef.current)
        autoCollapsedRef.current = false
      }
    }
    // Run once on mount so desktops starting ≤1100px auto-collapse
    onChange(mql)
    if (typeof (mql as any).addEventListener === 'function') {
      mql.addEventListener('change', onChange as EventListener)
      return () => mql.removeEventListener('change', onChange as EventListener)
    } else if (typeof (mql as any).addListener === 'function') {
      ;(mql as any).addListener(onChange)
      return () => (mql as any).removeListener(onChange)
    }
    return
  }, [isMobile])

  // Track user's last explicit preference for restoration
  useEffect(() => {
    prevPanelOpenRef.current = isPanelOpen
  }, [isPanelOpen])

  // On mobile entry/exit:
  // - default to Project on narrow if chat was auto-collapsed
  // - preserve header-driven selection via context otherwise
  useEffect(() => {
    if (isMobile) {
      if (autoCollapsedRef.current) {
        setMobileView('project')
      }
    } else if (mobileView === 'chat') setIsPanelOpen(true)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isMobile])

  // No route-based override; respect context-driven selection

  // Header uses context to set view directly

  // No need to emit events; header reads from context

  const effectivePanelOpen = isMobile ? true : isPanelOpen

  return (
    <div
      className="w-full h-full flex transition-colors bg-gray-200 dark:bg-blue-800 pt-12 md:pt-12"
      style={{
        paddingTop:
          isMobile &&
          location.pathname.startsWith('/chat') &&
          mobileView === 'project'
            ? ('6rem' as any)
            : undefined,
      }}
    >
      <ProjectUpgradeBanner />
      <div
        ref={chatPanelRef}
        className={`h-full ${isDragging ? 'transition-none' : 'transition-all duration-300'} relative ${
          isMobile
            ? mobileView === 'chat'
              ? 'w-full'
              : 'hidden'
            : effectivePanelOpen
              ? 'min-w-[360px]'
              : 'w-[47px]'
        }`}
        style={
          !isMobile && effectivePanelOpen
            ? {
                width: chatWidthPx != null ? `${chatWidthPx}px` : undefined,
                maxWidth: '820px',
              }
            : undefined
        }
      >
        <Chatbox
          isPanelOpen={effectivePanelOpen}
          setIsPanelOpen={setIsPanelOpen}
          initialMessage={initialMessage}
        />
        {/* Desktop drag handle for resizing chat panel */}
        {!isMobile && effectivePanelOpen ? (
          <div
            onMouseDown={startDrag}
            className="hidden md:flex items-center justify-center absolute right-0 bottom-0 top-10 w-3 sm:w-4 cursor-col-resize"
            role="separator"
            aria-label="Resize chat panel"
            title="Drag to resize chat"
          >
            <div className="w-[2px] sm:w-[3px] h-12 rounded-full bg-border hover:bg-primary/60 transition-colors" />
          </div>
        ) : null}
      </div>

      <div
        className={`h-full ${
          isMobile
            ? mobileView === 'project'
              ? 'w-full'
              : 'hidden'
            : effectivePanelOpen
              ? 'flex-1'
              : 'flex-1'
        } text-gray-900 dark:text-white px-6 pt-6 overflow-auto min-h-0`}
      >
        <Outlet />
      </div>

      {/* Mobile switcher moved to header */}
    </div>
  )
}

export default Chat
