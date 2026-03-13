interface CommsLostBannerProps {
  connected: boolean
  restored: boolean
  onDismissRestored: () => void
}

export function CommsLostBanner({ connected, restored, onDismissRestored }: CommsLostBannerProps) {
  if (connected && !restored) return null

  if (!connected) {
    return (
      <div className="absolute top-12 left-0 right-0 z-[3000] mx-4">
        <div className="flex items-center gap-3 px-4 py-3 rounded-lg bg-[#3d1c1c] border border-[#5a2a2a]">
          <div className="w-3 h-3 rounded-full bg-status-critical animate-pulse flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-[#d4a0a0]">COMMS LOST</p>
            <p className="text-xs text-[#8a6060]">Drone operating autonomously — data queued</p>
          </div>
        </div>
      </div>
    )
  }

  if (restored) {
    return (
      <div className="absolute top-12 left-0 right-0 z-[3000] mx-4 animate-fade-in">
        <button
          onClick={onDismissRestored}
          className="w-full flex items-center gap-3 px-4 py-3 rounded-lg bg-[#1c3d2a] border border-[#2a5a3a] text-left"
        >
          <div className="w-3 h-3 rounded-full bg-status-good flex-shrink-0" />
          <div>
            <p className="text-sm font-medium text-[#a0d4b0]">COMMS RESTORED</p>
            <p className="text-xs text-[#608a6a]">Tap to dismiss</p>
          </div>
        </button>
      </div>
    )
  }

  return null
}
