interface PIPThumbnailProps {
  type: 'map' | 'feed'
  onClick: () => void
  droneName?: string
}

export function PIPThumbnail({ type, onClick, droneName }: PIPThumbnailProps) {
  return (
    <button
      onClick={onClick}
      className="w-32 h-24 rounded-lg overflow-hidden border-2 border-white/30 hover:border-white/60 transition-all hover:scale-105 bg-surface-overlay shadow-lg"
    >
      {type === 'map' ? (
        <div className="w-full h-full bg-surface-raised flex items-center justify-center relative">
          {/* Simplified map representation */}
          <svg className="w-full h-full opacity-30" viewBox="0 0 100 100">
            <rect fill="#1a1a1a" width="100" height="100" />
            <path d="M10 30 L90 30 M10 50 L90 50 M10 70 L90 70" stroke="#333" strokeWidth="0.5" />
            <path d="M30 10 L30 90 M50 10 L50 90 M70 10 L70 90" stroke="#333" strokeWidth="0.5" />
          </svg>
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="text-xs text-gray-400">Map View</span>
          </div>
          <div className="absolute top-1 right-1 bg-black/60 px-1 text-[10px] text-white rounded">
            TAP
          </div>
        </div>
      ) : (
        <div className="w-full h-full bg-surface-base flex items-center justify-center relative">
          {/* Simplified feed representation */}
          <svg className="w-8 h-8 text-gray-600" fill="currentColor" viewBox="0 0 24 24">
            <path d="M17 10.5V7c0-.55-.45-1-1-1H4c-.55 0-1 .45-1 1v10c0 .55.45 1 1 1h12c.55 0 1-.45 1-1v-3.5l4 4v-11l-4 4z"/>
          </svg>
          {droneName && (
            <div className="absolute bottom-1 left-1 bg-black/60 px-1 text-[10px] text-white rounded">
              {droneName}
            </div>
          )}
          <div className="absolute top-1 right-1 bg-black/60 px-1 text-[10px] text-white rounded">
            TAP
          </div>
        </div>
      )}
    </button>
  )
}
