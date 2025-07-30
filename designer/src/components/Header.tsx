import React from 'react'
import FontIcon from '../common/FontIcon'

function Header() {
  return (
    <header className="fixed top-0 left-0 z-50 w-full bg-[#040C1D] border-b border-blue-400/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-12">
          {/* Logo */}
          <div className="font-serif text-white text-base font-medium">
            🦙 LlaMaFarm
          </div>

          {/* Right side controls */}
          <div className="flex items-center gap-3">
            {/* Theme toggle */}
            <div className="flex rounded-lg border border-blue-400/50 overflow-hidden">
              <button className="w-8 h-7 bg-[#263052] flex items-center justify-center text-[#85B1FF] hover:bg-blue-700 transition-colors">
                <FontIcon type="sun" className="w-4 h-4" />
              </button>
              <button className="w-8 h-7 flex items-center justify-center text-white hover:bg-blue-800/50 transition-colors">
                <FontIcon type="moon-filled" className="w-4 h-4" />
              </button>
            </div>

            {/* User avatar placeholder */}
            <FontIcon type="user-avatar" className="w-6 h-6 text-white" />
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
