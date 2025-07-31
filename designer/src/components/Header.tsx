import { useState } from 'react'
import FontIcon from '../common/FontIcon'
import useHeader from '../hooks/useHeader.tsx'

function Header() {
  const [isBuilding, setIsBuilding] = useState(true)
  const { setBuildType } = useHeader()

  return (
    <header className="fixed top-0 left-0 z-50 w-full bg-[#040C1D] border-b border-blue-400/30 max-w-7xl">
      <div className="flex items-center h-12">
        <div className="w-1/4 pl-4 font-serif text-white text-base font-medium ">
          🦙 LlaMaFarm
        </div>

        <div
          className={`flex items-center w-3/4 justify-end pr-4 ${
            isBuilding ? 'justify-between' : 'justify-end'
          }`}
        >
          {isBuilding && (
            <div className="flex items-center gap-4">
              <button
                className="flex items-center gap-2 hover:bg-[#263052] transition-colors rounded-lg p-2"
                onClick={() => setBuildType('dashboard')}
              >
                <FontIcon type="dashboard" className="w-6 h-6 text-white" />
                <span className="text-white">Dashboard</span>
              </button>
              <button
                className="flex items-center gap-2 hover:bg-[#263052] transition-colors rounded-lg p-2"
                onClick={() => setBuildType('data')}
              >
                <FontIcon type="data" className="w-6 h-6 text-white" />
                <span className="text-white">Data</span>
              </button>
              <button
                className="flex items-center gap-2 hover:bg-[#263052] transition-colors rounded-lg p-2"
                onClick={() => setBuildType('prompt')}
              >
                <FontIcon type="prompt" className="w-6 h-6 text-white" />
                <span className="text-white">Prompt</span>
              </button>
              <button
                className="flex items-center gap-2 hover:bg-[#263052] transition-colors rounded-lg p-2"
                onClick={() => setBuildType('test')}
              >
                <FontIcon type="test" className="w-6 h-6 text-white" />
                <span className="text-white">Test</span>
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 justify-end">
            <div className="flex rounded-lg border border-blue-400/50 overflow-hidden">
              <button className="w-8 h-7 bg-[#263052] flex items-center justify-center text-[#85B1FF] hover:bg-blue-700 transition-colors">
                <FontIcon type="sun" className="w-4 h-4" />
              </button>
              <button className="w-8 h-7 flex items-center justify-center text-white hover:bg-blue-800/50 transition-colors">
                <FontIcon type="moon-filled" className="w-4 h-4" />
              </button>
            </div>
            <FontIcon type="user-avatar" className="w-6 h-6 text-white" />
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
