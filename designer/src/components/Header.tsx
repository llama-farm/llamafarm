import { useEffect, useState } from 'react'
import FontIcon from '../common/FontIcon'
import { useLocation, useNavigate } from 'react-router-dom'

function Header() {
  const [isBuilding, setIsBuilding] = useState(false)
  const navigate = useNavigate()
  const isSelected = useLocation().pathname.split('/')[2]

  useEffect(() => {
    if (window.location.pathname !== '/') {
      setIsBuilding(true)
    } else {
      setIsBuilding(false)
    }
  }, [window.location.pathname])

  return (
    <header className="fixed top-0 left-0 z-50 w-full bg-blue-700 border-b border-blue-400/30">
      <div className="w-full flex items-center h-12">
        <div
          className="w-1/4 pl-4 font-serif text-white text-base font-medium cursor-pointer"
          onClick={() => navigate('/')}
        >
          🦙 LlaMaFarm
        </div>

        <div
          className={`flex items-center w-3/4 justify-end pr-4 ${
            isBuilding ? 'justify-between' : 'justify-end'
          }`}
        >
          {isBuilding && (
            <div className="flex items-center gap-4 w-2/3">
              <button
                className={`w-full flex items-center justify-center gap-2 hover:bg-blue-600 transition-colors rounded-lg p-2 ${
                  isSelected === 'dashboard' ? 'bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/dashboard')}
              >
                <FontIcon type="dashboard" className="w-6 h-6 text-white" />
                <span className="text-white">Dashboard</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 hover:bg-blue-600 transition-colors rounded-lg p-2 ${
                  isSelected === 'data' ? 'bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/data')}
              >
                <FontIcon type="data" className="w-6 h-6 text-white" />
                <span className="text-white">Data</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 hover:bg-blue-600 transition-colors rounded-lg p-2 ${
                  isSelected === 'prompt' ? 'bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/prompt')}
              >
                <FontIcon type="prompt" className="w-6 h-6 text-white" />
                <span className="text-white">Prompt</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 hover:bg-blue-600 transition-colors rounded-lg p-2 ${
                  isSelected === 'test' ? 'bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/test')}
              >
                <FontIcon type="test" className="w-6 h-6 text-white" />
                <span className="text-white">Test</span>
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 justify-end">
            <div className="flex rounded-lg border border-blue-400/50 overflow-hidden">
              <button className="w-8 h-7 bg-blue-600 flex items-center justify-center text-blue-100 hover:bg-blue-700 transition-colors">
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
