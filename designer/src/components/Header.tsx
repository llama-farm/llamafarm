import { useEffect, useState } from 'react'
import FontIcon from '../common/FontIcon'
import { useLocation, useNavigate } from 'react-router-dom'
import { useTheme } from '../contexts/ThemeContext'

function Header() {
  const [isBuilding, setIsBuilding] = useState(false)
  const navigate = useNavigate()
  const isSelected = useLocation().pathname.split('/')[2]
  const { theme, setTheme } = useTheme()

  useEffect(() => {
    if (window.location.pathname !== '/') {
      setIsBuilding(true)
    } else {
      setIsBuilding(false)
    }
  }, [window.location.pathname])

  return (
    <header className="fixed top-0 left-0 z-50 w-full border-b transition-colors bg-white border-gray-200 dark:bg-blue-700 dark:border-blue-400/30">
      <div className="w-full flex items-center h-12">
        <div
          className="w-1/4 pl-4 font-serif text-base font-medium cursor-pointer transition-colors text-gray-900 dark:text-white"
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
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-gray-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'dashboard'
                    ? 'bg-gray-100 dark:bg-blue-600'
                    : ''
                }`}
                onClick={() => navigate('/chat/dashboard')}
              >
                <FontIcon
                  type="dashboard"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Dashboard</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-gray-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'data' ? 'bg-gray-100 dark:bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/data')}
              >
                <FontIcon
                  type="data"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Data</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-gray-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'prompt' ? 'bg-gray-100 dark:bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/prompt')}
              >
                <FontIcon
                  type="prompt"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Prompt</span>
              </button>
              <button
                className={`w-full flex items-center justify-center gap-2 transition-colors rounded-lg p-2 hover:bg-gray-100 dark:hover:bg-blue-600 dark:hover:opacity-80 ${
                  isSelected === 'test' ? 'bg-gray-100 dark:bg-blue-600' : ''
                }`}
                onClick={() => navigate('/chat/test')}
              >
                <FontIcon
                  type="test"
                  className="w-6 h-6 text-gray-700 dark:text-white"
                />
                <span className="text-gray-700 dark:text-white">Test</span>
              </button>
            </div>
          )}

          <div className="flex items-center gap-3 justify-end">
            <div className="flex rounded-lg border overflow-hidden border-gray-300 dark:border-blue-400/50">
              <button
                className={`w-8 h-7 flex items-center justify-center transition-colors bg-gray-100 text-gray-700 hover:bg-gray-200 dark:bg-blue-600 dark:text-blue-100 dark:hover:bg-blue-700`}
                onClick={() => setTheme('light')}
              >
                <FontIcon type="sun" className="w-4 h-4" />
              </button>
              <button
                className={`w-8 h-7 flex items-center justify-center transition-colors text-gray-700 hover:bg-gray-100 dark:text-white dark:hover:bg-blue-800/50`}
                onClick={() => setTheme('dark')}
              >
                <FontIcon type="moon-filled" className="w-4 h-4" />
              </button>
            </div>
            <FontIcon
              type="user-avatar"
              className="w-6 h-6 text-gray-700 dark:text-white"
            />
          </div>
        </div>
      </div>
    </header>
  )
}

export default Header
