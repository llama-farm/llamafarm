import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import llamaLogo from './assets/logos/llamafarm-logo.svg'

function Home() {
  const [inputValue, setInputValue] = useState('')
  const navigate = useNavigate()

  const projectOptions = [
    { id: 1, text: 'AI Agent for Enterprise Product' },
    { id: 2, text: 'AI-Powered Chatbot for Customer Support' },
    { id: 3, text: 'AI Model for Predicting Equipment Failures' },
    { id: 4, text: 'Recommendation System for E-commerce' },
  ]

  const handleOptionClick = (option: { id: number; text: string }) => {
    setInputValue(option.text)
  }

  const handleSendClick = () => {
    navigate('/chat/data')
  }

  return (
    <div className="h-full flex flex-col items-center justify-center px-4 sm:px-6 lg:px-8 py-8 transition-colors bg-gradient-to-br from-blue-50 to-indigo-100 dark:from-blue-900 dark:to-indigo-900">
      <div className="max-w-4xl w-full text-center space-y-8">
        <div className="space-y-4">
          <p className="text-sm font-medium tracking-wide text-gray-700 dark:text-white">
            Welcome to LlaMaFarm 🦙
          </p>

          <h1 className="font-serif text-2xl sm:text-3xl lg:text-4xl font-normal leading-tight text-gray-900 dark:text-white">
            What are you building?
          </h1>
        </div>
        <div className="max-w-3xl mx-auto">
          <div className="backdrop-blur-sm rounded-lg border p-1 relative bg-white/80 border-gray-300 shadow-lg dark:bg-black/10 dark:border-blue-400/50">
            <textarea
              value={inputValue}
              onChange={e => setInputValue(e.target.value)}
              className="w-full h-24 sm:h-28 bg-transparent border-none resize-none p-4 pr-12 placeholder-opacity-60 focus:outline-none focus:ring-0 font-sans text-sm sm:text-base leading-relaxed text-gray-900 placeholder-gray-500 dark:text-white dark:placeholder-white/60"
              placeholder="I'm building an agent that will work with my app..."
            />
            <button
              onClick={handleSendClick}
              className="absolute bottom-2 right-2 w-8 h-8 bg-blue-100 dark:text-blue-100 dark:bg-blue-800 hover:bg-blue-500 rounded-md flex items-center justify-center text-white transition-colors duration-200 shadow-sm hover:shadow-md"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth="2"
                  d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                />
              </svg>
            </button>
          </div>
        </div>

        <p className="max-w-2xl mx-auto text-sm sm:text-base leading-relaxed text-gray-600 dark:text-white/90">
          We'll help you bring your AI project dreams to life, all while showing
          you how we're doing it.
        </p>

        {/* Project option buttons */}
        <div className="max-w-4xl mx-auto space-y-4">
          {/* First row - stacks on mobile */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => handleOptionClick(projectOptions[0])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[0].text}
            </button>

            <button
              onClick={() => handleOptionClick(projectOptions[1])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[1].text}
            </button>
          </div>

          {/* Second row */}
          <div className="flex flex-col sm:flex-row gap-3 sm:gap-4 justify-center">
            <button
              onClick={() => handleOptionClick(projectOptions[2])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[2].text}
            </button>

            <button
              onClick={() => handleOptionClick(projectOptions[3])}
              className="px-4 py-2 backdrop-blur-sm rounded-full border font-serif text-sm sm:text-base transition-all duration-200 whitespace-nowrap bg-white/60 border-gray-300 text-gray-700 hover:bg-white/80 hover:border-gray-400 dark:bg-slate-800/20 dark:border-blue-300/40 dark:text-white dark:hover:bg-slate-700/30 dark:hover:border-blue-300/60"
            >
              {projectOptions[3].text}
            </button>
          </div>
        </div>
      </div>

      {/* Decorative llama mascot - hidden on mobile for better UX */}
      <div className="lg:block absolute bottom-0 right-0">
        <img src={llamaLogo} alt="llama" className="w-40 h-40" />
      </div>
    </div>
  )
}

export default Home
