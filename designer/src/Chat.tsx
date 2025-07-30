import { Link } from 'react-router-dom'

function Chat() {
  return (
    <div
      className="min-h-screen relative overflow-hidden"
      style={{
        background:
          'radial-gradient(88.43% 88.43% at 49.96% 88.43%, #007FFF 0%, #000D3E 100%)',
      }}
    >
      <div className="absolute inset-0 bg-black/10" />

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-4 sm:px-6 lg:px-8 py-8">
        <div className="max-w-4xl w-full text-center space-y-8">
          <div className="space-y-4">
            <h1 className="font-serif text-white text-2xl sm:text-3xl lg:text-4xl font-normal leading-tight">
              Blank Page
            </h1>
            <p className="text-white/80 text-sm font-medium tracking-wide">
              This is a blank page you navigated to!
            </p>
          </div>

          <div className="mt-8">
            <Link
              to="/"
              className="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-lg text-white font-medium transition-colors duration-200 shadow-sm hover:shadow-md"
            >
              Go Back Home
            </Link>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Chat
