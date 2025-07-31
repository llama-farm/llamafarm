import Dashboard from './Dashboard'
import Data from './Data'
import Prompt from './Prompt'
import Test from './Test'
import useHeader from '../hooks/useHeader.tsx'

function BuildArea() {
  const { buildType } = useHeader()

  return (
    <div className="bg-[#000B1B] h-full text-white">
      {buildType === 'dashboard' && <Dashboard />}
      {buildType === 'data' && <Data />}
      {buildType === 'prompt' && <Prompt />}
      {buildType === 'test' && <Test />}
    </div>
  )
}

export default BuildArea
