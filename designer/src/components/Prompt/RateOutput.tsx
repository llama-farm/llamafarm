import { useState } from 'react'
import FontIcon from '../../common/FontIcon'

interface RateOutputProps {
  output: string
  tag: string
}

const RateOutput = ({ output, tag }: RateOutputProps) => {
  const [isThumbsUpFilled, setIsThumbsUpFilled] = useState(false)
  const [isThumbsDownFilled, setIsThumbsDownFilled] = useState(false)

  return (
    <div
      className={`flex flex-col border-[1px] border-solid border-blue-100 dark:border-blue-400 rounded-lg p-4 gap-2 ${
        isThumbsUpFilled ? 'border-green-100' : ''
      }`}
    >
      <div className="flex flex-row gap-2 justify-between">
        <div>{output}</div>
        <div className="flex flex-row gap-2 ml-4">
          <FontIcon
            type={isThumbsUpFilled ? 'thumbs-up-filled' : 'thumbs-up'}
            className="w-6 h-6 text-blue-200 dark:text-white"
            isButton
            handleOnClick={() => {
              if (isThumbsUpFilled) {
                setIsThumbsUpFilled(false)
              } else {
                setIsThumbsUpFilled(true)
                setIsThumbsDownFilled(false)
              }
            }}
          />
          <FontIcon
            type={isThumbsDownFilled ? 'thumbs-down-filled' : 'thumbs-down'}
            className="w-6 h-6 text-blue-200 dark:text-white"
            isButton
            handleOnClick={() => {
              if (isThumbsDownFilled) {
                setIsThumbsDownFilled(false)
              } else {
                setIsThumbsDownFilled(true)
                setIsThumbsUpFilled(false)
              }
            }}
          />
        </div>
      </div>
      <div className="w-fit py-1 px-3 bg-blue-50 dark:bg-blue-600 rounded-2xl text-sm">
        {tag}
      </div>
    </div>
  )
}

export default RateOutput
