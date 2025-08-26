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
      className={`flex flex-col border border-border bg-card rounded-lg p-4 gap-2 ${
        isThumbsUpFilled ? 'ring-2 ring-primary/40' : ''
      } focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary`}
      tabIndex={0}
      aria-label="Rate output"
      role="group"
    >
      <div className="flex flex-row gap-2 justify-between">
        <div>{output}</div>
        <div className="flex flex-row gap-2 ml-4">
          <FontIcon
            type={isThumbsUpFilled ? 'thumbs-up-filled' : 'thumbs-up'}
            className="w-6 h-6 text-muted-foreground"
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
            className="w-6 h-6 text-muted-foreground"
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
      <div className="w-fit py-1 px-3 bg-accent text-accent-foreground rounded-2xl text-sm">
        {tag}
      </div>
    </div>
  )
}

export default RateOutput
