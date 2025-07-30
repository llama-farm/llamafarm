import { lazy, Suspense, useCallback } from 'react'

const Sun = lazy(() => import('../assets/icons/Sun'))
const MoonFilled = lazy(() => import('../assets/icons/MoonFilled'))
const UserAvatar = lazy(() => import('../assets/icons/UserAvatar'))
const ArrowFilled = lazy(() => import('../assets/icons/ArrowFilled'))

type FontIconTypes = 'sun' | 'moon-filled' | 'user-avatar' | 'arrow-filled'

export interface FontIconProps {
  className?: string
  type: FontIconTypes
  isButton?: boolean
  handleOnClick?: () => void
  stopPropagation?: boolean
}

const FontIcon: React.FC<FontIconProps> = ({
  className,
  type = 'close',
  isButton = false,
  handleOnClick = () => undefined,
  stopPropagation = false,
}) => {
  const renderIcon = useCallback(() => {
    switch (type) {
      case 'sun':
        return <Sun />
      case 'moon-filled':
        return <MoonFilled />
      case 'user-avatar':
        return <UserAvatar />
      case 'arrow-filled':
        return <ArrowFilled />
    }
  }, [type])

  if (isButton) {
    return (
      <button
        type="button"
        onClick={e => {
          if (stopPropagation) {
            e.stopPropagation()
          }
          handleOnClick()
        }}
        className={`${className} cursor-pointer hover:bg-blue-400/20 rounded-full`}
      >
        <Suspense fallback={<></>}>{renderIcon()}</Suspense>
      </button>
    )
  }

  return (
    <Suspense fallback={<div className={className} />}>
      <div className={className}>{renderIcon()}</div>
    </Suspense>
  )
}

export default FontIcon
