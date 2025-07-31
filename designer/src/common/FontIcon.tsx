import { lazy, Suspense, useCallback } from 'react'

const Sun = lazy(() => import('../assets/icons/Sun'))
const MoonFilled = lazy(() => import('../assets/icons/MoonFilled'))
const UserAvatar = lazy(() => import('../assets/icons/UserAvatar'))
const ArrowFilled = lazy(() => import('../assets/icons/ArrowFilled'))
const ClosePanel = lazy(() => import('../assets/icons/ClosePanel'))
const OpenPanel = lazy(() => import('../assets/icons/OpenPanel'))

type FontIconTypes =
  | 'sun'
  | 'moon-filled'
  | 'user-avatar'
  | 'arrow-filled'
  | 'close-panel'
  | 'open-panel'

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
      case 'close-panel':
        return <ClosePanel />
      case 'open-panel':
        return <OpenPanel />
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
        className={`${className} cursor-pointer hover:bg-blue-400/20 rounded-sm`}
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
