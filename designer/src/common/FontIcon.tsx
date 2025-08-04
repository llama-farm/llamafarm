import { lazy, Suspense, useCallback } from 'react'

const Sun = lazy(() => import('../assets/icons/Sun'))
const MoonFilled = lazy(() => import('../assets/icons/MoonFilled'))
const UserAvatar = lazy(() => import('../assets/icons/UserAvatar'))
const ArrowFilled = lazy(() => import('../assets/icons/ArrowFilled'))
const ClosePanel = lazy(() => import('../assets/icons/ClosePanel'))
const OpenPanel = lazy(() => import('../assets/icons/OpenPanel'))
const Dashboard = lazy(() => import('../assets/icons/Dashboard'))
const Data = lazy(() => import('../assets/icons/Data'))
const Prompt = lazy(() => import('../assets/icons/Prompt'))
const Test = lazy(() => import('../assets/icons/Test'))
const Integration = lazy(() => import('../assets/icons/Integration'))
const RecentlyViewed = lazy(() => import('../assets/icons/RecentlyViewed'))
const Code = lazy(() => import('../assets/icons/Code'))
const Edit = lazy(() => import('../assets/icons/Edit'))
const Add = lazy(() => import('../assets/icons/Add'))
const Upload = lazy(() => import('../assets/icons/Upload'))
const CheckmarkFilled = lazy(() => import('../assets/icons/CheckmarkFilled'))
const Trashcan = lazy(() => import('../assets/icons/Trashcan'))
const Fade = lazy(() => import('../assets/icons/Fade'))
const ChevronDown = lazy(() => import('../assets/icons/ChevronDown'))

type FontIconTypes =
  | 'sun'
  | 'moon-filled'
  | 'user-avatar'
  | 'arrow-filled'
  | 'close-panel'
  | 'open-panel'
  | 'dashboard'
  | 'data'
  | 'prompt'
  | 'test'
  | 'integration'
  | 'recently-viewed'
  | 'code'
  | 'edit'
  | 'add'
  | 'upload'
  | 'checkmark-filled'
  | 'trashcan'
  | 'fade'
  | 'chevron-down'

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
      case 'dashboard':
        return <Dashboard />
      case 'data':
        return <Data />
      case 'prompt':
        return <Prompt />
      case 'test':
        return <Test />
      case 'integration':
        return <Integration />
      case 'recently-viewed':
        return <RecentlyViewed />
      case 'code':
        return <Code />
      case 'edit':
        return <Edit />
      case 'add':
        return <Add />
      case 'upload':
        return <Upload />
      case 'checkmark-filled':
        return <CheckmarkFilled />
      case 'trashcan':
        return <Trashcan />
      case 'fade':
        return <Fade />
      case 'chevron-down':
        return <ChevronDown />
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
