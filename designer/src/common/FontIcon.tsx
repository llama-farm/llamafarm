import { lazy, Suspense, useCallback } from 'react'

const Add = lazy(() => import('../assets/icons/Add'))
const ArrowFilled = lazy(() => import('../assets/icons/ArrowFilled'))
const ArrowRight = lazy(() => import('../assets/icons/ArrowRight'))
const ChevronDown = lazy(() => import('../assets/icons/ChevronDown'))
const CheckmarkFilled = lazy(() => import('../assets/icons/CheckmarkFilled'))
const CheckmarkOutline = lazy(() => import('../assets/icons/CheckmarkOutline'))
const Close = lazy(() => import('../assets/icons/Close'))
const ClosePanel = lazy(() => import('../assets/icons/ClosePanel'))
const Code = lazy(() => import('../assets/icons/Code'))
const Dashboard = lazy(() => import('../assets/icons/Dashboard'))
const ToolsAlt = lazy(() => import('../assets/icons/ToolsAlt'))
const Data = lazy(() => import('../assets/icons/Data'))
const Edit = lazy(() => import('../assets/icons/Edit'))
const Fade = lazy(() => import('../assets/icons/Fade'))
const Integration = lazy(() => import('../assets/icons/Integration'))
const MoonFilled = lazy(() => import('../assets/icons/MoonFilled'))
const OpenPanel = lazy(() => import('../assets/icons/OpenPanel'))
const Prompt = lazy(() => import('../assets/icons/Prompt'))
const RecentlyViewed = lazy(() => import('../assets/icons/RecentlyViewed'))
const Search = lazy(() => import('../assets/icons/Search'))
const Sun = lazy(() => import('../assets/icons/Sun'))
const Test = lazy(() => import('../assets/icons/Test'))
const ThumbsDown = lazy(() => import('../assets/icons/ThumbsDown'))
const ThumbsDownFilled = lazy(() => import('../assets/icons/ThumbsDownFilled'))
const ThumbsUp = lazy(() => import('../assets/icons/ThumbsUp'))
const ThumbsUpFilled = lazy(() => import('../assets/icons/ThumbsUpFilled'))
const Trashcan = lazy(() => import('../assets/icons/Trashcan'))
const Upload = lazy(() => import('../assets/icons/Upload'))
const UserAvatar = lazy(() => import('../assets/icons/UserAvatar'))

type FontIconTypes =
  | 'add'
  | 'arrow-filled'
  | 'arrow-right'
  | 'chevron-down'
  | 'checkmark-filled'
  | 'checkmark-outline'
  | 'close'
  | 'close-panel'
  | 'code'
  | 'dashboard'
  | 'data'
  | 'tools-alt'
  | 'edit'
  | 'fade'
  | 'integration'
  | 'moon-filled'
  | 'open-panel'
  | 'prompt'
  | 'recently-viewed'
  | 'search'
  | 'sun'
  | 'test'
  | 'thumbs-down'
  | 'thumbs-down-filled'
  | 'thumbs-up'
  | 'thumbs-up-filled'
  | 'trashcan'
  | 'upload'
  | 'user-avatar'

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
      case 'add':
        return <Add />
      case 'arrow-filled':
        return <ArrowFilled />
      case 'arrow-right':
        return <ArrowRight />
      case 'chevron-down':
        return <ChevronDown />
      case 'checkmark-filled':
        return <CheckmarkFilled />
      case 'checkmark-outline':
        return <CheckmarkOutline />
      case 'close':
        return <Close />
      case 'close-panel':
        return <ClosePanel />
      case 'code':
        return <Code />
      case 'dashboard':
        return <Dashboard />
      case 'data':
        return <Data />
      case 'tools-alt':
        return <ToolsAlt />
      case 'edit':
        return <Edit />
      case 'fade':
        return <Fade />
      case 'integration':
        return <Integration />
      case 'moon-filled':
        return <MoonFilled />
      case 'open-panel':
        return <OpenPanel />
      case 'prompt':
        return <Prompt />
      case 'recently-viewed':
        return <RecentlyViewed />
      case 'search':
        return <Search />
      case 'sun':
        return <Sun />
      case 'test':
        return <Test />
      case 'thumbs-down':
        return <ThumbsDown />
      case 'thumbs-down-filled':
        return <ThumbsDownFilled />
      case 'thumbs-up':
        return <ThumbsUp />
      case 'thumbs-up-filled':
        return <ThumbsUpFilled />
      case 'trashcan':
        return <Trashcan />
      case 'upload':
        return <Upload />
      case 'user-avatar':
        return <UserAvatar />
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
        className={`${className} cursor-pointer rounded-sm hover:opacity-80`}
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
