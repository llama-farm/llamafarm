import { lazy, Suspense, useCallback } from 'react'

const Add = lazy(() => import('../assets/icons/Add'))
const ArrowFilled = lazy(() => import('../assets/icons/ArrowFilled'))
const ArrowRight = lazy(() => import('../assets/icons/ArrowRight'))
const BuildAssistant = lazy(() => import('../assets/icons/BuildAssistant'))
const ChevronDown = lazy(() => import('../assets/icons/ChevronDown'))
const CheckmarkFilled = lazy(() => import('../assets/icons/CheckmarkFilled'))
const CheckmarkOutline = lazy(() => import('../assets/icons/CheckmarkOutline'))
const Close = lazy(() => import('../assets/icons/Close'))
const ClosePanel = lazy(() => import('../assets/icons/ClosePanel'))
const Code = lazy(() => import('../assets/icons/Code'))
const Copy = lazy(() => import('../assets/icons/Copy'))
const Dashboard = lazy(() => import('../assets/icons/Dashboard'))
const ToolsAlt = lazy(() => import('../assets/icons/ToolsAlt'))
const Data = lazy(() => import('../assets/icons/Data'))
const Edit = lazy(() => import('../assets/icons/Edit'))
const Fade = lazy(() => import('../assets/icons/Fade'))
const Integration = lazy(() => import('../assets/icons/Integration'))
const MoonFilled = lazy(() => import('../assets/icons/MoonFilled'))
const OpenPanel = lazy(() => import('../assets/icons/OpenPanel'))
const Prompt = lazy(() => import('../assets/icons/Prompt'))
const Model = lazy(() => import('../assets/icons/Model'))
const RecentlyViewed = lazy(() => import('../assets/icons/RecentlyViewed'))
const Search = lazy(() => import('../assets/icons/Search'))
const Sun = lazy(() => import('../assets/icons/Sun'))
const Test = lazy(() => import('../assets/icons/Test'))
const Overflow = lazy(() => import('../assets/icons/Overflow'))
const ThumbsDown = lazy(() => import('../assets/icons/ThumbsDown'))
const ThumbsDownFilled = lazy(() => import('../assets/icons/ThumbsDownFilled'))
const ThumbsUp = lazy(() => import('../assets/icons/ThumbsUp'))
const ThumbsUpFilled = lazy(() => import('../assets/icons/ThumbsUpFilled'))
const Trashcan = lazy(() => import('../assets/icons/Trashcan'))
const Upload = lazy(() => import('../assets/icons/Upload'))
const UserAvatar = lazy(() => import('../assets/icons/UserAvatar'))
const Info = lazy(() => import('../assets/icons/Info'))
const Eye = lazy(() => import('../assets/icons/Eye'))
const EyeOff = lazy(() => import('../assets/icons/EyeOff'))
const Rag = lazy(() => import('../assets/icons/Rag'))
const Folder = lazy(() => import('../assets/icons/Folder'))

type FontIconTypes =
  | 'add'
  | 'alert-triangle'
  | 'arrow-filled'
  | 'arrow-right'
  | 'build-assistant'
  | 'chevron-down'
  | 'chevron-up'
  | 'checkmark-filled'
  | 'checkmark-outline'
  | 'copy'
  | 'close'
  | 'close-panel'
  | 'code'
  | 'dashboard'
  | 'data'
  | 'tools-alt'
  | 'edit'
  | 'fade'
  | 'integration'
  | 'loading'
  | 'moon-filled'
  | 'open-panel'
  | 'prompt'
  | 'model'
  | 'recently-viewed'
  | 'save'
  | 'search'
  | 'sun'
  | 'test'
  | 'overflow'
  | 'thumbs-down'
  | 'thumbs-down-filled'
  | 'thumbs-up'
  | 'thumbs-up-filled'
  | 'trashcan'
  | 'upload'
  | 'user-avatar'
  | 'info'
  | 'eye'
  | 'eye-off'
  | 'rag'
  | 'folder'

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
      case 'alert-triangle':
        return <Info /> // Use Info icon for alert-triangle
      case 'arrow-filled':
        return <ArrowFilled />
      case 'arrow-right':
        return <ArrowRight />
      case 'build-assistant':
        return <BuildAssistant />
      case 'chevron-down':
        return <ChevronDown />
      case 'chevron-up':
        return (
          <div style={{ transform: 'rotate(180deg)' }}>
            <ChevronDown />
          </div>
        ) // Rotate chevron-down for chevron-up
      case 'checkmark-filled':
        return <CheckmarkFilled />
      case 'checkmark-outline':
        return <CheckmarkOutline />
      case 'close':
        return <Close />
      case 'copy':
        return <Copy />
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
      case 'loading':
        return <RecentlyViewed /> // Use RecentlyViewed icon for loading (with animate-spin in className)
      case 'moon-filled':
        return <MoonFilled />
      case 'open-panel':
        return <OpenPanel />
      case 'prompt':
        return <Prompt />
      case 'model':
        return <Model />
      case 'recently-viewed':
        return <RecentlyViewed />
      case 'save':
        return <CheckmarkFilled /> // Use CheckmarkFilled icon for save
      case 'search':
        return <Search />
      case 'sun':
        return <Sun />
      case 'test':
        return <Test />
      case 'overflow':
        return <Overflow />
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
      case 'info':
        return <Info />
      case 'eye':
        return <Eye />
      case 'eye-off':
        return <EyeOff />
      case 'rag':
        return <Rag />
      case 'folder':
        return <Folder />
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
