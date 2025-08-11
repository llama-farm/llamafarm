import FontIcon from '../../../common/FontIcon'

const Model = () => {
  return (
    <div className="w-full flex flex-col gap-2 items-center justify-center bg-white dark:bg-blue-500 py-10 rounded-lg">
      <div className="font-bold">Model Tab Coming Soon</div>
      <div>
        You’ll soon be able to view and manage model settings directly in this
        tab.
      </div>
      <button className="px-4 py-2 w-fit flex flex-row gap-2 items-center bg-transparent border-[1px] border-solid border-blue-200 text-blue-200 rounded-lg">
        Manage models in code <FontIcon type="code" className="w-4 h-4" />
      </button>
    </div>
  )
}

export default Model
