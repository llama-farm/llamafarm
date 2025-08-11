const DataCards = () => {
  return (
    <div className="w-full flex flex-row gap-2">
      <div className="w-1/3 h-[103px] flex flex-col rounded-lg p-4 bg-white dark:bg-blue-500">
        <div className="text-sm text-gray-700 dark:text-white">
          Model accuracy
        </div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[40px] text-blue-200 dark:text-green-100">
            78.3% <span className="text-sm">up from last week </span>
          </div>
        </div>
      </div>
      <div className="w-1/3 h-[103px] flex flex-col rounded-lg p-4 bg-white dark:bg-blue-500">
        <div className="text-sm text-gray-700 dark:text-white">
          Response time
        </div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[40px] text-blue-200 dark:text-green-100">
            340ms <span className="text-sm">45ms faster </span>
          </div>
        </div>
      </div>
      <div className="w-1/3 h-[103px] flex flex-col rounded-lg p-4 bg-white dark:bg-blue-500">
        <div className="text-sm text-gray-700 dark:text-white">
          Monthly queries
        </div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[40px] text-blue-200 dark:text-green-100">
            30k <span className="text-sm">up 24% this month</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DataCards
