const DataCards = () => {
  return (
    <div className="w-full flex flex-row gap-4">
      <div className="w-1/3 min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card">
        <div className="text-sm text-foreground">Model accuracy</div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[40px]">
            <span className="text-teal-500 dark:text-teal-300">78.3%</span>{' '}
            <span className="text-sm text-muted-foreground">
              up from last week
            </span>
          </div>
        </div>
      </div>
      <div className="w-1/3 min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card">
        <div className="text-sm text-foreground">Response time</div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[40px]">
            <span className="text-teal-500 dark:text-teal-300">340ms</span>{' '}
            <span className="text-sm text-muted-foreground">45ms faster</span>
          </div>
        </div>
      </div>
      <div className="w-1/3 min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card">
        <div className="text-sm text-foreground">Monthly queries</div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[40px]">
            <span className="text-teal-500 dark:text-teal-300">30k</span>{' '}
            <span className="text-sm text-muted-foreground">
              up 24% this month
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DataCards
