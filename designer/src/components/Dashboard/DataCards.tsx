const DataCards = () => {
  return (
    <div className="w-full grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card min-w-0 overflow-hidden">
        <div className="text-sm text-foreground">Model accuracy</div>
        <div className="flex flex-row gap-2 items-end min-w-0">
          <div className="text-[40px] min-w-0">
            <span className="text-teal-500 dark:text-teal-300">78.3%</span>{' '}
            <span className="text-sm text-muted-foreground truncate inline-block align-bottom max-w-full">
              up from last week
            </span>
          </div>
        </div>
      </div>
      <div className="min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card min-w-0 overflow-hidden">
        <div className="text-sm text-foreground">Response time</div>
        <div className="flex flex-row gap-2 items-end min-w-0">
          <div className="text-[40px] min-w-0">
            <span className="text-teal-500 dark:text-teal-300">340ms</span>{' '}
            <span className="text-sm text-muted-foreground truncate inline-block align-bottom max-w-full">
              45ms faster
            </span>
          </div>
        </div>
      </div>
      <div className="min-h-[103px] flex flex-col rounded-lg p-6 pb-8 bg-card min-w-0 overflow-hidden">
        <div className="text-sm text-foreground">Monthly queries</div>
        <div className="flex flex-row gap-2 items-end min-w-0">
          <div className="text-[40px] min-w-0">
            <span className="text-teal-500 dark:text-teal-300">30k</span>{' '}
            <span className="text-sm text-muted-foreground truncate inline-block align-bottom max-w-full">
              up 24% this month
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DataCards
