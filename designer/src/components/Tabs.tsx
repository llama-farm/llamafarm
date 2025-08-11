interface TabsProps {
  activeTab: string
  setActiveTab: (tab: string) => void
  tabs: { id: string; label: string }[]
}

const Tabs = ({ activeTab, setActiveTab, tabs }: TabsProps) => {
  return (
    <div className="w-full flex flex-row items-end">
      {tabs.map(tab => (
        <button
          key={tab.id}
          className={`h-fit border-b-2 border-solid ${
            activeTab === tab.id
              ? 'border-blue-200 dark:border-green-100'
              : 'border-blue-50 dark:border-blue-600'
          } pb-1 pl-4 w-full text-left`}
          onClick={() => setActiveTab(tab.id)}
        >
          {tab.label}
        </button>
      ))}
    </div>
  )
}

export default Tabs
