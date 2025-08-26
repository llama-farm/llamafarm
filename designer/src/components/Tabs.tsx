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
          className={`h-fit border-b-2 ${
            activeTab === tab.id
              ? 'border-primary text-foreground'
              : 'border-border text-muted-foreground hover:border-primary hover:text-foreground'
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
