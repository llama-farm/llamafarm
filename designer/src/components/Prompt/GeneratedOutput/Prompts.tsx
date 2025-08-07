import FontIcon from '../../../common/FontIcon'

const Prompts = () => {
  const tempPrompts = [
    {
      version: '1.0',
      status: 'Active',
      preview:
        'You are an experienced aircraft maintenance technician with 15+ years of experience working on military and commercial aircraft. You specialize in...',
      settings: '[ ]',
    },
    {
      version: '1.1',
      status: 'Active',
      preview:
        'You are a senior aircraft maintenance specialist focused on rapid diagnosis and...',
      settings: '[ ]',
    },
    {
      version: '1.2',
      status: 'Active',
      preview:
        'You are an aircraft maintenance worker focused on diagnosis and error handling...',
      settings: '[ ]',
    },
  ]

  return (
    <div className="w-full h-full">
      <table className="w-full">
        <thead className="bg-white dark:bg-blue-600 font-normal">
          <tr>
            <th className="text-left w-[10%] py-2 px-3">Version</th>
            <th className="text-left w-[10%] py-2 px-3">Status</th>
            <th className="text-left w-[50%] py-2 px-3">Preview</th>
            <th className="text-left w-[10%] py-2 px-3">Settings</th>
            <th className="text-left w-[10%] py-2 px-3">Actions</th>
          </tr>
        </thead>
        <tbody>
          {tempPrompts.map((prompt, index) => (
            <tr
              key={index}
              className={`border-b border-solid border-white dark:border-blue-600 text-sm font-mono leading-4 tracking-[0.32px]${
                index === tempPrompts.length - 1 ? 'border-b-0' : 'border-b'
              }`}
            >
              <td className="align-top p-3">{prompt.version}</td>
              <td className="align-top p-3">
                <FontIcon
                  type="checkmark-outline"
                  className="w-6 h-6 text-blue-100 dark:text-green-100"
                />
              </td>
              <td className="align-top p-3">{prompt.preview}</td>
              <td className="align-top p-3">{prompt.settings}</td>
              <td className="flex flex-row gap-4 align-top p-3">
                <FontIcon
                  type="edit"
                  className="w-6 h-6 text-blue-100"
                  isButton
                />
                <FontIcon
                  type="trashcan"
                  className="w-6 h-6 text-blue-100"
                  isButton
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default Prompts
