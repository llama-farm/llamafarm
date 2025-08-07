import FontIcon from '../../common/FontIcon'
import GitlabLogo from '../../assets/logos/gitlab-logo.svg'
import GithubLogo from '../../assets/logos/github-logo.svg'
import SlackLogo from '../../assets/logos/slack-logo.svg'

const DataCards = () => {
  return (
    <div className="w-full flex flex-row gap-2">
      <div className="w-1/3 h-[103px] flex flex-col gap-2 rounded-lg p-4 bg-white shadow-md dark:bg-blue-500">
        <div className="text-sm text-gray-700 dark:text-white">
          Model accuracy
        </div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[30px] text-green-600 dark:text-green-100">
            78.3% <span className="text-sm">up from last week </span>
          </div>
        </div>
      </div>
      <div className="w-1/3 h-[103px] flex flex-col gap-2 rounded-lg p-4 bg-white shadow-md dark:bg-blue-500">
        <div className="text-sm text-gray-700 dark:text-white">
          Model accuracy
        </div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[30px] text-green-600 dark:text-green-100">
            78.3% <span className="text-sm">up from last week </span>
          </div>
        </div>
      </div>
      <div className="w-1/3 h-[103px] flex flex-col gap-2 rounded-lg p-4 bg-white shadow-md dark:bg-blue-500">
        <div className="text-sm text-gray-700 dark:text-white">
          Model accuracy
        </div>
        <div className="flex flex-row gap-2 items-end">
          <div className="text-[30px] text-green-600 dark:text-green-100">
            78.3% <span className="text-sm">up from last week </span>
          </div>
        </div>
      </div>
    </div>
  )
}

const Dashboard = () => {
  return (
    <div className="w-full flex flex-col gap-2">
      <DataCards />
      <div className="w-full flex flex-row gap-2">
        <div className="w-3/5 flex flex-col gap-2">
          <div className="flex flex-col">
            <div className="flex flex-row gap-2 items-center h-[40px] px-2 rounded-tl-lg rounded-tr-lg justify-between bg-gray-200 dark:bg-blue-600">
              <div className="flex flex-row gap-2 items-center text-gray-700 dark:text-white">
                <FontIcon type="data" className="w-4 h-4" />
                Data
              </div>
              <button className="text-xs text-green-600 dark:text-green-100">
                View and add
              </button>
            </div>
            <div className="p-2 flex flex-col gap-2 rounded-bl-lg rounded-br-lg bg-gray-50 dark:bg-blue-500">
              <div className="py-1 px-2 rounded-lg flex flex-row gap-2 items-center justify-between bg-white shadow-sm dark:bg-blue-700">
                <div className="text-gray-700 dark:text-white">
                  dataset-1-aircraft-logs
                </div>
                <div className="text-xs text-gray-500 dark:text-blue-100">
                  Updated 3 hours ago
                </div>
              </div>
              <div className="py-1 px-2 rounded-lg flex flex-row gap-2 items-center justify-between bg-white shadow-sm dark:bg-blue-700">
                <div className="text-gray-700 dark:text-white">
                  data-set-2-aircraft-maintenance-rules
                </div>
                <div className="text-xs text-gray-500 dark:text-blue-100">
                  Updated 6 hours ago
                </div>
              </div>
              <div className="text-xs text-gray-500 dark:text-blue-100">
                3 datasets total, showing last updated
              </div>
            </div>
          </div>
          <div className="flex flex-row gap-2 rounded-lg">
            <div className="w-1/2">
              <div className="h-[40px] px-2 flex items-center rounded-tl-lg rounded-tr-lg bg-gray-200 dark:bg-blue-600">
                <span className="text-gray-700 dark:text-white">Models</span>
              </div>
              <div className="p-4 flex flex-col min-h-[325px] rounded-bl-lg rounded-br-lg bg-gray-50 dark:bg-blue-500">
                <div className="mb-6">
                  <label className="text-xs text-gray-600 dark:text-gray-100">
                    Current model
                  </label>
                  <div className="w-full flex flex-row gap-2 items-center justify-between">
                    <div className="rounded-xl px-3 py-1 my-1 w-full bg-white shadow-sm dark:bg-blue-600">
                      <span className="text-gray-700 dark:text-white">
                        TinyLlama
                      </span>
                    </div>
                    <FontIcon
                      type="edit"
                      className="w-6 h-6 text-gray-600 dark:text-blue-100"
                    />
                  </div>
                  <div className="text-xs text-gray-500 dark:text-blue-100">
                    Why TinyLama?
                  </div>
                </div>
                <div>
                  <label className="text-xs text-gray-600 dark:text-gray-100">
                    OpenAI API Key
                  </label>
                  <div className="w-full flex flex-row gap-2 items-center justify-between my-1">
                    <div className="rounded-xl px-3 py-1 w-full bg-white shadow-sm dark:bg-blue-600">
                      <span className="text-gray-500 dark:text-gray-300">
                        Enter here
                      </span>
                    </div>
                    <button className="rounded-lg p-1 w-10 h-8 flex items-center justify-center bg-blue-500 dark:bg-blue-200">
                      <FontIcon type="add" className="w-4 h-4 text-white" />
                    </button>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-blue-100">
                    Connect your project to OpenAI
                  </div>
                </div>
              </div>
            </div>
            <div className="w-1/2">
              <div className="flex flex-row gap-2 items-center justify-between h-[40px] px-2 rounded-tl-lg rounded-tr-lg bg-gray-200 dark:bg-blue-600">
                <div className="flex flex-row gap-2 items-center text-gray-700 dark:text-white">
                  <FontIcon type="integration" className="w-4 h-4" />
                  Integrations
                </div>
                <button className="text-xs text-green-600 dark:text-green-100 flex flex-row gap-1 items-center">
                  Add
                  <FontIcon type="add" className="w-4 h-4" />
                </button>
              </div>
              <div className="p-4 flex flex-col min-h-[325px] justify-between rounded-bl-lg rounded-br-lg bg-gray-50 dark:bg-blue-500">
                <div className="flex flex-col gap-2">
                  <div className="flex flex-row gap-2 items-center border-[1px] border-solid rounded-lg py-1 px-2 justify-between border-gray-300 bg-white dark:bg-blue-600 dark:border-blue-600">
                    <div className="flex flex-row gap-2 items-center text-gray-700 dark:text-white">
                      <img
                        src={GitlabLogo}
                        alt="integrations"
                        className="w-4 h-4"
                      />
                      <div>Gitlab</div>
                    </div>
                    <div className="flex flex-row gap-1 items-center">
                      <div className="w-2 h-2 bg-green-600 dark:bg-green-100 rounded-full"></div>
                      <div className="text-xs text-green-600 dark:text-green-100">
                        Connected
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-row gap-2 items-center border-[1px] border-solid rounded-lg py-1 px-2 justify-between border-gray-300 bg-white dark:bg-blue-600 dark:border-blue-600">
                    <div className="flex flex-row gap-2 items-center text-gray-700 dark:text-white">
                      <img
                        src={GithubLogo}
                        alt="integrations"
                        className="w-4 h-4"
                      />
                      <div>Github</div>
                    </div>
                    <div className="flex flex-row gap-1 items-center">
                      <div className="w-2 h-2 bg-green-600 dark:bg-green-100 rounded-full"></div>
                      <div className="text-xs text-green-600 dark:text-green-100">
                        Connected
                      </div>
                    </div>
                  </div>
                  <div className="flex flex-row gap-2 items-center border-[1px] border-solid rounded-lg py-1 px-2 justify-between border-gray-300 bg-white dark:bg-blue-600 dark:border-blue-600">
                    <div className="flex flex-row gap-2 items-center text-gray-700 dark:text-white">
                      <img
                        src={SlackLogo}
                        alt="integrations"
                        className="w-4 h-4"
                      />
                      <div>Slack</div>
                    </div>
                    <div className="flex flex-row gap-1 items-center">
                      <div className="w-2 h-2 bg-green-600 dark:bg-green-100 rounded-full"></div>
                      <div className="text-xs text-green-600 dark:text-green-100">
                        Connected
                      </div>
                    </div>
                  </div>
                </div>
                <div>
                  <button className="text-xs text-gray-600 dark:text-blue-100 flex flex-row gap-1 items-center justify-center">
                    Edit
                    <FontIcon type="edit" className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
        <div className="w-2/5">
          <div className="flex flex-row gap-2 items-center justify-between h-[40px] px-2 rounded-tl-lg rounded-tr-lg bg-gray-200 dark:bg-blue-600">
            <span className="text-gray-700 dark:text-white">
              Project versions
            </span>
            <FontIcon type="recently-viewed" className="w-4 h-4" />
          </div>
          <div className="p-4 flex flex-col rounded-bl-lg rounded-br-lg bg-gray-50 dark:bg-blue-500">
            <div className="flex flex-col gap-2 h-[400px] overflow-y-auto">
              <div className="text-xs text-gray-500 dark:text-blue-100">
                Today
              </div>
              {Array.from({ length: 4 }).map((_, index) => (
                <div key={index} className="flex flex-col mb-2">
                  <div className="flex flex-row gap-2 items-center justify-between">
                    <div className="text-gray-700 dark:text-white">
                      Version 1.0.0
                    </div>
                    <div className="text-gray-500 dark:text-blue-100">
                      7:45PM
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-blue-100">
                    RAG and prompt model updates
                  </div>
                </div>
              ))}
              <div className="text-xs text-gray-500 dark:text-blue-100">
                July 30
              </div>
              {Array.from({ length: 2 }).map((_, index) => (
                <div key={index} className="flex flex-col">
                  <div className="flex flex-row gap-2 items-center justify-between">
                    <div className="text-gray-700 dark:text-white">
                      Version 1.0.0
                    </div>
                    <div className="text-gray-500 dark:text-blue-100">
                      7:45PM
                    </div>
                  </div>
                  <div className="text-xs text-gray-500 dark:text-blue-100">
                    RAG and prompt model updates
                  </div>
                </div>
              ))}
            </div>
            <div className="w-full flex justify-center items-center mt-4">
              <button className="w-full rounded-lg py-1 border-[1px] border-solid flex flex-row gap-2 items-center justify-center border-green-600 text-green-600 hover:bg-green-50 dark:border-green-100 dark:text-green-100">
                View config
                <FontIcon type="code" className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
