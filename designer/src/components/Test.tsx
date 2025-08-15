import { useMemo, useState } from 'react'
import FontIcon from '../common/FontIcon'
import ModeToggle, { Mode } from './ModeToggle'
import ConfigEditor from './ConfigEditor'

interface TestCase {
  id: number
  name: string
  source: string
  score: number
  environment: 'Local' | 'Production' | 'Staging'
  lastRun: string
  input?: string
  expected?: string
}

const scorePillClasses = (score: number) => {
  if (score >= 95) return 'bg-green-200 text-black'
  if (score >= 75) return 'bg-blue-200 text-black'
  return 'bg-amber-300 text-black'
}

const Test = () => {
  const [tests, setTests] = useState<TestCase[]>([
    {
      id: 1,
      name: 'Aircraft maintenance queries',
      source: 'From prompts',
      score: 99.5,
      environment: 'Local',
      lastRun: '2hr ago',
      input:
        'The hydraulic pump on the F-16 showed a pressure drop during taxi. What are the most likely causes and the next steps for inspection?',
      expected:
        'A pressure drop in the hydraulic pump during taxi on an F-16 could be caused by fluid leakage, air in the system, or a failing pressure sensor. Recommended next steps include inspecting hydraulic lines for leaks, checking fluid levels, and running a diagnostic on the pressure sensor.',
    },
    {
      id: 2,
      name: 'Basic user queries',
      source: 'From prompts',
      score: 82.5,
      environment: 'Local',
      lastRun: '1d ago',
    },
    {
      id: 3,
      name: 'Aircraft maintenance queries',
      source: 'From prompts',
      score: 76.5,
      environment: 'Production',
      lastRun: '8/1/25',
    },
    {
      id: 4,
      name: 'API integration',
      source: 'Custom',
      score: 99.5,
      environment: 'Production',
      lastRun: '7/30/25',
    },
    {
      id: 5,
      name: 'Aircraft maintenance queries',
      source: 'From chat history',
      score: 54,
      environment: 'Local',
      lastRun: '7/30/25',
    },
    {
      id: 6,
      name: 'Security validation',
      source: 'Custom',
      score: 99.5,
      environment: 'Staging',
      lastRun: '7/30/25',
    },
  ])

  const [isNewOpen, setIsNewOpen] = useState<boolean>(false)
  const [form, setForm] = useState<{
    name: string
    input: string
    expected: string
  }>({ name: '', input: '', expected: '' })

  const [isEditOpen, setIsEditOpen] = useState<boolean>(false)
  const [editId, setEditId] = useState<number | null>(null)
  const [editForm, setEditForm] = useState<{
    name: string
    input: string
    expected: string
  }>({
    name: '',
    input: '',
    expected: '',
  })

  const envOptions = useMemo(
    () => ['Local', 'Production', 'Staging'] as const,
    []
  )

  const isFormValid =
    form.name.trim().length > 0 &&
    form.input.trim().length > 0 &&
    form.expected.trim().length > 0

  const nextId = useMemo(
    () => tests.reduce((max, t) => (t.id > max ? t.id : max), 0) + 1,
    [tests]
  )

  const handleRun = (id: number) => {
    setTests(prev =>
      prev.map(t =>
        t.id === id
          ? {
              ...t,
              lastRun: 'just now',
              // Tiny nudge to the score to simulate a run
              score: Math.max(
                0,
                Math.min(
                  100,
                  Number((t.score + (Math.random() - 0.5) * 2).toFixed(1))
                )
              ),
            }
          : t
      )
    )
  }

  const handleEnvChange = (id: number, env: TestCase['environment']) => {
    setTests(prev =>
      prev.map(t => (t.id === id ? { ...t, environment: env } : t))
    )
  }

  const handleAddTest = () => {
    if (!isFormValid) return
    const newRow: TestCase = {
      id: nextId,
      name: form.name.trim(),
      source: 'Custom',
      score: 0,
      environment: 'Local',
      lastRun: '-',
      input: form.input.trim(),
      expected: form.expected.trim(),
    }
    setTests(prev => [newRow, ...prev])
    setForm({ name: '', input: '', expected: '' })
    setIsNewOpen(false)
  }

  const openEdit = (id: number) => {
    const row = tests.find(t => t.id === id)
    if (!row) return
    setEditId(id)
    setEditForm({
      name: row.name,
      input: row.input ?? '',
      expected: row.expected ?? '',
    })
    setIsEditOpen(true)
  }

  const saveEdit = (runAfterSave: boolean) => {
    if (editId == null) return
    setTests(prev =>
      prev.map(t =>
        t.id === editId
          ? {
              ...t,
              name: editForm.name.trim(),
              input: editForm.input.trim(),
              expected: editForm.expected.trim(),
            }
          : t
      )
    )
    if (runAfterSave) {
      handleRun(editId)
    }
    setIsEditOpen(false)
  }

  const [mode, setMode] = useState<Mode>('designer')

  return (
    <div className="w-full flex flex-col gap-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl ">
          {mode === 'designer' ? 'Test' : 'Config editor'}
        </h2>
        <div className="flex items-center gap-3">
          <ModeToggle mode={mode} onToggle={setMode} />
          <button className="opacity-50 cursor-not-allowed text-sm px-3 py-2 rounded-lg border border-blue-50 text-blue-50 dark:text-blue-100 dark:border-blue-400">
            Deploy
          </button>
        </div>
      </div>

      {/* New test case accordion */}
      {mode !== 'designer' ? (
        <ConfigEditor />
      ) : (
        <div className="w-full rounded-xl bg-white dark:bg-blue-500 p-4">
          <div className="flex items-center justify-between">
            <button
              type="button"
              className="flex items-center gap-2 text-left"
              onClick={() => setIsNewOpen(v => !v)}
            >
              <span className="text-sm">New test case</span>
              <FontIcon
                type="chevron-down"
                className={`w-4 h-4 text-gray-800 dark:text-white transition-transform ${
                  isNewOpen ? 'rotate-180' : ''
                }`}
              />
            </button>
            <button
              type="button"
              disabled={!isFormValid}
              onClick={handleAddTest}
              className={`text-sm px-3 py-2 rounded-lg ${
                isFormValid
                  ? 'bg-blue-200 text-white hover:opacity-90'
                  : 'opacity-50 cursor-not-allowed border border-blue-50 text-blue-50 dark:text-blue-100 dark:border-blue-400'
              }`}
            >
              Add test case
            </button>
          </div>

          {isNewOpen && (
            <div className="mt-3 flex flex-col gap-3">
              <div>
                <label className="text-xs text-blue-100">Test name</label>
                <input
                  type="text"
                  placeholder="Test name here"
                  value={form.name}
                  onChange={e => setForm({ ...form, name: e.target.value })}
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100"
                />
              </div>
              <div>
                <label className="text-xs text-blue-100">Input</label>
                <textarea
                  rows={3}
                  placeholder="Enter the input prompt to test"
                  value={form.input}
                  onChange={e => setForm({ ...form, input: e.target.value })}
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100 code-like"
                />
              </div>
              <div>
                <label className="text-xs text-blue-100">
                  Expected output (baseline)
                </label>
                <textarea
                  rows={3}
                  placeholder="Enter the input prompt to test"
                  value={form.expected}
                  onChange={e => setForm({ ...form, expected: e.target.value })}
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100 code-like"
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Table of tests */}
      {mode === 'designer' && (
        <div className="w-full rounded-md overflow-hidden border border-blue-50 dark:border-blue-700">
          <table className="w-full text-sm">
            <thead className="bg-[#131d45]/10 dark:bg-blue-700/40">
              <tr>
                <th className="text-left px-4 py-2">Test name</th>
                <th className="text-left px-4 py-2">Match score</th>
                <th className="text-left px-4 py-2">Environment</th>
                <th className="text-left px-4 py-2">Last run</th>
                <th className="text-left px-4 py-2">Actions</th>
              </tr>
            </thead>
            <tbody>
              {tests.map(test => (
                <tr
                  key={test.id}
                  className="bg-white dark:bg-blue-500 border-t border-blue-50 dark:border-blue-700"
                >
                  <td className="px-4 py-3 align-top">
                    <div className="text-sm">{test.name}</div>
                    <div className="text-xs text-blue-200 dark:text-blue-100">
                      {test.source}
                    </div>
                  </td>
                  <td className="px-4 py-3 align-top">
                    <span
                      className={`px-2 py-0.5 rounded-2xl text-xs ${scorePillClasses(test.score)}`}
                    >
                      {test.score}%
                    </span>
                  </td>
                  <td className="px-4 py-3 align-top">
                    <select
                      className="w-[160px] bg-transparent rounded-md px-3 py-1 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100"
                      value={test.environment}
                      onChange={e =>
                        handleEnvChange(
                          test.id,
                          e.target.value as TestCase['environment']
                        )
                      }
                    >
                      {envOptions.map(opt => (
                        <option key={opt} value={opt} className="text-gray-900">
                          {opt}
                        </option>
                      ))}
                    </select>
                  </td>
                  <td className="px-4 py-3 align-top">{test.lastRun}</td>
                  <td className="px-4 py-3 align-top">
                    <div className="flex items-center gap-2">
                      <button
                        className="px-3 py-1 rounded-md border text-xs border-emerald-300 text-emerald-300 hover:bg-emerald-300 hover:text-black transition-colors"
                        onClick={() => handleRun(test.id)}
                      >
                        Run
                      </button>
                      <FontIcon
                        type="edit"
                        isButton
                        handleOnClick={() => openEdit(test.id)}
                        className="w-5 h-5 text-blue-200 dark:text-blue-100"
                      />
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {isEditOpen && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/50">
          <div className="w-[860px] max-w-[95vw] rounded-xl overflow-hidden bg-white dark:bg-blue-600 text-gray-900 dark:text-white shadow-xl">
            <div className="flex items-center justify-between px-5 py-4 border-b border-blue-50/50 dark:border-blue-400/30">
              <div className="text-sm">Edit test case</div>
              <FontIcon
                type="close"
                isButton
                handleOnClick={() => setIsEditOpen(false)}
                className="w-5 h-5 text-gray-700 dark:text-white"
              />
            </div>

            <div className="p-5 flex flex-col gap-3">
              <div>
                <label className="text-xs text-blue-100">Test name</label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={e =>
                    setEditForm({ ...editForm, name: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100"
                />
              </div>
              <div>
                <label className="text-xs text-blue-100">Input</label>
                <textarea
                  rows={3}
                  value={editForm.input}
                  onChange={e =>
                    setEditForm({ ...editForm, input: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100 code-like"
                />
              </div>
              <div>
                <label className="text-xs text-blue-100">
                  Expected output (baseline)
                </label>
                <textarea
                  rows={3}
                  value={editForm.expected}
                  onChange={e =>
                    setEditForm({ ...editForm, expected: e.target.value })
                  }
                  className="w-full mt-1 bg-transparent rounded-lg py-2 px-3 border border-blue-50 text-gray-800 dark:text-white dark:border-blue-100 code-like"
                />
              </div>
            </div>

            <div className="px-5 py-4 flex items-center justify-between bg-blue-50/30 dark:bg-blue-700">
              <div className="flex items-center gap-2">
                <span className="text-sm">Match score</span>
                <span
                  className={`px-2 py-0.5 rounded-2xl text-xs ${scorePillClasses(tests.find(t => t.id === editId)?.score ?? 0)}`}
                >
                  {tests.find(t => t.id === editId)?.score ?? 0}%
                </span>
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="px-3 py-2 rounded-md bg-blue-200 text-white hover:opacity-90 text-sm"
                  onClick={() => saveEdit(true)}
                >
                  Save and run
                </button>
                <button
                  className="px-3 py-2 rounded-md bg-blue-200/10 text-blue-200 border border-blue-200/50 hover:bg-blue-200/20 text-sm dark:text-green-100 dark:border-green-100/40"
                  onClick={() => saveEdit(false)}
                >
                  Save changes
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default Test
