import { useState } from 'react'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Label } from '../ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../ui/select'
import type { MeshNode, TestNode } from '../../types/atmosphere'
import { Plus, Send, Trash2, X } from 'lucide-react'

interface TestPanelProps {
  nodes: MeshNode[]
  onAddNode: (node: TestNode) => void
  onRemoveNode: (nodeId: string) => void
  onSendIntent: (intent: { content: string; targetCapability?: string }) => void
  isOpen: boolean
  onToggle: () => void
}

export default function TestPanel({
  nodes,
  onAddNode,
  onRemoveNode,
  onSendIntent,
  isOpen,
  onToggle,
}: TestPanelProps) {
  const [newNodeName, setNewNodeName] = useState('')
  const [newNodeType, setNewNodeType] = useState<MeshNode['type']>('agent')
  const [newNodeCapabilities, setNewNodeCapabilities] = useState('')
  const [intentContent, setIntentContent] = useState('')
  const [targetCapability, setTargetCapability] = useState('')

  const handleAddNode = () => {
    if (!newNodeName.trim()) return

    const capabilities = newNodeCapabilities
      .split(',')
      .map(c => c.trim())
      .filter(c => c.length > 0)

    onAddNode({
      name: newNodeName,
      type: newNodeType,
      capabilities,
    })

    // Reset form
    setNewNodeName('')
    setNewNodeCapabilities('')
  }

  const handleSendIntent = () => {
    if (!intentContent.trim()) return

    onSendIntent({
      content: intentContent,
      targetCapability: targetCapability || undefined,
    })

    // Reset form
    setIntentContent('')
    setTargetCapability('')
  }

  // Extract all unique capabilities from nodes
  const allCapabilities = Array.from(
    new Set(nodes.flatMap(node => node.capabilities))
  ).sort()

  if (!isOpen) {
    return (
      <div className="fixed top-20 right-4 z-10">
        <Button
          onClick={onToggle}
          className="bg-purple-600 hover:bg-purple-700 text-white shadow-lg"
        >
          Open Test Panel
        </Button>
      </div>
    )
  }

  return (
    <div className="fixed top-20 right-4 w-96 max-h-[calc(100vh-6rem)] bg-gray-800 rounded-lg shadow-2xl overflow-hidden z-10 flex flex-col">
      {/* Header */}
      <div className="bg-purple-600 px-4 py-3 flex items-center justify-between">
        <h3 className="text-white font-semibold">Test Mode</h3>
        <button
          onClick={onToggle}
          className="text-white hover:bg-purple-700 rounded p-1"
        >
          <X size={18} />
        </button>
      </div>

      <div className="flex-1 overflow-y-auto p-4 space-y-6">
        {/* Add Node Section */}
        <section className="space-y-3">
          <h4 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
            <Plus size={16} />
            Add Test Node
          </h4>
          
          <div className="space-y-2">
            <div>
              <Label htmlFor="nodeName" className="text-gray-300 text-xs">
                Node Name
              </Label>
              <Input
                id="nodeName"
                value={newNodeName}
                onChange={(e) => setNewNodeName(e.target.value)}
                placeholder="e.g., Test Agent 1"
                className="bg-gray-700 border-gray-600 text-white mt-1"
              />
            </div>

            <div>
              <Label htmlFor="nodeType" className="text-gray-300 text-xs">
                Node Type
              </Label>
              <Select value={newNodeType} onValueChange={(v) => setNewNodeType(v as MeshNode['type'])}>
                <SelectTrigger className="bg-gray-700 border-gray-600 text-white mt-1">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="gateway">Gateway</SelectItem>
                  <SelectItem value="agent">Agent</SelectItem>
                  <SelectItem value="skill">Skill</SelectItem>
                  <SelectItem value="service">Service</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div>
              <Label htmlFor="capabilities" className="text-gray-300 text-xs">
                Capabilities (comma-separated)
              </Label>
              <Input
                id="capabilities"
                value={newNodeCapabilities}
                onChange={(e) => setNewNodeCapabilities(e.target.value)}
                placeholder="e.g., chat, image, search"
                className="bg-gray-700 border-gray-600 text-white mt-1"
              />
            </div>

            <Button
              onClick={handleAddNode}
              disabled={!newNodeName.trim()}
              className="w-full bg-green-600 hover:bg-green-700"
            >
              <Plus size={16} className="mr-2" />
              Add Node
            </Button>
          </div>
        </section>

        {/* Send Intent Section */}
        <section className="space-y-3 border-t border-gray-700 pt-4">
          <h4 className="text-sm font-semibold text-gray-200 flex items-center gap-2">
            <Send size={16} />
            Send Test Intent
          </h4>
          
          <div className="space-y-2">
            <div>
              <Label htmlFor="intentContent" className="text-gray-300 text-xs">
                Intent Content
              </Label>
              <Input
                id="intentContent"
                value={intentContent}
                onChange={(e) => setIntentContent(e.target.value)}
                placeholder="What do you want to do?"
                className="bg-gray-700 border-gray-600 text-white mt-1"
              />
            </div>

            <div>
              <Label htmlFor="targetCap" className="text-gray-300 text-xs">
                Target Capability (optional)
              </Label>
              <Select value={targetCapability} onValueChange={setTargetCapability}>
                <SelectTrigger className="bg-gray-700 border-gray-600 text-white mt-1">
                  <SelectValue placeholder="Any capability" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="">Any capability</SelectItem>
                  {allCapabilities.map(cap => (
                    <SelectItem key={cap} value={cap}>
                      {cap}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <Button
              onClick={handleSendIntent}
              disabled={!intentContent.trim()}
              className="w-full bg-blue-600 hover:bg-blue-700"
            >
              <Send size={16} className="mr-2" />
              Send Intent
            </Button>
          </div>
        </section>

        {/* Active Nodes Section */}
        <section className="space-y-3 border-t border-gray-700 pt-4">
          <h4 className="text-sm font-semibold text-gray-200">
            Active Nodes ({nodes.length})
          </h4>
          
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {nodes.length === 0 ? (
              <p className="text-gray-400 text-xs italic">No nodes yet. Add some to get started!</p>
            ) : (
              nodes.map(node => (
                <div
                  key={node.id}
                  className="bg-gray-700 rounded p-2 flex items-start justify-between group"
                >
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-2 h-2 rounded-full flex-shrink-0"
                        style={{
                          backgroundColor:
                            node.status === 'online' ? '#10b981' :
                            node.status === 'busy' ? '#f59e0b' :
                            node.status === 'error' ? '#ef4444' : '#6b7280'
                        }}
                      />
                      <span className="text-white text-sm font-medium truncate">
                        {node.name}
                      </span>
                    </div>
                    <div className="text-xs text-gray-400 mt-1">
                      {node.type}
                      {node.capabilities.length > 0 && (
                        <> · {node.capabilities.join(', ')}</>
                      )}
                    </div>
                    {node.sessions && node.sessions > 0 && (
                      <div className="text-xs text-purple-400 mt-1">
                        {node.sessions} active session{node.sessions > 1 ? 's' : ''}
                      </div>
                    )}
                  </div>
                  
                  <button
                    onClick={() => onRemoveNode(node.id)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity ml-2 text-red-400 hover:text-red-300"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))
            )}
          </div>
        </section>
      </div>
    </div>
  )
}
