import { useMemo, useRef } from 'react'
import yaml from 'yaml'
import type { ParsedNode } from 'yaml'
import type { TOCNode, ConfigStructureResult } from '../types/config-toc'

const MAX_YAML_LENGTH = 500_000

type LineRange = {
  start: number
  end: number
}

const clampLine = (value: number, max: number) => {
  if (!Number.isFinite(value)) return 1
  return Math.max(1, Math.min(Math.floor(value), max))
}

const normaliseRange = (
  range: LineRange | null,
  fallbackStart: number,
  fallbackEnd: number,
  max: number
): LineRange => {
  const start = clampLine(range?.start ?? fallbackStart, max)
  const end = clampLine(range?.end ?? fallbackEnd, max)
  return {
    start,
    end: Math.max(start, end),
  }
}

const rangeFromNode = (
  node: ParsedNode | null | undefined,
  counter: yaml.LineCounter | undefined
): LineRange | null => {
  if (!node || !node.range || !counter) return null
  const [startIdx, endIdx] = node.range
  const start = counter.linePos(Math.max(0, startIdx)).line
  const end = counter.linePos(Math.max(startIdx, endIdx - 1)).line
  return { start, end }
}

const safePointer = (pointer: string | null | undefined) =>
  typeof pointer === 'string' && pointer.length > 0 ? pointer : '/'

interface BuildArrayParams {
  items: unknown[]
  basePath: (string | number)[]
  pointerPrefix: string
  idPrefix: string
  level: number
  labelForItem: (item: any, index: number) => string
  parentFallback: LineRange
}

export function useConfigStructure(
  yamlContent: string,
  shouldUpdate: boolean = true
): ConfigStructureResult {
  const lastResultRef = useRef<ConfigStructureResult>({
    nodes: [],
    success: true,
  })

  return useMemo(() => {
    if (!shouldUpdate) {
      return lastResultRef.current
    }

    if (!yamlContent.trim()) {
      const result: ConfigStructureResult = {
        nodes: [],
        success: false,
        error: 'Empty content',
      }
      lastResultRef.current = result
      return result
    }

    if (yamlContent.length > MAX_YAML_LENGTH) {
      const result: ConfigStructureResult = {
        nodes: [],
        success: false,
        error: 'Config content exceeds safe size limit',
      }
      lastResultRef.current = result
      return result
    }

    const yamlLines = yamlContent.split(/\r?\n/)
    const totalLines = Math.max(yamlLines.length, 1)

    try {
      const lineCounter = new yaml.LineCounter()
      const doc = yaml.parseDocument(yamlContent, {
        lineCounter,
        uniqueKeys: true,
      })

      if (doc.errors.length > 0) {
        const result: ConfigStructureResult = {
          nodes: [],
          success: false,
          error: doc.errors[0]?.message || 'Invalid YAML',
        }
        lastResultRef.current = result
        return result
      }

      const config = doc.toJS({})
      if (!config || typeof config !== 'object') {
        const result: ConfigStructureResult = {
          nodes: [],
          success: false,
          error: 'Invalid config format',
        }
        lastResultRef.current = result
        return result
      }

      const sections: TOCNode[] = []

      const docRange = normaliseRange(
        rangeFromNode(doc.contents as ParsedNode, lineCounter),
        1,
        totalLines,
        totalLines
      )

      const buildArrayChildren = ({
        items,
        basePath,
        pointerPrefix,
        idPrefix,
        level,
        labelForItem,
        parentFallback,
      }: BuildArrayParams) => {
        const arrayNode = doc.getIn(basePath, true) as ParsedNode | undefined
        const sectionRange = normaliseRange(
          rangeFromNode(arrayNode, lineCounter),
          parentFallback.start,
          parentFallback.end,
          totalLines
        )

        const children = items.map((item, index) => {
          const itemNode = doc.getIn([...basePath, index], true) as
            | ParsedNode
            | undefined
          const itemRange = normaliseRange(
            rangeFromNode(itemNode, lineCounter),
            sectionRange.start,
            sectionRange.end,
            totalLines
          )

          return {
            id: `${idPrefix}-${index}`,
            label: labelForItem(item, index),
            jsonPointer: safePointer(`${pointerPrefix}/${index}`),
            lineStart: itemRange.start,
            lineEnd: itemRange.end,
            level,
            isCollapsible: false,
          }
        })

        return {
          children,
          sectionRange,
        }
      }

      sections.push({
        id: 'overview',
        label: 'Overview',
        jsonPointer: '/',
        lineStart: 1,
        lineEnd: clampLine(Math.max(docRange.start + 2, 3), totalLines),
        level: 0,
        isCollapsible: false,
      })

      // Prompts ---------------------------------------------------------------
      const prompts = Array.isArray((config as any).prompts)
        ? (config as any).prompts
        : []
      if (prompts.length > 0) {
        const promptsNode = doc.get('prompts', true) as ParsedNode | undefined
        const promptsRange = normaliseRange(
          rangeFromNode(promptsNode, lineCounter),
          docRange.start,
          docRange.end,
          totalLines
        )

        const { children: promptChildren, sectionRange } = buildArrayChildren({
          items: prompts,
          basePath: ['prompts'],
          pointerPrefix: '/prompts',
          idPrefix: 'prompt',
          level: 1,
          labelForItem: (item, index) => item?.name || `Prompt ${index + 1}`,
          parentFallback: promptsRange,
        })

        sections.push({
          id: 'prompts',
          label: 'Prompts',
          jsonPointer: '/prompts',
          lineStart: sectionRange.start,
          lineEnd:
            promptChildren.length > 0
              ? Math.max(
                  promptChildren[promptChildren.length - 1].lineEnd,
                  promptsRange.end
                )
              : sectionRange.end,
          level: 0,
          isCollapsible: true,
          children: promptChildren,
        })
      }

      // RAG -------------------------------------------------------------------
      const rag = (config as any).rag
      if (rag && typeof rag === 'object') {
        const ragNode = doc.get('rag', true) as ParsedNode | undefined
        const ragRange = normaliseRange(
          rangeFromNode(ragNode, lineCounter),
          docRange.start,
          docRange.end,
          totalLines
        )

        const ragChildren: TOCNode[] = []

        const databases = Array.isArray(rag.databases) ? rag.databases : []
        if (databases.length > 0) {
          const { children: databaseChildren, sectionRange: databasesRange } =
            buildArrayChildren({
              items: databases,
              basePath: ['rag', 'databases'],
              pointerPrefix: '/rag/databases',
              idPrefix: 'database',
              level: 2,
              labelForItem: (item, index) =>
                item?.name || `Database ${index + 1}`,
              parentFallback: ragRange,
            })

          ragChildren.push({
            id: 'databases',
            label: 'Databases',
            jsonPointer: '/rag/databases',
            lineStart: databasesRange.start,
            lineEnd:
              databaseChildren.length > 0
                ? Math.max(
                    databaseChildren[databaseChildren.length - 1].lineEnd,
                    databasesRange.end
                  )
                : databasesRange.end,
            level: 1,
            isCollapsible: true,
            children: databaseChildren,
          })
        }

        const strategies = Array.isArray(rag.data_processing_strategies)
          ? rag.data_processing_strategies
          : []
        if (strategies.length > 0) {
          const { children: strategyChildren, sectionRange: strategiesRange } =
            buildArrayChildren({
              items: strategies,
              basePath: ['rag', 'data_processing_strategies'],
              pointerPrefix: '/rag/data_processing_strategies',
              idPrefix: 'strategy',
              level: 2,
              labelForItem: (item, index) =>
                item?.name || `Strategy ${index + 1}`,
              parentFallback: ragRange,
            })

          ragChildren.push({
            id: 'data_processing_strategies',
            label: 'Data processing strategies',
            jsonPointer: '/rag/data_processing_strategies',
            lineStart: strategiesRange.start,
            lineEnd:
              strategyChildren.length > 0
                ? Math.max(
                    strategyChildren[strategyChildren.length - 1].lineEnd,
                    strategiesRange.end
                  )
                : strategiesRange.end,
            level: 1,
            isCollapsible: true,
            children: strategyChildren,
          })
        }

        if (ragChildren.length > 0) {
          sections.push({
            id: 'rag',
            label: 'RAG',
            jsonPointer: '/rag',
            lineStart: ragRange.start,
            lineEnd: Math.max(
              ragChildren[ragChildren.length - 1].lineEnd,
              ragRange.end
            ),
            level: 0,
            isCollapsible: true,
            children: ragChildren,
          })
        } else {
          sections.push({
            id: 'rag',
            label: 'RAG',
            jsonPointer: '/rag',
            lineStart: ragRange.start,
            lineEnd: ragRange.end,
            level: 0,
            isCollapsible: false,
          })
        }
      }

      // Datasets --------------------------------------------------------------
      const datasets = Array.isArray((config as any).datasets)
        ? (config as any).datasets
        : []
      if (datasets.length > 0) {
        const datasetsNode = doc.get('datasets', true) as ParsedNode | undefined
        const datasetsRange = normaliseRange(
          rangeFromNode(datasetsNode, lineCounter),
          docRange.start,
          docRange.end,
          totalLines
        )

        const { children: datasetChildren, sectionRange } = buildArrayChildren({
          items: datasets,
          basePath: ['datasets'],
          pointerPrefix: '/datasets',
          idPrefix: 'dataset',
          level: 1,
          labelForItem: (item, index) => item?.name || `Dataset ${index + 1}`,
          parentFallback: datasetsRange,
        })

        sections.push({
          id: 'datasets',
          label: `Datasets${datasets.length > 0 ? ` (${datasets.length})` : ''}`,
          jsonPointer: '/datasets',
          lineStart: sectionRange.start,
          lineEnd:
            datasetChildren.length > 0
              ? Math.max(
                  datasetChildren[datasetChildren.length - 1].lineEnd,
                  datasetsRange.end
                )
              : sectionRange.end,
          level: 0,
          isCollapsible: true,
          children: datasetChildren,
        })
      }

      // Runtime ---------------------------------------------------------------
      const runtime = (config as any).runtime
      if (runtime && typeof runtime === 'object') {
        const runtimeNode = doc.get('runtime', true) as ParsedNode | undefined
        const runtimeRange = normaliseRange(
          rangeFromNode(runtimeNode, lineCounter),
          docRange.start,
          docRange.end,
          totalLines
        )

        const models = Array.isArray(runtime.models) ? runtime.models : []
        let runtimeLineEnd = runtimeRange.end
        let modelChildren: TOCNode[] | undefined

        if (models.length > 0) {
          const { children, sectionRange: modelsRange } = buildArrayChildren({
            items: models,
            basePath: ['runtime', 'models'],
            pointerPrefix: '/runtime/models',
            idPrefix: 'model',
            level: 1,
            labelForItem: (item, index) => item?.name || `Model ${index + 1}`,
            parentFallback: runtimeRange,
          })
          modelChildren = children
          runtimeLineEnd =
            children.length > 0
              ? Math.max(children[children.length - 1].lineEnd, modelsRange.end)
              : modelsRange.end
        }

        sections.push({
          id: 'runtime',
          label:
            models.length > 0 ? `Runtime (${models.length} models)` : 'Runtime',
          jsonPointer: '/runtime',
          lineStart: runtimeRange.start,
          lineEnd: runtimeLineEnd,
          level: 0,
          isCollapsible:
            Array.isArray(modelChildren) && modelChildren.length > 0,
          children: modelChildren,
        })
      }

      // MCP -------------------------------------------------------------------
      const mcp = (config as any).mcp
      if (mcp && typeof mcp === 'object') {
        const mcpNode = doc.get('mcp', true) as ParsedNode | undefined
        const mcpRange = normaliseRange(
          rangeFromNode(mcpNode, lineCounter),
          docRange.start,
          docRange.end,
          totalLines
        )
        const serverCount = Array.isArray(mcp.servers) ? mcp.servers.length : 0

        sections.push({
          id: 'mcp',
          label: serverCount > 0 ? `MCP (${serverCount} servers)` : 'MCP',
          jsonPointer: '/mcp',
          lineStart: mcpRange.start,
          lineEnd: mcpRange.end,
          level: 0,
          isCollapsible: false,
        })
      }

      const result: ConfigStructureResult = { nodes: sections, success: true }
      lastResultRef.current = result
      return result
    } catch (error) {
      console.error('Failed to parse config structure:', error)
      const result: ConfigStructureResult = {
        nodes: [],
        success: false,
        error: error instanceof Error ? error.message : 'Parse error',
      }
      lastResultRef.current = result
      return result
    }
  }, [yamlContent, shouldUpdate])
}
