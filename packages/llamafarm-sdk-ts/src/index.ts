/**
 * LlamaFarm TypeScript SDK — local AI infrastructure client.
 *
 * @example
 * ```ts
 * import { LlamaFarm } from 'llamafarm'
 *
 * const lf = new LlamaFarm()
 * const project = await lf.createProject('default', 'my-app', { model: 'Qwen/Qwen3-8B' })
 * const response = await project.chat('Hello!')
 * console.log(response.text)
 * ```
 *
 * @alpha
 * @packageDocumentation
 */

export { LlamaFarm } from './client.js'
export type {
  LlamaFarmOptions,
  ChatMessage,
  ChatResponse,
  ChatChoice,
  StreamChunk,
  Model,
  Project,
  ProjectConfig,
  Detection,
  Classification,
  DetectResponse,
  ClassifyResponse,
  FineTuneJob,
  CacheStats,
  RAGQueryRequest,
  RAGQueryResponse,
  RAGQueryResult,
} from './types.js'
export { LlamaFarmError, ConnectionError, NotFoundError, ValidationError, ServerError } from './errors.js'
